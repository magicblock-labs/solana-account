//! A memory-optimized, zero-copy account representation for Solana.
//!
//! This module provides `AccountSharedData`, an enum that manages Solana accounts either
//! in a memory-mapped region (`AccountBorrowed`) or on the heap (`AccountOwned`). The primary
//! goal is to maximize performance by directly manipulating account data in memory-mapped
//! files, avoiding deserialization overhead.
//!
//! ## Core Concepts
//!
//! - **`AccountBorrowed`**: A lightweight view into an account's data stored in a memory-mapped
//!   file. It uses raw pointers for direct field access and modification. This is the "zero-copy"
//!   part of the implementation.
//!
//! - **Copy-on-Write (CoW)**: To handle modifications safely (e.g., during a transaction that
//!   might fail), `AccountBorrowed` uses a shadow buffer. The first time a borrowed account is
//!   modified, its data is copied to an adjacent "shadow" memory region. All subsequent changes
//!   are applied to this shadow copy.
//!
//! - **`ShadowSwitch`**: An atomic counter that acts as a sequence lock. It tracks which buffer
//!   (primary or shadow) holds the most recent version of the account data. An even count
//!   points to the primary buffer, and an odd count points to the shadow buffer. This allows for
//!   safe, optimistic reads.
//!
//! - **`AccountOwned`**: A standard, heap-allocated representation of an account.
//!   An `AccountBorrowed` is promoted to `AccountOwned` if its data needs to grow
//!   beyond the capacity of its memory-mapped buffer.
//!
//! ## Memory Layout
//!
//! The efficiency of this module hinges on a precise and stable memory layout for serialized
//! accounts. Each account record consists of a small header, a primary data buffer, and an
//! identical shadow buffer for CoW operations.
//!
//! ```text
//! +---------------------------------------------------------------------------------------------+
//! | Section      | Field               | Size (bytes) | Offset | Description                    |
//! |--------------|---------------------|--------------|--------|--------------------------------|
//! |              | 1. Shadow Switch    | 4            | 0      | Atomic counter for CoW buffer. |
//! |     Meta     | 2. Shadow Offset    | 4            | 4      | Byte offset to the shadow copy.|
//! |--------------|---------------------|--------------|--------|--------------------------------|
//! |              | 3. Lamports         | 8            | 8      | Account balance.               |
//! |              | 4. Owner            | 32           | 16     | Public key of the owner.       |
//! |              | 5. Remote Slot      | 8            | 48     | Remote slot number.            |
//! |   Primary    | 6. Flags            | 4            | 56     | Bit-packed flags (e.g., exec). |
//! |    Buffer    | 7. Data Capacity    | 4            | 60     | Max capacity of account data.  |
//! |              | 8. Data Length      | 4            | 64     | Current length of data.        |
//! |              | 9. Data             | Variable     | 68     | Account data payload.          |
//! |              | 10. Padding         | Variable     | -      | Aligns buffer to 8 bytes.      |
//! |--------------|---------------------|--------------|--------|--------------------------------|
//! | Shadow Buf   | Fields 3-10 cloned  | eq primary   | -      | Shadow copy for modifications. |
//! +---------------------------------------------------------------------------------------------+
//! ```
//!
use std::{
    mem::{align_of, size_of},
    ops::{Deref, DerefMut, Div, Mul},
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    },
};

use solana_clock::Epoch;
use solana_pubkey::Pubkey;

use crate::AccountSharedData;

/// The byte offset to find the data length (`u32`) relative to the data pointer.
/// This is a performance optimization (a "dirty hack") that relies on the fixed
/// memory layout where `data_len` is stored immediately before `data`
///
/// NOTE: Any changes to the memory layout must be reflected here.
const RELATIVE_DATA_LEN_POINTER_OFFSET: isize = -4;
#[cfg(test)]
const RELATIVE_DATA_CAP_POINTER_OFFSET: isize = -8;

// --- Flag bit indices ---
pub(crate) const EXECUTABLE_FLAG_INDEX: u32 = 0;
pub(crate) const DELEGATED_FLAG_INDEX: u32 = 1;
pub(crate) const PRIVILEGED_FLAG_INDEX: u32 = 2;

// --- Memory Layout Offsets ---
// NOTE: These constants define the memory layout of a serialized account and must
// be kept in sync with the `serialize_to_mmap` and `deserialize_from_mmap` functions.
const LAMPORTS_OFFSET: usize = 0;
const OWNER_OFFSET: usize = LAMPORTS_OFFSET + size_of::<u64>();
const REMOTE_SLOT_OFFSET: usize = OWNER_OFFSET + size_of::<Pubkey>();
const FLAGS_OFFSET: usize = REMOTE_SLOT_OFFSET + size_of::<u64>();
const DATA_CAPACITY_OFFSET: usize = FLAGS_OFFSET + size_of::<u32>();
const DATA_LEN_OFFSET: usize = DATA_CAPACITY_OFFSET + size_of::<u32>();

/// A memory-optimized, "borrowed" view of a Solana account residing in a memory-mapped region.
///
/// This struct uses raw pointers to directly access and manipulate the serialized account data,
/// avoiding deserialization costs. It implements a Copy-on-Write (CoW) mechanism to handle
/// modifications: changes are written to a separate "shadow" buffer, ensuring the original
/// data remains untouched until the changes are committed.
#[derive(Clone, PartialEq, Eq)]
pub struct AccountBorrowed {
    /// An atomic counter that determines which buffer (primary or shadow) is active.
    /// Even values indicate the primary buffer, odd values indicate the shadow buffer.
    shadow_switch: ShadowSwitch,
    /// The byte offset to the shadow buffer. A positive value means the primary buffer is active,
    /// and a negative value means the shadow buffer is active. A value of `0` indicates
    /// that a CoW has already occurred and no further copies are needed.
    shadow_offset: isize,
    /// A raw pointer to the account's lamports (`u64`).
    pub(crate) lamports: *mut u64,
    /// A view into the account's data slice within the memory map.
    pub(crate) data: DataSlice,
    /// A raw pointer to the account's owner (`Pubkey`).
    pub(crate) owner: *mut Pubkey,
    /// A raw pointer to the remote slot number (`u64`).
    pub(crate) remote_slot: *mut u64,
    /// A flag indicating if the account's owner has been changed.
    pub owner_changed: bool,
    /// A flag indicating if any field in the account has been modified.
    pub is_dirty: bool,
    /// A wrapper around a raw pointer to the bit-packed flags field (`u32`).
    /// Flags include:
    /// - `0`: `executable`
    /// - `1`: `delegated`
    pub(crate) flags: BitFlags,
}

/// A wrapper for a raw pointer to a `u32` used for bit-packed flags.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct BitFlags(*mut u32);

impl From<AccountBorrowed> for AccountSharedData {
    fn from(value: AccountBorrowed) -> Self {
        Self::Borrowed(value)
    }
}

// SAFETY: `AccountBorrowed` is safe to send across threads because the underlying
// data is in a shared memory map. Synchronization for writes is handled by the CoW
// mechanism and the atomic `ShadowSwitch`. Concurrent writes to the same account
// are undefined behavior and must be prevented by higher-level locking.
unsafe impl Send for AccountBorrowed {}
// SAFETY: See `Send`. Access is synchronized via atomics for the switch, and writes
// are isolated via CoW. Concurrent writes are UB.
unsafe impl Sync for AccountBorrowed {}

impl AccountBorrowed {
    /// Performs a Copy-on-Write (CoW) by copying the active account buffer to its shadow buffer.
    ///
    /// After the copy, all internal pointers are updated to point to the new shadow buffer.
    /// This function is idempotent; it does nothing if a CoW has already occurred (i.e.,
    /// `shadow_offset` is 0).
    ///
    /// # Safety
    ///
    /// The `AccountBorrowed` instance must be properly initialized, pointing to a valid
    /// memory layout that includes a correctly sized and located shadow buffer at `shadow_offset`.
    pub unsafe fn cow(&mut self) {
        // If shadow_offset is 0, CoW has already happened.
        if self.shadow_offset == 0 {
            return;
        }
        self.is_dirty = true;

        // The source is the start of the current active buffer (lamports field).
        let src = self.lamports as *mut u8;
        // The destination is the shadow buffer, calculated using the offset.
        let dst = src.offset(self.shadow_offset);

        // Bulk copy the entire active buffer to the shadow buffer.
        dst.copy_from_nonoverlapping(src, self.shadow_offset.unsigned_abs());

        // Translate all internal pointers to point to the new shadow buffer.
        self.lamports = (self.lamports as *mut u8).offset(self.shadow_offset) as *mut u64;
        self.owner = (self.owner as *mut u8).offset(self.shadow_offset) as *mut Pubkey;
        self.remote_slot = (self.remote_slot as *mut u8).offset(self.shadow_offset) as *mut u64;
        self.flags.translate(self.shadow_offset);
        self.data.translate(self.shadow_offset);

        // Set shadow_offset to 0 to prevent subsequent CoWs.
        self.shadow_offset = 0;
    }

    /// Commits the changes made in the shadow buffer by incrementing the `shadow_switch` counter.
    ///
    /// This atomically makes the shadow buffer the new primary buffer for all subsequent readers.
    /// If no CoW occurred (`shadow_offset != 0`), this is a no-op.
    #[inline(always)]
    pub fn commit(&self) {
        // Only increment the switch if a CoW has actually happened.
        if self.shadow_offset == 0 {
            self.shadow_switch.increment();
        }
    }

    /// Checks if the owner of a serialized account matches any in the provided slice.
    ///
    /// This function performs a direct memory read without full deserialization.
    /// It returns `None` if the account has zero lamports.
    ///
    /// # Safety
    ///
    /// `memptr` must point to the beginning of a valid, initialized account record
    /// in memory, including its metadata header.
    pub unsafe fn any_owner_matches(memptr: *mut u8, others: &[Pubkey]) -> Option<usize> {
        // Determine the correct buffer to read from based on the shadow switch.
        let Deserialization {
            mut deserializer, ..
        } = AccountSharedData::init_deserialization(memptr);

        // An account with zero lamports is considered non-existent for this check.
        if deserializer.read_val::<u64>() == 0 {
            return None;
        }

        let owner = deserializer.read::<Pubkey>();
        others.iter().position(|o| *owner == *o)
    }

    /// Creates a sequence lock guard for safe, optimistic reads.
    ///
    /// A reader can `lock()` the account, perform its reads,
    /// and then use `AccountSeqLock::changed()` to verify
    /// that no writes occurred during the read.
    pub fn lock(&self) -> AccountSeqLock {
        let counter = self.shadow_switch.clone();
        let current = counter.counter();
        AccountSeqLock { counter, current }
    }

    /// Re-deserializes the account from its original memory location.
    ///
    /// This is used with the sequence lock pattern. If a write occurred during a read,
    /// the reader can call `reinit` to get a fresh, consistent view of the account.
    pub fn reinit(&self) -> AccountSharedData {
        let memptr = self.shadow_switch.inner();
        // SAFETY: The pointer from `shadow_switch` is guaranteed to point to the
        // start of the valid memory allocation for this account.
        unsafe { AccountSharedData::deserialize_from_mmap(memptr).into() }
    }
}

/// A standard, heap-allocated representation of a Solana account.
///
/// This is the "owned" variant of `AccountSharedData`, used when an account's data
/// needs to grow beyond its memory-mapped capacity or when a non-mmap-backed
/// account is needed.
#[derive(Clone, PartialEq, Eq, Default)]
pub struct AccountOwned {
    /// Lamports in the account.
    pub(crate) lamports: u64,
    /// Data held in this account, wrapped in an `Arc` for cheap cloning.
    pub(crate) data: Arc<Vec<u8>>,
    /// The program that owns this account.
    pub(crate) owner: Pubkey,
    /// Whether this account's data contains a loaded program.
    pub(crate) executable: bool,
    /// The epoch at which this account will next owe rent.
    pub(crate) rent_epoch: Epoch,
    /// Remote slot number.
    pub(crate) remote_slot: u64,
    /// A flag to track if the account has been delegated.
    pub(crate) delegated: bool,
    /// A flag to track if the account is privileged.
    /// It is used to determine if certain checks can be bypassed when this account is
    /// the signing feepayer of a transaction.
    pub(crate) privileged: bool,
}

impl Default for AccountSharedData {
    fn default() -> Self {
        Self::Owned(AccountOwned::default())
    }
}

/// A helper struct for deserializing an account from a raw memory pointer.
struct Deserialization {
    deserializer: BytesSerDe,
    shadow_switch: ShadowSwitch,
    shadow_offset: isize,
}

/// A helper for serializing/deserializing data from a raw byte pointer.
/// It acts as a cursor over the memory region.
struct BytesSerDe {
    ptr: *mut u8,
}

impl BytesSerDe {
    fn new(ptr: *mut u8) -> Self {
        Self { ptr }
    }

    /// Writes a value of type `T` to the current position and advances the pointer.
    #[inline(always)]
    unsafe fn write<T: Sized>(&mut self, val: T) {
        (self.ptr as *mut T).write_unaligned(val);
        self.ptr = self.ptr.add(size_of::<T>());
    }

    /// Writes a slice's length (`u32`) followed by its data, and advances the pointer.
    #[inline(always)]
    unsafe fn write_slice(&mut self, slice: &[u8]) {
        self.write(slice.len() as u32);
        self.ptr
            .copy_from_nonoverlapping(slice.as_ptr(), slice.len());
        self.ptr = self.ptr.add(slice.len());
    }

    /// Returns a mutable pointer to type `T` at the current position and advances the cursor.
    #[inline(always)]
    unsafe fn read<T: Sized>(&mut self) -> *mut T {
        let value = self.ptr as *mut T;
        self.ptr = self.ptr.add(size_of::<T>());
        value
    }

    /// Reads a `Copy`-able value of type `T` from the current position and advances the cursor.
    #[inline(always)]
    unsafe fn read_val<T: Sized + Copy>(&mut self) -> T {
        let value = (self.ptr as *mut T).read_unaligned();
        self.ptr = self.ptr.add(size_of::<T>());
        value
    }

    /// Reads data capacity and length, then returns a `DataSlice` pointing to the data.
    #[inline(always)]
    unsafe fn read_slice(&mut self) -> DataSlice {
        let cap = self.read_val::<u32>();
        let len = self.read_val::<u32>();
        let ptr = self.ptr;
        // Advance pointer past the data itself
        self.ptr = self.ptr.add(len as usize);
        DataSlice { ptr, len, cap }
    }

    /// Moves the internal pointer by a specified distance.
    #[inline(always)]
    unsafe fn jump(&mut self, distance: isize) {
        self.ptr = self.ptr.offset(distance);
    }
}

impl AccountSharedData {
    /// Size of the metadata header (shadow switch and offset).
    const SERIALIZED_META_SIZE: u32 = (size_of::<u32>() * 2) as u32;
    /// The required memory alignment for the account buffers.
    const SERIALIZATION_ALIGNMENT: u32 = align_of::<u64>() as u32;
    /// Total size of the fixed-size fields in a serialized account buffer.
    const ACCOUNT_STATIC_SIZE: u32 = (DATA_LEN_OFFSET + size_of::<u32>()) as u32;

    /// Calculates the total serialized size for an account, including its shadow buffer
    /// and metadata, aligned to a specified boundary.
    pub const fn serialized_size_aligned(data_len: u32, alignment: u32) -> u32 {
        const fn align_up(size: u32, align: u32) -> u32 {
            // Equivalent to `(size + align - 1) / align * align`
            size.div_ceil(align).wrapping_mul(align)
        }
        // Size of one buffer (static fields + data).
        let base_size = Self::ACCOUNT_STATIC_SIZE + data_len;

        // Align the buffer size up to ensure 8-byte alignment for fields.
        let aligned_buffer_size = align_up(base_size, Self::SERIALIZATION_ALIGNMENT);

        // Total size is metadata + two aligned buffers (primary and shadow).
        let total_size = Self::SERIALIZED_META_SIZE + aligned_buffer_size * 2;

        // Align the final total size to the user-requested alignment.
        align_up(total_size, alignment)
    }

    /// Calculates the optimal data capacity for a single buffer given a total allocation size.
    /// This function derives the usable space per buffer from the total size provided
    /// by `serialized_size_aligned`.
    #[inline]
    fn calculate_capacity(total_capacity: u32) -> u32 {
        total_capacity
            // 1. Subtract metadata size
            .saturating_sub(Self::SERIALIZED_META_SIZE)
            // 2. Halve it for one buffer (primary or shadow)
            .div(2)
            // 3. Floor to the nearest alignment boundary (integer division trick)
            .div(Self::SERIALIZATION_ALIGNMENT)
            .mul(Self::SERIALIZATION_ALIGNMENT)
    }

    /// Serializes an `AccountOwned` into a memory-mapped region.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// 1. `memptr` points to a valid, exclusively owned memory region.
    /// 2. The memory region is aligned to at least 8 bytes (`SERIALIZATION_ALIGNMENT`).
    /// 3. The region has sufficient `capacity`, which must be a value returned from
    ///    `serialized_size_aligned` to ensure the correct layout and space for the shadow buffer.
    ///    Mismatching capacity can lead to buffer overflows and undefined behavior.
    pub unsafe fn serialize_to_mmap(acc: &AccountOwned, memptr: *mut u8, capacity: u32) {
        // Calculate the actual size of a single buffer based on the total allocation size.
        let single_buffer_capacity = Self::calculate_capacity(capacity);

        let mut serializer = BytesSerDe::new(memptr);

        // --- Write Metadata Header ---
        // 1. Shadow Switch: Starts at 0, pointing to the first (primary) buffer.
        serializer.write(0u32);
        // 2. Shadow Offset: The size of a single buffer, which is the distance to the shadow copy.
        serializer.write(single_buffer_capacity);

        // --- Write Primary Buffer ---
        // 3. Lamports
        serializer.write(acc.lamports);
        // 4. Owner
        serializer.write(acc.owner);
        // 5. Remote Slot
        serializer.write(acc.remote_slot);
        // 6. Flags (bit-packed)
        let flags = (acc.executable as u32) << EXECUTABLE_FLAG_INDEX
            | (acc.delegated as u32) << DELEGATED_FLAG_INDEX
            | (acc.privileged as u32) << PRIVILEGED_FLAG_INDEX;
        serializer.write(flags);
        // 7. Data Capacity
        let data_capacity = single_buffer_capacity.saturating_sub(Self::ACCOUNT_STATIC_SIZE);
        serializer.write(data_capacity);
        // 8. Data Length and Payload
        serializer.write_slice(&acc.data);
    }

    /// Deserializes an `AccountBorrowed` from a memory-mapped region.
    ///
    /// # Safety
    ///
    /// `memptr` must be aligned to 8 bytes and point to a memory region containing a
    /// correctly serialized account, including metadata and a shadow buffer. The layout
    /// must match the one produced by `serialize_to_mmap`.
    pub unsafe fn deserialize_from_mmap(memptr: *mut u8) -> AccountBorrowed {
        let Deserialization {
            mut deserializer,
            shadow_switch,
            shadow_offset,
        } = Self::init_deserialization(memptr);

        // Read pointers to the fields in the active buffer.
        let lamports = deserializer.read::<u64>();
        let owner = deserializer.read::<Pubkey>();
        let remote_slot = deserializer.read::<u64>();
        let flags = deserializer.read::<u32>();
        let data = deserializer.read_slice();

        AccountBorrowed {
            lamports,
            owner,
            remote_slot,
            data,
            shadow_offset,
            shadow_switch,
            flags: BitFlags(flags),
            owner_changed: false,
            is_dirty: false,
        }
    }

    /// Initializes a deserializer by reading the metadata header and positioning
    /// the cursor at the start of the active buffer (primary or shadow).
    unsafe fn init_deserialization(memptr: *mut u8) -> Deserialization {
        let mut deserializer = BytesSerDe::new(memptr);

        // Read metadata header.
        let shadow_switch = ShadowSwitch::from(deserializer.read::<AtomicU32>());
        let mut shadow_offset = deserializer.read_val::<u32>() as isize;

        // Check if the shadow switch is odd. An odd value means the shadow buffer is active.
        let is_shadow_active = shadow_switch.counter() % 2 == 1;

        if is_shadow_active {
            // The shadow buffer is active. Jump the deserializer to the start of it.
            deserializer.jump(shadow_offset);
            // Invert the offset. Now, a CoW will copy from the shadow buffer *back*
            // to the primary buffer.
            shadow_offset = -shadow_offset;
        }

        Deserialization {
            deserializer,
            shadow_offset,
            shadow_switch,
        }
    }

    /// Returns `true` if the account has been modified.
    pub fn is_dirty(&self) -> bool {
        match self {
            Self::Borrowed(acc) => acc.is_dirty,
            // Owned accounts are heap-allocated copies, so they are always considered "dirty".
            Self::Owned(_) => true,
        }
    }
}

impl BitFlags {
    /// Checks if the bit at `index` is set.
    pub(crate) fn is_set(&self, index: u32) -> bool {
        // SAFETY: The pointer `self.0` is guaranteed to be valid for the lifetime
        // of the parent `AccountBorrowed`. The read is atomic-like in practice on
        // supported platforms for aligned u32.
        (unsafe { *self.0 } >> index) & 1 == 1
    }

    /// Sets or clears the bit at `index`.
    pub(crate) fn set(&mut self, val: bool, index: u32) {
        // SAFETY: The pointer `self.0` is valid. Modifications only occur after a CoW,
        // so we are writing to a private shadow buffer, preventing data races.
        unsafe {
            if val {
                *self.0 |= 1 << index;
            } else {
                *self.0 &= !(1 << index);
            }
        }
    }

    /// Translates the internal pointer by a given offset, used during CoW.
    fn translate(&mut self, offset: isize) {
        // SAFETY: This is only called during CoW, where `offset` is a valid
        // distance to the corresponding field in the shadow buffer.
        self.0 = unsafe { (self.0 as *mut u8).offset(offset) as *mut u32 };
    }
}

/// An atomic counter used as a sequence lock to select the active account buffer.
///
/// This struct wraps a pointer to an `AtomicU32`. Even counter values indicate the
/// primary buffer is active, while odd values indicate the shadow buffer is active.
/// This allows readers to optimistically read from a buffer and later check if a
/// write occurred concurrently.
#[repr(transparent)]
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct ShadowSwitch(*const AtomicU32);

impl From<*mut AtomicU32> for ShadowSwitch {
    fn from(value: *mut AtomicU32) -> Self {
        Self(value)
    }
}

impl ShadowSwitch {
    /// Atomically loads the current value of the counter.
    #[inline(always)]
    pub(crate) fn counter(&self) -> u32 {
        // SAFETY: `self.0` points to a valid `AtomicU32` within the mmap region.
        // Atomic operations ensure thread safety. `Acquire` ordering ensures that
        // any writes made before the counter was incremented are visible to this thread.
        unsafe { (*self.0).load(Ordering::Acquire) }
    }

    /// Atomically increments the counter.
    #[inline(always)]
    fn increment(&self) {
        // SAFETY: `self.0` points to a valid `AtomicU32`.
        // `Release` ordering ensures that all writes to the shadow buffer are completed
        // before the counter is incremented, making them visible to other threads
        // that see the new counter value.
        unsafe { (*self.0).fetch_add(1, Ordering::Release) };
    }

    /// Returns the raw pointer to the start of the account's memory allocation.
    #[inline(always)]
    fn inner(&self) -> *mut u8 {
        self.0 as *mut u8
    }
}

/// A "fat pointer" representing a view into the account's data in the mmap region.
#[derive(Clone, PartialEq, Eq)]
pub struct DataSlice {
    pub(crate) ptr: *mut u8,
    pub(crate) len: u32,
    pub(crate) cap: u32,
}

impl Deref for DataSlice {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        // SAFETY: `ptr` and `len` are initialized from the serialized account data.
        // The lifetime is tied to the parent `AccountBorrowed`, which holds the mmap view.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len as usize) }
    }
}

impl DerefMut for DataSlice {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: Similar to `Deref`. Mutable access is safe because modifications
        // only happen after a CoW, on a private shadow buffer.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len as usize) }
    }
}

impl DataSlice {
    /// Translates the internal data pointer by `offset`, used during CoW.
    fn translate(&mut self, offset: isize) {
        // SAFETY: This is only called from `AccountBorrowed::cow`, where `offset` is
        // a valid, in-bounds distance to the shadow buffer's data field.
        self.ptr = unsafe { self.ptr.offset(offset) }
    }

    /// Updates the length of the data slice, both in this struct and in the underlying memory.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `len` is less than or equal to the capacity (`self.cap`)
    /// and that the bytes from `0..len` in the backing store are properly initialized.
    pub(crate) unsafe fn set_len(&mut self, len: usize) {
        self.len = len as u32;
        // This is a performance hack that directly writes the new length to its known
        // location in memory, which is `RELATIVE_DATA_LEN_POINTER_OFFSET` bytes
        // before the data pointer itself. This avoids needing a separate pointer.
        (self.ptr.offset(RELATIVE_DATA_LEN_POINTER_OFFSET) as *mut u32).write_unaligned(len as u32);
    }
}

/// A guard used for optimistic reads, implementing a sequence lock pattern.
///
/// A reader creates this lock, reads data, and then calls `changed()` to see if
/// a writer modified the data during the read.
#[derive(Clone)]
pub struct AccountSeqLock {
    counter: ShadowSwitch,
    current: u32,
}

impl AccountSeqLock {
    /// Re-reads the current value of the atomic counter.
    pub fn relock(&mut self) {
        self.current = self.counter.counter();
    }

    /// Checks if the atomic counter has changed since the lock was created or last relocked.
    ///
    /// A change indicates a concurrent write occurred, and the data read may be inconsistent.
    pub fn changed(&self) -> bool {
        self.counter.counter() != self.current
    }
}

// SAFETY: `AccountSeqLock` is safe to send and sync across threads because it only
// contains a `ShadowSwitch`, which internally uses atomic operations for thread safety.
unsafe impl Send for AccountSeqLock {}
unsafe impl Sync for AccountSeqLock {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::Layout;

    const DATA_LEN: u32 = 16384;
    const ALIGNMENT: u32 = 256;
    const BUFFER_SIZE: u32 = AccountSharedData::serialized_size_aligned(DATA_LEN, ALIGNMENT);
    const LAMPORTS: u64 = 2342525;
    const SPACE: usize = 2525;
    const OWNER: Pubkey = Pubkey::new_from_array([5; 32]);

    struct BufferArea {
        ptr: *mut u8,
        layout: Layout,
    }

    impl BufferArea {
        fn new() -> Self {
            let layout = Layout::from_size_align(BUFFER_SIZE as usize, ALIGNMENT as usize).unwrap();
            let ptr = unsafe { std::alloc::alloc(layout) };
            assert!(!ptr.is_null(), "Allocation failed");
            Self { ptr, layout }
        }
    }

    impl Drop for BufferArea {
        fn drop(&mut self) {
            unsafe { std::alloc::dealloc(self.ptr, self.layout) };
        }
    }

    use crate::{accounts_equal, ReadableAccount, WritableAccount};
    macro_rules! setup {
        () => {{
            let buffer = BufferArea::new();
            let owned = account();
            let AccountSharedData::Owned(ref acc) = owned else {
                panic!("invalid AccountSharedData initialization");
            };
            unsafe { AccountSharedData::serialize_to_mmap(acc, buffer.ptr, BUFFER_SIZE) };
            let borrowed = unsafe { AccountSharedData::deserialize_from_mmap(buffer.ptr) };
            let borrowed = AccountSharedData::Borrowed(borrowed);
            (buffer, owned, borrowed)
        }};
    }

    fn account() -> AccountSharedData {
        AccountSharedData::new_rent_epoch(LAMPORTS, SPACE, &OWNER, Epoch::MAX)
    }

    #[test]
    fn test_serialized_size() {
        for align in [128, 256, 512] {
            for s in 128..16384 {
                let size = AccountSharedData::serialized_size_aligned(s, align);
                assert_eq!(
                    size % AccountSharedData::SERIALIZATION_ALIGNMENT,
                    0,
                    "Size should be aligned to internal alignment"
                );
                assert_eq!(
                    size % align,
                    0,
                    "Size should be aligned to requested alignment"
                );
                let single_buf_size = (size - AccountSharedData::SERIALIZED_META_SIZE) / 2;
                assert!(
                    single_buf_size >= AccountSharedData::ACCOUNT_STATIC_SIZE + s,
                    "Size must be large enough for data"
                );
            }
        }
    }

    #[test]
    fn test_serde() {
        let (_buffer, owned, borrowed) = setup!();
        assert!(
            accounts_equal(&borrowed, &owned),
            "deserialization of serialized account should result in the same account"
        );
    }

    #[test]
    fn test_shadow_switch() {
        let (buffer, _, mut borrowed) = setup!();

        let shadow_switch = buffer.ptr as *const u32;
        let offset = AccountSharedData::calculate_capacity(BUFFER_SIZE) as isize;
        // The start of the primary buffer's fields, after the meta header.
        let primary_buffer_start = unsafe {
            buffer
                .ptr
                .add(AccountSharedData::SERIALIZED_META_SIZE as usize)
        };

        assert_eq!(borrowed.lamports(), LAMPORTS);
        unsafe { assert_eq!(*shadow_switch, 0, "Initial shadow switch should be 0") };
        borrowed.set_lamports(42);
        assert_eq!(
            unsafe { *AccountSharedData::deserialize_from_mmap(buffer.ptr).lamports },
            LAMPORTS,
            "lamports change should have been written to shadow buffer, not primary"
        );
        assert_eq!(
            borrowed.lamports(),
            42,
            "expected lamports to be updated to new value in borrowed view"
        );
        if let AccountSharedData::Borrowed(bacc) = &borrowed {
            assert_eq!(
                bacc.lamports,
                // Lamports are at the start of the buffer. The shadow buffer starts at `offset` from the primary.
                unsafe { primary_buffer_start.offset(offset) as *mut u64 },
                "expected lamports pointer to be translated to shadow buffer"
            );
            bacc.commit();
        } else {
            panic!("expected AccountSharedDataBorrowed, found Owned version");
        }
        unsafe {
            assert_eq!(
                *shadow_switch, 1,
                "shadow_switch should have been incremented after commit"
            )
        }
        // After commit, deserializing again should read from the shadow buffer.
        let new_borrowed = unsafe { AccountSharedData::deserialize_from_mmap(buffer.ptr) };
        assert_eq!(
            unsafe { *new_borrowed.lamports },
            42,
            "New deserialization should see committed value"
        );
    }

    #[test]
    fn test_switch_back() {
        let (buffer, _, mut borrowed) = setup!();

        let shadow_switch = buffer.ptr as *const u32;
        let primary_buffer_start = unsafe {
            buffer
                .ptr
                .add(AccountSharedData::SERIALIZED_META_SIZE as usize)
        };

        // First modification: primary -> shadow
        unsafe { assert_eq!(*shadow_switch, 0) };
        borrowed.set_lamports(42);
        if let AccountSharedData::Borrowed(bacc) = &borrowed {
            bacc.commit();
        }
        borrowed = unsafe { AccountSharedData::deserialize_from_mmap(buffer.ptr).into() };
        assert_eq!(borrowed.lamports(), 42);
        unsafe { assert_eq!(*shadow_switch, 1, "Switch should be 1 after first commit") };

        // Second modification: shadow -> primary
        borrowed.set_lamports(43);
        if let AccountSharedData::Borrowed(bacc) = &borrowed {
            assert_eq!(
                bacc.lamports, primary_buffer_start as *mut u64,
                "expected lamports pointer to be translated back to primary buffer"
            );
            bacc.commit();
        }
        assert_eq!(borrowed.lamports(), 43, "Lamports should be 43 in the view");
        unsafe { assert_eq!(*shadow_switch, 2, "Switch should be 2 after second commit") };

        // After second commit, deserializing again should read from the primary buffer.
        let new_borrowed = unsafe { AccountSharedData::deserialize_from_mmap(buffer.ptr) };
        assert_eq!(
            unsafe { *new_borrowed.lamports },
            43,
            "New deserialization should see the latest value"
        );
    }

    #[test]
    fn test_upgrade_to_owned() {
        let (_buffer, owned, mut borrowed) = setup!();
        let msg = vec![42; DATA_LEN as usize];
        println!("LEN: {}", borrowed.data().len());
        let mut new_data = borrowed.data().to_vec();
        new_data.extend_from_slice(&msg);

        // Extend beyond capacity to trigger upgrade
        borrowed.resize(borrowed.data().len() + msg.len(), 0);
        borrowed.set_data_from_slice(&new_data);

        assert_eq!(
            borrowed.data(),
            new_data.as_slice(),
            "data should match extended slice"
        );
        assert!(
            matches!(borrowed, AccountSharedData::Owned(_)),
            "Borrowed account should have been upgraded to Owned upon slice extension"
        );
        assert_ne!(
            owned, borrowed,
            "two accounts should be different objects in memory"
        );
    }

    #[test]
    fn test_setting_data_from_slice() {
        let (_buffer, _, mut borrowed) = setup!();
        let old_len = borrowed.data().len();
        let msg = b"hello world?";
        borrowed.set_data_from_slice(msg);

        assert_eq!(borrowed.data(), msg, "account data should have changed");
        assert_ne!(
            borrowed.data().len(),
            old_len,
            "account data length should have changed"
        );
        assert_eq!(
            borrowed.data().len(),
            msg.len(),
            "account data len should be equal to that of slice"
        );

        let AccountSharedData::Borrowed(b) = borrowed else {
            panic!("Borrowed account should not upgrade if new data fits in capacity");
        };

        // Check that the length field in memory was updated correctly using the relative offset hack.
        assert_eq!(
            unsafe { *(b.data.ptr.offset(RELATIVE_DATA_LEN_POINTER_OFFSET) as *const u32) }
                as usize,
            msg.len(),
            "data length in memory must have been updated"
        );
    }

    #[test]
    fn test_account_shrinking() {
        let (_buffer, _, mut borrowed) = setup!();
        let original_cap = if let AccountSharedData::Borrowed(b) = &borrowed {
            b.data.cap
        } else {
            0
        };

        borrowed.resize(0, 0);
        assert_eq!(borrowed.data().len(), 0, "account data should be empty");

        let AccountSharedData::Borrowed(b) = borrowed else {
            panic!("Account should not upgrade when shrinking");
        };
        assert_eq!(
            unsafe { *(b.data.ptr.offset(RELATIVE_DATA_LEN_POINTER_OFFSET) as *const u32) },
            0,
            "data length in memory must be zero"
        );
        assert_eq!(
            unsafe { *(b.data.ptr.offset(RELATIVE_DATA_CAP_POINTER_OFFSET) as *const u32) },
            original_cap,
            "data capacity should not have been overwritten"
        );
    }

    #[test]
    fn test_account_growth() {
        let (_buffer, _, mut borrowed) = setup!();
        let original_cap = if let AccountSharedData::Borrowed(b) = &borrowed {
            b.data.cap
        } else {
            0
        };

        // Grow within capacity
        borrowed.resize(SPACE * 2, 0);
        assert_eq!(
            borrowed.data().len(),
            SPACE * 2,
            "account data should have grown"
        );

        if let AccountSharedData::Borrowed(ref b) = borrowed {
            assert_eq!(
                unsafe { *(b.data.ptr.offset(RELATIVE_DATA_LEN_POINTER_OFFSET) as *const u32) },
                SPACE as u32 * 2,
                "data length in memory must have been doubled"
            );
            assert_eq!(
                unsafe { *(b.data.ptr.offset(RELATIVE_DATA_CAP_POINTER_OFFSET) as *const u32) },
                original_cap,
                "data capacity should not have been overwritten"
            );
        } else {
            panic!("Should not have upgraded yet");
        };

        // Grow beyond capacity
        borrowed.resize((original_cap + 1) as usize, 0);
        let AccountSharedData::Owned(ref owned) = borrowed else {
            panic!("borrowed account should have converted to owned after large resize")
        };
        assert_eq!(
            owned.data.len(),
            (original_cap + 1) as usize,
            "data must have been resized beyond original capacity"
        );
    }

    #[test]
    fn test_owner_changed() {
        let (_buffer, _, mut borrowed) = setup!();
        borrowed.set_owner(Pubkey::default());

        let AccountSharedData::Borrowed(b) = borrowed else {
            panic!("Account should not have been upgraded after owner change");
        };
        assert_eq!(
            unsafe { *b.owner },
            Pubkey::default(),
            "account owner must have changed"
        );
        assert!(b.owner_changed, "owner_changed flag must have been set");
    }

    #[test]
    fn test_bitflags() {
        let (_buffer, _, mut borrowed) = setup!();
        assert!(
            !borrowed.delegated(),
            "account should not be delegated by default"
        );
        assert!(
            !borrowed.executable(),
            "account should not be executable by default"
        );
        assert!(
            !borrowed.privileged(),
            "account should not be privileged by default"
        );

        borrowed.set_executable(true);
        assert!(
            borrowed.executable(),
            "account should have become executable"
        );
        // Ensure other flags are unaffected
        assert!(!borrowed.delegated());

        borrowed.set_delegated(true);
        assert!(borrowed.delegated(), "account should have become delegated");
        // Ensure other flags are unaffected
        assert!(borrowed.executable());

        borrowed.set_privileged(true);
        assert!(
            borrowed.privileged(),
            "account should have become privileged"
        );
        // Ensure other flags are unaffected
        assert!(borrowed.executable());
        assert!(borrowed.delegated());

        borrowed.set_executable(false);
        assert!(!borrowed.executable(), "account should be non-executable");
        assert!(borrowed.delegated(), "delegated flag should remain set");
        assert!(borrowed.privileged(), "privileged flag should remain set");
    }
}
