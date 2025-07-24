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

/// Offset (in bytes) from data field pointer where len field can be located
const DATA_LENGTH_POINTER_OFFSET: isize = -4;

pub(crate) const EXECUTABLE_FLAG_INDEX: u32 = 0;
pub(crate) const IS_DELEGATED_FLAG_INDEX: u32 = 1;

/// Memory optimized version of account shared data, which internally uses raw pointers to
/// manipulate database (memory mapped) directly. If the account is modified, the modification
/// triggers Copy on Write and all the changes are written to shadow buffer which can be discarded
/// if necessary (e.g. transaction fails).
#[derive(Clone, PartialEq, Eq)]
pub struct AccountBorrowed {
    /// shadow buffer switch counter
    shadow_switch: ShadowSwitch,
    /// number of bytes to jump (and the direction) at which shadow counterpart
    /// is located. This field is used to perform copy on write.
    shadow_offset: isize,
    /// lamports in the account
    pub(crate) lamports: *mut u64,
    /// data held in this account
    pub(crate) data: DataSlice,
    /// the program that owns this account. If executable, the program that loads this account.
    pub(crate) owner: *mut Pubkey,
    /// a boolean flag to track whether account has changed its owner
    pub owner_changed: bool,
    /// a boolean flag indicating whether any of the account's fields has been modified
    pub is_dirty: bool,
    /// various bitpacked flags
    /// 0. whether the account is executable
    /// 1. whether the account is delegated
    pub(crate) flags: BitFlags,
}

#[derive(Clone, PartialEq, Eq)]
pub(crate) struct BitFlags(*mut u32);

impl From<AccountBorrowed> for AccountSharedData {
    fn from(value: AccountBorrowed) -> Self {
        Self::Borrowed(value)
    }
}

unsafe impl Send for AccountBorrowed {}
// NOTE!: this variant of AccountSharedData should not be write accessed from different threads
// without some kind of synchronization like locks, otherwise it's a UB
unsafe impl Sync for AccountBorrowed {}

impl AccountBorrowed {
    /// Copies the modifiable section of account (serialized form) into shadow buffer
    ///
    /// # Safety
    /// This method should only be called on properly initialized AccountSharedDataBorrowed,
    /// pointing to correct memory layout of account record with shadow buffer at `shadow_offset`
    pub unsafe fn cow(&mut self) {
        // CoW has already happened and we are
        // currently operating on shadow buffer
        if self.shadow_offset == 0 {
            return;
        }
        self.is_dirty = true;
        // bulk copy the modifiable section to shadow buffer
        // we start with lamports address and copy the entire current buffer over
        let src = self.lamports as *mut u8;
        let dst = src.offset(self.shadow_offset);

        dst.copy_from_nonoverlapping(src, self.shadow_offset.unsigned_abs());

        // translate the pointers after copying the data over to shadow buffer

        // translate lamports
        self.lamports = (self.lamports as *mut u8).offset(self.shadow_offset) as *mut u64;
        // translate owner
        self.owner = (self.owner as *mut u8).offset(self.shadow_offset) as *mut Pubkey;
        // translate data
        self.data.translate(self.shadow_offset);
        // prevent further copy on writes
        self.shadow_offset = 0;
    }
    /// make current shadow buffer active by advancing the counter
    #[inline(always)]
    pub fn commit(&self) {
        // if cow didn't take place, then we never
        // touched this account, leave it be
        if self.shadow_offset != 0 {
            return;
        }
        self.shadow_switch.increment();
    }

    /// Performs direct memory jump to owner field and compares
    /// owner with `others`, no deserialization is performed
    /// _Note_: if account has 0 lamports, None is returned
    ///
    /// # Safety
    /// The memptr should point to initialized memory region where account is laid out
    /// along with shadow buffer
    pub unsafe fn any_owner_matches(memptr: *mut u8, others: &[Pubkey]) -> Option<usize> {
        // get the correct buffer to read the owner from
        let Deserialization {
            mut deserializer, ..
        } = AccountSharedData::init_deserialization(memptr);
        // check non-zero lamports
        (deserializer.read_val::<u64>() != 0).then_some(())?;
        let owner = deserializer.read::<Pubkey>();
        for (i, o) in others.iter().enumerate() {
            if *owner == *o {
                return Some(i);
            }
        }
        None
    }

    /// Remember the state of the account using Sequence Lock pattern
    pub fn lock(&self) -> AccountSeqLock {
        let counter = self.shadow_switch.clone();
        let current = counter.counter();
        AccountSeqLock { counter, current }
    }

    /// Re-read the account state from the database, this is used in Sequence Lock
    /// pattern, if the account has been modified during the read operation
    pub fn reinit(&mut self) {
        let memptr = self.shadow_switch.inner();
        // SAFETY: the invocation is safe as we are reusing the pointer from
        // shadow_switch which points to the beginning of the valid allocation
        *self = unsafe { AccountSharedData::deserialize_from_mmap(memptr) };
    }
}

/// Solana backward compatible version of account shared data AccountBorrowed will be promoted to
/// AccountOwned, if data field is grown
#[derive(Clone, PartialEq, Eq, Default)]
pub struct AccountOwned {
    /// lamports in the account
    pub(crate) lamports: u64,
    /// data held in this account
    pub(crate) data: Arc<Vec<u8>>,
    /// the program that owns this account. If executable, the program that loads this account.
    pub(crate) owner: Pubkey,
    /// this account's data contains a loaded program (and is now read-only)
    pub(crate) executable: bool,
    /// the epoch at which this account will next owe rent
    pub(crate) rent_epoch: Epoch,
    /// a boolean flag to track whether account has been delegated to the host ER node
    pub(crate) is_delegated: bool,
}

impl Default for AccountSharedData {
    fn default() -> Self {
        Self::Owned(AccountOwned::default())
    }
}

struct Deserialization {
    deserializer: BytesSerDe,
    shadow_switch: ShadowSwitch,
    shadow_offset: isize,
}

struct BytesSerDe {
    ptr: *mut u8,
}

impl BytesSerDe {
    fn new(ptr: *mut u8) -> Self {
        Self { ptr }
    }
    /// write value and advance the pointer
    #[inline(always)]
    unsafe fn write<T: Sized>(&mut self, val: T) {
        (self.ptr as *mut T).write(val);
        self.ptr = self.ptr.add(size_of::<T>());
    }
    /// write length of slice followed by slice itself, and advance the pointer
    #[inline(always)]
    unsafe fn write_slice(&mut self, slice: &[u8]) {
        self.write(slice.len() as u32);
        self.ptr
            .copy_from_nonoverlapping(slice.as_ptr(), slice.len());
        self.ptr = self.ptr.add(slice.len());
    }

    /// return pointer to specified type and advance the cursor by the type's size
    #[inline(always)]
    unsafe fn read<T: Sized>(&mut self) -> *mut T {
        let value = self.ptr as *mut T;
        self.ptr = self.ptr.add(size_of::<T>());
        value
    }
    /// read the actual Copy value from memory location and advance the cursor
    #[inline(always)]
    unsafe fn read_val<T: Sized + Copy>(&mut self) -> T {
        let value = (self.ptr as *mut T).read();
        self.ptr = self.ptr.add(size_of::<T>());
        value
    }
    /// read length of slice followed by slice pointer itself, advance the cursor
    #[inline(always)]
    unsafe fn read_slice(&mut self) -> DataSlice {
        let cap = self.read_val::<u32>();
        let len = self.read_val::<u32>();
        let ptr = self.ptr;
        DataSlice { ptr, len, cap }
    }

    #[inline(always)]
    unsafe fn jump(&mut self, distance: usize) {
        self.ptr = self.ptr.add(distance);
    }
}

/// # Account Record Layout
///
/// The account record is stored in memory-mapped format with the following structure:
/// ```text
/// +-----------------------------------------------------------------------------------------+
/// | Field                   | Size (bytes) | Offset | Description                           |
/// |-------------------------|--------------|--------|---------------------------------------|
/// | 1. Shadow Switch        | 4            | 0      | Counter for shadow copy switching     |
/// | 2. Shadow Offset        | 4            | 4      | Offset to shadow copy location        |
/// | 3. Lamports             | 8            | 8      | Account balance in lamports           |
/// | 4. Owner                | 32           | 16     | Public key of the account owner       |
/// | 5. Executable           | 4            | 48     | Whether  account is executable        |
/// | 6. Data Capacity        | 4            | 52     | Maximum capacity of account data      |
/// | 7. Data Length          | 4            | 56     | Current length of account data        |
/// | 8. Data                 | Variable     | 60     | Account data payload                  |
/// | 9. Slack space          | Variable     | -      | 8-byte aligned extra capacity         |
/// | 10. Shadow Copy         | Variable     | -      | Shadow copy of fields 3-8             |
/// +-----------------------------------------------------------------------------------------+
/// ```
impl AccountSharedData {
    const SERIALIZED_META_SIZE: u32 = size_of::<u64>() as u32;
    const SERIALIZATION_ALIGNMENT: u32 = align_of::<u64>() as u32;
    const ACCOUNT_STATIC_SIZE: u32 =
        // lamports
        (size_of::<u64>() +
        // owner
        size_of::<Pubkey>() +
        // data capacity
        size_of::<u32>() +
        // data length
        size_of::<u32>() +
        // executable and other flags
        size_of::<u32>()) as u32;

    /// Get the size of serialization of account along with shadow
    /// buffer which is aligned to the provided `alignment` value
    pub const fn serialized_size_aligned(data_len: u32, alignment: u32) -> u32 {
        // Helper function for alignment
        const fn align_up(size: u32, align: u32) -> u32 {
            size.div_ceil(align) * align
        }
        // Step 1: Calculate base size
        let base_size = Self::ACCOUNT_STATIC_SIZE + data_len;

        // Step 2: Align up to SERIALIZATION_ALIGNMENT and double it (for shadow buffer)
        let aligned = align_up(base_size, Self::SERIALIZATION_ALIGNMENT) * 2;

        // Step 3: Add meta size
        let with_meta = aligned + Self::SERIALIZED_META_SIZE;

        // Step 4: Align up to requested alignment
        align_up(with_meta, alignment)
    }
    /// Calculate optimal size which uses as much of the available
    /// space as possible, while maintaining the 8 byte alignment
    #[inline]
    fn calculate_capacity(capacity: u32) -> u32 {
        capacity
            .saturating_sub(Self::SERIALIZED_META_SIZE)
            .div(2)
            .div(Self::SERIALIZATION_ALIGNMENT)
            .mul(Self::SERIALIZATION_ALIGNMENT)
    }

    /// # Safety
    /// memptr should point to valid and exclusively owned memory region, memory should be properly
    /// aligned to 8 bytes, memory should have enough capacity to fit account' data and static
    /// fields (use [Self::serialized_size_aligned]) + 8 bytes of metadata, capacity must be
    /// constructed through a call to the [Self::serialized_size_aligned], otherwise it might result
    /// in UB caused by misaligned layout of data
    pub unsafe fn serialize_to_mmap(acc: &AccountOwned, memptr: *mut u8, mut capacity: u32) {
        // figure out optimal aligned capacity
        capacity = Self::calculate_capacity(capacity);

        let mut serializer = BytesSerDe::new(memptr);
        // write shadow buffer switch counter, we start with 0 for the first buffer
        // and then keep incrementing it on each modification, with even values
        // indicating first buffer and odd values indicating adjacent second one
        serializer.write(0u32);
        // write shadow offset, it is equal to the
        // size of serialized data (with padding)
        serializer.write(capacity);
        // write 8 bytes for lamports
        serializer.write(acc.lamports);
        // write 32 bytes for owner
        serializer.write(acc.owner);
        // write various flags into next 32 bits (for alignment purposes),
        // bit 0 is "executable" flag
        // bit 1 is "is_delegated" flag
        // also the remaining upper 30 bits can bit used for various future extensions
        let flags = acc.executable as u32 | ((acc.is_delegated as u32) << 1);
        serializer.write(flags);
        // write the capacity allocated for the data field
        serializer.write(capacity.saturating_sub(Self::ACCOUNT_STATIC_SIZE));
        // finally write binary data
        serializer.write_slice(&acc.data);
    }

    /// # Safety
    /// memptr should be aligned to 8 bytes and contain properly serialized account data
    /// along with correcly reserved shadow buffer for copy on write operations
    pub unsafe fn deserialize_from_mmap(memptr: *mut u8) -> AccountBorrowed {
        let Deserialization {
            mut deserializer,
            shadow_switch,
            shadow_offset,
        } = Self::init_deserialization(memptr);
        // read 8 bytes for lamports
        let lamports = deserializer.read::<u64>();
        // read 32 bytes for owner
        let owner = deserializer.read::<Pubkey>();
        // read the boolean flags
        let flags = deserializer.read::<u32>();
        // read the data slice
        let data = deserializer.read_slice();
        AccountBorrowed {
            lamports,
            owner,
            data,
            shadow_offset,
            shadow_switch,
            owner_changed: false,
            is_dirty: false,
            flags: BitFlags(flags),
        }
    }

    unsafe fn init_deserialization(memptr: *mut u8) -> Deserialization {
        let mut deserializer = BytesSerDe::new(memptr);
        let shadow_switch = ShadowSwitch::from(deserializer.read::<AtomicU32>());
        let is_odd = shadow_switch.counter() % 2 == 1;
        let mut shadow_offset = deserializer.read_val::<u32>() as isize;

        // check shadow switch to determine which buffer to use
        //
        // use data from second buffer for odd value (or keep reading from first otherwise)
        if is_odd {
            // jump to second buffer to start deserializing from there
            deserializer.jump(shadow_offset as usize);
            // set shadow_offset to negative value, so
            // it will jump back to first buffer upon CoW
            shadow_offset = -shadow_offset;
        }
        Deserialization {
            deserializer,
            shadow_offset,
            shadow_switch,
        }
    }

    /// Whether account has been modified by a transaction
    pub fn is_dirty(&self) -> bool {
        match self {
            Self::Borrowed(acc) => acc.is_dirty,
            // owned accounts are always modified
            Self::Owned(_) => true,
        }
    }
}

impl BitFlags {
    pub(crate) fn is_set(&self, index: u32) -> bool {
        (unsafe { *self.0 } >> index) == 1
    }

    pub(crate) fn set(&self, val: bool, index: u32) {
        unsafe {
            if val {
                *self.0 |= 1 << index;
            } else {
                *self.0 &= !(1 << index);
            }
        }
    }
}

/// SeqLock kind of atomic counter to track the buffer containing latest
/// version of account buffer: even numbers point to first buffer, while
/// odd ones point to second one Readers can optimistically read pointed
/// buffer and if the counter value doesn't change at the end of read,
/// then they can proceed with read data, otherwise they should retry, as
/// the changed pointer indicates that the buffer has been changed
#[repr(transparent)]
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct ShadowSwitch(*const AtomicU32);

impl From<*mut AtomicU32> for ShadowSwitch {
    fn from(value: *mut AtomicU32) -> Self {
        Self(value)
    }
}

impl ShadowSwitch {
    #[inline(always)]
    pub(crate) fn counter(&self) -> u32 {
        unsafe { &*self.0 }.load(Ordering::Relaxed)
    }

    #[inline(always)]
    fn increment(&self) {
        unsafe { &*self.0 }.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    fn inner(&self) -> *mut u8 {
        self.0 as *mut u8
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct DataSlice {
    pub(crate) ptr: *mut u8,
    pub(crate) len: u32,
    pub(crate) cap: u32,
}

impl Deref for DataSlice {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len as usize) }
    }
}

impl DerefMut for DataSlice {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len as usize) }
    }
}

impl DataSlice {
    fn translate(&mut self, offset: isize) {
        self.ptr = unsafe { self.ptr.offset(offset) }
    }

    // # Safety
    // the caller must ensure that backing store does indeed have enough capacity
    // indicated by len and that the data before ..len is initialized
    pub(crate) unsafe fn set_len(&mut self, len: usize) {
        self.len = len as u32;
        // dirty hack: len u32 is guaranteed to be located 4 bytes before the
        // data pointer, we jump back to it and modify
        (self.ptr.offset(DATA_LENGTH_POINTER_OFFSET) as *mut u32).write(len as u32);
    }
}

pub struct AccountSeqLock {
    counter: ShadowSwitch,
    current: u32,
}

impl AccountSeqLock {
    pub fn relock(&mut self) {
        self.current = self.counter.counter();
    }

    pub fn changed(&self) -> bool {
        self.counter.counter() != self.current
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::Layout;

    const BUFFER_SIZE: u32 = AccountSharedData::serialized_size_aligned(16384, 256);
    const LAMPORTS: u64 = 2342525;
    const SPACE: usize = 2525;
    const OWNER: Pubkey = Pubkey::new_from_array([5; 32]);

    struct BufferArea {
        ptr: *mut u8,
        layout: Layout,
    }

    impl BufferArea {
        fn new() -> Self {
            let layout = Layout::from_size_align(BUFFER_SIZE as usize, 256).unwrap();
            let ptr = unsafe { std::alloc::alloc(layout) };
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
                assert_eq!(size % AccountSharedData::SERIALIZATION_ALIGNMENT, 0);
                assert!(size >= s * 2 + AccountSharedData::SERIALIZED_META_SIZE);
            }
        }
    }

    #[test]
    fn test_serde() {
        let (_, owned, borrowed) = setup!();
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

        assert_eq!(borrowed.lamports(), LAMPORTS);
        unsafe { assert_eq!(*shadow_switch, 0) };
        borrowed.set_lamports(42);
        assert_eq!(
            unsafe { *AccountSharedData::deserialize_from_mmap(buffer.ptr).lamports },
            LAMPORTS,
            "lamports change should have been written to shadow buffer"
        );
        assert_eq!(
            borrowed.lamports(),
            42,
            "expected lamports to updated to new value"
        );
        if let AccountSharedData::Borrowed(bacc) = borrowed {
            assert_eq!(
                bacc.lamports,
                unsafe { buffer.ptr.offset(8 + offset) as *mut u64 },
                "expected lamports pointer to be translated to shadow buffer"
            );
            bacc.commit();
        } else {
            panic!("expected AccountSharedDataBorrowed, found Owned version");
        }
        unsafe {
            assert_eq!(
                *shadow_switch, 1,
                "shadow_switch should have been incrmented"
            )
        }
    }

    #[test]
    fn test_switch_back() {
        let (buffer, _, mut borrowed) = setup!();

        let shadow_switch = buffer.ptr as *const u32;

        unsafe { assert_eq!(*shadow_switch, 0) };
        borrowed.set_lamports(42);
        if let AccountSharedData::Borrowed(bacc) = borrowed {
            bacc.commit();
        }
        borrowed = unsafe { AccountSharedData::deserialize_from_mmap(buffer.ptr).into() };
        assert_eq!(borrowed.lamports(), 42);
        unsafe { assert_eq!(*shadow_switch, 1) };
        borrowed.set_lamports(43);
        if let AccountSharedData::Borrowed(bacc) = borrowed {
            assert_eq!(
                bacc.lamports,
                unsafe { buffer.ptr.offset(8) as *mut u64 },
                "expected lamports pointer to be translated to shadow(first) buffer"
            );
            bacc.commit();
        }
        // shadow_switch should have been incremented
        // and now points to first buffer again
        unsafe { assert_eq!(*shadow_switch % 2, 0) };
    }

    #[test]
    fn test_upgrade_to_owned() {
        let (_, owned, mut borrowed) = setup!();
        let len = borrowed.data().len();
        let msg = b"hello world?";
        borrowed.extend_from_slice(msg);
        assert_eq!(
            &borrowed.data()[len..],
            msg,
            "message should have been extended with new slice"
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
        let (_, _, mut borrowed) = setup!();
        let len = borrowed.data().len();
        let msg = b"hello world?";
        borrowed.set_data_from_slice(msg);
        assert_eq!(borrowed.data(), msg, "account data should have changed");
        assert_ne!(
            borrowed.data().len(),
            len,
            "account data length should have decreased"
        );
        assert_eq!(
            borrowed.data().len(),
            msg.len(),
            "account data len should be equal to that of slice"
        );
        let AccountSharedData::Borrowed(borrowed) = borrowed else {
            panic!("Borrowed account should not have been upgraded to Owned when slice length is less than the original");
        };
        assert_eq!(
            unsafe { *(borrowed.data.ptr.offset(-4) as *mut u32) } as usize,
            msg.len(),
            "data must have been shrunk"
        );
    }

    #[test]
    fn test_account_shrinking() {
        let (_, _, mut borrowed) = setup!();
        borrowed.resize(0, 0);
        assert_eq!(
            borrowed.data(),
            b"",
            "account data should have been truncated"
        );
        assert_eq!(
            borrowed.data().len(),
            0,
            "account data should have been truncated"
        );
        let AccountSharedData::Borrowed(borrowed) = borrowed else {
            panic!("Borrowed account should not have been upgraded to Owned when slice length is less than the original");
        };
        assert_eq!(
            unsafe { *(borrowed.data.ptr.offset(-4) as *mut u32) },
            0,
            "data must have been truncated"
        );
        assert_eq!(
            unsafe { *(borrowed.data.ptr.offset(-8) as *mut u32) },
            AccountSharedData::calculate_capacity(BUFFER_SIZE)
                - AccountSharedData::ACCOUNT_STATIC_SIZE,
            "data capacity should not have been overwritten"
        );
    }

    #[test]
    fn test_account_growth() {
        let (_, _, mut borrowed) = setup!();
        borrowed.resize(SPACE * 2, 0);
        assert_eq!(
            borrowed.data().len(),
            SPACE * 2,
            "account data should have grown"
        );
        if let AccountSharedData::Borrowed(ref borrowed) = borrowed {
            assert_eq!(
                unsafe { *(borrowed.data.ptr.offset(-4) as *mut u32) },
                SPACE as u32 * 2,
                "data must have been doubled in size"
            );
            assert_eq!(
                unsafe { *(borrowed.data.ptr.offset(-8) as *mut u32) },
                AccountSharedData::calculate_capacity(BUFFER_SIZE)
                    - AccountSharedData::ACCOUNT_STATIC_SIZE,
                "data capacity should not have been overwritten"
            );
        };
        borrowed.resize(BUFFER_SIZE as usize, 0);
        let AccountSharedData::Owned(ref owned) = borrowed else {
            panic!("borrowed account should have been converted to owned after large resize")
        };
        assert_eq!(
            owned.data.len(),
            BUFFER_SIZE as usize,
            "data must have been resized to BUFFER_SIZE"
        );
    }

    #[test]
    fn test_owner_changed() {
        let (_, _, mut borrowed) = setup!();
        borrowed.set_owner(Pubkey::default());
        let AccountSharedData::Borrowed(borrowed) = borrowed else {
            panic!("Borrowed account should not have been upgraded to Owned after owner change");
        };
        assert_eq!(
            unsafe { *borrowed.owner },
            Pubkey::default(),
            "account owner must have changed"
        );
        assert!(
            borrowed.owner_changed,
            "owner_changed flag must have been set"
        );
    }

    #[test]
    fn test_bitflags() {
        let (_, _, mut borrowed) = setup!();
        assert!(
            !borrowed.is_delegated(),
            "account should not be delegated by default"
        );
        assert!(
            !borrowed.executable(),
            "account should not be executable by default"
        );
        borrowed.set_executable(true);
        assert!(
            borrowed.executable(),
            "account should have become executable after change"
        );
        borrowed.set_delegated(true);
        assert!(
            borrowed.is_delegated(),
            "account should have become delegated after change"
        );
    }
}
