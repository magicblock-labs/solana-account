use std::{
    ops::{Deref, DerefMut},
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    },
};

use solana_clock::Epoch;
use solana_pubkey::Pubkey;

use crate::AccountSharedData;

#[derive(Clone, PartialEq, Eq)]
pub struct AccountSharedDataBorrowed {
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
    /// this account's data contains a loaded program (and is now read-only)
    pub(crate) executable: bool, // we don't use pointer as this field is pretty much static once set (on chain)
}

impl AccountSharedDataBorrowed {
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
    // make current shadow buffer active by advancing the counter
    #[inline(always)]
    pub fn commit(self) {
        self.shadow_switch.increment();
    }
}

#[derive(Clone, PartialEq, Eq, Default)]
pub struct AccountSharedDataOwned {
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
}

impl Default for AccountSharedData {
    fn default() -> Self {
        Self::Owned(AccountSharedDataOwned::default())
    }
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

    fn skip(&mut self, step: usize) {
        self.ptr = unsafe { self.ptr.add(step) };
    }

    /// return pointer to specified type and advance the cursor by the type's size
    #[inline(always)]
    unsafe fn read<T: Sized>(&mut self) -> *mut T {
        let value = self.ptr as *mut T;
        self.ptr = self.ptr.add(size_of::<T>());
        value
    }
    /// read the actuall value Copy from memory location and advance the cursor
    unsafe fn read_val<T: Sized + Copy>(&mut self) -> T {
        let value = (self.ptr as *mut T).read();
        self.ptr = self.ptr.add(size_of::<T>());
        value
    }
    /// read length of slice followed by slice pointer itself, advance the cursor
    #[inline(always)]
    unsafe fn read_slice(&mut self) -> DataSlice {
        let len = self.read_val::<u32>() as usize;
        let ptr = self.ptr;
        DataSlice { ptr, len }
    }
}

// Implementation related to custom memory mapped AccountsDB
/// Account Record Layout
/// 1. | shadow switch counter: 4 |
/// 2. | shadow offset: 4         |
/// 3. | lamports: 8              |
/// 4. | owner: 32                |
/// 5. | executable: 4            |
/// 6. | datalen: 4               |
/// 7. | data: var                |
/// 8. | 8 byte align padding     |
/// 9. | shadow copy of 3-7       |
impl AccountSharedData {
    const SERIALIZATION_ALIGNMENT: usize = size_of::<u64>();
    const ACCOUNT_STATIC_SIZE: usize =
        // lamports
        size_of::<u64>() +
        // owner
        size_of::<Pubkey>() +
        // data length
        size_of::<u32>() +
        // executable and other flags
        size_of::<u32>();

    /// Get the size of serialization with extra padding to reach SERIALIZATION_ALIGNMENT
    pub fn serialized_size_aligned(data_len: usize) -> usize {
        let size = Self::ACCOUNT_STATIC_SIZE + data_len;
        let extra = size % Self::SERIALIZATION_ALIGNMENT;
        size + (Self::SERIALIZATION_ALIGNMENT - extra) * (extra != 0) as usize
    }

    /// # Safety
    /// memptr should point to valid and exclusively owned memory region, memory should be
    /// properly aligned to 8 bytes, memory should have enough capacity to fit account' data and
    /// static fields (use [Self::serialized_size_aligned]) + 8 bytes of metadata
    pub unsafe fn serialize_to_mmap(acc: &AccountSharedDataOwned, memptr: *mut u8) {
        let size = Self::serialized_size_aligned(acc.data.len()) as u32;
        println!("size: {size}");
        let mut serializer = BytesSerDe::new(memptr);
        // write shadow buffer switch counter, we start with 0 for first buffer
        // and then keep incrementing it on each modification, with even values
        // indicating first buffer and odd values indicating adjacent second one
        serializer.write(0_u32);
        // write shadow offset, it is equal to the
        // size of serialized data (with padding)
        serializer.write(size);
        // write 8 bytes for lamports
        serializer.write(acc.lamports);
        // write 32 bytes for owner
        serializer.write(acc.owner);
        // write executable 32 bytes (for alignment purposes), also
        // upper 31 bits can bit used for various future extensions
        serializer.write(acc.executable as u32);
        // finally write data binary data
        serializer.write_slice(&acc.data);
    }

    /// # Safety
    /// memptr should be properly aligned to 8 bytes and contain properly serialized account data
    /// along with correcly reserved shadow buffer for copy on write operations
    pub unsafe fn deserialize_from_mmap(memptr: *mut u8) -> Self {
        let mut deserializer = BytesSerDe::new(memptr);
        let shadow_switch = ShadowSwitch::from(deserializer.read::<AtomicU32>());
        let is_odd = shadow_switch.counter() % 2 == 1;
        let mut shadow_offset = deserializer.read_val::<u32>() as isize;

        // check shadow switch to determine which buffer to use
        //
        // use data from second buffer for odd value (or keep reading from first otherwise)
        if is_odd {
            // jump to second buffer to start deserializing from there
            deserializer = BytesSerDe::new(memptr.offset(shadow_offset));
            // set shadow_offset to negative value, will jump back to first buffer upon CoW
            shadow_offset = -shadow_offset;
        }
        // read 8 bytes for lamports
        let lamports = deserializer.read::<u64>();
        // read 32 bytes for owner
        let owner = deserializer.read::<Pubkey>();
        // skip 3 bytes of padding before boolean flag
        deserializer.skip(3);
        // read a single byte for boolean executable flag
        let executable = *deserializer.read::<bool>();
        // read the slice b
        let data = deserializer.read_slice();
        let account = AccountSharedDataBorrowed {
            lamports,
            owner,
            data,
            executable,
            shadow_offset,
            shadow_switch,
        };
        Self::Borrowed(account)
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
    fn counter(&self) -> u32 {
        unsafe { &*self.0 }.load(Ordering::Relaxed)
    }

    #[inline(always)]
    fn increment(&self) {
        unsafe { &*self.0 }.fetch_add(1, Ordering::Relaxed);
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct DataSlice {
    pub(crate) ptr: *mut u8,
    pub(crate) len: usize,
}

impl Deref for DataSlice {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl DerefMut for DataSlice {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl DataSlice {
    fn translate(&mut self, offset: isize) {
        self.ptr = unsafe { self.ptr.offset(offset) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::Layout;

    const BUFFER_SIZE: usize = 2252525;
    const LAMPORTS: u64 = 23425252;
    const SPACE: usize = 2525;
    const OWNER: Pubkey = Pubkey::new_from_array([5; 32]);

    struct BufferArea {
        ptr: *mut u8,
        layout: Layout,
    }

    impl BufferArea {
        fn new() -> Self {
            let layout = Layout::from_size_align(BUFFER_SIZE, 256).unwrap();
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

    fn account() -> AccountSharedData {
        AccountSharedData::new_rent_epoch(LAMPORTS, SPACE, &OWNER, Epoch::MAX)
    }
    #[test]
    fn test_serde() {
        let buffer = BufferArea::new();

        let owned = account();
        let AccountSharedData::Owned(ref acc) = owned else {
            panic!("invalid AccountSharedData initialization");
        };
        unsafe { AccountSharedData::serialize_to_mmap(acc, buffer.ptr) };
        let borrowed = unsafe { AccountSharedData::deserialize_from_mmap(buffer.ptr) };
        println!("OWNED: {owned:?}");
        println!("BORROWED: {borrowed:?}");
        assert!(accounts_equal(&borrowed, &owned));
    }

    #[test]
    fn test_shadow_switch() {
        let buffer = BufferArea::new();

        let owned = account();
        let AccountSharedData::Owned(ref acc) = owned else {
            panic!("invalid AccountSharedData initialization");
        };
        unsafe { AccountSharedData::serialize_to_mmap(acc, buffer.ptr) };
        let mut borrowed = unsafe { AccountSharedData::deserialize_from_mmap(buffer.ptr) };

        let shadow_switch = buffer.ptr as *const u32;
        let offset = AccountSharedData::serialized_size_aligned(borrowed.data().len()) as isize;

        assert_eq!(borrowed.lamports(), LAMPORTS);
        unsafe { assert_eq!(*shadow_switch, 0) };
        borrowed.set_lamports(42);
        assert_eq!(borrowed.lamports(), 42);
        if let AccountSharedData::Borrowed(bacc) = borrowed {
            assert_eq!(bacc.lamports, unsafe {
                buffer.ptr.offset(8 + offset) as *mut u64
            });
            bacc.commit();
        } else {
            panic!("expected AccountSharedDataBorrowed, found Owned version");
        }
        unsafe { assert_eq!(*shadow_switch, 1) }
    }
}
