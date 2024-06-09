use core::{
    mem  ::size_of           ,
    ops  ::Deref             ,
    slice::from_raw_parts_mut,
    mem  ::MaybeUninit
};

pub struct Loader< D: Deref< Target = [u8] > > {
    slice : D,
    offset: usize
}

impl< D: Deref< Target = [u8] > > Loader< D > {
    pub fn from(slice: D) -> Self {
        Self {
            slice,
            offset: 0
        }
    }

    pub fn load_to< T >(&mut self, target: &mut T) -> Option< () > {
        target.load_to_impl(&self.slice, &mut self.offset)
    }

    pub fn load< T >(&mut self) -> Option< T > {
        let mut target = MaybeUninit::uninit();
        self.load_to(&mut target)?;
        Some(unsafe { target.assume_init() })
    }

    pub fn end(&self) -> bool {
        self.offset == self.slice.len()
    }
}

trait LoadToImpl {
    fn load_to_impl< D: Deref< Target = [u8] > >(&mut self, slice: &D, offset: &mut usize) -> Option< () >;
}

impl< T > LoadToImpl for T {
    fn load_to_impl< D: Deref< Target = [u8] > >(&mut self, slice: &D, offset: &mut usize) -> Option< () > {
        let to_ptr    = self as *mut T as *mut u8;
        let to_len    = size_of::< T >();
        let to_slice  = unsafe { from_raw_parts_mut(to_ptr, to_len) };
        let loaded    = slice.get(*offset..*offset + to_len)?;
        *offset      += to_len;
        to_slice.copy_from_slice(loaded);
        Some(())
    }
}

impl< T > LoadToImpl for [T] {
    fn load_to_impl< D: Deref< Target = [u8] > >(&mut self, slice: &D, offset: &mut usize) -> Option< () > {
        let to_ptr    = self.as_ptr() as *mut u8;
        let to_len    = size_of::< T >() * self.len();
        let to_slice  = unsafe { from_raw_parts_mut(to_ptr, to_len) };
        let loaded    = slice.get(*offset..*offset + to_len)?;
        *offset      += to_len;
        to_slice.copy_from_slice(loaded);
        Some(())
    }
}
