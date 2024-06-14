use super::Pod;
use core::{
    mem  ::{ size_of, transmute_copy, MaybeUninit },
    ops  ::Deref                                   ,
    slice::from_raw_parts_mut
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

    pub fn load< T: LoadImpl >(&mut self) -> Option< T > {
        LoadImpl::load_impl(&self.slice, &mut self.offset)
    }

    pub fn end(&self) -> Option< () > {
        if self.offset == self.slice.len() {
            Some(())
        }
        else {
            None
        }
    }
}

pub trait LoadImpl {
    fn load_impl< D: Deref< Target = [u8] > >(slice: &D, offset: &mut usize) -> Option< Self > where Self: Sized;
}

impl< T: Pod > LoadImpl for T {
    fn load_impl< D: Deref< Target = [u8] > >(slice: &D, offset: &mut usize) -> Option< Self > {
        let mut ret   = MaybeUninit::uninit();
        let     ptr   = ret.as_mut_ptr() as *mut u8;
        let     len   = size_of::< T >();
        let     s     = unsafe { from_raw_parts_mut(ptr, len) };
        let     read  = slice.get(*offset..*offset + len)?;
        *offset      += len;
        s.copy_from_slice(read);
        Some(unsafe { ret.assume_init() })
    }
}

impl LoadImpl for String {
    fn load_impl< D: Deref< Target = [u8] > >(slice: &D, offset: &mut usize) -> Option< Self > {
        let len   = < u32 as LoadImpl >::load_impl(slice, offset)? as usize;
        let read  = slice.get(*offset..*offset + len)?;
        *offset  += len;
        std::str::from_utf8(read).ok().map(|x| x.to_string())
    }
}

impl< const N: usize > LoadImpl for [String; N] {
    fn load_impl< D: Deref< Target = [u8] > >(slice: &D, offset: &mut usize) -> Option< Self > {
        let mut ret: [MaybeUninit::< String >; N] = unsafe { MaybeUninit::uninit().assume_init() };

        for x in ret.iter_mut() {
            x.write(LoadImpl::load_impl(slice, offset)?);
        }

        Some(unsafe { transmute_copy(&ret) })
    }
}
