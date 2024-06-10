use core::{
    mem  ::{ size_of, transmute_copy },
    ops  ::Deref                      ,
    slice::from_raw_parts_mut         ,
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

impl< T > LoadImpl for T {
    default fn load_impl< D: Deref< Target = [u8] > >(slice: &D, offset: &mut usize) -> Option< Self > {
        let mut ret        = MaybeUninit::uninit();
        let     ptr        = ret.as_mut_ptr() as *mut u8;
        let     len        = size_of::< T >();
        let     ret_slice  = unsafe { from_raw_parts_mut(ptr, len) };
        ret_slice.copy_from_slice(slice.get(*offset..*offset + len)?);
        *offset           += len;
        Some(unsafe { ret.assume_init() })
    }
}

impl LoadImpl for String {
    fn load_impl< D: Deref< Target = [u8] > >(slice: &D, offset: &mut usize) -> Option< Self > {
        let tail  = slice.get(*offset..)?;
        let pos   = tail.iter().position(|&x| x == 0)?;
        *offset  += pos + 1;
        std::str::from_utf8(&tail[ ..pos ]).ok().map(std::string::String::from)
    }
}

impl< T: LoadImpl, const N: usize > LoadImpl for [T; N] {
    fn load_impl< D: Deref< Target = [u8] > >(slice: &D, offset: &mut usize) -> Option< Self > {
        let mut ret: [MaybeUninit::< T >; N] = unsafe { MaybeUninit::uninit().assume_init() };

        for x in ret.iter_mut() {
            x.write(LoadImpl::load_impl(slice, offset)?);
        }

        Some(unsafe { transmute_copy(&ret) })
    }
}
