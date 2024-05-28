use core::intrinsics::size_of;

pub struct Loader {
    ptr   : *const char,
    offset: usize
}

impl Loader {
    pub fn new(ptr: *const char) -> Self {
        Self {
            ptr,
            offset: 0
        }
    }

    pub fn load< T: LoaderImpl >(&mut self, target: &mut T) {
        target.load(self);
    }
}

trait LoaderImpl {
    fn load(&mut self, loader: &mut Loader);
}

impl LoaderImpl for f32 {
    fn load(&mut self, loader: &mut Loader) {
        unsafe {
            *self = *(loader.ptr.add(loader.offset) as *const f32);
        }

        loader.offset += size_of::< f32 >()
    }
}

impl< T: LoaderImpl, const N: usize > LoaderImpl for [T; N] {
    fn load(&mut self, loader: &mut Loader) {
        for i in 0..N {
            self[ i ].load(loader);
        }
    }
}
