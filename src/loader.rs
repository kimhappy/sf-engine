pub struct Loader {
    ptr   : *const f32,
    offset: usize
}

impl Loader {
    pub fn new(ptr: *const f32) -> Self {
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
            *self = *loader.ptr.add(loader.offset);
        }

        loader.offset += 1;
    }
}

impl< const N: usize > LoaderImpl for [f32; N] {
    fn load(&mut self, loader: &mut Loader) {
        unsafe {
            let src = loader.ptr.add(loader.offset);
            let dst = self.as_mut_ptr();
            core::ptr::copy_nonoverlapping(src, dst, N);
        }

        loader.offset += N;
    }
}

impl< const N: usize, const M: usize > LoaderImpl for [[f32; N]; M] {
    fn load(&mut self, loader: &mut Loader) {
        unsafe {
            let src = loader.ptr.add(loader.offset);
            let dst = self.as_mut_ptr() as *mut f32;
            core::ptr::copy_nonoverlapping(src, dst, N * M);
        }

        loader.offset += N * M;
    }
}
