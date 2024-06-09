use core::intrinsics::expf32;

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + unsafe { expf32(-x) })
}

pub fn tanh(x: f32) -> f32 {
    let e2x = unsafe { expf32(2.0 * x) };
    (e2x - 1.0) / (e2x + 1.0)
}
