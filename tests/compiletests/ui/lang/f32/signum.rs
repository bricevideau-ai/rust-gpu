// Test that `signum` works.
// build-pass
// ignore-opencl1.2
// ignore-opencl2.0

use spirv_std::num_traits::Float;
use spirv_std::spirv;

#[spirv(fragment)]
pub fn main(i: f32, o: &mut f32) {
    *o = i.signum();
}
