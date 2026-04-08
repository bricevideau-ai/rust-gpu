// build-pass
// ignore-opencl1.2
// ignore-opencl2.0

use spirv_std::spirv;

#[spirv(fragment)]
pub fn main() {
    let vector = glam::BVec2::new(false, true);
    assert!(spirv_std::arch::any(vector));
}
