// Tests that the invariant attribute works
// build-pass
// ignore-opencl1.2
// ignore-opencl2.0

use spirv_std::spirv;

#[spirv(vertex)]
pub fn main(#[spirv(invariant)] output: &mut f32) {}
