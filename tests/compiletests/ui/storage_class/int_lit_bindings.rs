// build-pass
// ignore-opencl1.2
// ignore-opencl2.0
use spirv_std::spirv;

#[spirv(compute(threads(1, 2u32, 3i128)))]
pub fn main(#[spirv(storage_buffer, descriptor_set = 42u8, binding = 69u64)] value: &mut f32) {
    *value = *value + 1.;
}
