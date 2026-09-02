// build-pass
// ignore-opencl1.2
// ignore-opencl2.0

use spirv_std::image::Image2d;
use spirv_std::spirv;

#[spirv(vertex)]
pub fn main(
    #[spirv(descriptor_set = 0, binding = 0)] implicit: &Image2d,
    #[spirv(descriptor_set = 0, binding = 1, uniform_constant)] explicit: &Image2d,
) {
}
