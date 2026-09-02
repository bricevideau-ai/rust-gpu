// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// ignore-spv1.5
// ignore-spv1.6

use spirv_std::glam::Vec4;
use spirv_std::spirv;

#[spirv(kernel)]
pub fn main() {
    let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
    spirv_std::printf!("%v4hlf\n", v);
}
