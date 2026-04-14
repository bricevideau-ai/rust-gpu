// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

use spirv_std::glam::IVec2;
use spirv_std::spirv;

#[spirv(kernel)]
pub fn main() {
    let v = IVec2::new(10, 20);
    spirv_std::printf!("%v2hld\n", v);
}
