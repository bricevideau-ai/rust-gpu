// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// ignore-spv1.5
// ignore-spv1.6

use spirv_std::glam::{IVec3, UVec2, Vec4};
use spirv_std::spirv;

#[spirv(kernel)]
pub fn main() {
    let fv = Vec4::new(1.0, 2.0, 3.0, 4.0);
    let iv = IVec3::new(1, 2, 3);
    let uv = UVec2::new(10, 20);
    spirv_std::printf!("%v4f\n", fv);
    spirv_std::printf!("%v3d\n", iv);
    spirv_std::printf!("%v2u\n", uv);
}
