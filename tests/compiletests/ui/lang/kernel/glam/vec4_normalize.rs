// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// ignore-spv1.5
// ignore-spv1.6

use spirv_std::glam;
use spirv_std::spirv;

#[spirv(kernel)]
pub fn main(#[spirv(cross_workgroup)] out: &mut f32) {
    let v = glam::Vec4::new(1.0, 1.0, 1.0, 1.0);
    let n = v.normalize();
    *out = n.length();
}
