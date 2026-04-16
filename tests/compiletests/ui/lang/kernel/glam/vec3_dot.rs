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
    let a = glam::Vec3::new(1.0, 0.0, 0.0);
    let b = glam::Vec3::new(0.0, 1.0, 0.0);
    *out = a.dot(b);
}
