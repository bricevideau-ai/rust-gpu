// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

use spirv_std::glam;
use spirv_std::spirv;

#[spirv(kernel)]
pub fn main(#[spirv(cross_workgroup)] out: &mut f32) {
    let a = glam::Vec3::new(1.0, 2.0, 3.0);
    let b = glam::Vec3::new(4.0, 5.0, 6.0);
    let c = a + b;
    *out = c.x + c.y + c.z;
}
