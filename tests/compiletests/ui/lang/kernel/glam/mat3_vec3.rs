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
pub fn main(
    #[spirv(cross_workgroup)] out_x: &mut f32,
    #[spirv(cross_workgroup)] out_y: &mut f32,
    #[spirv(cross_workgroup)] out_z: &mut f32,
) {
    let m = glam::Mat3::IDENTITY;
    let v = glam::Vec3::new(1.0, 2.0, 3.0);
    let result = m * v;
    *out_x = result.x;
    *out_y = result.y;
    *out_z = result.z;
}
