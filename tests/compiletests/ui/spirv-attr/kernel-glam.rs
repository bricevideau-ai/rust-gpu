// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

// Test glam math library operations in kernel context.

use spirv_std::glam;
use spirv_std::spirv;

#[spirv(kernel)]
pub fn test_vec3_ops(#[spirv(cross_workgroup)] out: &mut f32) {
    let a = glam::Vec3::new(1.0, 2.0, 3.0);
    let b = glam::Vec3::new(4.0, 5.0, 6.0);
    let c = a + b;
    *out = c.x + c.y + c.z;
}

#[spirv(kernel)]
pub fn test_vec3_dot(#[spirv(cross_workgroup)] out: &mut f32) {
    let a = glam::Vec3::new(1.0, 0.0, 0.0);
    let b = glam::Vec3::new(0.0, 1.0, 0.0);
    *out = a.dot(b);
}

#[spirv(kernel)]
pub fn test_vec3_cross(
    #[spirv(cross_workgroup)] out_x: &mut f32,
    #[spirv(cross_workgroup)] out_y: &mut f32,
    #[spirv(cross_workgroup)] out_z: &mut f32,
) {
    let a = glam::Vec3::new(1.0, 0.0, 0.0);
    let b = glam::Vec3::new(0.0, 1.0, 0.0);
    let c = a.cross(b);
    *out_x = c.x;
    *out_y = c.y;
    *out_z = c.z;
}

#[spirv(kernel)]
pub fn test_mat3_vec3_multiply(
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

#[spirv(kernel)]
pub fn test_vec2_length(#[spirv(cross_workgroup)] out: &mut f32) {
    let v = glam::Vec2::new(3.0, 4.0);
    *out = v.length();
}

#[spirv(kernel)]
pub fn test_vec4_normalize(#[spirv(cross_workgroup)] out: &mut f32) {
    let v = glam::Vec4::new(1.0, 1.0, 1.0, 1.0);
    let n = v.normalize();
    *out = n.length();
}
