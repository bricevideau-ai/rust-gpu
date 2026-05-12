// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

use spirv_std::spirv;

#[spirv(kernel)]
pub fn test_saturating_add_u32(
    #[spirv(cross_workgroup)] a: &u32,
    #[spirv(cross_workgroup)] b: &u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    *out = a.saturating_add(*b);
}

#[spirv(kernel)]
pub fn test_saturating_sub_u32(
    #[spirv(cross_workgroup)] a: &u32,
    #[spirv(cross_workgroup)] b: &u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    *out = a.saturating_sub(*b);
}

#[spirv(kernel)]
pub fn test_saturating_add_i32(
    #[spirv(cross_workgroup)] a: &i32,
    #[spirv(cross_workgroup)] b: &i32,
    #[spirv(cross_workgroup)] out: &mut i32,
) {
    *out = a.saturating_add(*b);
}

#[spirv(kernel)]
pub fn test_saturating_sub_i32(
    #[spirv(cross_workgroup)] a: &i32,
    #[spirv(cross_workgroup)] b: &i32,
    #[spirv(cross_workgroup)] out: &mut i32,
) {
    *out = a.saturating_sub(*b);
}
