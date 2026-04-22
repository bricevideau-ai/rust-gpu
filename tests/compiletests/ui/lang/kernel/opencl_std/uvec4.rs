// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

use spirv_std::arch::opencl_std as ocl;
use spirv_std::glam::UVec4;
use spirv_std::spirv;

#[spirv(kernel)]
pub fn main(
    #[spirv(cross_workgroup)] a: &UVec4,
    #[spirv(cross_workgroup)] b: &UVec4,
    #[spirv(cross_workgroup)] out: &mut UVec4,
) {
    *out = ocl::u_min(*a, *b)
        + ocl::u_max(*a, *b)
        + ocl::u_clamp(*a, UVec4::ZERO, *b)
        + ocl::popcount(*a);
}
