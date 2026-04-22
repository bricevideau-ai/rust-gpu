// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

use spirv_std::arch::opencl_std as ocl;
use spirv_std::glam::IVec3;
use spirv_std::spirv;

#[spirv(kernel)]
pub fn main(
    #[spirv(cross_workgroup)] a: &IVec3,
    #[spirv(cross_workgroup)] b: &IVec3,
    #[spirv(cross_workgroup)] out: &mut IVec3,
) {
    *out = ocl::s_abs(*a)
        + ocl::s_min(*a, *b)
        + ocl::s_max(*a, *b)
        + ocl::s_clamp(*a, IVec3::ZERO, *b)
        + ocl::popcount(*a)
        + ocl::clz(*a)
        + ocl::ctz(*a);
}
