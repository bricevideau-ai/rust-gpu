// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// ignore-spv1.5
// ignore-spv1.6

use spirv_std::arch::opencl_std as ocl;
use spirv_std::spirv;

#[spirv(kernel)]
pub fn main(
    #[spirv(cross_workgroup)] a: &u32,
    #[spirv(cross_workgroup)] b: &u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    *out = ocl::popcount(*a)
        + ocl::clz(*a)
        + ocl::ctz(*a)
        + ocl::u_min(*a, *b)
        + ocl::u_max(*a, *b)
        + ocl::u_clamp(*a, 0, 100);
}
