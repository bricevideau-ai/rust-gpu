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
    #[spirv(cross_workgroup)] a: &i32,
    #[spirv(cross_workgroup)] b: &i32,
    #[spirv(cross_workgroup)] out: &mut i32,
) {
    *out = ocl::s_abs(*a) + ocl::s_min(*a, *b) + ocl::s_max(*a, *b) + ocl::s_clamp(*a, 0, 100);
}
