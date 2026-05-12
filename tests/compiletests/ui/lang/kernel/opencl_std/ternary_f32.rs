// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

use spirv_std::arch::opencl_std as ocl;
use spirv_std::spirv;

#[spirv(kernel)]
pub fn main(
    #[spirv(cross_workgroup)] a: &f32,
    #[spirv(cross_workgroup)] b: &f32,
    #[spirv(cross_workgroup)] c: &f32,
    #[spirv(cross_workgroup)] out: &mut f32,
) {
    *out = ocl::fma(*a, *b, *c)
        + ocl::mad(*a, *b, *c)
        + ocl::clamp(*a, 0.0, 1.0)
        + ocl::mix(*a, *b, *c)
        + ocl::smoothstep(0.0, 1.0, *a);
}
