// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// compile-flags: -C target-feature=+Float64

use spirv_std::arch::opencl_std as ocl;
use spirv_std::glam::DVec3;
use spirv_std::spirv;

#[spirv(kernel)]
pub fn main(
    #[spirv(cross_workgroup)] a: &DVec3,
    #[spirv(cross_workgroup)] b: &DVec3,
    #[spirv(cross_workgroup)] out: &mut DVec3,
) {
    *out = ocl::sqrt(*a) + ocl::pow(*a, *b) + ocl::fma(*a, *b, *a);
}
