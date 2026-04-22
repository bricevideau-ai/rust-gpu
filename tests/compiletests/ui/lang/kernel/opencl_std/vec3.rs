// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

use spirv_std::arch::opencl_std as ocl;
use spirv_std::glam::Vec3;
use spirv_std::spirv;

#[spirv(kernel)]
pub fn main(
    #[spirv(cross_workgroup)] a: &Vec3,
    #[spirv(cross_workgroup)] b: &Vec3,
    #[spirv(cross_workgroup)] t: &f32,
    #[spirv(cross_workgroup)] out: &mut Vec3,
) {
    *out = ocl::sqrt(*a)
        + ocl::sin(*a)
        + ocl::fmin(*a, *b)
        + ocl::pow(*a, *b)
        + ocl::clamp(*a, *b, *a)
        + ocl::mix(*a, *b, Vec3::splat(*t))
        + ocl::fma(*a, *b, *a);
}
