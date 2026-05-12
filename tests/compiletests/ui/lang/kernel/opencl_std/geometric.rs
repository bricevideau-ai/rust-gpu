// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// ignore-spv1.5
// ignore-spv1.6

// Geometric ops — `dot`/`length`/`distance`/`fast_length`/`fast_distance`
// collapse a vector to its scalar; `normalize`/`cross`/`fast_normalize`
// keep the vector type. `cross` is vec3/vec4-only. Note: `dot` lowers
// to core SPIR-V `OpDot`, the others to `OpExtInst %OpenCL.std`.

use spirv_std::arch::opencl_std as ocl;
use spirv_std::glam::Vec3;
use spirv_std::spirv;

#[spirv(kernel)]
pub fn main(
    #[spirv(cross_workgroup)] a: &Vec3,
    #[spirv(cross_workgroup)] b: &Vec3,
    #[spirv(cross_workgroup)] out_scalar: &mut f32,
    #[spirv(cross_workgroup)] out_vec: &mut Vec3,
) {
    *out_scalar = ocl::dot(*a, *b)
        + ocl::length(*a)
        + ocl::distance(*a, *b)
        + ocl::fast_length(*a)
        + ocl::fast_distance(*a, *b);
    *out_vec = ocl::normalize(*a) + ocl::cross(*a, *b) + ocl::fast_normalize(*b);
}
