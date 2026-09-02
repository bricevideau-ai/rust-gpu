// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// ignore-spv1.5
// ignore-spv1.6

// `spirv_std::cl::Float4` — native OpenCL 4-wide single-precision vector
// type. Verifies construction, splat, accessors, the four arithmetic
// operators, and round-tripping through `to_array`/`from_array` all
// codegen on Kernel targets.

use spirv_std::cl::Float4;
use spirv_std::spirv;

#[spirv(kernel)]
pub fn main(
    #[spirv(cross_workgroup)] a: &Float4,
    #[spirv(cross_workgroup)] b: &Float4,
    #[spirv(cross_workgroup)] out: &mut Float4,
) {
    let x = *a + *b;
    let y = *a - *b;
    let z = *a * *b;
    let w = *a / Float4::splat(2.0);
    let arr = (x + y + z + w).to_array();
    *out = Float4::from_array([arr[0], arr[1], arr[2], arr[3]]);
}
