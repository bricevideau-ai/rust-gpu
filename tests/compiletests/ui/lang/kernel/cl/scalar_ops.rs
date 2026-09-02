// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// ignore-spv1.5
// ignore-spv1.6

// Scalar↔vector arithmetic on `cl::*` types — both directions.
// Float `Mul` lowers to `OpVectorTimesScalar`; everything else
// splats and falls back to the existing vector op.

use spirv_std::cl::{Float4, Int4};
use spirv_std::spirv;

#[spirv(kernel)]
pub fn floats(
    #[spirv(cross_workgroup)] a: &Float4,
    #[spirv(cross_workgroup)] s: &f32,
    #[spirv(cross_workgroup)] out: &mut Float4,
) {
    let x = *a * *s;
    let y = *s * *a;
    let z = *a + *s;
    let w = *s - *a;
    *out = (x + y + z) / *s + w;
}

#[spirv(kernel)]
pub fn ints(
    #[spirv(cross_workgroup)] a: &Int4,
    #[spirv(cross_workgroup)] s: &i32,
    #[spirv(cross_workgroup)] out: &mut Int4,
) {
    let x = *a * *s;
    let y = *s * *a;
    let z = *a + *s;
    let w = *s - *a;
    *out = (x + y + z) / *s + w;
}
