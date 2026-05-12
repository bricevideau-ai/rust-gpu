// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

// Compound-assignment operators (`+=`/`-=`/`*=`/`/=`) for `cl::*`
// vectors, both vector-vector and vector-scalar. Each one delegates
// to the matching non-assign op so codegen is identical to `x = x op y`.

use spirv_std::cl::{Float4, Int4};
use spirv_std::spirv;

#[spirv(kernel)]
pub fn floats(
    #[spirv(cross_workgroup)] a: &Float4,
    #[spirv(cross_workgroup)] s: &f32,
    #[spirv(cross_workgroup)] out: &mut Float4,
) {
    let mut acc = *a;
    acc += *a;
    acc -= *a;
    acc *= *a;
    acc /= *a;
    acc += *s;
    acc -= *s;
    acc *= *s;
    acc /= *s;
    *out = acc;
}

#[spirv(kernel)]
pub fn ints(
    #[spirv(cross_workgroup)] a: &Int4,
    #[spirv(cross_workgroup)] s: &i32,
    #[spirv(cross_workgroup)] out: &mut Int4,
) {
    let mut acc = *a;
    acc += *a;
    acc -= *a;
    acc *= *a;
    acc /= *a;
    acc += *s;
    acc -= *s;
    acc *= *s;
    acc /= *s;
    *out = acc;
}
