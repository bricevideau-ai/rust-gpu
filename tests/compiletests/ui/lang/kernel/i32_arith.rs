// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

// i32 arithmetic and signed comparison in kernel context. Exercises that
// signedness=0 stripping at the type emission level doesn't break signed
// codegen.

use spirv_std::spirv;

#[spirv(kernel)]
pub fn neg(#[spirv(cross_workgroup)] a: &i32, #[spirv(cross_workgroup)] out: &mut i32) {
    *out = -*a;
}

#[spirv(kernel)]
pub fn lt(
    #[spirv(cross_workgroup)] a: &i32,
    #[spirv(cross_workgroup)] b: &i32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    *out = if *a < *b { 1 } else { 0 };
}
