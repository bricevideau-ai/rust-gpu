// Verify that arithmetic on `spirv_std::cl::Float4` lowers to native
// 4-wide vector `OpFAdd`/`OpFSub`/`OpFMul`/`OpFDiv` — i.e. the type
// is recognised as `OpTypeVector f32 4` by the codegen, not as a
// 4-field struct that gets scalarised.

// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// ignore-spv1.5
// ignore-spv1.6

// compile-flags: -C llvm-args=--disassemble-entry=main
// normalize-stderr-test "OpLine .*\n" -> ""

use spirv_std::cl::Float4;
use spirv_std::spirv;

#[spirv(kernel)]
pub fn main(
    #[spirv(cross_workgroup)] a: &Float4,
    #[spirv(cross_workgroup)] b: &Float4,
    #[spirv(cross_workgroup)] out: &mut Float4,
) {
    let s = *a + *b;
    let d = *a - *b;
    let p = *a * *b;
    let q = *a / *b;
    *out = (s + d) + (p + q);
}
