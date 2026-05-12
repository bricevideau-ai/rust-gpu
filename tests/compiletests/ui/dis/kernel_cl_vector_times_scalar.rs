// Verify that `Float4 * f32` (and `f32 * Float4`) lowers to a single
// `OpVectorTimesScalar`, SPIR-V's purpose-built op — not to a splat
// followed by a vector multiply.

// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

// compile-flags: -C llvm-args=--disassemble-entry=main
// normalize-stderr-test "OpLine .*\n" -> ""

use spirv_std::cl::Float4;
use spirv_std::spirv;

#[spirv(kernel)]
pub fn main(
    #[spirv(cross_workgroup)] v: &Float4,
    #[spirv(cross_workgroup)] s: &f32,
    #[spirv(cross_workgroup)] out_a: &mut Float4,
    #[spirv(cross_workgroup)] out_b: &mut Float4,
) {
    *out_a = *v * *s;
    *out_b = *s * *v;
}
