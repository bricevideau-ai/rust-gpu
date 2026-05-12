// Verifies the libm-intercept fix for Kernel targets: calling
// `Float::sqrt` / `Float::powf` (the idiomatic Rust paths) on a kernel
// must lower to `OpExtInst <OpenCL.std> {sqrt, pow}`, NOT
// `OpExtInst <GLSL.std.450> {Sqrt, Pow}`, since GLSL.std.450 is not
// part of the OpenCL SPIR-V Environment Spec.
//
// Pre-fix this would emit `OpExtInstImport "GLSL.std.450"` (wrong);
// post-fix it imports only `OpenCL.std`. `pow` in particular catches
// a separate bug class — GLSL `Pow` is opcode 26 but OpenCL `pow` is
// opcode 48 (26 is `fma` in OpenCL.std), so a naive substitution would
// silently emit `fma` instead of `pow`.

// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

// compile-flags: -C llvm-args=--disassemble-globals
// normalize-stderr-test "OpString .*\n" -> ""
// normalize-stderr-test "OpSource .*\n" -> ""

use spirv_std::num_traits::Float;
use spirv_std::spirv;

#[spirv(kernel)]
pub fn main(
    #[spirv(cross_workgroup)] x: &f32,
    #[spirv(cross_workgroup)] e: &f32,
    #[spirv(cross_workgroup)] out_sqrt: &mut f32,
    #[spirv(cross_workgroup)] out_pow: &mut f32,
) {
    *out_sqrt = (*x).sqrt();
    *out_pow = (*x).powf(*e);
}
