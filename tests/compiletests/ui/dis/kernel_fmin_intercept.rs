// Verifies Path A1: `f32::min` / `f32::max` (and the f64 forms) lower
// to `OpExtInst <OpenCL.std> {fmin_common, fmax_common}` (opcodes 98/97)
// on Kernel targets via the `minimum_number_nsz_*` / `maximum_number_nsz_*`
// rustc intrinsics. These OpenCL ops are NaN-ignoring (return the
// non-NaN operand), matching Rust's `f32::min` semantics — unlike
// `OpExtInst <GLSL.std.450> FMin` which is NaN-undefined per spec.
//
// On Vulkan/Shader targets there's no `fmin_common`/`fmax_common`
// equivalent in `GLSL.std.450`, so the intercept is gated on
// `Capability::Kernel` and Vulkan still inlines the branchy core
// implementation (lower-quality codegen, but no NaN semantics regression).

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

use spirv_std::spirv;

#[spirv(kernel)]
pub fn main(
    #[spirv(cross_workgroup)] a: &f32,
    #[spirv(cross_workgroup)] b: &f32,
    #[spirv(cross_workgroup)] out_min: &mut f32,
    #[spirv(cross_workgroup)] out_max: &mut f32,
) {
    *out_min = (*a).min(*b);
    *out_max = (*a).max(*b);
}
