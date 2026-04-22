// Companion test to `kernel_libm_sqrt.rs` — uses `--disassemble-fn`
// to verify the actual `OpExtInst` opcode for `Float::powf` is 48
// (`OpenCL.std::pow`), not 26 (`OpenCL.std::fma`). Catches a subtle
// bug class where GLSL/OpenCL opcode-number coincidence (`Pow` is 26
// in GLSL.std.450, but 26 is `fma` in OpenCL.std) would silently
// substitute the wrong instruction.

// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

// compile-flags: -C llvm-args=--disassemble-entry=main
// normalize-stderr-test "OpLine .*\n" -> ""

use spirv_std::num_traits::Float;
use spirv_std::spirv;

#[spirv(kernel)]
pub fn main(
    #[spirv(cross_workgroup)] x: &f32,
    #[spirv(cross_workgroup)] e: &f32,
    #[spirv(cross_workgroup)] out: &mut f32,
) {
    *out = (*x).powf(*e);
}
