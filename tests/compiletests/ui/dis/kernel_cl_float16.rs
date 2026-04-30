// Verify that `spirv_std::cl::Float16` lowers to a 16-wide
// `OpTypeVector f32 16` (Vector16 capability), and that arithmetic
// on it emits a single `OpFAdd`/`OpFMul` per source operator —
// proving the count widening at `abi.rs:1129` is wired up correctly.

// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

// compile-flags: -C llvm-args=--disassemble-entry=main
// normalize-stderr-test "OpLine .*\n" -> ""

use spirv_std::cl::Float16;
use spirv_std::spirv;

#[spirv(kernel)]
pub fn main(
    #[spirv(cross_workgroup)] a: &Float16,
    #[spirv(cross_workgroup)] b: &Float16,
    #[spirv(cross_workgroup)] out: &mut Float16,
) {
    *out = (*a + *b) * *b;
}
