// Verify that integer-vector division on `cl::Int4` emits `OpSDiv`
// and `cl::UInt4` emits `OpUDiv` — i.e. the macro selects the correct
// signed/unsigned opcode per scalar type.

// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

// compile-flags: -C llvm-args=--disassemble-entry=main
// normalize-stderr-test "OpLine .*\n" -> ""

use spirv_std::cl::{Int4, UInt4};
use spirv_std::spirv;

#[spirv(kernel)]
pub fn main(
    #[spirv(cross_workgroup)] s: &Int4,
    #[spirv(cross_workgroup)] u: &UInt4,
    #[spirv(cross_workgroup)] out_s: &mut Int4,
    #[spirv(cross_workgroup)] out_u: &mut UInt4,
) {
    *out_s = *s / Int4::splat(3);
    *out_u = *u / UInt4::splat(3);
}
