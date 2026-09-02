// Verify that the `cl::s!` macro lowers to a single `OpVectorShuffle`
// (multi-index) or `OpCompositeExtract` (single-index) per call, with
// the indices baked in as literal SPIR-V operands.

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

use spirv_std::cl::{Float2, Float4, Float8, s};
use spirv_std::spirv;

#[spirv(kernel)]
pub fn main(
    #[spirv(cross_workgroup)] v4: &Float4,
    #[spirv(cross_workgroup)] v8: &Float8,
    #[spirv(cross_workgroup)] out_scalar: &mut f32,
    #[spirv(cross_workgroup)] out2: &mut Float2,
    #[spirv(cross_workgroup)] out4: &mut Float4,
) {
    *out_scalar = s!(*v4, z); // OpCompositeExtract %f32 %v4 2
    *out2 = s!(*v4, yx); // OpVectorShuffle %v2f32 %v4 %v4 1 0
    *out4 = s!(*v8, s7654); // OpVectorShuffle %v4f32 %v8 %v8 7 6 5 4
}
