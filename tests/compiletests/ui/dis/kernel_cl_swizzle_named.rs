// Verify that named-group swizzles (`lo`/`hi`/`even`/`odd`) emit the
// same single-instruction SPIR-V as explicit-index swizzles — one
// `OpVectorShuffle` per call (or `OpCompositeExtract` for source
// width 2 → scalar).

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

use spirv_std::cl::{Float2, Float4, Float8, Float16, s};
use spirv_std::spirv;

#[spirv(kernel)]
pub fn main(
    #[spirv(cross_workgroup)] v2: &Float2,
    #[spirv(cross_workgroup)] v4: &Float4,
    #[spirv(cross_workgroup)] v8: &Float8,
    #[spirv(cross_workgroup)] v16: &Float16,
    #[spirv(cross_workgroup)] out_scalar: &mut f32,
    #[spirv(cross_workgroup)] out2: &mut Float2,
    #[spirv(cross_workgroup)] out4: &mut Float4,
    #[spirv(cross_workgroup)] out8: &mut Float8,
) {
    *out_scalar = s!(*v2, hi); // OpCompositeExtract %f32 %v2 1
    *out2 = s!(*v4, even); // OpVectorShuffle %v2f32 %v4 %v4 0 2
    *out4 = s!(*v8, hi); // OpVectorShuffle %v4f32 %v8 %v8 4 5 6 7
    *out8 = s!(*v16, odd); // OpVectorShuffle %v8f32 %v16 %v16 1 3 5 7 9 11 13 15
}
