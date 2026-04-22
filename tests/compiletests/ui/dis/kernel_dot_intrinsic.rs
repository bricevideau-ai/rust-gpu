// Verify that `spirv_std::arch::opencl_std::dot` lowers to a single
// `OpDot` (core SPIR-V opcode 148, no extended-instruction wrapping).
// Without this intrinsic, glam's `Vec3::dot()` would lower to 3
// OpFMul + 2 OpFAdd on scalars (because rust-gpu doesn't recognize the
// glam method), missing native vector hardware.

// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

// compile-flags: -C llvm-args=--disassemble-entry=main
// normalize-stderr-test "OpLine .*\n" -> ""

use spirv_std::arch::opencl_std as ocl;
use spirv_std::glam::Vec3;
use spirv_std::spirv;

#[spirv(kernel)]
pub fn main(
    #[spirv(cross_workgroup)] a: &Vec3,
    #[spirv(cross_workgroup)] b: &Vec3,
    #[spirv(cross_workgroup)] out: &mut f32,
) {
    *out = ocl::dot(*a, *b);
}
