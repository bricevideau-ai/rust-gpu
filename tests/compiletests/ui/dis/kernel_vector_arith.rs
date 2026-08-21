// Verify that the explicit vector arithmetic intrinsics
// (`opencl_std::add`/`sub`/`mul`/`div`) lower to single core SPIR-V
// `OpFAdd`/`OpFSub`/`OpFMul`/`OpFDiv` on vector operands — without
// the OpCompositeExtract/Construct round-trip that glam's overloaded
// operators sometimes scalarise into.

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

use spirv_std::arch::opencl_std as ocl;
use spirv_std::glam::Vec3;
use spirv_std::spirv;

#[spirv(kernel)]
pub fn main(
    #[spirv(cross_workgroup)] a: &Vec3,
    #[spirv(cross_workgroup)] b: &Vec3,
    #[spirv(cross_workgroup)] out: &mut Vec3,
) {
    let s = ocl::add(*a, *b);
    let d = ocl::sub(*a, *b);
    let p = ocl::mul(*a, *b);
    let q = ocl::div(*a, *b);
    *out = ocl::add(ocl::add(s, d), ocl::add(p, q));
}
