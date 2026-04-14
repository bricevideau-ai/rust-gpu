// Verify the printf-format-string + glam-vector lowering for `%v4f`:
// the format string captures the vector spec verbatim and the kernel
// passes the vector through to the OpenCL.std `printf` extended op.

// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

// compile-flags: -C llvm-args=--disassemble-globals
// normalize-stderr-test "OpSource .*\n" -> ""
// normalize-stderr-test "ui/dis/" -> "$$DIR/"
// normalize-stderr-test "%\d+ = OpString .\S*/glam-\S*.\n" -> ""

use spirv_std::glam::Vec4;
use spirv_std::spirv;

#[spirv(kernel)]
pub fn main() {
    let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
    spirv_std::printf!("%v4f\n", v);
}
