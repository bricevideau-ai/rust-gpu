// Verify the codegen of an OpenCL printf call: the kernel calls into a
// helper function that performs `OpExtInst <result-id> <opencl.std-import>
// printf <format-string-var>`. This is the post-codegen shape; the actual
// `184` opcode is encoded as the symbolic name `printf` in the OpenCL.std
// extended instruction set.

// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

// compile-flags: -C llvm-args=--disassemble
// normalize-stderr-test "OpLine .*\n" -> ""
// normalize-stderr-test "OpSource .*\n" -> ""
// normalize-stderr-test "ui/dis/" -> "$$DIR/"

use spirv_std::spirv;

#[spirv(kernel)]
pub fn main() {
    spirv_std::printf!("hi\n");
}
