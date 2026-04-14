// Verify the module-level shape of a basic OpenCL printf:
//   - `OpExtInstImport "OpenCL.std"` (commit "Add OpenCL printf support
//     via OpenCL.std extended instruction set")
//   - format string emitted as `[u8; N]` `OpConstantComposite` in
//     `UniformConstant` storage
//   - kernel exposes `Int8` capability for the byte array

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

use spirv_std::spirv;

#[spirv(kernel)]
pub fn main() {
    spirv_std::printf!("hello\n");
}
