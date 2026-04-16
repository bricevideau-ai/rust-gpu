// Verify the OpCapability set, OpMemoryModel and OpEntryPoint emitted
// for an OpenCL Kernel target.

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
pub fn main() {}
