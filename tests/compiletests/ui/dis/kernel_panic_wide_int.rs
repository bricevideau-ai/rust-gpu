// Verify that a `panic!` formatting a `u64` value lowers to a DebugPrintf
// format string with `%lu` (commit "Extend DebugPrintf format specifiers
// for non-32-bit integer and f64 types"). Prior to that change the spec
// rendered as the empty placeholder.

// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

// compile-flags: -C llvm-args=--abort-strategy=debug-printf
// compile-flags: -C llvm-args=--disassemble-globals
// normalize-stderr-test "OpSource .*\n" -> ""
// normalize-stderr-test "ui/dis/" -> "$$DIR/"

use spirv_std::spirv;

#[spirv(kernel)]
pub fn main(value: u64) {
    if value == 0 {
        panic!("got u64 zero: {}", value);
    }
}
