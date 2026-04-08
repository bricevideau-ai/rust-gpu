// build-pass
// compile-flags: -C llvm-args=--disassemble-fn=add_two_ints::add_two_ints
// ignore-opencl1.2
// ignore-opencl2.0

use spirv_std::spirv;

fn add_two_ints(x: u32, y: u32) -> u32 {
    x + y
}
#[spirv(fragment)]
pub fn main() {
    add_two_ints(2, 3);
}
