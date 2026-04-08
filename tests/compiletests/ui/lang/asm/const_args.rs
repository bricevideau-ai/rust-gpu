// Tests using `asm!` with a const argument.
// build-pass
// ignore-opencl1.2
// ignore-opencl2.0

use core::arch::asm;
use spirv_std::spirv;

fn asm() {
    unsafe {
        const N: usize = 3;
        asm!(
            "%int = OpTypeInt 32 0",
            "%value = OpConstant %int {len}",
            len = const N,
        );
    }
}

#[spirv(fragment)]
pub fn main() {
    asm();
}
