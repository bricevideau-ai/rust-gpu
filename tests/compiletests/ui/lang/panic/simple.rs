// Test that calling `panic!` works.
// build-pass
// ignore-opencl1.2
// ignore-opencl2.0

use spirv_std::spirv;

#[spirv(fragment)]
pub fn main() {
    panic!("aaa");
}
