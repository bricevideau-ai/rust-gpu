// Test panics coming from the Rust language such as `1 / 0`.
// build-pass
// ignore-opencl1.2
// ignore-opencl2.0

use spirv_std::spirv;

fn int_div(x: usize) -> usize {
    1 / x
}

#[spirv(fragment)]
pub fn main() {
    int_div(0);
}
