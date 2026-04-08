// Test using `&&` operator.
// build-pass
// ignore-opencl1.2
// ignore-opencl2.0

use spirv_std::spirv;

fn f(x: bool, y: bool) -> bool {
    x && y
}

#[spirv(fragment)]
pub fn main() {
    f(false, true);
}
