// build-pass
// ignore-opencl1.2
// ignore-opencl2.0

use spirv_std::spirv;

#[spirv(fragment)]
pub fn main(#[spirv(flat)] i: i32) {
    if i < 10 {
        return;
    } else {
    }
}
