// build-pass
// ignore-opencl1.2
// ignore-opencl2.0

use spirv_std::spirv;

#[spirv(fragment)]
pub fn main(#[spirv(flat)] i: i32) {
    while i < 10 {
        if i == 0 {
            break;
        }
        if i == 1 {
            break;
        }
    }
}
