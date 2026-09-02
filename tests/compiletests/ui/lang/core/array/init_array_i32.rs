// Test creating an array.
// build-pass
// ignore-opencl1.2
// ignore-opencl2.0

use spirv_std::macros::spirv;

#[spirv(fragment)]
pub fn main(o: &mut i32) {
    let array = [0i32; 4];
    *o = array[1];
}
