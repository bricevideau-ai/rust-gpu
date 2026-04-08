// build-pass
// ignore-opencl1.2
// ignore-opencl2.0

#![allow(unconditional_panic)]

use spirv_std::spirv;

#[spirv(fragment)]
pub fn const_fold_div(out: &mut u32) {
    *out = 7u32 / 0;
}
