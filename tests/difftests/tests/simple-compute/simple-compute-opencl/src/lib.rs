#![cfg_attr(target_arch = "spirv", no_std)]

use spirv_std::spirv;

#[spirv(kernel)]
pub fn main_kernel(#[spirv(cross_workgroup)] output: &mut [u32]) {
    output[0] = 42;
}
