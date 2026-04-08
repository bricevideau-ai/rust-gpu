// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

// Test const evaluation and static data in kernel context.

use spirv_std::spirv;

const OFFSETS: [f32; 8] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

#[spirv(kernel)]
pub fn test_const_array_len(#[spirv(cross_workgroup)] out: &mut u32) {
    *out = OFFSETS.len() as u32;
}

#[spirv(kernel)]
pub fn test_const_array_access(#[spirv(cross_workgroup)] out: &mut f32) {
    *out = OFFSETS[3];
}

const MAGIC: u32 = 0xDEAD_BEEF;

#[spirv(kernel)]
pub fn test_const_scalar(#[spirv(cross_workgroup)] out: &mut u32) {
    *out = MAGIC;
}

const fn fibonacci(n: u32) -> u32 {
    let mut a = 0u32;
    let mut b = 1u32;
    let mut i = 0;
    while i < n {
        let tmp = b;
        b = a + b;
        a = tmp;
        i += 1;
    }
    a
}

const FIB_10: u32 = fibonacci(10);

#[spirv(kernel)]
pub fn test_const_fn(#[spirv(cross_workgroup)] out: &mut u32) {
    *out = FIB_10;
}

#[spirv(kernel)]
pub fn test_const_fold_div(#[spirv(cross_workgroup)] out: &mut u32) {
    *out = 100 / 3;
}

// NOTE: &'static references require StorageClass.Private which needs the
// Shader capability. This is a known gap for Kernel targets.
