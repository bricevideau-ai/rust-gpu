// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

use spirv_std::spirv;

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
pub fn main(#[spirv(cross_workgroup)] out: &mut u32) {
    *out = FIB_10;
}
