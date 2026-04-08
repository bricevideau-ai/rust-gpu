// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// ignore-spv1.5
// ignore-spv1.6

use spirv_std::spirv;

fn fold_range<F: FnMut(u32, u32) -> u32>(n: u32, init: u32, mut f: F) -> u32 {
    let mut acc = init;
    for i in 0..n {
        acc = f(acc, i);
    }
    acc
}

#[spirv(kernel)]
pub fn main(#[spirv(cross_workgroup)] out: &mut u32, n: u32) {
    *out = fold_range(n, 0, |acc, i| acc + i);
}
