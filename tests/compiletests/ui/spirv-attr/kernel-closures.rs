// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

// Test closures and higher-order functions in kernel context.

use spirv_std::spirv;

fn apply<F: FnOnce(u32) -> u32>(f: F, x: u32) -> u32 {
    f(x)
}

#[spirv(kernel)]
pub fn test_closure_basic(#[spirv(cross_workgroup)] out: &mut u32, x: u32) {
    *out = apply(|v| v * 2, x);
}

fn fold_range<F: FnMut(u32, u32) -> u32>(n: u32, init: u32, mut f: F) -> u32 {
    let mut acc = init;
    for i in 0..n {
        acc = f(acc, i);
    }
    acc
}

#[spirv(kernel)]
pub fn test_closure_fold(#[spirv(cross_workgroup)] out: &mut u32, n: u32) {
    *out = fold_range(n, 0, |acc, i| acc + i);
}

#[spirv(kernel)]
pub fn test_closure_capture(#[spirv(cross_workgroup)] out: &mut u32, x: u32, y: u32) {
    let sum = x + y;
    *out = apply(|v| v + sum, 10);
}

#[spirv(kernel)]
pub fn test_closure_mut(#[spirv(cross_workgroup)] out: &mut u32, n: u32) {
    let mut count = 0u32;
    let mut counter = |_: u32| {
        count += 1;
        count
    };
    for i in 0..n {
        counter(i);
    }
    *out = count;
}
