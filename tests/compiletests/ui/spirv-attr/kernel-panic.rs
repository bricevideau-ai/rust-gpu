// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

// Test that panic paths compile for kernel targets (they become unreachable
// abort intrinsics in SPIR-V, but the code must still compile).

use spirv_std::spirv;

#[spirv(kernel)]
pub fn test_panic_simple() {
    panic!("kernel panic");
}

fn array_bounds_check(x: [u32; 4], i: usize) -> u32 {
    x[i]
}

#[spirv(kernel)]
pub fn test_bounds_check(#[spirv(cross_workgroup)] out: &mut u32) {
    *out = array_bounds_check([10, 20, 30, 40], 2);
}

#[spirv(kernel)]
pub fn test_unwrap(#[spirv(cross_workgroup)] out: &mut u32) {
    *out = Some(42u32).unwrap();
}

#[spirv(kernel)]
pub fn test_unwrap_or(#[spirv(cross_workgroup)] out: &mut u32) {
    *out = None.unwrap_or(15);
}
