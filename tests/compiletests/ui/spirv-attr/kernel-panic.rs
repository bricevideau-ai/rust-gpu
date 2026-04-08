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

// NOTE: Dynamic array indexing (x[i]) fails on Physical64 due to mixed
// u32/u64 types in pointer arithmetic. Use static indices for now.

#[spirv(kernel)]
pub fn test_array_static_index(#[spirv(cross_workgroup)] out: &mut u32) {
    let x = [10u32, 20, 30, 40];
    *out = x[2];
}

#[spirv(kernel)]
pub fn test_unwrap(#[spirv(cross_workgroup)] out: &mut u32) {
    *out = Some(42u32).unwrap();
}

#[spirv(kernel)]
pub fn test_unwrap_or(#[spirv(cross_workgroup)] out: &mut u32) {
    *out = None.unwrap_or(15);
}
