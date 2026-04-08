// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// ignore-spv1.5
// ignore-spv1.6

// Regression: `is_multiple_of` previously triggered a spirv-opt
// DeadBranchElimPass SIGSEGV. See KhronosGroup/SPIRV-Tools#6632.

use spirv_std::spirv;

#[spirv(kernel)]
pub fn main(#[spirv(cross_workgroup)] buf: &mut u32, n: u32) {
    let mut sum = 0u32;
    for i in 0..n {
        if i.is_multiple_of(2) {
            continue;
        }
        sum += i;
    }
    *buf = sum;
}
