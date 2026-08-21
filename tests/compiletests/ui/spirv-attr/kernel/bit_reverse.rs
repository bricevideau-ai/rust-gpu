// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// ignore-spv1.5
// ignore-spv1.6
// compile-flags: -C target-feature=+Int8,+Int16,+Int64

use spirv_std::spirv;

#[spirv(kernel)]
pub fn bit_reverse_u32(
    #[spirv(cross_workgroup)] input: &[u32],
    #[spirv(cross_workgroup)] output: &mut [u32],
) {
    output[0] = input[0].reverse_bits();
}

#[spirv(kernel)]
pub fn bit_reverse_u64(
    #[spirv(cross_workgroup)] input: &[u64],
    #[spirv(cross_workgroup)] output: &mut [u64],
) {
    output[0] = input[0].reverse_bits();
}

#[spirv(kernel)]
pub fn bit_reverse_i32(
    #[spirv(cross_workgroup)] input: &[i32],
    #[spirv(cross_workgroup)] output: &mut [i32],
) {
    output[0] = input[0].reverse_bits();
}

#[spirv(kernel)]
pub fn bit_reverse_opencl_std(
    #[spirv(cross_workgroup)] input: &[u32],
    #[spirv(cross_workgroup)] output: &mut [u32],
) {
    output[0] = spirv_std::arch::opencl_std::bit_reverse(input[0]);
}
