// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

// Test array operations in kernel context.

use spirv_std::spirv;

#[spirv(kernel)]
pub fn test_array_init_i32(#[spirv(cross_workgroup)] out: &mut i32) {
    let array = [0i32; 4];
    *out = array[1];
}

#[spirv(kernel)]
pub fn test_array_init_u32(#[spirv(cross_workgroup)] out: &mut u32) {
    let array = [10u32, 20, 30, 40];
    *out = array[2];
}

#[spirv(kernel)]
pub fn test_array_init_f32(#[spirv(cross_workgroup)] out: &mut f32) {
    let array = [1.0f32, 2.0, 3.0, 4.0];
    *out = array[0] + array[3];
}

#[spirv(kernel)]
pub fn test_array_write(#[spirv(cross_workgroup)] out: &mut u32) {
    let mut array = [0u32; 4];
    array[0] = 10;
    array[1] = 20;
    *out = array[0] + array[1];
}

#[spirv(kernel)]
pub fn test_array_loop(#[spirv(cross_workgroup)] out: &mut u32) {
    let array = [1u32, 2, 3, 4, 5];
    let mut sum = 0u32;
    for i in 0..5 {
        sum += array[i];
    }
    *out = sum;
}

#[spirv(kernel)]
pub fn test_array_dyn_index(#[spirv(cross_workgroup)] out: &mut u32, i: u32) {
    let array = [10u32, 20, 30, 40];
    *out = array[i as usize];
}

#[spirv(kernel)]
pub fn test_array_param(#[spirv(cross_workgroup)] out: &mut u32) {
    fn sum_array(a: [u32; 4]) -> u32 {
        a[0] + a[1] + a[2] + a[3]
    }
    *out = sum_array([10, 20, 30, 40]);
}
