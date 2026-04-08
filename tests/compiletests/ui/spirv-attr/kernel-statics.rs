// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

// Test static and static mut variables in kernel context.
// Mutable statics use CrossWorkgroup storage class (program-scope globals).
// Immutable static references use UniformConstant.

use spirv_std::spirv;

// Mutable static — program-scope global variable.
static mut COUNTER: u32 = 0;

#[spirv(kernel)]
pub fn test_static_mut(#[spirv(cross_workgroup)] out: &mut u32) {
    unsafe {
        COUNTER += 1;
        *out = COUNTER;
    }
}

// Mutable static with non-zero initializer.
static mut ACCUMULATOR: u32 = 100;

#[spirv(kernel)]
pub fn test_static_mut_init(#[spirv(cross_workgroup)] out: &mut u32, value: u32) {
    unsafe {
        ACCUMULATOR += value;
        *out = ACCUMULATOR;
    }
}

// Immutable static reference (const-promoted).
#[inline(never)]
fn load_static(r: &'static u32) -> u32 {
    *r
}

#[spirv(kernel)]
pub fn test_static_ref(#[spirv(cross_workgroup)] out: &mut u32) {
    *out = load_static(&42);
}

// Static array.
static LOOKUP: [u32; 4] = [10, 20, 30, 40];

#[spirv(kernel)]
pub fn test_static_array(#[spirv(cross_workgroup)] out: &mut u32, index: u32) {
    *out = LOOKUP[index as usize];
}
