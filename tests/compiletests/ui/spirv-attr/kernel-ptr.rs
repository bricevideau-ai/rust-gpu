// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

use spirv_std::spirv;

// Basic pointer dereference.

#[spirv(kernel)]
pub fn test_deref_read(
    #[spirv(cross_workgroup)] val: &u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    *out = *val;
}

#[spirv(kernel)]
pub fn test_deref_write(#[spirv(cross_workgroup)] out: &mut u32) {
    *out = 42;
}

// Pointer arithmetic via ptr.add().

#[spirv(kernel)]
pub fn test_ptr_add_const(#[spirv(cross_workgroup)] data: &mut [u32]) {
    unsafe {
        let ptr = data.as_mut_ptr();
        *ptr.add(0) = 10;
        *ptr.add(1) = 20;
    }
}

#[spirv(kernel)]
pub fn test_ptr_add_dynamic(#[spirv(cross_workgroup)] data: &mut [u32], index: u32) {
    unsafe {
        let ptr = data.as_mut_ptr().add(index as usize);
        *ptr = 99;
    }
}

// Read-then-write through pointer.

#[spirv(kernel)]
pub fn test_ptr_read_write(#[spirv(cross_workgroup)] data: &mut [u32], index: u32) {
    unsafe {
        let ptr = data.as_mut_ptr().add(index as usize);
        let val = *ptr;
        *ptr = val * 3 + 1;
    }
}

// Multiple buffers with pointer ops.

#[spirv(kernel)]
pub fn test_ptr_copy(
    #[spirv(cross_workgroup)] src: &[u32],
    #[spirv(cross_workgroup)] dst: &mut [u32],
    index: u32,
) {
    unsafe {
        let s = src.as_ptr().add(index as usize);
        let d = dst.as_mut_ptr().add(index as usize);
        *d = *s;
    }
}

// Pointer to different types.

#[spirv(kernel)]
pub fn test_ptr_u64(#[spirv(cross_workgroup)] data: &mut u64) {
    *data = *data + 1;
}

#[spirv(kernel)]
pub fn test_ptr_i32(#[spirv(cross_workgroup)] data: &mut i32) {
    *data = -*data;
}

#[spirv(kernel)]
pub fn test_ptr_f32(#[spirv(cross_workgroup)] data: &mut f32) {
    *data = *data * 2.0;
}
