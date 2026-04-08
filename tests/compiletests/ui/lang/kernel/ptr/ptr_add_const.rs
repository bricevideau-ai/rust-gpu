// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

use spirv_std::spirv;

#[spirv(kernel)]
pub fn main(#[spirv(cross_workgroup)] data: &mut [u32]) {
    unsafe {
        let ptr = data.as_mut_ptr();
        *ptr.add(0) = 10;
        *ptr.add(1) = 20;
    }
}
