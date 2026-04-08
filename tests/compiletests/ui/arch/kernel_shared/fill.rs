// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

use spirv_std::arch::workgroup_memory_barrier_with_group_sync;
use spirv_std::glam::U64Vec3;
use spirv_std::spirv;

#[spirv(kernel(threads(64)))]
pub fn main(
    #[spirv(local_invocation_id)] local_id: U64Vec3,
    #[spirv(workgroup)] shared: &mut [u32; 64],
    #[spirv(cross_workgroup)] out: &mut [u32],
    value: u32,
) {
    let id = local_id.x as usize;
    shared[id] = value;
    workgroup_memory_barrier_with_group_sync();
    out[id] = shared[id];
}
