// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// ignore-spv1.5
// ignore-spv1.6

use spirv_std::arch::workgroup_memory_barrier_with_group_sync;
use spirv_std::glam::U64Vec3;
use spirv_std::spirv;

#[spirv(kernel(threads(32)))]
pub fn main(
    #[spirv(local_invocation_id)] local_id: U64Vec3,
    #[spirv(cross_workgroup)] out: &mut [u32],
    #[spirv(workgroup)] shared: &mut [u32; 32],
) {
    let id = local_id.x as usize;
    shared[id] = id as u32 * 10;
    workgroup_memory_barrier_with_group_sync();
    out[id] = shared[id];
}
