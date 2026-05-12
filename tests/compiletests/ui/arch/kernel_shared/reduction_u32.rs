// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// ignore-spv1.5
// ignore-spv1.6

// Tree reduction in shared memory.

use spirv_std::arch::workgroup_memory_barrier_with_group_sync;
use spirv_std::glam::U64Vec3;
use spirv_std::spirv;

const WG_SIZE: usize = 32;

#[spirv(kernel(threads(32)))]
pub fn main(
    #[spirv(local_invocation_id)] local_id: U64Vec3,
    #[spirv(cross_workgroup)] input: &[u32],
    #[spirv(cross_workgroup)] output: &mut u32,
    #[spirv(workgroup)] shared: &mut [u32; WG_SIZE],
) {
    let id = local_id.x as usize;
    shared[id] = input[id];
    workgroup_memory_barrier_with_group_sync();

    let mut stride = WG_SIZE / 2;
    while stride > 0 {
        if id < stride {
            shared[id] += shared[id + stride];
        }
        workgroup_memory_barrier_with_group_sync();
        stride /= 2;
    }

    if id == 0 {
        *output = shared[0];
    }
}
