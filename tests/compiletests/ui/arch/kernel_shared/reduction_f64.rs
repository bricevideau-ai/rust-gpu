// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// compile-flags: -C target-feature=+Float64

// f64 tree reduction in shared memory (HPC pattern).

use spirv_std::arch::workgroup_memory_barrier_with_group_sync;
use spirv_std::glam::U64Vec3;
use spirv_std::spirv;

const WG_SIZE: usize = 32;

#[spirv(kernel(threads(32)))]
pub fn main(
    #[spirv(local_invocation_id)] local_id: U64Vec3,
    #[spirv(cross_workgroup)] input: &[f64],
    #[spirv(cross_workgroup)] output: &mut f64,
    #[spirv(workgroup)] shared: &mut [f64; WG_SIZE],
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
