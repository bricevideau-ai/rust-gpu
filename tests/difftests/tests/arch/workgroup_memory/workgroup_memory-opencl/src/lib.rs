#![cfg_attr(target_arch = "spirv", no_std)]

use spirv_std::arch::workgroup_memory_barrier_with_group_sync;
use spirv_std::spirv;

// Same hierarchical reduction as the workgroup_memory-rust variant, but
// expressed as an OpenCL Kernel: `compute(threads(N))` becomes
// `kernel(threads(N))`, and bound storage buffers become `cross_workgroup`
// pointer parameters (the codegen lowers `&[T]` to `(*T, usize)`).
#[spirv(kernel(threads(64)))]
pub fn main_kernel(
    #[spirv(cross_workgroup)] input: &[u32],
    #[spirv(cross_workgroup)] output: &mut [u32],
    #[spirv(local_invocation_id)] local_id: spirv_std::glam::UVec3,
    #[spirv(workgroup)] shared: &mut [u32; 64],
) {
    let lid = local_id.x as usize;

    shared[lid] = input[lid];

    workgroup_memory_barrier_with_group_sync();

    if lid < 32 {
        shared[lid] += shared[lid + 32];
    }
    workgroup_memory_barrier_with_group_sync();

    if lid < 16 {
        shared[lid] += shared[lid + 16];
    }
    workgroup_memory_barrier_with_group_sync();

    if lid < 8 {
        shared[lid] += shared[lid + 8];
    }
    workgroup_memory_barrier_with_group_sync();

    if lid < 4 {
        shared[lid] += shared[lid + 4];
    }
    workgroup_memory_barrier_with_group_sync();

    if lid < 2 {
        shared[lid] += shared[lid + 2];
    }
    workgroup_memory_barrier_with_group_sync();

    if lid < 1 {
        shared[lid] += shared[lid + 1];
    }
    workgroup_memory_barrier_with_group_sync();

    if lid == 0 {
        output[0] = shared[0];
    }
}
