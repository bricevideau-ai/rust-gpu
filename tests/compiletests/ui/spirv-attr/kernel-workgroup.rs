// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// compile-flags: -C target-feature=+Float64

// Test workgroup (shared/local) memory operations in kernel context.
// These are the OpenCL equivalent of Vulkan's shared memory patterns.

use spirv_std::arch::workgroup_memory_barrier_with_group_sync;
use spirv_std::glam::U64Vec3;
use spirv_std::spirv;

// Basic shared memory: write and read back.
#[spirv(kernel(threads(32)))]
pub fn test_shared_write_read(
    #[spirv(local_invocation_id)] local_id: U64Vec3,
    #[spirv(cross_workgroup)] out: &mut [u32],
    #[spirv(workgroup)] shared: &mut [u32; 32],
) {
    let id = local_id.x as usize;
    shared[id] = id as u32 * 10;
    workgroup_memory_barrier_with_group_sync();
    out[id] = shared[id];
}

// Shared memory: neighbor exchange.
#[spirv(kernel(threads(32)))]
pub fn test_shared_neighbor(
    #[spirv(local_invocation_id)] local_id: U64Vec3,
    #[spirv(cross_workgroup)] out: &mut [u32],
    #[spirv(workgroup)] shared: &mut [u32; 32],
) {
    let id = local_id.x as usize;
    shared[id] = id as u32;
    workgroup_memory_barrier_with_group_sync();
    let neighbor = (id + 1) % 32;
    out[id] = shared[neighbor];
}

// Parallel reduction in shared memory.
pub const WG_SIZE: usize = 32;

#[spirv(kernel(threads(32)))]
pub fn test_reduction(
    #[spirv(local_invocation_id)] local_id: U64Vec3,
    #[spirv(cross_workgroup)] input: &[u32],
    #[spirv(cross_workgroup)] output: &mut u32,
    #[spirv(workgroup)] shared: &mut [u32; WG_SIZE],
) {
    let id = local_id.x as usize;

    shared[id] = input[id];
    workgroup_memory_barrier_with_group_sync();

    // Tree reduction.
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

// f64 reduction in shared memory (HPC pattern).
#[spirv(kernel(threads(32)))]
pub fn test_reduction_f64(
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

// Shared memory with scalar parameter.
#[spirv(kernel(threads(64)))]
pub fn test_shared_fill(
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
