#![cfg_attr(target_arch = "spirv", no_std)]
#![deny(warnings)]
#![allow(clippy::too_many_arguments)]

use glam::USizeVec3;
use spirv_std::arch::{
    group_exclusive_i_add, group_i_add, work_group_all, work_group_any, work_group_broadcast_u32,
    work_group_exclusive_i_add, work_group_i_add, work_group_inclusive_i_add,
    workgroup_memory_barrier_with_group_sync,
};
use spirv_std::{glam, spirv};

// ── Test 1: subgroup builtins ────────────────────────────────────────

/// Write `subgroup_id`, `subgroup_local_id`, `num_subgroups`, `subgroup_size` for each work item.
/// Uses `global_id` for indexing — works with multiple workgroups.
#[spirv(kernel(threads(32)))]
pub fn test_subgroup_builtins(
    #[spirv(global_invocation_id)] global_id: USizeVec3,
    #[spirv(subgroup_id)] subgroup_id: u32,
    #[spirv(subgroup_local_invocation_id)] subgroup_local_id: u32,
    #[spirv(num_subgroups)] num_subgroups: u32,
    #[spirv(subgroup_size)] subgroup_size: u32,
    #[spirv(cross_workgroup)] out_sg_id: &mut [u32],
    #[spirv(cross_workgroup)] out_sg_lid: &mut [u32],
    #[spirv(cross_workgroup)] out_num_sg: &mut [u32],
    #[spirv(cross_workgroup)] out_sg_size: &mut [u32],
) {
    let gid = global_id.x;
    out_sg_id[gid] = subgroup_id;
    out_sg_lid[gid] = subgroup_local_id;
    out_num_sg[gid] = num_subgroups;
    out_sg_size[gid] = subgroup_size;
}

// ── Test 2: shared memory + barrier ──────────────────────────────────

/// Each item writes its `local_id` to shared, barrier, then reads the
/// reversed element: `out[gid] = shared[WG_SIZE - 1 - lid]`.
#[spirv(kernel(threads(32)))]
pub fn test_shared_barrier(
    #[spirv(global_invocation_id)] global_id: USizeVec3,
    #[spirv(local_invocation_id)] local_id: USizeVec3,
    #[spirv(cross_workgroup)] output: &mut [u32],
    #[spirv(workgroup)] shared: &mut [u32; 32],
) {
    let gid = global_id.x;
    let lid = local_id.x;
    shared[lid] = lid as u32;

    workgroup_memory_barrier_with_group_sync();

    output[gid] = shared[31 - lid];
}

// ── Test 3: group_i_add (reduce) ─────────────────────────────────────

/// Each item contributes its `local_id+1`, reduce gives sum within the subgroup.
#[spirv(kernel(threads(32)))]
pub fn test_group_reduce(
    #[spirv(global_invocation_id)] global_id: USizeVec3,
    #[spirv(local_invocation_id)] local_id: USizeVec3,
    #[spirv(cross_workgroup)] output: &mut [u32],
) {
    let gid = global_id.x;
    let lid = local_id.x as u32;
    let reduced = group_i_add(lid + 1);
    output[gid] = reduced;
}

// ── Test 4: group_exclusive_i_add (scan) ─────────────────────────────

/// Exclusive prefix sum of `(lid+1)` within each subgroup.
#[spirv(kernel(threads(32)))]
pub fn test_group_scan(
    #[spirv(global_invocation_id)] global_id: USizeVec3,
    #[spirv(local_invocation_id)] local_id: USizeVec3,
    #[spirv(cross_workgroup)] output: &mut [u32],
) {
    let gid = global_id.x;
    let lid = local_id.x as u32;
    let scanned = group_exclusive_i_add(lid + 1);
    output[gid] = scanned;
}

// ── Test 5: shared + subgroup builtins ───────────────────────────────

/// Each subgroup writes `subgroup_id*1000` to shared, barrier, read back.
#[spirv(kernel(threads(32)))]
pub fn test_shared_with_subgroup_builtins(
    #[spirv(global_invocation_id)] global_id: USizeVec3,
    #[spirv(subgroup_id)] subgroup_id: u32,
    #[spirv(subgroup_local_invocation_id)] subgroup_local_id: u32,
    #[spirv(cross_workgroup)] output: &mut [u32],
    #[spirv(workgroup)] shared: &mut [u32; 32],
) {
    let gid = global_id.x;

    if subgroup_local_id == 0 {
        shared[subgroup_id as usize] = subgroup_id * 1000;
    }

    workgroup_memory_barrier_with_group_sync();

    output[gid] = shared[subgroup_id as usize];
}

// ── Test 6: subgroup ops + shared memory (no builtins) ───────────────

/// `group_i_add` reduce, store to `shared[0]` by item 0, barrier, all read back.
#[spirv(kernel(threads(32)))]
pub fn test_subgroup_ops_with_shared(
    #[spirv(global_invocation_id)] global_id: USizeVec3,
    #[spirv(local_invocation_id)] local_id: USizeVec3,
    #[spirv(cross_workgroup)] output: &mut [u32],
    #[spirv(workgroup)] shared: &mut [u32; 32],
) {
    let gid = global_id.x;
    let lid = local_id.x;
    let total = group_i_add(lid as u32 + 1);

    if lid == 0 {
        shared[0] = total;
    }

    workgroup_memory_barrier_with_group_sync();

    output[gid] = shared[0];
}

// ── Test 7: all three combined ───────────────────────────────────────

/// Subgroup ops + subgroup builtins + shared memory.
#[spirv(kernel(threads(32)))]
pub fn test_all_combined(
    #[spirv(global_invocation_id)] global_id: USizeVec3,
    #[spirv(local_invocation_id)] local_id: USizeVec3,
    #[spirv(subgroup_id)] subgroup_id: u32,
    #[spirv(subgroup_local_invocation_id)] subgroup_local_id: u32,
    #[spirv(cross_workgroup)] output: &mut [u32],
    #[spirv(workgroup)] shared: &mut [u32; 32],
) {
    let gid = global_id.x;
    let lid = local_id.x;
    let total = group_i_add(lid as u32 + 1);

    if subgroup_local_id == 0 {
        shared[subgroup_id as usize] = total;
    }

    workgroup_memory_barrier_with_group_sync();

    output[gid] = shared[subgroup_id as usize];
}

// ── Test 8: work_group_i_add (reduce across whole work-group) ────────

/// Each item contributes `lid+1`; reduce gives sum across the whole
/// work-group (1+2+…+32 = 528 for WG=32).
#[spirv(kernel(threads(32)))]
pub fn test_work_group_reduce(
    #[spirv(global_invocation_id)] global_id: USizeVec3,
    #[spirv(local_invocation_id)] local_id: USizeVec3,
    #[spirv(cross_workgroup)] output: &mut [u32],
) {
    let gid = global_id.x;
    let lid = local_id.x as u32;
    output[gid] = work_group_i_add(lid + 1);
}

// ── Test 9: work_group_inclusive_i_add (inclusive scan) ──────────────

/// Inclusive prefix sum of `(lid+1)` over the whole work-group.
#[spirv(kernel(threads(32)))]
pub fn test_work_group_inclusive_scan(
    #[spirv(global_invocation_id)] global_id: USizeVec3,
    #[spirv(local_invocation_id)] local_id: USizeVec3,
    #[spirv(cross_workgroup)] output: &mut [u32],
) {
    let gid = global_id.x;
    let lid = local_id.x as u32;
    output[gid] = work_group_inclusive_i_add(lid + 1);
}

// ── Test 10: work_group_exclusive_i_add (exclusive scan) ─────────────

/// Exclusive prefix sum of `(lid+1)` over the whole work-group.
#[spirv(kernel(threads(32)))]
pub fn test_work_group_exclusive_scan(
    #[spirv(global_invocation_id)] global_id: USizeVec3,
    #[spirv(local_invocation_id)] local_id: USizeVec3,
    #[spirv(cross_workgroup)] output: &mut [u32],
) {
    let gid = global_id.x;
    let lid = local_id.x as u32;
    output[gid] = work_group_exclusive_i_add(lid + 1);
}

// ── Test 11: work_group_broadcast ────────────────────────────────────

/// Each item writes its own `lid` to its slot, but the whole work-group
/// receives the value from item 7 broadcast back via `work_group_broadcast`.
#[spirv(kernel(threads(32)))]
pub fn test_work_group_broadcast_kernel(
    #[spirv(global_invocation_id)] global_id: USizeVec3,
    #[spirv(local_invocation_id)] local_id: USizeVec3,
    #[spirv(cross_workgroup)] output: &mut [u32],
) {
    let gid = global_id.x;
    let lid = local_id.x as u32;
    output[gid] = work_group_broadcast_u32(lid * 100, 7);
}

// ── Test 12: work_group_all / work_group_any ─────────────────────────

/// Two voting predicates. `lid < 32` is true for every invocation in the
/// WG, so `work_group_all` returns 1. `lid == 5` is true for one invocation,
/// so `work_group_any` returns 1 and `work_group_all` returns 0.
#[spirv(kernel(threads(32)))]
pub fn test_work_group_vote(
    #[spirv(global_invocation_id)] global_id: USizeVec3,
    #[spirv(local_invocation_id)] local_id: USizeVec3,
    #[spirv(cross_workgroup)] out_all_true: &mut [u32],
    #[spirv(cross_workgroup)] out_all_one: &mut [u32],
    #[spirv(cross_workgroup)] out_any_one: &mut [u32],
) {
    let gid = global_id.x;
    let lid = local_id.x as u32;
    out_all_true[gid] = work_group_all(lid < 32) as u32;
    out_all_one[gid] = work_group_all(lid == 5) as u32;
    out_any_one[gid] = work_group_any(lid == 5) as u32;
}
