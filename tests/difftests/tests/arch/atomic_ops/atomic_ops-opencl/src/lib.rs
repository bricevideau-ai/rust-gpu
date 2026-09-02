#![cfg_attr(target_arch = "spirv", no_std)]

use spirv_std::arch::{atomic_i_add, atomic_i_sub, atomic_u_max, atomic_u_min};
use spirv_std::memory::{Scope, Semantics};
use spirv_std::spirv;

#[spirv(kernel(threads(32)))]
pub fn main_kernel(
    #[spirv(cross_workgroup)] counters: &mut [u32],
    #[spirv(cross_workgroup)] output: &mut [u32],
    #[spirv(global_invocation_id)] global_id: spirv_std::glam::UVec3,
) {
    const SCOPE: u32 = Scope::Workgroup as u32;
    const SEMANTICS: u32 = Semantics::NONE.bits();

    let tid = global_id.x;

    unsafe { atomic_i_add::<_, SCOPE, SEMANTICS>(&mut counters[0], 1) };
    unsafe { atomic_i_sub::<_, SCOPE, SEMANTICS>(&mut counters[1], 1) };
    unsafe { atomic_u_min::<_, SCOPE, SEMANTICS>(&mut counters[2], tid) };
    unsafe { atomic_u_max::<_, SCOPE, SEMANTICS>(&mut counters[3], tid) };

    if tid == 0 {
        spirv_std::arch::workgroup_memory_barrier_with_group_sync();
        output[0] = counters[0];
        output[1] = counters[1];
        output[2] = counters[2];
        output[3] = counters[3];
        output[4] = counters[4];
    }
}
