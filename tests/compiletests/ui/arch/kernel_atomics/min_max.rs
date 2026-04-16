// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// ignore-spv1.5
// ignore-spv1.6

// Atomic min/max for signed and unsigned integers.

use spirv_std::arch;
use spirv_std::memory::{Scope, Semantics};
use spirv_std::spirv;

#[spirv(kernel)]
pub fn s_min(
    #[spirv(cross_workgroup)] val: &mut i32,
    operand: i32,
    #[spirv(cross_workgroup)] out: &mut i32,
) {
    *out = unsafe {
        arch::atomic_s_min::<_, { Scope::Workgroup as u32 }, { Semantics::NONE.bits() as u32 }>(
            val, operand,
        )
    };
}

#[spirv(kernel)]
pub fn s_max(
    #[spirv(cross_workgroup)] val: &mut i32,
    operand: i32,
    #[spirv(cross_workgroup)] out: &mut i32,
) {
    *out = unsafe {
        arch::atomic_s_max::<_, { Scope::Workgroup as u32 }, { Semantics::NONE.bits() as u32 }>(
            val, operand,
        )
    };
}

#[spirv(kernel)]
pub fn u_min(
    #[spirv(cross_workgroup)] val: &mut u32,
    operand: u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    *out = unsafe {
        arch::atomic_u_min::<_, { Scope::Workgroup as u32 }, { Semantics::NONE.bits() as u32 }>(
            val, operand,
        )
    };
}

#[spirv(kernel)]
pub fn u_max(
    #[spirv(cross_workgroup)] val: &mut u32,
    operand: u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    *out = unsafe {
        arch::atomic_u_max::<_, { Scope::Workgroup as u32 }, { Semantics::NONE.bits() as u32 }>(
            val, operand,
        )
    };
}
