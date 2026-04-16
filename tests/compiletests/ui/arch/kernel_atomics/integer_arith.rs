// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

// Integer atomic arithmetic: i_increment / i_decrement / i_add / i_sub,
// covering Workgroup + Device scopes and NONE + SeqCst semantics.

use spirv_std::arch;
use spirv_std::memory::{Scope, Semantics};
use spirv_std::spirv;

#[spirv(kernel)]
pub fn i_increment(
    #[spirv(cross_workgroup)] val: &mut u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    *out = unsafe {
        arch::atomic_i_increment::<_, { Scope::Workgroup as u32 }, { Semantics::NONE.bits() as u32 }>(
            val,
        )
    };
}

#[spirv(kernel)]
pub fn i_decrement(
    #[spirv(cross_workgroup)] val: &mut u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    *out = unsafe {
        arch::atomic_i_decrement::<_, { Scope::Workgroup as u32 }, { Semantics::NONE.bits() as u32 }>(
            val,
        )
    };
}

#[spirv(kernel)]
pub fn i_add(
    #[spirv(cross_workgroup)] val: &mut u32,
    operand: u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    *out = unsafe {
        arch::atomic_i_add::<_, { Scope::Workgroup as u32 }, { Semantics::NONE.bits() as u32 }>(
            val, operand,
        )
    };
}

#[spirv(kernel)]
pub fn i_sub(
    #[spirv(cross_workgroup)] val: &mut u32,
    operand: u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    *out = unsafe {
        arch::atomic_i_sub::<_, { Scope::Workgroup as u32 }, { Semantics::NONE.bits() as u32 }>(
            val, operand,
        )
    };
}

#[spirv(kernel)]
pub fn i_add_device_scope(
    #[spirv(cross_workgroup)] val: &mut u32,
    operand: u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    *out = unsafe {
        arch::atomic_i_add::<_, { Scope::Device as u32 }, { Semantics::NONE.bits() as u32 }>(
            val, operand,
        )
    };
}

#[spirv(kernel)]
pub fn i_add_seqcst(
    #[spirv(cross_workgroup)] val: &mut u32,
    operand: u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    *out = unsafe {
        arch::atomic_i_add::<
            _,
            { Scope::Workgroup as u32 },
            { Semantics::SEQUENTIALLY_CONST.bits() as u32 },
        >(val, operand)
    };
}
