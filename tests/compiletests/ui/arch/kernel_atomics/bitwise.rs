// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

// Atomic bitwise: and / or / xor.

use spirv_std::arch;
use spirv_std::memory::{Scope, Semantics};
use spirv_std::spirv;

#[spirv(kernel)]
pub fn and(
    #[spirv(cross_workgroup)] val: &mut u32,
    operand: u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    *out = unsafe {
        arch::atomic_and::<_, { Scope::Workgroup as u32 }, { Semantics::NONE.bits() as u32 }>(
            val, operand,
        )
    };
}

#[spirv(kernel)]
pub fn or(
    #[spirv(cross_workgroup)] val: &mut u32,
    operand: u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    *out = unsafe {
        arch::atomic_or::<_, { Scope::Workgroup as u32 }, { Semantics::NONE.bits() as u32 }>(
            val, operand,
        )
    };
}

#[spirv(kernel)]
pub fn xor(
    #[spirv(cross_workgroup)] val: &mut u32,
    operand: u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    *out = unsafe {
        arch::atomic_xor::<_, { Scope::Workgroup as u32 }, { Semantics::NONE.bits() as u32 }>(
            val, operand,
        )
    };
}
