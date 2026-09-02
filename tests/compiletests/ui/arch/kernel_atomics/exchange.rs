// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// ignore-spv1.5
// ignore-spv1.6

// `arch::atomic_exchange` / `arch::atomic_compare_exchange`, with various
// scopes and memory semantics.

use spirv_std::arch;
use spirv_std::memory::{Scope, Semantics};
use spirv_std::spirv;

#[spirv(kernel)]
pub fn exchange(#[spirv(cross_workgroup)] val: &mut u32, #[spirv(cross_workgroup)] out: &mut u32) {
    *out = unsafe {
        arch::atomic_exchange::<_, { Scope::Workgroup as u32 }, { Semantics::NONE.bits() as u32 }>(
            val, 99u32,
        )
    };
}

#[spirv(kernel)]
pub fn compare_exchange(
    #[spirv(cross_workgroup)] val: &mut u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    *out = unsafe {
        arch::atomic_compare_exchange::<
            _,
            { Scope::Workgroup as u32 },
            { Semantics::NONE.bits() as u32 },
            { Semantics::NONE.bits() as u32 },
        >(val, 1u32, 0u32)
    };
}

#[spirv(kernel)]
pub fn exchange_device_scope(
    #[spirv(cross_workgroup)] val: &mut u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    *out = unsafe {
        arch::atomic_exchange::<_, { Scope::Device as u32 }, { Semantics::NONE.bits() as u32 }>(
            val, 99u32,
        )
    };
}

#[spirv(kernel)]
pub fn exchange_acq_rel(
    #[spirv(cross_workgroup)] val: &mut u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    *out = unsafe {
        arch::atomic_exchange::<
            _,
            { Scope::Workgroup as u32 },
            { Semantics::ACQUIRE_RELEASE.bits() as u32 },
        >(val, 99u32)
    };
}
