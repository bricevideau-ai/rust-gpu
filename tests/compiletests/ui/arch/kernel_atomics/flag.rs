// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// ignore-spv1.5
// ignore-spv1.6

// Kernel-only `OpAtomicFlag*` operations.

use spirv_std::arch;
use spirv_std::memory::{Scope, Semantics};
use spirv_std::spirv;

#[spirv(kernel)]
pub fn test_and_set(
    #[spirv(cross_workgroup)] flag: &mut u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    let was_set = unsafe {
        arch::atomic_flag_test_and_set::<
            { Scope::Workgroup as u32 },
            { Semantics::NONE.bits() as u32 },
        >(flag)
    };
    *out = was_set as u32;
}

#[spirv(kernel)]
pub fn clear(#[spirv(cross_workgroup)] flag: &mut u32) {
    unsafe {
        arch::atomic_flag_clear::<{ Scope::Workgroup as u32 }, { Semantics::NONE.bits() as u32 }>(
            flag,
        );
    }
}

#[spirv(kernel)]
pub fn spinlock(#[spirv(cross_workgroup)] flag: &mut u32, #[spirv(cross_workgroup)] val: &mut u32) {
    unsafe {
        arch::atomic_flag_test_and_set::<
            { Scope::Device as u32 },
            { Semantics::ACQUIRE.bits() as u32 },
        >(flag);
    }
    *val += 1;
    unsafe {
        arch::atomic_flag_clear::<{ Scope::Device as u32 }, { Semantics::RELEASE.bits() as u32 }>(
            flag,
        );
    }
}
