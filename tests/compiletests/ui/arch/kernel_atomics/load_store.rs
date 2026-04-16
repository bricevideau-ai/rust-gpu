// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

// `arch::atomic_load` / `arch::atomic_store`, with NONE and SeqCst semantics.

use spirv_std::arch;
use spirv_std::memory::{Scope, Semantics};
use spirv_std::spirv;

#[spirv(kernel)]
pub fn load(#[spirv(cross_workgroup)] val: &u32, #[spirv(cross_workgroup)] out: &mut u32) {
    *out = unsafe {
        arch::atomic_load::<_, { Scope::Workgroup as u32 }, { Semantics::NONE.bits() as u32 }>(val)
    };
}

#[spirv(kernel)]
pub fn store(#[spirv(cross_workgroup)] val: &mut u32) {
    unsafe {
        arch::atomic_store::<_, { Scope::Workgroup as u32 }, { Semantics::NONE.bits() as u32 }>(
            val, 42u32,
        );
    }
}

#[spirv(kernel)]
pub fn load_seqcst(#[spirv(cross_workgroup)] val: &u32, #[spirv(cross_workgroup)] out: &mut u32) {
    *out = unsafe {
        arch::atomic_load::<
            _,
            { Scope::Workgroup as u32 },
            { Semantics::SEQUENTIALLY_CONST.bits() as u32 },
        >(val)
    };
}

#[spirv(kernel)]
pub fn store_seqcst(#[spirv(cross_workgroup)] val: &mut u32) {
    unsafe {
        arch::atomic_store::<
            _,
            { Scope::Workgroup as u32 },
            { Semantics::SEQUENTIALLY_CONST.bits() as u32 },
        >(val, 42u32);
    }
}
