// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4

// Test all atomic operations in kernel context.
// These cover integer atomics, flag operations, various scopes and memory
// semantics including SeqCst (valid on OpenCL, forbidden on Vulkan).

use spirv_std::arch;
use spirv_std::memory::{Scope, Semantics};
use spirv_std::spirv;

// --- Category A: All integer atomics (Workgroup scope, NONE semantics) ---

#[spirv(kernel)]
pub fn test_atomic_load(
    #[spirv(cross_workgroup)] val: &u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    let loaded = unsafe {
        arch::atomic_load::<_, { Scope::Workgroup as u32 }, { Semantics::NONE.bits() as u32 }>(val)
    };
    *out = loaded;
}

#[spirv(kernel)]
pub fn test_atomic_store(#[spirv(cross_workgroup)] val: &mut u32) {
    unsafe {
        arch::atomic_store::<_, { Scope::Workgroup as u32 }, { Semantics::NONE.bits() as u32 }>(
            val, 42u32,
        );
    }
}

#[spirv(kernel)]
pub fn test_atomic_exchange(
    #[spirv(cross_workgroup)] val: &mut u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    let old = unsafe {
        arch::atomic_exchange::<_, { Scope::Workgroup as u32 }, { Semantics::NONE.bits() as u32 }>(
            val, 99u32,
        )
    };
    *out = old;
}

#[spirv(kernel)]
pub fn test_atomic_compare_exchange(
    #[spirv(cross_workgroup)] val: &mut u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    let old = unsafe {
        arch::atomic_compare_exchange::<
            _,
            { Scope::Workgroup as u32 },
            { Semantics::NONE.bits() as u32 },
            { Semantics::NONE.bits() as u32 },
        >(val, 1u32, 0u32)
    };
    *out = old;
}

#[spirv(kernel)]
pub fn test_atomic_i_increment(
    #[spirv(cross_workgroup)] val: &mut u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    let old = unsafe {
        arch::atomic_i_increment::<_, { Scope::Workgroup as u32 }, { Semantics::NONE.bits() as u32 }>(
            val,
        )
    };
    *out = old;
}

#[spirv(kernel)]
pub fn test_atomic_i_decrement(
    #[spirv(cross_workgroup)] val: &mut u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    let old = unsafe {
        arch::atomic_i_decrement::<_, { Scope::Workgroup as u32 }, { Semantics::NONE.bits() as u32 }>(
            val,
        )
    };
    *out = old;
}

#[spirv(kernel)]
pub fn test_atomic_i_add(
    #[spirv(cross_workgroup)] val: &mut u32,
    operand: u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    let old = unsafe {
        arch::atomic_i_add::<_, { Scope::Workgroup as u32 }, { Semantics::NONE.bits() as u32 }>(
            val, operand,
        )
    };
    *out = old;
}

#[spirv(kernel)]
pub fn test_atomic_i_sub(
    #[spirv(cross_workgroup)] val: &mut u32,
    operand: u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    let old = unsafe {
        arch::atomic_i_sub::<_, { Scope::Workgroup as u32 }, { Semantics::NONE.bits() as u32 }>(
            val, operand,
        )
    };
    *out = old;
}

#[spirv(kernel)]
pub fn test_atomic_s_min(
    #[spirv(cross_workgroup)] val: &mut i32,
    operand: i32,
    #[spirv(cross_workgroup)] out: &mut i32,
) {
    let old = unsafe {
        arch::atomic_s_min::<_, { Scope::Workgroup as u32 }, { Semantics::NONE.bits() as u32 }>(
            val, operand,
        )
    };
    *out = old;
}

#[spirv(kernel)]
pub fn test_atomic_s_max(
    #[spirv(cross_workgroup)] val: &mut i32,
    operand: i32,
    #[spirv(cross_workgroup)] out: &mut i32,
) {
    let old = unsafe {
        arch::atomic_s_max::<_, { Scope::Workgroup as u32 }, { Semantics::NONE.bits() as u32 }>(
            val, operand,
        )
    };
    *out = old;
}

#[spirv(kernel)]
pub fn test_atomic_u_min(
    #[spirv(cross_workgroup)] val: &mut u32,
    operand: u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    let old = unsafe {
        arch::atomic_u_min::<_, { Scope::Workgroup as u32 }, { Semantics::NONE.bits() as u32 }>(
            val, operand,
        )
    };
    *out = old;
}

#[spirv(kernel)]
pub fn test_atomic_u_max(
    #[spirv(cross_workgroup)] val: &mut u32,
    operand: u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    let old = unsafe {
        arch::atomic_u_max::<_, { Scope::Workgroup as u32 }, { Semantics::NONE.bits() as u32 }>(
            val, operand,
        )
    };
    *out = old;
}

#[spirv(kernel)]
pub fn test_atomic_and(
    #[spirv(cross_workgroup)] val: &mut u32,
    operand: u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    let old = unsafe {
        arch::atomic_and::<_, { Scope::Workgroup as u32 }, { Semantics::NONE.bits() as u32 }>(
            val, operand,
        )
    };
    *out = old;
}

#[spirv(kernel)]
pub fn test_atomic_or(
    #[spirv(cross_workgroup)] val: &mut u32,
    operand: u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    let old = unsafe {
        arch::atomic_or::<_, { Scope::Workgroup as u32 }, { Semantics::NONE.bits() as u32 }>(
            val, operand,
        )
    };
    *out = old;
}

#[spirv(kernel)]
pub fn test_atomic_xor(
    #[spirv(cross_workgroup)] val: &mut u32,
    operand: u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    let old = unsafe {
        arch::atomic_xor::<_, { Scope::Workgroup as u32 }, { Semantics::NONE.bits() as u32 }>(
            val, operand,
        )
    };
    *out = old;
}

// --- Category B: Device scope ---

#[spirv(kernel)]
pub fn test_atomic_i_add_device_scope(
    #[spirv(cross_workgroup)] val: &mut u32,
    operand: u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    let old = unsafe {
        arch::atomic_i_add::<_, { Scope::Device as u32 }, { Semantics::NONE.bits() as u32 }>(
            val, operand,
        )
    };
    *out = old;
}

#[spirv(kernel)]
pub fn test_atomic_exchange_device_scope(
    #[spirv(cross_workgroup)] val: &mut u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    let old = unsafe {
        arch::atomic_exchange::<_, { Scope::Device as u32 }, { Semantics::NONE.bits() as u32 }>(
            val, 99u32,
        )
    };
    *out = old;
}

// --- Category C: SeqCst semantics (valid on OpenCL, forbidden on Vulkan) ---

#[spirv(kernel)]
pub fn test_atomic_load_seqcst(
    #[spirv(cross_workgroup)] val: &u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    let loaded = unsafe {
        arch::atomic_load::<
            _,
            { Scope::Workgroup as u32 },
            { Semantics::SEQUENTIALLY_CONST.bits() as u32 },
        >(val)
    };
    *out = loaded;
}

#[spirv(kernel)]
pub fn test_atomic_store_seqcst(#[spirv(cross_workgroup)] val: &mut u32) {
    unsafe {
        arch::atomic_store::<
            _,
            { Scope::Workgroup as u32 },
            { Semantics::SEQUENTIALLY_CONST.bits() as u32 },
        >(val, 42u32);
    }
}

#[spirv(kernel)]
pub fn test_atomic_i_add_seqcst(
    #[spirv(cross_workgroup)] val: &mut u32,
    operand: u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    let old = unsafe {
        arch::atomic_i_add::<
            _,
            { Scope::Workgroup as u32 },
            { Semantics::SEQUENTIALLY_CONST.bits() as u32 },
        >(val, operand)
    };
    *out = old;
}

// --- Category D: AcquireRelease semantics ---

#[spirv(kernel)]
pub fn test_atomic_exchange_acq_rel(
    #[spirv(cross_workgroup)] val: &mut u32,
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    let old = unsafe {
        arch::atomic_exchange::<
            _,
            { Scope::Workgroup as u32 },
            { Semantics::ACQUIRE_RELEASE.bits() as u32 },
        >(val, 99u32)
    };
    *out = old;
}

// --- Category E: Workgroup (shared) memory atomics ---

#[spirv(kernel(threads(32)))]
pub fn test_atomic_i_add_workgroup_mem(
    #[spirv(workgroup)] shared: &mut [u32; 1],
    #[spirv(cross_workgroup)] out: &mut u32,
) {
    let old = unsafe {
        arch::atomic_i_add::<_, { Scope::Workgroup as u32 }, { Semantics::NONE.bits() as u32 }>(
            &mut shared[0],
            1u32,
        )
    };
    *out = old;
}

// --- Category F: Flag operations (Kernel-only) ---

#[spirv(kernel)]
pub fn test_atomic_flag_test_and_set(
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
pub fn test_atomic_flag_clear(#[spirv(cross_workgroup)] flag: &mut u32) {
    unsafe {
        arch::atomic_flag_clear::<{ Scope::Workgroup as u32 }, { Semantics::NONE.bits() as u32 }>(
            flag,
        );
    }
}

#[spirv(kernel)]
pub fn test_atomic_flag_spinlock(
    #[spirv(cross_workgroup)] flag: &mut u32,
    #[spirv(cross_workgroup)] val: &mut u32,
) {
    // Acquire the lock.
    unsafe {
        arch::atomic_flag_test_and_set::<
            { Scope::Device as u32 },
            { Semantics::ACQUIRE.bits() as u32 },
        >(flag);
    }

    // Critical section.
    *val += 1;

    // Release the lock.
    unsafe {
        arch::atomic_flag_clear::<{ Scope::Device as u32 }, { Semantics::RELEASE.bits() as u32 }>(
            flag,
        );
    }
}
