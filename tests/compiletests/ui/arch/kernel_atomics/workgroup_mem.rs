// build-pass
// ignore-vulkan1.1
// ignore-vulkan1.2
// ignore-vulkan1.3
// ignore-vulkan1.4
// ignore-spv1.3
// ignore-spv1.4
// ignore-spv1.5
// ignore-spv1.6

// Atomic op against a Workgroup-storage-class variable rather than a
// CrossWorkgroup buffer.

use spirv_std::arch;
use spirv_std::memory::{Scope, Semantics};
use spirv_std::spirv;

#[spirv(kernel(threads(32)))]
pub fn main(#[spirv(workgroup)] shared: &mut [u32; 1], #[spirv(cross_workgroup)] out: &mut u32) {
    *out = unsafe {
        arch::atomic_i_add::<_, { Scope::Workgroup as u32 }, { Semantics::NONE.bits() as u32 }>(
            &mut shared[0],
            1u32,
        )
    };
}
