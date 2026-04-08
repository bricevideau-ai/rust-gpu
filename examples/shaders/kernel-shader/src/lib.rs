#![cfg_attr(target_arch = "spirv", no_std)]
// HACK(eddyb) can't easily see warnings otherwise from `spirv-builder` builds.
#![deny(warnings)]

use glam::USizeVec3;
use spirv_std::{glam, spirv};

// Adapted from the compute-shader example for OpenCL kernels.

/// Returns the length of the Collatz sequence (excluding the starting number) for `n`. Returns
/// `None` if (a) `n` is zero, or (b) a number in the sequence overflows a `u32`.
///
/// # Examples
///
/// The sequence for 3 (excluding the starting number) is `[10, 5, 16, 8, 4, 2, 1]`, which has
/// length 7.
/// ```
/// # use kernel_shader::collatz;
/// assert_eq!(collatz(3), Some(7));
/// ```
pub fn collatz(mut n: u32) -> Option<u32> {
    let mut i = 0;
    if n == 0 {
        return None;
    }
    while n != 1 {
        n = if n.is_multiple_of(2) {
            n / 2
        } else {
            if n >= 0x5555_5555 {
                return None;
            }
            3 * n + 1
        };
        i += 1;
    }
    Some(i)
}

// OpenCL kernel entry point. Unlike Vulkan compute shaders, kernel entry points:
// - Use the Kernel execution model (not GLCompute)
// - Do not require threads()/LocalSize at compile time (set at dispatch)
// - Use CrossWorkgroup storage class for global memory buffers
//
// Slices (&mut [u32]) are decomposed into two kernel arguments: a pointer to
// the element type and a length. The host sets both via clSetKernelArg.
//
#[spirv(kernel)]
pub fn main_kernel(
    #[spirv(global_invocation_id)] id: USizeVec3,
    #[spirv(cross_workgroup)] prime_indices: &mut [u32],
) {
    let index = id.x;
    prime_indices[index] = collatz(prime_indices[index]).unwrap_or(u32::MAX);
}

/// Regression test for kernel argument ordering. The runner sets args in
/// Rust source order and verifies the slice received `(scalar_a, scalar_b)`
/// at indices `[0, 1]`. If the linker mis-orders the `(slice_ptr, slice_len,
/// scalar_a, scalar_b)` parameters, the kernel reads `scalar_b` into the
/// `scalar_a` slot or vice versa, and the runtime check fails.
#[spirv(kernel)]
pub fn arg_ordering_test(
    #[spirv(global_invocation_id)] id: USizeVec3,
    #[spirv(cross_workgroup)] data: &mut [u32],
    scalar_a: u32,
    scalar_b: u32,
) {
    if id.x == 0 {
        data[0] = scalar_a;
        data[1] = scalar_b;
    }
}
