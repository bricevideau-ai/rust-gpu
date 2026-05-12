#![cfg_attr(target_arch = "spirv", no_std)]

use spirv_std::arch::opencl_std;

pub const N: usize = 256;
pub const THREADS: u32 = 64;
pub const WORKGROUPS: u32 = 4;

const _: () = assert!(
    (THREADS as usize) * (WORKGROUPS as usize) == N,
    "THREADS * WORKGROUPS must equal N (one work item per input element)",
);

/// Number of u32 outputs per work-item: 4 saturating ops
/// (`s_add`, `u_add`, `s_sub`, `u_sub`).
pub const FUNCS: usize = 4;

/// Run the four saturating ops using **Rust stdlib** intrinsics
/// (`i32::saturating_add` / `u32::saturating_add` / `_sub` variants).
/// Single source compiled for SPIR-V Kernel and host CPU.
pub fn rust_compute(a_signed: i32, b_signed: i32, out: &mut [u32]) {
    let a_unsigned = a_signed as u32;
    let b_unsigned = b_signed as u32;
    out[0] = a_signed.saturating_add(b_signed) as u32;
    out[1] = a_unsigned.saturating_add(b_unsigned);
    out[2] = a_signed.saturating_sub(b_signed) as u32;
    out[3] = a_unsigned.saturating_sub(b_unsigned);
}

/// Run the four saturating ops using **OpenCL.std** intrinsics
/// (`opencl_std::s_add_sat` / `u_add_sat` / `s_sub_sat` / `u_sub_sat`).
/// On Kernel these lower to `OpExtInst <OpenCL.std> *_sat`; on CPU they
/// use spirv-std's host fallback (`num_traits::Saturating`).
pub fn opencl_std_compute(a_signed: i32, b_signed: i32, out: &mut [u32]) {
    let a_unsigned = a_signed as u32;
    let b_unsigned = b_signed as u32;
    out[0] = opencl_std::s_add_sat(a_signed, b_signed) as u32;
    out[1] = opencl_std::u_add_sat(a_unsigned, b_unsigned);
    out[2] = opencl_std::s_sub_sat(a_signed, b_signed) as u32;
    out[3] = opencl_std::u_sub_sat(a_unsigned, b_unsigned);
}

/// Build the deterministic input vectors of length `N` covering edge cases
/// (MAX, MIN, ±1, 0) and a deterministic non-trivial pattern. Repeated values
/// across arms are intentional — each `i % 8` slot maps to a meaningful pair.
#[cfg(not(target_arch = "spirv"))]
#[allow(clippy::match_same_arms)]
pub fn make_inputs() -> (Vec<i32>, Vec<i32>) {
    let input_a: Vec<i32> = (0..N)
        .map(|i| match i % 8 {
            0 => i32::MAX,
            1 => i32::MAX - 1,
            2 => i32::MIN,
            3 => i32::MIN + 1,
            4 => 0,
            5 => 1,
            6 => -1,
            _ => (i as i32).wrapping_mul(0x0135_7BDF),
        })
        .collect();
    let input_b: Vec<i32> = (0..N)
        .map(|i| match i % 8 {
            0 => 1,
            1 => i32::MAX,
            2 => -1,
            3 => i32::MIN,
            4 => i32::MAX,
            5 => i32::MIN,
            6 => 0,
            _ => (i as i32).wrapping_mul(0x0246_8ACE),
        })
        .collect();
    (input_a, input_b)
}
