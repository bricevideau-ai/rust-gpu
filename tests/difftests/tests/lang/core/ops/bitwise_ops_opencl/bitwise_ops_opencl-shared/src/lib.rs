#![cfg_attr(target_arch = "spirv", no_std)]

pub const N: usize = 256;
pub const THREADS: u32 = 64;
pub const WORKGROUPS: u32 = 4;

const _: () = assert!(
    (THREADS as usize) * (WORKGROUPS as usize) == N,
    "THREADS * WORKGROUPS must equal N (one work item per input element)",
);

/// Compute one of 12 bitwise / shift / count operations selected by `tid % 12`.
/// Single source compiled for SPIR-V Kernel and host CPU.
pub fn compute(tid: usize, a: u32, b: u32) -> u32 {
    match tid % 12 {
        0 => a & b,
        1 => a | b,
        2 => a ^ b,
        3 => !a,
        4 => a << (b % 32),
        5 => a >> (b % 32),
        6 => a.rotate_left(b % 32),
        7 => a.rotate_right(b % 32),
        8 => a.count_ones(),
        9 => a.leading_zeros(),
        10 => a.trailing_zeros(),
        11 => a.reverse_bits(),
        _ => 0,
    }
}

#[cfg(not(target_arch = "spirv"))]
pub fn make_inputs() -> (Vec<u32>, Vec<u32>) {
    let input_a: Vec<u32> = (0..N as u32).map(|i| i.wrapping_mul(0x9E37_79B9)).collect();
    let input_b: Vec<u32> = (0..N as u32).collect();
    (input_a, input_b)
}
