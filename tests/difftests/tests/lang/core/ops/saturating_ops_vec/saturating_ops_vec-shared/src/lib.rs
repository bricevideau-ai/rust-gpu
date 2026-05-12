#![cfg_attr(target_arch = "spirv", no_std)]

use spirv_std::arch::opencl_std;
use spirv_std::cl::{Int4, UInt4};

pub const N: usize = 256;
pub const THREADS: u32 = 64;
pub const WORKGROUPS: u32 = 1;

const _: () = assert!(
    (THREADS as usize) * (WORKGROUPS as usize) * 4 == N,
    "THREADS * WORKGROUPS * 4 must equal N (one Int4 per work item)",
);

/// Number of u32 outputs per work-item: 4 ops × 4 lanes
/// (`s_add`, `u_add`, `s_sub`, `u_sub`).
pub const FUNCS: usize = 16;

/// Run the four saturating ops per-lane using **Rust stdlib** intrinsics
/// (`i32::saturating_add` / `u32::saturating_*` per lane, no vectorisation).
/// Single source compiled for SPIR-V Kernel and host CPU.
pub fn rust_compute(a: [i32; 4], b: [i32; 4], out: &mut [u32]) {
    for c in 0..4 {
        out[c] = a[c].saturating_add(b[c]) as u32;
    }
    for c in 0..4 {
        out[4 + c] = (a[c] as u32).saturating_add(b[c] as u32);
    }
    for c in 0..4 {
        out[8 + c] = a[c].saturating_sub(b[c]) as u32;
    }
    for c in 0..4 {
        out[12 + c] = (a[c] as u32).saturating_sub(b[c] as u32);
    }
}

/// Run the four saturating ops on `Int4` / `UInt4` vectors using
/// **OpenCL.std** vector intrinsics. On Kernel these lower to a single
/// `OpExtInst <OpenCL.std> *_sat`; on CPU they use spirv-std's host
/// fallback (`num_traits::Saturating` per lane).
pub fn opencl_std_compute(a: [i32; 4], b: [i32; 4], out: &mut [u32]) {
    let va = Int4::new(a[0], a[1], a[2], a[3]);
    let vb = Int4::new(b[0], b[1], b[2], b[3]);
    let ua = UInt4::new(a[0] as u32, a[1] as u32, a[2] as u32, a[3] as u32);
    let ub = UInt4::new(b[0] as u32, b[1] as u32, b[2] as u32, b[3] as u32);

    let s_add = opencl_std::s_add_sat(va, vb).to_array();
    let u_add = opencl_std::u_add_sat(ua, ub).to_array();
    let s_sub = opencl_std::s_sub_sat(va, vb).to_array();
    let u_sub = opencl_std::u_sub_sat(ua, ub).to_array();
    // copy_from_slice doesn't lower in SPIR-V Kernel ("cannot memcpy
    // dynamically sized data") — keep the explicit per-lane loops.
    #[allow(clippy::manual_memcpy)]
    for c in 0..4 {
        out[c] = s_add[c] as u32;
    }
    #[allow(clippy::manual_memcpy)]
    for c in 0..4 {
        out[4 + c] = u_add[c];
    }
    #[allow(clippy::manual_memcpy)]
    for c in 0..4 {
        out[8 + c] = s_sub[c] as u32;
    }
    #[allow(clippy::manual_memcpy)]
    for c in 0..4 {
        out[12 + c] = u_sub[c];
    }
}

/// Build the deterministic input vectors of length `N`. Repeated values
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
