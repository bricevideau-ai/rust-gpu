#![cfg_attr(target_arch = "spirv", no_std)]

use spirv_std::arch::opencl_std;
use spirv_std::cl::Double4;

pub const N: usize = 256;
pub const THREADS: u32 = 64;
pub const WORKGROUPS: u32 = 1;

const _: () = assert!(
    (THREADS as usize) * (WORKGROUPS as usize) * 4 == N,
    "THREADS * WORKGROUPS * 4 must equal N (one Double4 per work item)",
);

pub const UNARY: usize = 31;
pub const BINARY: usize = 10;
pub const TERNARY: usize = 4;
pub const FUNCS: usize = UNARY + BINARY + TERNARY;

fn write4(output: &mut [f64], offset: usize, v: Double4) {
    let a = v.to_array();
    output[offset] = a[0];
    output[offset + 1] = a[1];
    output[offset + 2] = a[2];
    output[offset + 3] = a[3];
}

/// Run all `FUNCS` Double4 intrinsics into `vec_out` (length `4 * FUNCS`).
/// Single source compiled for SPIR-V Kernel and host CPU.
pub fn compute(va: Double4, vb: Double4, vc: Double4, vec_out: &mut [f64]) {
    let mut fi = 0usize;

    // --- Unary (31) ---
    write4(vec_out, fi * 4, opencl_std::acos(va));
    fi += 1;
    write4(vec_out, fi * 4, opencl_std::acosh(vb));
    fi += 1;
    write4(vec_out, fi * 4, opencl_std::asin(va));
    fi += 1;
    write4(vec_out, fi * 4, opencl_std::asinh(va));
    fi += 1;
    write4(vec_out, fi * 4, opencl_std::atan(va));
    fi += 1;
    write4(vec_out, fi * 4, opencl_std::atanh(va));
    fi += 1;
    write4(vec_out, fi * 4, opencl_std::cbrt(va));
    fi += 1;
    write4(vec_out, fi * 4, opencl_std::ceil(va));
    fi += 1;
    write4(vec_out, fi * 4, opencl_std::cos(va));
    fi += 1;
    write4(vec_out, fi * 4, opencl_std::cosh(va));
    fi += 1;
    write4(vec_out, fi * 4, opencl_std::exp(va));
    fi += 1;
    write4(vec_out, fi * 4, opencl_std::exp2(va));
    fi += 1;
    write4(vec_out, fi * 4, opencl_std::exp10(va));
    fi += 1;
    write4(vec_out, fi * 4, opencl_std::fabs(va));
    fi += 1;
    write4(vec_out, fi * 4, opencl_std::floor(va));
    fi += 1;
    write4(vec_out, fi * 4, opencl_std::log(vb));
    fi += 1;
    write4(vec_out, fi * 4, opencl_std::log2(vb));
    fi += 1;
    write4(vec_out, fi * 4, opencl_std::log10(vb));
    fi += 1;
    write4(vec_out, fi * 4, opencl_std::round(va));
    fi += 1;
    write4(vec_out, fi * 4, opencl_std::rsqrt(vb));
    fi += 1;
    write4(vec_out, fi * 4, opencl_std::sin(va));
    fi += 1;
    write4(vec_out, fi * 4, opencl_std::sinh(va));
    fi += 1;
    write4(vec_out, fi * 4, opencl_std::sqrt(vb));
    fi += 1;
    write4(vec_out, fi * 4, opencl_std::tan(va));
    fi += 1;
    write4(vec_out, fi * 4, opencl_std::tanh(va));
    fi += 1;
    write4(vec_out, fi * 4, opencl_std::trunc(va));
    fi += 1;
    write4(vec_out, fi * 4, opencl_std::sign(va));
    fi += 1;
    write4(vec_out, fi * 4, opencl_std::erf(va));
    fi += 1;
    write4(vec_out, fi * 4, opencl_std::erfc(va));
    fi += 1;
    write4(vec_out, fi * 4, opencl_std::lgamma(vb));
    fi += 1;
    write4(vec_out, fi * 4, opencl_std::tgamma(vb));
    fi += 1;

    // --- Binary (10) ---
    write4(vec_out, fi * 4, opencl_std::atan2(va, vb));
    fi += 1;
    write4(vec_out, fi * 4, opencl_std::copysign(va, vb));
    fi += 1;
    write4(vec_out, fi * 4, opencl_std::fmax(va, vb));
    fi += 1;
    write4(vec_out, fi * 4, opencl_std::fmin(va, vb));
    fi += 1;
    write4(vec_out, fi * 4, opencl_std::fmod(va, vb));
    fi += 1;
    write4(vec_out, fi * 4, opencl_std::hypot(va, vb));
    fi += 1;
    write4(vec_out, fi * 4, opencl_std::pow(va, vb));
    fi += 1;
    write4(vec_out, fi * 4, opencl_std::fdim(va, vb));
    fi += 1;
    write4(vec_out, fi * 4, opencl_std::nextafter(va, vb));
    fi += 1;
    write4(vec_out, fi * 4, opencl_std::remainder(va, vb));
    fi += 1;

    // --- Ternary (4) ---
    write4(vec_out, fi * 4, opencl_std::fma(va, vb, vc));
    fi += 1;
    write4(
        vec_out,
        fi * 4,
        opencl_std::clamp(va, Double4::splat(0.0), Double4::splat(1.0)),
    );
    fi += 1;
    write4(vec_out, fi * 4, opencl_std::mix(va, vb, vc));
    fi += 1;
    write4(
        vec_out,
        fi * 4,
        opencl_std::smoothstep(Double4::splat(0.0), Double4::splat(1.0), va),
    );
    fi += 1;

    #[cfg(not(target_arch = "spirv"))]
    debug_assert_eq!(fi, FUNCS);
    #[cfg(target_arch = "spirv")]
    let _ = fi;
}

#[cfg(not(target_arch = "spirv"))]
pub fn make_inputs() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mk = |lo: f64, hi: f64| -> Vec<f64> {
        (0..N)
            .map(|i| {
                let t = (i as f64) / (N - 1) as f64;
                lo + t * (hi - lo)
            })
            .collect()
    };
    (mk(0.1, 0.9), mk(1.0, 3.0), mk(0.0, 1.0))
}
