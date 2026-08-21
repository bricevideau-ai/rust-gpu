#![cfg_attr(target_arch = "spirv", no_std)]

#[allow(unused_imports)]
use spirv_std::num_traits::Float;

use spirv_std::arch::opencl_std;

pub const N: usize = 256;
pub const THREADS: u32 = 64;
pub const WORKGROUPS: u32 = 4;

const _: () = assert!(
    (THREADS as usize) * (WORKGROUPS as usize) == N,
    "THREADS * WORKGROUPS must equal N (one work item per input element)",
);

/// Number of f64 outputs per work-item: 33 functions in the intersection of
/// `num_traits::Float` trait methods and `opencl_std::*` functions.
pub const FUNCS: usize = 33;

/// Run all `FUNCS` math operations on `(a, b, c)` using **idiomatic Rust**
/// calls (`a.sin()`, `a.powf(b)`, etc.). Slot order matches `opencl_std_compute`.
pub fn rust_compute(a: f64, b: f64, c: f64, out: &mut [f64]) {
    let mut fi = 0usize;

    out[fi] = a.sin();
    fi += 1;
    out[fi] = a.cos();
    fi += 1;
    out[fi] = a.tan();
    fi += 1;
    out[fi] = a.asin();
    fi += 1;
    out[fi] = a.acos();
    fi += 1;
    out[fi] = a.atan();
    fi += 1;
    out[fi] = a.sinh();
    fi += 1;
    out[fi] = a.cosh();
    fi += 1;
    out[fi] = a.tanh();
    fi += 1;
    out[fi] = a.asinh();
    fi += 1;
    out[fi] = b.acosh();
    fi += 1;
    out[fi] = a.atanh();
    fi += 1;
    out[fi] = a.exp();
    fi += 1;
    out[fi] = a.exp2();
    fi += 1;
    out[fi] = b.ln();
    fi += 1;
    out[fi] = b.log2();
    fi += 1;
    out[fi] = b.log10();
    fi += 1;
    out[fi] = b.sqrt();
    fi += 1;
    out[fi] = a.cbrt();
    fi += 1;
    out[fi] = a.abs();
    fi += 1;
    out[fi] = a.floor();
    fi += 1;
    out[fi] = a.ceil();
    fi += 1;
    out[fi] = a.round();
    fi += 1;
    out[fi] = a.trunc();
    fi += 1;
    out[fi] = a.signum();
    fi += 1;
    out[fi] = a.atan2(b);
    fi += 1;
    out[fi] = a.hypot(b);
    fi += 1;
    out[fi] = a.powf(b);
    fi += 1;
    out[fi] = a.copysign(b);
    fi += 1;
    out[fi] = a.max(b);
    fi += 1;
    out[fi] = a.min(b);
    fi += 1;
    out[fi] = a.mul_add(b, c);
    fi += 1;
    out[fi] = a.clamp(0.0, 1.0);
    fi += 1;

    #[cfg(not(target_arch = "spirv"))]
    debug_assert_eq!(fi, FUNCS);
    #[cfg(target_arch = "spirv")]
    let _ = fi;
}

/// Run all `FUNCS` math operations on `(a, b, c)` using explicit
/// `opencl_std::*` calls. Slot order matches `rust_compute`.
pub fn opencl_std_compute(a: f64, b: f64, c: f64, out: &mut [f64]) {
    let mut fi = 0usize;

    out[fi] = opencl_std::sin(a);
    fi += 1;
    out[fi] = opencl_std::cos(a);
    fi += 1;
    out[fi] = opencl_std::tan(a);
    fi += 1;
    out[fi] = opencl_std::asin(a);
    fi += 1;
    out[fi] = opencl_std::acos(a);
    fi += 1;
    out[fi] = opencl_std::atan(a);
    fi += 1;
    out[fi] = opencl_std::sinh(a);
    fi += 1;
    out[fi] = opencl_std::cosh(a);
    fi += 1;
    out[fi] = opencl_std::tanh(a);
    fi += 1;
    out[fi] = opencl_std::asinh(a);
    fi += 1;
    out[fi] = opencl_std::acosh(b);
    fi += 1;
    out[fi] = opencl_std::atanh(a);
    fi += 1;
    out[fi] = opencl_std::exp(a);
    fi += 1;
    out[fi] = opencl_std::exp2(a);
    fi += 1;
    out[fi] = opencl_std::log(b);
    fi += 1;
    out[fi] = opencl_std::log2(b);
    fi += 1;
    out[fi] = opencl_std::log10(b);
    fi += 1;
    out[fi] = opencl_std::sqrt(b);
    fi += 1;
    out[fi] = opencl_std::cbrt(a);
    fi += 1;
    out[fi] = opencl_std::fabs(a);
    fi += 1;
    out[fi] = opencl_std::floor(a);
    fi += 1;
    out[fi] = opencl_std::ceil(a);
    fi += 1;
    out[fi] = opencl_std::round(a);
    fi += 1;
    out[fi] = opencl_std::trunc(a);
    fi += 1;
    out[fi] = opencl_std::sign(a);
    fi += 1;
    out[fi] = opencl_std::atan2(a, b);
    fi += 1;
    out[fi] = opencl_std::hypot(a, b);
    fi += 1;
    out[fi] = opencl_std::pow(a, b);
    fi += 1;
    out[fi] = opencl_std::copysign(a, b);
    fi += 1;
    out[fi] = opencl_std::fmax(a, b);
    fi += 1;
    out[fi] = opencl_std::fmin(a, b);
    fi += 1;
    out[fi] = opencl_std::fma(a, b, c);
    fi += 1;
    out[fi] = opencl_std::clamp(a, 0.0f64, 1.0f64);
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
