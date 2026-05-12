#![cfg_attr(target_arch = "spirv", no_std)]

use spirv_std::arch::opencl_std;

pub const N: usize = 256;
pub const THREADS: u32 = 64;
pub const WORKGROUPS: u32 = 4;

const _: () = assert!(
    (THREADS as usize) * (WORKGROUPS as usize) == N,
    "THREADS * WORKGROUPS must equal N (one work item per input element)",
);

/// Number of f32 outputs per work-item: `opencl_std` functions that have NO
/// equivalent in `num_traits::Float`. The intersection (sin/cos/exp/...) is
/// covered by the 4-way `math_ops_opencl` test.
pub const FUNCS: usize = 16;

/// Run all `FUNCS` `opencl_std`-only intrinsics on `(a, b, c)` into `out`.
/// Single source compiled for SPIR-V Kernel and host CPU.
pub fn compute(a: f32, b: f32, c: f32, out: &mut [f32]) {
    let mut fi = 0usize;

    // Unary opencl_std-only
    out[fi] = opencl_std::exp10(a);
    fi += 1;
    out[fi] = opencl_std::rsqrt(b);
    fi += 1;
    out[fi] = opencl_std::erf(a);
    fi += 1;
    out[fi] = opencl_std::erfc(a);
    fi += 1;
    out[fi] = opencl_std::lgamma(b);
    fi += 1;
    out[fi] = opencl_std::tgamma(b);
    fi += 1;

    // Binary opencl_std-only
    out[fi] = opencl_std::fmod(a, b);
    fi += 1;
    out[fi] = opencl_std::fdim(a, b);
    fi += 1;
    out[fi] = opencl_std::nextafter(a, b);
    fi += 1;
    out[fi] = opencl_std::remainder(a, b);
    fi += 1;

    // Ternary opencl_std-only
    out[fi] = opencl_std::mix(a, b, c);
    fi += 1;
    out[fi] = opencl_std::smoothstep(0.0f32, 1.0f32, a);
    fi += 1;

    // Scalar-only multi-output / int-result
    out[fi] = opencl_std::ilogb(a) as f32;
    fi += 1;
    out[fi] = opencl_std::ldexp(a, b as i32);
    fi += 1;
    let (lg_val, _lg_sign) = opencl_std::lgamma_r(a);
    out[fi] = lg_val;
    fi += 1;
    let (rq_rem, _rq_quo) = opencl_std::remquo(a, b);
    out[fi] = rq_rem;
    fi += 1;

    #[cfg(not(target_arch = "spirv"))]
    debug_assert_eq!(fi, FUNCS);
    #[cfg(target_arch = "spirv")]
    let _ = fi;
}

#[cfg(not(target_arch = "spirv"))]
pub fn make_inputs() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mk = |lo: f32, hi: f32| -> Vec<f32> {
        (0..N)
            .map(|i| {
                let t = (i as f32) / (N - 1) as f32;
                lo + t * (hi - lo)
            })
            .collect()
    };
    (mk(0.1, 0.9), mk(1.0, 3.0), mk(0.0, 1.0))
}
