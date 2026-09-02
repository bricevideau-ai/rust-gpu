use difftest::config::{Config, TestMetadata};
use opencl_std_math_f64_vec_shared::{FUNCS, THREADS, compute, make_inputs};
use spirv_std::cl::Double4;

fn main() {
    let config = Config::from_path(std::env::args().nth(1).unwrap()).unwrap();

    let threads = THREADS as usize;
    let (input_a, input_b, input_c) = make_inputs();

    let out_len = threads * 4 * FUNCS;
    let mut output = vec![0.0f64; out_len];

    for tid in 0..threads {
        let base = tid * 4;
        let va = Double4::new(
            input_a[base],
            input_a[base + 1],
            input_a[base + 2],
            input_a[base + 3],
        );
        let vb = Double4::new(
            input_b[base],
            input_b[base + 1],
            input_b[base + 2],
            input_b[base + 3],
        );
        let vc = Double4::new(
            input_c[base],
            input_c[base + 1],
            input_c[base + 2],
            input_c[base + 3],
        );

        let out_base = tid * 4 * FUNCS;
        compute(va, vb, vc, &mut output[out_base..out_base + 4 * FUNCS]);
    }

    config.write_result(&output).unwrap();
    config.write_metadata(&TestMetadata::f64(1e-10)).unwrap();
}
