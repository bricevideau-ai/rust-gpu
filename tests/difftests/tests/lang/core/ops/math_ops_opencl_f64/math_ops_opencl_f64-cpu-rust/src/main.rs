use difftest::config::{Config, TestMetadata};
use math_ops_opencl_f64_shared::{FUNCS, N, make_inputs, rust_compute};

fn main() {
    let config = Config::from_path(std::env::args().nth(1).unwrap()).unwrap();

    let (input_a, input_b, input_c) = make_inputs();
    let mut output = vec![0.0f64; N * FUNCS];

    for tid in 0..N {
        let base = tid * FUNCS;
        rust_compute(
            input_a[tid],
            input_b[tid],
            input_c[tid],
            &mut output[base..base + FUNCS],
        );
    }

    config.write_result(&output).unwrap();
    config.write_metadata(&TestMetadata::f64(1e-10)).unwrap();
}
