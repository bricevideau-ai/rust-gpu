use difftest::config::{Config, TestMetadata};
use opencl_std_math_shared::{FUNCS, N, compute, make_inputs};

fn main() {
    let config = Config::from_path(std::env::args().nth(1).unwrap()).unwrap();

    let (input_a, input_b, input_c) = make_inputs();
    let mut output = vec![0.0f32; N * FUNCS];

    for tid in 0..N {
        let base = tid * FUNCS;
        compute(
            input_a[tid],
            input_b[tid],
            input_c[tid],
            &mut output[base..base + FUNCS],
        );
    }

    config.write_result(&output).unwrap();
    config.write_metadata(&TestMetadata::f32(1e-4)).unwrap();
}
