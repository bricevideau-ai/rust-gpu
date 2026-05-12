use difftest::config::{Config, TestMetadata};
use opencl_std_math_vec_shared::{FUNCS, THREADS, compute, make_inputs};
use spirv_std::cl::Float4;

fn main() {
    let config = Config::from_path(std::env::args().nth(1).unwrap()).unwrap();

    let threads = THREADS as usize;
    let (input_a, input_b, input_c) = make_inputs();

    let out_len = threads * 4 * FUNCS;
    let mut output = vec![0.0f32; out_len];

    for tid in 0..threads {
        let base = tid * 4;
        let va = Float4::new(
            input_a[base],
            input_a[base + 1],
            input_a[base + 2],
            input_a[base + 3],
        );
        let vb = Float4::new(
            input_b[base],
            input_b[base + 1],
            input_b[base + 2],
            input_b[base + 3],
        );
        let vc = Float4::new(
            input_c[base],
            input_c[base + 1],
            input_c[base + 2],
            input_c[base + 3],
        );

        let out_base = tid * 4 * FUNCS;
        compute(va, vb, vc, &mut output[out_base..out_base + 4 * FUNCS]);
    }

    config.write_result(&output).unwrap();
    config.write_metadata(&TestMetadata::f32(1e-4)).unwrap();
}
