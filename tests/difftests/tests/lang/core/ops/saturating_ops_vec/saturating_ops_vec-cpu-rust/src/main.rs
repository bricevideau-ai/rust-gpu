use difftest::config::{Config, TestMetadata};
use saturating_ops_vec_shared::{FUNCS, THREADS, make_inputs, rust_compute};

fn main() {
    let config = Config::from_path(std::env::args().nth(1).unwrap()).unwrap();

    let threads = THREADS as usize;
    let (input_a, input_b) = make_inputs();
    let mut output = vec![0u32; threads * FUNCS];

    for tid in 0..threads {
        let base = tid * 4;
        let a = [
            input_a[base],
            input_a[base + 1],
            input_a[base + 2],
            input_a[base + 3],
        ];
        let b = [
            input_b[base],
            input_b[base + 1],
            input_b[base + 2],
            input_b[base + 3],
        ];
        let out_base = tid * FUNCS;
        rust_compute(a, b, &mut output[out_base..out_base + FUNCS]);
    }

    config.write_result(&output).unwrap();
    config.write_metadata(&TestMetadata::raw()).unwrap();
}
