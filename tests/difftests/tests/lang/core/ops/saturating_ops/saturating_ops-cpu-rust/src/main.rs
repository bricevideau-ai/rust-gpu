use difftest::config::{Config, TestMetadata};
use saturating_ops_shared::{FUNCS, N, make_inputs, rust_compute};

fn main() {
    let config = Config::from_path(std::env::args().nth(1).unwrap()).unwrap();

    let (input_a, input_b) = make_inputs();
    let mut output = vec![0u32; N * FUNCS];

    for tid in 0..N {
        let base = tid * FUNCS;
        rust_compute(input_a[tid], input_b[tid], &mut output[base..base + FUNCS]);
    }

    config.write_result(&output).unwrap();
    config.write_metadata(&TestMetadata::raw()).unwrap();
}
