use bitwise_ops_opencl_shared::{N, compute, make_inputs};
use difftest::config::Config;

fn main() {
    let config = Config::from_path(std::env::args().nth(1).unwrap()).unwrap();

    let (input_a, input_b) = make_inputs();
    let output: Vec<u32> = (0..N)
        .map(|tid| compute(tid, input_a[tid], input_b[tid]))
        .collect();

    config.write_result(&output).unwrap();
}
