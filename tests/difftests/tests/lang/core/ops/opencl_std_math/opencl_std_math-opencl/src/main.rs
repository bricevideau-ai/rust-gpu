#[cfg(not(target_arch = "spirv"))]
fn main() {
    use difftest::config::{Config, TestMetadata};
    use opencl_std_math_shared::{FUNCS, N, WORKGROUPS, make_inputs};

    let config = Config::from_path(std::env::args().nth(1).unwrap()).unwrap();

    #[cfg(not(target_os = "linux"))]
    {
        difftest::scaffold::Skip::new("OpenCL backend is Linux-only")
            .run_test(&config)
            .unwrap();
    }

    #[cfg(target_os = "linux")]
    {
        use difftest::scaffold::compute::{BufferConfig, run_opencl_test_default};

        let (input_a, input_b, input_c) = make_inputs();
        let out_count = N * FUNCS;

        let buffers = vec![
            BufferConfig::read_only(&input_a),
            BufferConfig::read_only(&input_b),
            BufferConfig::read_only(&input_c),
            BufferConfig::writeback(out_count * std::mem::size_of::<f32>()),
        ];

        config.write_metadata(&TestMetadata::f32(1e-4)).unwrap();

        run_opencl_test_default(&config, [WORKGROUPS, 1, 1], buffers).unwrap();
    }
}

#[cfg(target_arch = "spirv")]
fn main() {}
