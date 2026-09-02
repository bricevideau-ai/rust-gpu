#[cfg(not(target_arch = "spirv"))]
fn main() {
    use difftest::config::Config;

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

        let input_data: Vec<[f32; 4]> = (0..64)
            .map(|i| {
                let base = i as f32 * 4.0;
                [base + 1.0, base + 2.0, base + 3.0, base + 4.0]
            })
            .collect();

        let buffers = vec![
            BufferConfig::read_only(&input_data),
            BufferConfig::writeback(std::mem::size_of_val(input_data.as_slice())),
        ];

        run_opencl_test_default(&config, [1, 1, 1], buffers).unwrap();
    }
}

#[cfg(target_arch = "spirv")]
fn main() {}
