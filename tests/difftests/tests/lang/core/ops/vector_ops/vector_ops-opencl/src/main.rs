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
        use difftest::scaffold::compute::{BufferConfig, BufferUsage, run_opencl_test_default};

        let input_data: Vec<f32> = (0..128)
            .map(|i| match i % 16 {
                0 => 0.0,
                1 => 1.0,
                2 => -1.0,
                3 => 0.5,
                4 => -0.5,
                5 => 2.0,
                6 => -2.0,
                7 => 3.0,
                8 => std::f32::consts::PI,
                9 => std::f32::consts::E,
                10 => 0.1,
                11 => -0.1,
                12 => 4.0,
                13 => -4.0,
                14 => 0.25,
                15 => -0.25,
                _ => unreachable!(),
            })
            .collect();

        let input_bytes = bytemuck::cast_slice(&input_data).to_vec();

        let buffers = vec![
            BufferConfig {
                size: 512,
                usage: BufferUsage::StorageReadOnly,
                initial_data: Some(input_bytes),
                element_size: std::mem::size_of::<f32>(),
            },
            BufferConfig {
                size: 6400,
                usage: BufferUsage::Storage,
                initial_data: None,
                element_size: std::mem::size_of::<f32>(),
            },
        ];

        config
            .write_metadata(&difftest::config::TestMetadata::f32(1e-5))
            .unwrap();

        run_opencl_test_default(&config, [1, 1, 1], buffers).unwrap();
    }
}

#[cfg(target_arch = "spirv")]
fn main() {}
