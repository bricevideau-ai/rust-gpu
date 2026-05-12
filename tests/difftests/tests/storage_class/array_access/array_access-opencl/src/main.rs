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

        let mut input_data = vec![0u32; 256];
        for (i, slot) in input_data.iter_mut().enumerate().take(256) {
            *slot = i as u32;
        }
        input_data[0] = 5;
        input_data[1] = 10;
        input_data[2] = 15;
        input_data[3] = 20;

        let input_bytes = bytemuck::cast_slice(&input_data).to_vec();

        let buffers = vec![
            BufferConfig {
                size: 1024,
                usage: BufferUsage::StorageReadOnly,
                initial_data: Some(input_bytes),
                element_size: std::mem::size_of::<u32>(),
            },
            BufferConfig {
                size: 1024,
                usage: BufferUsage::Storage,
                initial_data: None,
                element_size: std::mem::size_of::<u32>(),
            },
        ];

        run_opencl_test_default(&config, [1, 1, 1], buffers).unwrap();
    }
}

#[cfg(target_arch = "spirv")]
fn main() {}
