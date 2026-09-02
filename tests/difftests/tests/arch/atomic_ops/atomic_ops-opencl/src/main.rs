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

        let counter_data = vec![100u32, 50, 20, 5, 0];
        let counter_bytes = bytemuck::cast_slice(&counter_data).to_vec();

        let buffers = vec![
            BufferConfig {
                size: 20,
                usage: BufferUsage::Storage,
                initial_data: Some(counter_bytes),
                element_size: std::mem::size_of::<u32>(),
            },
            BufferConfig {
                size: 20,
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
