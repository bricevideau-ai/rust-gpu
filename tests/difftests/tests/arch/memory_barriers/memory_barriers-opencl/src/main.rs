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

        let buffer_size = 256;
        let initial_data: Vec<u32> = (0..64).collect();
        let initial_bytes: Vec<u8> = initial_data.iter().flat_map(|&x| x.to_ne_bytes()).collect();

        let buffers = vec![
            BufferConfig {
                size: buffer_size,
                usage: BufferUsage::StorageReadOnly,
                initial_data: Some(initial_bytes),
                element_size: std::mem::size_of::<u32>(),
            },
            BufferConfig {
                size: buffer_size,
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
