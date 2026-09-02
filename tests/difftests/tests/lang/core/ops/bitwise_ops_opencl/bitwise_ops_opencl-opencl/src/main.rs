#[cfg(not(target_arch = "spirv"))]
fn main() {
    use bitwise_ops_opencl_shared::{N, WORKGROUPS, make_inputs};
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

        let (input_a, input_b) = make_inputs();
        let byte_len = (N * std::mem::size_of::<u32>()) as u64;

        let buffers = vec![
            BufferConfig {
                size: byte_len,
                usage: BufferUsage::StorageReadOnly,
                initial_data: Some(bytemuck::cast_slice(&input_a).to_vec()),
                element_size: std::mem::size_of::<u32>(),
            },
            BufferConfig {
                size: byte_len,
                usage: BufferUsage::StorageReadOnly,
                initial_data: Some(bytemuck::cast_slice(&input_b).to_vec()),
                element_size: std::mem::size_of::<u32>(),
            },
            BufferConfig {
                size: byte_len,
                usage: BufferUsage::Storage,
                initial_data: None,
                element_size: std::mem::size_of::<u32>(),
            },
        ];

        run_opencl_test_default(&config, [WORKGROUPS, 1, 1], buffers).unwrap();
    }
}

#[cfg(target_arch = "spirv")]
fn main() {}
