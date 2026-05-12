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
        use abi_vector_layout_opencl_cpu::layout::{LAYOUT_LEN, LAYOUT_RANGE};
        use difftest::scaffold::compute::{BufferConfig, BufferUsage, run_opencl_test_default};

        let buffers = vec![BufferConfig {
            size: (LAYOUT_LEN * std::mem::size_of::<u32>()) as u64,
            usage: BufferUsage::Storage,
            initial_data: None,
            element_size: std::mem::size_of::<u32>(),
        }];

        run_opencl_test_default(&config, [LAYOUT_RANGE.end as u32, 1, 1], buffers).unwrap();
    }
}

#[cfg(target_arch = "spirv")]
fn main() {}
