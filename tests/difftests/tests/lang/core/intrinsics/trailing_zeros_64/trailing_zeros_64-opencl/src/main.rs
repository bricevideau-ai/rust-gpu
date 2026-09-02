#[cfg(not(target_arch = "spirv"))]
fn main() {
    use difftest::config::{Config, TestMetadata};

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

        let test_data: [u64; 16] = [
            0x0000000000000000,
            0x0000000000000001,
            0x8000000000000000,
            0xFFFFFFFFFFFFFFFE,
            0x1234000000000000,
            0x0000000100000000,
            0x0000000000001000,
            0x0000000080000000,
            0x0000000000000010,
            0x0000000000000100,
            0x0000000000010000,
            0x0001000000000000,
            0x0100000000000000,
            0xFFFFFFFFFFFFFFFF,
            0x8000000000000001,
            0x4000000000000000,
        ];

        let input_bytes: Vec<u8> = bytemuck::cast_slice(&test_data).to_vec();
        let output_size = test_data.len() * std::mem::size_of::<u32>();

        let buffers = vec![
            BufferConfig {
                size: input_bytes.len() as u64,
                usage: BufferUsage::StorageReadOnly,
                initial_data: Some(input_bytes),
                element_size: std::mem::size_of::<u64>(),
            },
            BufferConfig {
                size: output_size as u64,
                usage: BufferUsage::Storage,
                initial_data: None,
                element_size: std::mem::size_of::<u32>(),
            },
        ];

        config
            .write_metadata(&TestMetadata::u32())
            .expect("Failed to write metadata");

        run_opencl_test_default(&config, [test_data.len() as u32, 1, 1], buffers).unwrap();
    }
}

#[cfg(target_arch = "spirv")]
fn main() {}
