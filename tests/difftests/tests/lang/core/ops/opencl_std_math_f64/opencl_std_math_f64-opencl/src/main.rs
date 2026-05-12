#[cfg(not(target_arch = "spirv"))]
fn main() {
    use difftest::config::Config;
    use opencl_std_math_f64_shared::{FUNCS, N, WORKGROUPS, make_inputs};

    let config = Config::from_path(std::env::args().nth(1).unwrap()).unwrap();

    #[cfg(not(target_os = "linux"))]
    {
        difftest::scaffold::Skip::new("OpenCL backend is Linux-only")
            .run_test(&config)
            .unwrap();
    }

    #[cfg(target_os = "linux")]
    {
        use difftest::scaffold::compute::{BufferConfig, run_opencl_test};
        use difftest::scaffold::shader::RustComputeShader;

        let (input_a, input_b, input_c) = make_inputs();
        let out_count = N * FUNCS;

        let buffers = vec![
            BufferConfig::read_only(&input_a),
            BufferConfig::read_only(&input_b),
            BufferConfig::read_only(&input_c),
            BufferConfig::writeback(out_count * std::mem::size_of::<f64>()),
        ];

        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
        let shader = RustComputeShader::with_target(manifest_dir, "spirv-unknown-opencl1.2")
            .with_capability(spirv_builder::Capability::Float64);

        config
            .write_metadata(&difftest::config::TestMetadata::f64(1e-10))
            .unwrap();

        match run_opencl_test(&config, shader, [WORKGROUPS, 1, 1], buffers) {
            Ok(()) => {}
            Err(e) => {
                let msg = format!("{e:?}");
                if msg.contains("Double type is not supported")
                    || msg.contains("cl_khr_fp64")
                    || msg.contains("BUILD_PROGRAM_FAILURE")
                {
                    eprintln!("fp64 not supported on this device, skipping: {e}");
                    difftest::scaffold::Skip::new("Device does not support fp64")
                        .run_test(&config)
                        .unwrap();
                } else {
                    panic!("OpenCL test failed: {e:?}");
                }
            }
        }
    }
}

#[cfg(target_arch = "spirv")]
fn main() {}
