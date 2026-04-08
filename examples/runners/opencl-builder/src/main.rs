use spirv_builder::SpirvBuilder;
use std::error::Error;
use std::path::Path;

fn main() -> Result<(), Box<dyn Error>> {
    let path_to_crate = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../shaders/kernel-shader");
    let result = SpirvBuilder::new(path_to_crate, "spirv-unknown-opencl1.2").build()?;
    let module = result.module.unwrap_single();
    println!("OpenCL kernel compiled successfully: {module:?}");
    Ok(())
}
