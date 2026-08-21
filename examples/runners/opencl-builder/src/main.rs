use spirv_builder::{Capability, SpirvBuilder};
use std::error::Error;
use std::path::Path;

fn main() -> Result<(), Box<dyn Error>> {
    let shaders = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../shaders");

    let result =
        SpirvBuilder::new(shaders.join("kernel-shader"), "spirv-unknown-opencl1.2").build()?;
    let module = result.module.unwrap_single();
    println!("OpenCL kernel compiled successfully: {module:?}");

    let result = SpirvBuilder::new(
        shaders.join("kernel-image-shader"),
        "spirv-unknown-opencl1.2",
    )
    .capability(Capability::ImageBasic)
    .build()?;
    let module = result.module.unwrap_single();
    println!("OpenCL image kernel compiled successfully: {module:?}");

    let result = SpirvBuilder::new(
        shaders.join("kernel-sampler-shader"),
        "spirv-unknown-opencl1.2",
    )
    .capability(Capability::LiteralSampler)
    .build()?;
    let module = result.module.unwrap_single();
    println!("OpenCL sampler kernel compiled successfully: {module:?}");

    Ok(())
}
