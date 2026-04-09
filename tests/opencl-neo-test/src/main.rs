use opencl3::context::Context;
use opencl3::device::{CL_DEVICE_TYPE_ALL, Device, get_all_devices};
use opencl3::program::Program;

fn test_spv(context: &Context, device_id: opencl3::types::cl_device_id, name: &str, spv: &[u8]) {
    match Program::create_from_il(context, spv) {
        Err(e) => println!("  {name}: FAIL create_from_il: {e}"),
        Ok(mut program) => match program.build(context.devices(), "") {
            Ok(_) => println!("  {name}: OK"),
            Err(e) => {
                let log = program
                    .get_build_log(device_id)
                    .unwrap_or_else(|_| "no log".into());
                println!("  {name}: FAIL build: {e} (log: {log})");
            }
        },
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device_id = *get_all_devices(CL_DEVICE_TYPE_ALL)?
        .first()
        .expect("no OpenCL devices");
    let device = Device::new(device_id);
    println!("Device: {} ({})", device.name()?, device.vendor()?);
    println!();

    let tests: &[(&str, &[u8])] = &[
        ("test1_minimal", include_bytes!("../spv/test1_minimal.spv")),
        ("test2_loop", include_bytes!("../spv/test2_loop.spv")),
        (
            "test3_struct_phi",
            include_bytes!("../spv/test3_struct_phi.spv"),
        ),
        (
            "test4_real_kernel",
            include_bytes!("../spv/test4_real_kernel.spv"),
        ),
    ];

    let context = Context::from_device(&device)?;
    for (name, spv) in tests {
        test_spv(&context, device_id, name, spv);
    }

    Ok(())
}
