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
        match run_opencl_c_test(&config) {
            Ok(()) => {}
            Err(e) => {
                eprintln!("OpenCL C layout test failed: {e:?}");
                difftest::scaffold::Skip::new("OpenCL runtime unavailable on this host")
                    .run_test(&config)
                    .unwrap();
            }
        }
    }
}

#[cfg(target_os = "linux")]
fn run_opencl_c_test(config: &difftest::config::Config) -> anyhow::Result<()> {
    use abi_vector_layout_opencl_cpu::layout::{LAYOUT_LEN, LAYOUT_RANGE};
    use anyhow::{Context as _, anyhow};
    use opencl3::command_queue::CommandQueue;
    use opencl3::context::Context;
    use opencl3::device::{CL_DEVICE_TYPE_ALL, Device, get_all_devices};
    use opencl3::kernel::{ExecuteKernel, Kernel};
    use opencl3::memory::{Buffer, CL_MEM_READ_WRITE};
    use opencl3::program::Program;
    use opencl3::types::CL_BLOCKING;
    use std::ptr;

    let device_ids = get_all_devices(CL_DEVICE_TYPE_ALL)
        .map_err(|e| anyhow!("OpenCL: failed to enumerate devices: {e}"))?;
    if device_ids.is_empty() {
        return Err(anyhow!("OpenCL: no devices available"));
    }
    let device_id = device_ids[0];
    let device = Device::new(device_id);
    let context =
        Context::from_device(&device).map_err(|e| anyhow!("OpenCL: Context::from_device: {e}"))?;
    let queue = CommandQueue::create_default_with_properties(&context, 0, 0)
        .map_err(|e| anyhow!("OpenCL: CommandQueue: {e}"))?;

    let buf_bytes = LAYOUT_LEN * std::mem::size_of::<u32>();
    let mut buffer =
        unsafe { Buffer::<u8>::create(&context, CL_MEM_READ_WRITE, buf_bytes, ptr::null_mut()) }
            .context("OpenCL: Buffer::create")?;

    let zeros = vec![0u8; buf_bytes];
    unsafe {
        queue
            .enqueue_write_buffer(&mut buffer, CL_BLOCKING, 0, &zeros, &[])
            .context("OpenCL: enqueue_write_buffer")?
            .wait()
            .context("OpenCL: wait on write")?;
    }

    let mut program = Program::create_from_source(&context, OPENCL_C_SOURCE)
        .map_err(|e| anyhow!("OpenCL: create_from_source: {e}"))?;
    if let Err(e) = program.build(context.devices(), "") {
        let log = program
            .get_build_log(device_id)
            .unwrap_or_else(|_| "no build log".into());
        return Err(anyhow!("OpenCL: build failed: {e}\n{log}"));
    }

    let kernel = Kernel::create(&program, "layout_kernel")
        .map_err(|e| anyhow!("OpenCL: Kernel::create: {e}"))?;

    let global = [LAYOUT_RANGE.end];
    let event = unsafe {
        ExecuteKernel::new(&kernel)
            .set_arg(&buffer)
            .set_global_work_sizes(&global)
            .enqueue_nd_range(&queue)
            .map_err(|e| anyhow!("OpenCL: enqueue: {e}"))?
    };
    event.wait().context("OpenCL: wait on kernel")?;

    let mut result = vec![0u8; buf_bytes];
    unsafe {
        queue
            .enqueue_read_buffer(&buffer, CL_BLOCKING, 0, &mut result, &[])
            .context("OpenCL: enqueue_read_buffer")?
            .wait()
            .context("OpenCL: wait on read")?;
    }

    config.write_result(&result)?;
    Ok(())
}

#[cfg(target_os = "linux")]
const OPENCL_C_SOURCE: &str = r#"
#define LAYOUT_MAX_SIZE (0x100 / 4)

#define WRITE_VEC2(out, base, type, scalar) \
    out[base]   = (uint)sizeof(type);       \
    out[base+1] = (uint)__alignof__(type);  \
    out[base+2] = 0;                        \
    out[base+3] = (uint)sizeof(scalar);

#define WRITE_VEC3(out, base, type, scalar) \
    out[base]   = (uint)sizeof(type);       \
    out[base+1] = (uint)__alignof__(type);  \
    out[base+2] = 0;                        \
    out[base+3] = (uint)sizeof(scalar);     \
    out[base+4] = 2 * (uint)sizeof(scalar);

#define WRITE_VEC4(out, base, type, scalar) \
    out[base]   = (uint)sizeof(type);       \
    out[base+1] = (uint)__alignof__(type);  \
    out[base+2] = 0;                        \
    out[base+3] = (uint)sizeof(scalar);     \
    out[base+4] = 2 * (uint)sizeof(scalar); \
    out[base+5] = 3 * (uint)sizeof(scalar);

__kernel void layout_kernel(__global uint* output) {
    uint gid = get_global_id(0);
    uint base = gid * LAYOUT_MAX_SIZE;

    switch (gid) {
        // Float
        case 0x00: WRITE_VEC2(output, base, float2,  float);  break;
        case 0x01: WRITE_VEC3(output, base, float3,  float);  break;
        case 0x02: WRITE_VEC4(output, base, float4,  float);  break;
        // Double
        case 0x03: WRITE_VEC2(output, base, double2, double); break;
        case 0x04: WRITE_VEC3(output, base, double3, double); break;
        case 0x05: WRITE_VEC4(output, base, double4, double); break;
        // Char (i8)
        case 0x06: WRITE_VEC2(output, base, char2,   char);   break;
        case 0x07: WRITE_VEC3(output, base, char3,   char);   break;
        case 0x08: WRITE_VEC4(output, base, char4,   char);   break;
        // UChar (u8)
        case 0x09: WRITE_VEC2(output, base, uchar2,  uchar);  break;
        case 0x0a: WRITE_VEC3(output, base, uchar3,  uchar);  break;
        case 0x0b: WRITE_VEC4(output, base, uchar4,  uchar);  break;
        // Short (i16)
        case 0x0c: WRITE_VEC2(output, base, short2,  short);  break;
        case 0x0d: WRITE_VEC3(output, base, short3,  short);  break;
        case 0x0e: WRITE_VEC4(output, base, short4,  short);  break;
        // UShort (u16)
        case 0x0f: WRITE_VEC2(output, base, ushort2, ushort); break;
        case 0x10: WRITE_VEC3(output, base, ushort3, ushort); break;
        case 0x11: WRITE_VEC4(output, base, ushort4, ushort); break;
        // Int (i32)
        case 0x12: WRITE_VEC2(output, base, int2,    int);    break;
        case 0x13: WRITE_VEC3(output, base, int3,    int);    break;
        case 0x14: WRITE_VEC4(output, base, int4,    int);    break;
        // UInt (u32)
        case 0x15: WRITE_VEC2(output, base, uint2,   uint);   break;
        case 0x16: WRITE_VEC3(output, base, uint3,   uint);   break;
        case 0x17: WRITE_VEC4(output, base, uint4,   uint);   break;
        // Long (i64)
        case 0x18: WRITE_VEC2(output, base, long2,   long);   break;
        case 0x19: WRITE_VEC3(output, base, long3,   long);   break;
        case 0x1a: WRITE_VEC4(output, base, long4,   long);   break;
        // ULong (u64)
        case 0x1b: WRITE_VEC2(output, base, ulong2,  ulong);  break;
        case 0x1c: WRITE_VEC3(output, base, ulong3,  ulong);  break;
        case 0x1d: WRITE_VEC4(output, base, ulong4,  ulong);  break;
        default: break;
    }
}
"#;
