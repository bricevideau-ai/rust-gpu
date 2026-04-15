use rspirv::spirv::MemoryModel;
use rustc_target::spec::{Arch, Cc, Env, LinkerFlavor, PanicStrategy, Target, TargetOptions};
use spirv_tools::TargetEnv;

const ARCH: &str = "spirv";

pub struct SpirvTarget {
    env: TargetEnv,
    vendor: String,
}

impl SpirvTarget {
    pub fn memory_model(&self) -> MemoryModel {
        match self.env {
            TargetEnv::Universal_1_0
            | TargetEnv::Universal_1_1
            | TargetEnv::Universal_1_2
            | TargetEnv::Universal_1_3
            | TargetEnv::Universal_1_4
            | TargetEnv::Universal_1_5
            | TargetEnv::Universal_1_6 => MemoryModel::Simple,

            TargetEnv::OpenGL_4_0
            | TargetEnv::OpenGL_4_1
            | TargetEnv::OpenGL_4_2
            | TargetEnv::OpenGL_4_3
            | TargetEnv::OpenGL_4_5 => MemoryModel::GLSL450,

            TargetEnv::OpenCL_2_1
            | TargetEnv::OpenCL_2_2
            | TargetEnv::OpenCL_1_2
            | TargetEnv::OpenCLEmbedded_1_2
            | TargetEnv::OpenCL_2_0
            | TargetEnv::OpenCLEmbedded_2_0
            | TargetEnv::OpenCLEmbedded_2_1
            | TargetEnv::OpenCLEmbedded_2_2 => MemoryModel::OpenCL,

            TargetEnv::Vulkan_1_0
            | TargetEnv::Vulkan_1_1
            | TargetEnv::WebGPU_0_DEPRECATED
            | TargetEnv::Vulkan_1_1_Spirv_1_4
            | TargetEnv::Vulkan_1_2
            | TargetEnv::Vulkan_1_3
            | TargetEnv::Vulkan_1_4 => MemoryModel::Vulkan,
        }
    }

    pub fn spirv_version(&self) -> (u8, u8) {
        self.env.spirv_version()
    }

    pub fn target_env(&self) -> TargetEnv {
        self.env
    }

    pub fn is_opencl(&self) -> bool {
        self.memory_model() == MemoryModel::OpenCL
    }

    pub fn is_opencl_2_0_or_later(&self) -> bool {
        matches!(
            self.env,
            TargetEnv::OpenCL_2_0
                | TargetEnv::OpenCLEmbedded_2_0
                | TargetEnv::OpenCL_2_1
                | TargetEnv::OpenCLEmbedded_2_1
                | TargetEnv::OpenCL_2_2
                | TargetEnv::OpenCLEmbedded_2_2
        )
    }

    pub fn is_opencl_2_2_or_later(&self) -> bool {
        matches!(
            self.env,
            TargetEnv::OpenCL_2_2 | TargetEnv::OpenCLEmbedded_2_2
        )
    }

    fn init_target_opts(&self) -> TargetOptions {
        let mut o = TargetOptions::default();
        o.simd_types_indirect = false;
        o.allows_weak_linkage = false;
        o.crt_static_allows_dylibs = true;
        o.crt_static_respected = true;
        o.dll_prefix = "".into();
        o.dll_suffix = ".spv.json".into();
        o.dynamic_linking = true;
        o.emit_debug_gdb_scripts = false;
        o.linker_flavor = LinkerFlavor::Unix(Cc::No);
        o.panic_strategy = PanicStrategy::Abort;
        o.env = Env::Other(self.env.to_string().into());
        // Note(@firestar99): not sure if this does anything
        o.is_like_gpu = true;
        // TODO: Investigate if main_needs_argc_argv is useful (for building exes)
        o.main_needs_argc_argv = false;
        o
    }

    pub fn rustc_target(&self) -> Target {
        // OpenCL targets use Physical64 addressing with 64-bit pointers.
        let (pointer_width, data_layout) = if self.memory_model() == MemoryModel::OpenCL {
            (64, "e-m:e-p:64:64:64-i64:64-n8:16:32:64")
        } else {
            (32, "e-m:e-p:32:32:32-i64:64-n8:16:32:64")
        };
        Target {
            llvm_target: self.to_string().into(),
            metadata: Default::default(),
            pointer_width,
            data_layout: data_layout.into(),
            arch: Arch::Other(ARCH.into()),
            options: self.init_target_opts(),
        }
    }
}

impl std::str::FromStr for SpirvTarget {
    type Err = InvalidTarget;

    fn from_str(target: &str) -> Result<Self, Self::Err> {
        let mut iter = target.split('-');
        let error = || InvalidTarget(target.into());

        if iter.next() != Some(ARCH) {
            return Err(error());
        }

        let vendor = iter.next().map(From::from).ok_or_else(error)?;

        let env = iter
            .next()
            .and_then(|env| env.parse().ok())
            .ok_or_else(error)?;

        if iter.next().is_some() {
            return Err(error());
        }

        let result = Self { env, vendor };

        Ok(result)
    }
}

impl std::fmt::Display for SpirvTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}-{}-{}", ARCH, self.vendor, self.env)
    }
}

#[derive(Debug)]
pub struct InvalidTarget(String);

impl std::error::Error for InvalidTarget {}
impl std::fmt::Display for InvalidTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Invalid target `{}`.", self.0)
    }
}
