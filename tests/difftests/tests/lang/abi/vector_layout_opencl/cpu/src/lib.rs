#![cfg_attr(target_arch = "spirv", no_std)]

#[cfg(not(target_arch = "spirv"))]
pub mod cpu_driver;
pub mod layout;
