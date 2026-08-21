use crate::layout::{LAYOUT_LEN, LAYOUT_RANGE, eval_cl_layouts};
use difftest::config::Config;

pub fn run() {
    let config = Config::from_path(std::env::args().nth(1).unwrap()).unwrap();
    let mut out = vec![0u32; LAYOUT_LEN];
    for gid in LAYOUT_RANGE {
        eval_cl_layouts(gid as u32, &mut out);
    }
    let bytes: &[u8] = bytemuck::cast_slice(&out);
    config.write_result(bytes).unwrap();
}
