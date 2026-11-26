//! The per-block build of an IPIR+SP table on the Apple GPU (Metal), checked word for word against the CPU build.
pub mod common;
pub mod cpu_ref;
pub mod gpu;
pub mod rns32;

/// `pack_digest`'s digest of the GPU build's output words.
pub fn gpu_digest(g: &gpu::GpuBuilder, q: u64) -> (String, String, String, bool) {
    let masks = g.hint_masks();
    let slots = g.slot_rows();
    let d = masks.len();
    let (m, overflow) = g.matrix_i32();
    let matrix: Vec<u64> = m.iter().map(|&x| (x as i128).rem_euclid(q as i128) as u64).collect();
    let (all, h, p) = common::digest(&common::BlockOutput {
        hint_masks: &masks,
        mask: &slots[0],
        other_mask: &slots[d / 2],
        storage_label: if overflow { "i64" } else { "i32" },
        rows: d,
        cols: 2 * d,
        q,
        matrix: &matrix,
    });
    (all, h, p, overflow)
}
