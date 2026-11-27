//! The per-block build of an IPIR+SP table on the Apple GPU (Metal), checked word for word against the CPU build.
pub mod common;
pub mod cpu_ref;
pub mod fetch;
pub mod gpu;
pub mod rns32;

// The benchmark's own code, shared with its CPU and CUDA programs: its helpers (the fetch's table and requests, the
// calls-per-second measure), each block's compiled packing matrix, and the host's part of a fetch on a GPU.
#[path = "../../ipir/bench/src/lib.rs"]
pub mod bench;
#[path = "../../ipir/bench/src/fetch_host.rs"]
pub mod fetch_host;
#[path = "../../ipir/bench/src/packing.rs"]
pub mod packing;

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
