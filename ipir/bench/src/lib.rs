//! Shared pieces of the compile benchmarks: the production profile at one column block, random cells, the timed
//! build of one block, and lowering a thread's priority.
//!
//! The profile is the proof of concept's (`ipir/core/src/lib.rs`): d = 2,048, q = 2^54, p = 2^16, gadget 2^19 × 2,
//! Gaussian secrets, two-mask output, masks published at 29 bits, 49-bit queries. Setups come from patch 0001's
//! `NativePublicSetup::new_without_columns` with seed = snapshot = 32 zero bytes, as in this design.
use ipir_sp::native::*;
use rand_chacha::{
    rand_core::{RngCore, SeedableRng},
    ChaCha20Rng,
};
use reinspiring::native::{NativeParams, SecretDistribution};
use std::time::{Duration, Instant};

/// Columns of one column block.
pub const BLOCK_COLS: usize = 2048;

pub fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// The production profile for a table of `rows` rows and one column block.
pub fn profile(rows: usize) -> NativeProfile {
    let p = NativeProfile::new(
        NativeParams::new(2048, 54, 16, 19, 2, SecretDistribution::Gaussian).unwrap(),
        rows,
        BLOCK_COLS,
    )
    .unwrap()
    .with_two_mask_output()
    .unwrap()
    .with_published_mask_bits(29)
    .unwrap();
    assert_eq!(p.query_bits(), 49);
    p
}

/// The setup this design uses: no column count in the identifier, zero seed and snapshot.
pub fn setup(rows: usize) -> NativePublicSetup {
    NativePublicSetup::new_without_columns(profile(rows), [0; 32], [0; 32])
}

/// `rows × 2,048` random 16-bit cells (the time does not depend on them).
pub fn cells(rows: usize, seed: u64) -> Vec<u16> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    (0..rows * BLOCK_COLS).map(|_| rng.next_u32() as u16).collect()
}

/// One block's build, timed: `NativeServer::build_with_concurrency(setup, cells, 1)`, the call the add-on's
/// `ServerTable::build` makes with its default options (hint columns, then packing: mask trace and matrix
/// compilation).
pub struct BuildTiming {
    pub wall: Duration,
    pub hints: Duration,
    pub packing: Duration,
    pub layout: Duration,
    pub total: Duration,
}

pub fn build(setup: NativePublicSetup, cells: Vec<u16>) -> BuildTiming {
    let t = Instant::now();
    let server = NativeServer::build_with_concurrency(setup, cells, 1).unwrap();
    let wall = t.elapsed();
    let o = server.offline_timing();
    BuildTiming { wall, hints: o.hints, packing: o.packing, layout: o.layout, total: o.total }
}

impl BuildTiming {
    pub fn json(&self) -> String {
        format!(
            "\"build_ms\":{:.2},\"hints_ms\":{:.2},\"packing_ms\":{:.2},\"layout_ms\":{:.2},\"prepare_ms\":{:.2}",
            ms(self.wall),
            ms(self.hints),
            ms(self.packing),
            ms(self.layout),
            ms(self.total)
        )
    }
}

/// Lowers the calling thread's priority: nice 19 for this thread on Linux (`setpriority` on its thread id; Linux
/// schedules threads by their own nice value), the utility QoS class on macOS (only for trying the benchmark there).
/// Returns what was done.
pub fn lower_thread_priority() -> &'static str {
    #[cfg(target_os = "linux")]
    unsafe {
        let tid = libc::gettid();
        if libc::setpriority(libc::PRIO_PROCESS as _, tid as libc::id_t, 19) == 0 {
            return "linux: thread nice 19";
        }
        return "linux: setpriority failed, normal priority";
    }
    #[cfg(target_os = "macos")]
    unsafe {
        extern "C" {
            fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
        }
        const QOS_CLASS_UTILITY: u32 = 0x11;
        if pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0) == 0 {
            return "macos: QOS_CLASS_UTILITY";
        }
        return "macos: QoS change failed, normal priority";
    }
    #[allow(unreachable_code)]
    "unsupported OS: normal priority"
}

pub fn logical_cpus() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}
