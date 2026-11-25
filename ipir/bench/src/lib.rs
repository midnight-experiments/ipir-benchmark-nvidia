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
    profile_cols(rows, BLOCK_COLS)
}

/// The production profile for a table of `rows` rows and `cols` columns (a multiple of 2,048).
pub fn profile_cols(rows: usize, cols: usize) -> NativeProfile {
    let p = NativeProfile::new(
        NativeParams::new(2048, 54, 16, 19, 2, SecretDistribution::Gaussian).unwrap(),
        rows,
        cols,
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

/// The setup of a table of `rows × cols`, as `setup`.
pub fn setup_cols(rows: usize, cols: usize) -> NativePublicSetup {
    NativePublicSetup::new_without_columns(profile_cols(rows, cols), [0; 32], [0; 32])
}

/// The xorshift32 generator of `bench/lib.ts`, so that a table here has the same cells as the Deno scripts make.
pub struct Xorshift(u32);

impl Xorshift {
    pub fn new(seed: u32) -> Self {
        Self(if seed == 0 { 1 } else { seed })
    }
    pub fn next(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }
}

/// `rows × cols` column-major cells from `next() & 0xffff`, as `syntheticCells` in `bench/lib.ts`.
pub fn xorshift_cells(rows: usize, cols: usize, rng: &mut Xorshift) -> Vec<u16> {
    (0..rows * cols).map(|_| rng.next() as u16).collect()
}

/// `count` requests laid end to end: random bytes, then the magic "RNQ3" and `setup_id` at the start of each, as
/// `syntheticRequests` in `bench/lib.ts`. Every random key word is below 2^54 and so a valid one.
pub fn synthetic_requests(request_bytes: usize, count: usize, setup_id: &[u8; 32], rng: &mut Xorshift) -> Vec<u8> {
    let mut out: Vec<u8> = (0..request_bytes * count).map(|_| rng.next() as u8).collect();
    for r in out.chunks_exact_mut(request_bytes) {
        r[..4].copy_from_slice(b"RNQ3");
        r[4..36].copy_from_slice(setup_id);
    }
    out
}

/// Bytes of one request to a table of `rows` rows: magic, setup id, the key K_g (4,096 words of 54 bits), and one
/// selection word of 49 bits per row (`ipir/sizes.ts`).
pub fn request_bytes(rows: usize) -> usize {
    36 + (4096 * 54) / 8 + (rows * 49).div_ceil(8)
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
