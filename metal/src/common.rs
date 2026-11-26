//! The benchmark's profile, setup and cells (copied from the NVIDIA benchmark's `ipir/bench/src/lib.rs` at
//! d400e43b), and the `pack_digest` hash of a build's output words.
use ipir_sp::native::*;
use rand_chacha::{
    rand_core::{RngCore, SeedableRng},
    ChaCha20Rng,
};
use reinspiring::native::{NativeParams, SecretDistribution};
use sha2::{Digest, Sha256};
use std::time::Duration;

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

/// `rows × 2,048` random 16-bit cells, column-major.
pub fn cells(rows: usize, seed: u64) -> Vec<u16> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    (0..rows * BLOCK_COLS).map(|_| rng.next_u32() as u16).collect()
}

pub fn hex(d: impl AsRef<[u8]>) -> String {
    d.as_ref().iter().map(|b| format!("{b:02x}")).collect()
}

fn update(h: &mut Sha256, words: &[u64]) {
    for chunk in words.chunks(1 << 12) {
        h.update(chunk.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
    }
}

/// The output words of one column block's build, in the form `pack_digest` hashes.
pub struct BlockOutput<'a> {
    /// The hint mask of each column, d words each.
    pub hint_masks: &'a [Vec<u64>],
    pub mask: &'a [u64],
    pub other_mask: &'a [u64],
    /// `NativeMatrixStorage::label()` of the compiled matrix.
    pub storage_label: &'static str,
    pub rows: usize,
    pub cols: usize,
    pub q: u64,
    /// The compiled matrix's canonical words (`NativeMatrix::to_compiled().data`), row-major.
    pub matrix: &'a [u64],
}

/// `pack_digest`'s digest of a one-block table: (all, hints, packing).
pub fn digest(o: &BlockOutput) -> (String, String, String) {
    let (mut hints, mut packing) = (Sha256::new(), Sha256::new());
    for mask in o.hint_masks {
        update(&mut hints, mask);
    }
    update(&mut packing, o.mask);
    update(&mut packing, o.other_mask);
    packing.update(o.storage_label);
    update(&mut packing, &[o.rows as u64, o.cols as u64, o.q]);
    update(&mut packing, o.matrix);
    let (hints, packing) = (hints.finalize(), packing.finalize());
    let mut all = Sha256::new();
    all.update(hints);
    all.update(packing);
    (hex(all.finalize()), hex(hints), hex(packing))
}
