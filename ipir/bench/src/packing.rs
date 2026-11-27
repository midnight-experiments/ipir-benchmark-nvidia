//! Each column block's packing preprocessing and its compiled matrix, computed apart from a `NativeServer` (which
//! keeps them private) through the public calls its build makes.
//!
//! A module of the programs that use it (`#[path]`), not of the library: `matrix_words` needs patch 0002's
//! `NativePreprocessed::matrix()`, which the stock crates do not have.
use ipir_sp::native::NativePublicSetup;
use rayon::prelude::*;
use reinspiring::{lift_ntt::LiftContext, native::NativePreprocessed};

/// The packing preprocessing of whole column blocks, one after another, from their cells (column-major, a multiple
/// of `rows × 2,048` of them): the public calls `NativeServer::build_with_concurrency` makes for each block
/// (`LiftContext::public_dot` per column, then `NativePreprocessed::build_two_mask`). `each` gets a block's index in
/// `cells` and its result, which does not depend on where the block is in a table.
pub fn for_packing_blocks(setup: &NativePublicSetup, cells: &[u16], mut each: impl FnMut(usize, NativePreprocessed)) {
    let p = setup.profile();
    let (d, q, rows) = (p.packing().d(), p.packing().q(), p.rows());
    assert_eq!(cells.len() % (rows * d), 0, "whole column blocks");
    let lift = LiftContext::new(d, q).unwrap();
    let public = lift.prepare_public_dot(setup.query_masks(), (p.packing().p() - 1).min(q / 2)).unwrap();
    for (k, block) in cells.chunks_exact(rows * d).enumerate() {
        let masks: Vec<Vec<u64>> = block
            .par_chunks_exact(rows)
            .map(|col| {
                let coeffs: Vec<Vec<u64>> = col.chunks_exact(d).map(|poly| poly.iter().map(|&x| x.into()).collect()).collect();
                lift.public_dot(&public, &coeffs).unwrap()
            })
            .collect();
        each(k, NativePreprocessed::build_two_mask(setup.packing_setup(), &masks).unwrap());
    }
}

/// A block's compiled packing matrix as the GPU paths take it: 2,048 × 4,096 signed 32-bit words, row-major (the
/// canonical residues modulo q, centred).
pub fn matrix_words(pre: &NativePreprocessed, q: u64) -> Vec<i32> {
    let m = pre.matrix().to_compiled();
    assert_eq!((m.rows, m.cols), (2048, 4096), "a compiled packing matrix is 2,048 × 4,096");
    m.data
        .iter()
        .map(|&x| {
            let v = if x >= q / 2 { x as i128 - q as i128 } else { x as i128 };
            i32::try_from(v).expect("a compiled matrix entry fits in 32 bits")
        })
        .collect()
}
