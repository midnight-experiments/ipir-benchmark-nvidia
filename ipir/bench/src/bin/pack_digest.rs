//! Equality check of the block compile: SHA-256 over every output word.
//!
//!   pack_digest [--parts] [rows,...] [seeds,...]      (default: 2048,8192,16384,32768 and 1,2,3)
//!
//! For each row count and seed, computes the hints and the packing state that
//! `build_with_concurrency(setup(rows), cells(rows, seed), 1)` computes, through the same public calls, and prints
//! `rows seed digest`. The digest is SHA-256 over two part digests: one over the hint masks of every output block, one
//! over each block's packing state (both public masks, and the compiled matrix with its storage width). `--parts` also
//! prints the two part digests.
use ipir_bench::*;
use rayon::prelude::*;
use reinspiring::{lift_ntt::LiftContext, native::NativePreprocessed};
use sha2::{Digest, Sha256};

fn list(s: &str) -> Vec<u64> {
    s.split(',').map(|x| x.trim().parse().expect("a number")).collect()
}

fn hex(d: impl AsRef<[u8]>) -> String {
    d.as_ref().iter().map(|b| format!("{b:02x}")).collect()
}

fn update(h: &mut Sha256, words: &[u64]) {
    for chunk in words.chunks(1 << 12) {
        h.update(chunk.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
    }
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let parts = args.iter().any(|a| a == "--parts");
    args.retain(|a| a != "--parts");
    let all_rows = list(args.first().map_or("2048,8192,16384,32768", String::as_str));
    let seeds = list(args.get(1).map_or("1,2,3", String::as_str));
    for &rows in &all_rows {
        let rows = rows as usize;
        let setup = setup(rows);
        let p = setup.profile();
        assert!(p.is_two_mask(), "the benchmark profile has two-mask output");
        let (d, q) = (p.packing().d(), p.packing().q());
        let lift = LiftContext::new(d, q).unwrap();
        let public = lift.prepare_public_dot(setup.query_masks(), (p.packing().p() - 1).min(q / 2)).unwrap();
        for &seed in &seeds {
            let cells = cells(rows, seed);
            let (mut hints, mut packing) = (Sha256::new(), Sha256::new());
            for block in 0..p.cols() / d {
                // One hint mask per column: its rows/d polynomials dotted with the query masks.
                let masks: Vec<Vec<u64>> = (block * d..(block + 1) * d)
                    .into_par_iter()
                    .map(|col| {
                        let coeffs: Vec<Vec<u64>> = cells[col * rows..(col + 1) * rows]
                            .chunks_exact(d)
                            .map(|poly| poly.iter().map(|&x| x.into()).collect())
                            .collect();
                        lift.public_dot(&public, &coeffs).unwrap()
                    })
                    .collect();
                for mask in &masks {
                    update(&mut hints, mask);
                }
                let pre = NativePreprocessed::build_two_mask(setup.packing_setup(), &masks).unwrap();
                update(&mut packing, pre.mask());
                update(&mut packing, pre.other_mask().unwrap());
                packing.update(pre.matrix().storage().label());
                let matrix = pre.matrix().to_compiled();
                update(&mut packing, &[matrix.rows as u64, matrix.cols as u64, matrix.q]);
                update(&mut packing, &matrix.data);
            }
            let (hints, packing) = (hints.finalize(), packing.finalize());
            let mut all = Sha256::new();
            all.update(hints);
            all.update(packing);
            if parts {
                println!("{rows} {seed} {} hints={} packing={}", hex(all.finalize()), hex(hints), hex(packing));
            } else {
                println!("{rows} {seed} {}", hex(all.finalize()));
            }
        }
    }
}
