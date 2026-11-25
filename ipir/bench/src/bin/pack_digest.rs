//! Equality check of the block compile: SHA-256 over every output word.
//!
//!   pack_digest [rows,...] [seeds,...]      (default: 2048,8192,16384,32768 and 1,2,3)
//!
//! For each row count and seed, builds one block with `build_with_concurrency(setup(rows), cells(rows, seed), 1)`
//! and prints `rows seed digest`. The digest covers the hint masks of every block (computed by the same code the
//! build uses) and everything the built server retains: the database layout, each block's setup id, both public
//! masks and the compiled packing matrix in its stored form. With `PACK_DIGEST_PARTS=1` it also prints one digest per
//! part (hints, server) after the combined one.
use ipir_bench::*;
use ipir_sp::native::{digest_hint_masks, NativeServer};
use sha2::{Digest, Sha256};

fn list(s: &str) -> Vec<u64> {
    s.split(',').map(|x| x.trim().parse().expect("a number")).collect()
}

fn hex(d: impl AsRef<[u8]>) -> String {
    d.as_ref().iter().map(|b| format!("{b:02x}")).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let all_rows = list(args.first().map_or("2048,8192,16384,32768", String::as_str));
    let seeds = list(args.get(1).map_or("1,2,3", String::as_str));
    let parts = std::env::var("PACK_DIGEST_PARTS").is_ok_and(|v| v == "1");
    for &rows in &all_rows {
        let rows = rows as usize;
        for &seed in &seeds {
            let cells = cells(rows, seed);
            let mut hints = Sha256::new();
            for block in digest_hint_masks(&setup(rows), &cells).unwrap() {
                for mask in block {
                    hints.update(mask.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
                }
            }
            let server = NativeServer::build_with_concurrency(setup(rows), cells, 1).unwrap();
            let mut words = Sha256::new();
            server.digest_words(&mut |bytes| words.update(bytes));
            let (hints, words) = (hints.finalize(), words.finalize());
            let mut all = Sha256::new();
            all.update(hints);
            all.update(words);
            if parts {
                println!("{rows} {seed} {} hints={} server={}", hex(all.finalize()), hex(hints), hex(words));
            } else {
                println!("{rows} {seed} {}", hex(all.finalize()));
            }
        }
    }
}
