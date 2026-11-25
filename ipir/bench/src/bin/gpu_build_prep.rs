//! The tables the GPU block build needs for one table shape, and optionally one block's cells and the CPU's outputs
//! to check the GPU's against.
//!
//!   gpu_build_prep <rows> <out.bin> [--cells <seed> <cells.u16>] [--expect <seed> <dir> [--steps N]]
//!
//! `<out.bin>`: the tables (`gpu_prep::GpuPrep::write`). `--cells`: the cells `pack_digest` builds for
//! that seed (`cells(rows, seed)`), little-endian u16, column-major. `--expect`: the same cells as `cells.u16`, and
//! the CPU's `masks.u64` (hint masks), `slots-before.u64` and `slots-after.u64` (the slots around the chains),
//! `digits.i32` (the compile input) and `matrix.u64` (the compiled matrix's canonical words), with only the first N
//! steps of each chain when `--steps` is given (then no matrix).
// Modules of this program, not of the library: they need patch 0003's accessors, and the library also builds
// against the crates without patches 0002 and 0003 (`ipir/build-stock.sh`).
#[path = "../gpu_prep.rs"]
mod gpu_prep;
#[path = "../rns32.rs"]
mod rns32;

use gpu_prep::*;
use ipir_bench::*;
use std::time::Instant;

fn bytes<T: Copy, const N: usize>(v: &[T], f: impl Fn(T) -> [u8; N]) -> Vec<u8> {
    v.iter().flat_map(|&x| f(x)).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: gpu_build_prep <rows> <out.bin> [--cells <seed> <cells.u16>] [--expect <seed> <dir> [--steps N]]");
        std::process::exit(2);
    }
    let rows: usize = args[0].parse().expect("rows");
    let t = Instant::now();
    let setup = setup(rows);
    let prep = GpuPrep::new(&setup);
    prep.write(&args[1]).unwrap();
    eprintln!("gpu_build_prep {rows}: tables in {:.0} ms", ms(t.elapsed()));
    if let Some(i) = args.iter().position(|a| a == "--cells") {
        let seed: u64 = args[i + 1].parse().expect("seed");
        std::fs::write(&args[i + 2], bytes(&cells(rows, seed), u16::to_le_bytes)).unwrap();
    }
    if let Some(i) = args.iter().position(|a| a == "--expect") {
        let seed: u64 = args[i + 1].parse().expect("seed");
        let dir = &args[i + 2];
        let d = setup.profile().packing().d();
        let steps = args.iter().position(|a| a == "--steps").map_or(d / 2 - 1, |j| args[j + 1].parse().expect("steps"));
        std::fs::create_dir_all(dir).unwrap();
        let cells = cells(rows, seed);
        let r = reference(&setup, &cells, steps);
        let flat = |x: &[Vec<u64>]| bytes(&x.iter().flatten().copied().collect::<Vec<u64>>(), u64::to_le_bytes);
        std::fs::write(format!("{dir}/cells.u16"), bytes(&cells, u16::to_le_bytes)).unwrap();
        std::fs::write(format!("{dir}/masks.u64"), flat(&r.hint_masks)).unwrap();
        std::fs::write(format!("{dir}/slots-before.u64"), flat(&r.slots_before)).unwrap();
        std::fs::write(format!("{dir}/slots-after.u64"), flat(&r.slots_after)).unwrap();
        std::fs::write(format!("{dir}/digits.i32"), bytes(&r.digits, i32::to_le_bytes)).unwrap();
        // The canonical words, however this CPU stores the matrix (`r.storage_label`: i32, packed27, ...).
        if !r.matrix.is_empty() {
            std::fs::write(format!("{dir}/matrix.u64"), bytes(&r.matrix, u64::to_le_bytes)).unwrap();
        }
        eprintln!("gpu_build_prep {rows}: reference for seed {seed}, {steps} chain steps, in {dir}");
    }
}
