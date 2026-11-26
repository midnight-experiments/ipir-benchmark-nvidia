//! Equality check of the CUDA build of a column block: `pack_digest`'s SHA-256 over every output word, computed
//! from the GPU's output.
//!
//!   cuda_build_digest [--stages] [--together] [--times] [rows,...] [seeds,...]
//!                                                            (default: 2048,8192,16384,32768 and 1,2,3)
//!
//! Prints `rows seed digest hints=.. packing=..`, the line format of `pack_digest --parts` without its
//! ` storage=...` suffix, so the two outputs compare with `diff` once that suffix is cut. The matrix is hashed with
//! the storage label `i32`: the lines equal the CPU's on a host whose CPU build stores the matrix so.
//!
//! `--stages` also runs the CPU replica of the build (`gpu_prep::reference`) and compares each stage's output with
//! it, word for word: the hint masks, the slots after the aggregation, the slots and the compile input after the
//! chains, the compiled matrix. It reports each stage and names the first that differs (on standard error; exit
//! code 1 if one differs).
//!
//! `--together` builds one block of every listed table in one go for each seed (`launch_many`, the report's path
//! for several tables: every table's chains in one launch) and prints the same lines, seed by seed. `--times` prints
//! each build's stage times on standard error.
//!
//! Built only with `--features cuda-13` (or `cuda-12`); at run time it needs the CUDA driver library and NVRTC.
// Modules of this program, not of the library: they need patch 0003's accessors, and the library also builds
// against the crates without patches 0002 and 0003 (`ipir/build-stock.sh`).
#[path = "../cuda_build.rs"]
#[allow(dead_code)]
mod cuda_build;
#[path = "../gpu_prep.rs"]
#[allow(dead_code)]
mod gpu_prep;
#[path = "../rns32.rs"]
mod rns32;

use cuda_build::*;
use ipir_bench::*;

fn list(s: &str) -> Vec<u64> {
    s.split(',').map(|x| x.trim().parse().expect("a number")).collect()
}

fn run() -> Result<bool, String> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| args.iter().any(|a| a == name);
    let (stages, together, times) = (flag("--stages"), flag("--together"), flag("--times"));
    args.retain(|a| !a.starts_with("--"));
    let all_rows = list(args.first().map_or("2048,8192,16384,32768", String::as_str));
    let seeds = list(args.get(1).map_or("1,2,3", String::as_str));
    let device = CudaDevice::new(0)?;
    let mut equal = true;
    if together {
        let mut builders = all_rows
            .iter()
            .map(|&rows| CudaBuilder::with_device(&device, &setup(rows as usize)))
            .collect::<Result<Vec<_>, _>>()?;
        for &seed in &seeds {
            for (b, &rows) in builders.iter_mut().zip(&all_rows) {
                b.upload(&cells(rows as usize, seed))?;
            }
            let gpu = launch_many(&builders.iter().collect::<Vec<_>>())?;
            if times {
                eprintln!("  seed {seed}: {} tables together, device {gpu:.2} ms", builders.len());
            }
            for (b, &rows) in builders.iter().zip(&all_rows) {
                let (all, h, pk) = digest(b)?;
                println!("{rows} {seed} {all} hints={h} packing={pk}");
            }
        }
        return Ok(true);
    }
    for &rows in &all_rows {
        let rows = rows as usize;
        let setup = setup(rows);
        let p = setup.profile();
        let (d, q) = (p.packing().d(), p.packing().q());
        let mut g = CudaBuilder::with_device(&device, &setup)?;
        for &seed in &seeds {
            let cells = cells(rows, seed);
            if !stages {
                let t = g.build_staged(&cells)?;
                if times {
                    let f = g.build_fused(&cells)?;
                    eprintln!(
                        "  {rows} {seed}: upload {:.2} ms; device: hints {:.2}, aggregation {:.2}, chains {:.2}, \
                         compile {:.2} ms; in one go: device {:.2} ms, wall {:.2} ms",
                        ms(t.upload_cells),
                        ms(t.hints.gpu),
                        ms(t.aggregation.gpu),
                        ms(t.chains.gpu),
                        ms(t.compile.gpu),
                        ms(f.fused.gpu),
                        ms(f.fused.wall)
                    );
                }
            } else {
                // Stage by stage, each read back before the next overwrites it.
                let r = gpu_prep::reference(&setup, &cells, d / 2 - 1);
                let [h, a, c, m] = g.stage_levels();
                let mut first: Option<&str> = None;
                let mut note = |stage: &'static str, differs: bool| {
                    if differs && first.is_none() {
                        first = Some(stage);
                    }
                };
                g.upload(&cells)?;
                g.run(h)?;
                let bad = g.hint_masks()?.iter().zip(&r.hint_masks).filter(|(x, y)| x != y).count();
                eprintln!("  hints: {bad} of {d} columns differ");
                note("hints", bad > 0);
                g.run(a)?;
                let bad = g.slot_rows()?.iter().zip(&r.slots_before).filter(|(x, y)| x != y).count();
                eprintln!("  aggregation: {bad} of {d} slots differ");
                note("aggregation", bad > 0);
                g.run(c)?;
                let slots = g.slot_rows()?;
                let bad = slots.iter().zip(&r.slots_after).filter(|(x, y)| x != y).count();
                let bad_digits = g.compile_input()?.iter().zip(&r.digits).filter(|(x, y)| x != y).count();
                eprintln!(
                    "  chains: {bad} of {d} slots differ (mask {}, other mask {}), {bad_digits} of {} digits differ",
                    if slots[0] == r.slots_after[0] { "equal" } else { "DIFFERS" },
                    if slots[d / 2] == r.slots_after[d / 2] { "equal" } else { "DIFFERS" },
                    r.digits.len()
                );
                note("chains", bad > 0 || bad_digits > 0);
                g.run(m)?;
                let matrix = g.matrix_i32()?;
                assert_eq!(matrix.len(), r.matrix.len());
                let bad = matrix.iter().zip(&r.matrix).filter(|(&x, &y)| (x as i128).rem_euclid(q as i128) as u64 != y).count();
                eprintln!("  compile: {bad} of {} matrix words differ (CPU storage {})", r.matrix.len(), r.storage_label);
                note("compile", bad > 0);
                match first {
                    Some(stage) => {
                        eprintln!("  {rows} {seed}: the first stage that differs from the CPU replica is {stage}");
                        equal = false;
                    }
                    None => eprintln!("  {rows} {seed}: every stage equals the CPU replica"),
                }
            }
            let (all, h, pk) = digest(&g)?;
            println!("{rows} {seed} {all} hints={h} packing={pk}");
        }
    }
    Ok(equal)
}

fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(e) => {
            eprintln!("cuda_build_digest: {e}");
            std::process::exit(1);
        }
    }
}
