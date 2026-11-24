//! Task 2: compile one column block (2,048 columns) on the CPU.
//!
//!   compile_block <rows,rows,...> <reps> <tables at once,...>
//!
//! For every row count and every "at once" value, `reps` repetitions. Each repetition builds `at once` one-block
//! tables on that many OS threads at the same moment (as the Node add-on runs builds on libuv's pool), each with
//! `build_with_concurrency(.., 1)`; Rayon's global pool (all logical cores) is shared, as in the add-on. Setups and
//! cells are made before the clock starts. One JSON line per repetition: the wall time of all tables, and upstream's
//! offline timing of each (`hints`, `packing` = mask trace and matrix compilation, `layout`, `prepare` = total).
use ipir_bench::*;
use std::time::Instant;

fn list(s: &str) -> Vec<usize> {
    s.split(',').map(|x| x.trim().parse().expect("a number")).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        eprintln!("usage: compile_block <rows,...> <reps> <tables at once,...>");
        std::process::exit(2);
    }
    let (all_rows, reps, all_at_once) = (list(&args[0]), args[1].parse::<usize>().unwrap(), list(&args[2]));
    for &at_once in &all_at_once {
        for &rows in &all_rows {
            for rep in 0..reps {
                let inputs: Vec<_> = (0..at_once)
                    .map(|t| (setup(rows), cells(rows, (rep * 1000 + t) as u64)))
                    .collect();
                let start = Instant::now();
                let timings: Vec<BuildTiming> = std::thread::scope(|s| {
                    let handles: Vec<_> =
                        inputs.into_iter().map(|(setup, cells)| s.spawn(move || build(setup, cells))).collect();
                    handles.into_iter().map(|h| h.join().unwrap()).collect()
                });
                let wall = start.elapsed();
                let tables: Vec<String> = timings.iter().map(|t| format!("{{{}}}", t.json())).collect();
                println!(
                    "{{\"rows\":{rows},\"cols\":{BLOCK_COLS},\"rep\":{rep},\"tables_at_once\":{at_once},\"rayon_threads\":{},\"wall_ms\":{:.2},\"tables\":[{}]}}",
                    rayon::current_num_threads(),
                    ms(wall),
                    tables.join(",")
                );
            }
        }
    }
}
