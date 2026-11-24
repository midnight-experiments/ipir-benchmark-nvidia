//! Task 3, CPU side: a chain block's compile work for a set of tables, on command, while the GPU answers calls in
//! another process (`bench/budget.ts` drives it).
//!
//!   block_worker <rows,rows,...> [pool threads]
//!
//! One table per row count; each chain block changes one column block (2,048 columns) of every table. Prints
//! `{"ready":...}` once, then reads commands on stdin, one per line:
//!
//! - `go seq`: the tables' blocks one after another, at normal priority, on Rayon's global pool (all logical cores);
//! - `go pool`: all the blocks at once as tasks of a pool of `pool threads` threads (default: half the logical
//!   cores), each thread at the lowest priority (`lower_thread_priority`);
//! - `go pool-seq`: one after another, on that low-priority pool;
//! - `quit`.
//!
//! Each `go` prints one JSON line: the wall time of the whole block's compile, and per table the time to expand its
//! setup and copy its cells (`prep_ms`) and the build (`build_ms`, with upstream's `hints_ms` and `packing_ms`). The
//! build recomputes the block's whole hint columns: an upper bound for the hint updates of new records, which are not
//! built yet.
use ipir_bench::*;
use rayon::prelude::*;
use std::io::{BufRead, Write};
use std::time::Instant;

fn one(rows: usize, master: &[u16]) -> String {
    let t = Instant::now();
    let s = setup(rows);
    let c = master.to_vec();
    let prep = t.elapsed();
    let b = build(s, c);
    format!("{{\"rows\":{rows},\"prep_ms\":{:.2},{}}}", ms(prep), b.json())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: block_worker <rows,...> [pool threads]");
        std::process::exit(2);
    }
    let rows: Vec<usize> = args[0].split(',').map(|x| x.trim().parse().unwrap()).collect();
    let cpus = logical_cpus();
    let pool_threads = args.get(1).map(|x| x.parse().unwrap()).unwrap_or((cpus / 2).max(1));
    let priority = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let set_priority = priority.clone();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(pool_threads)
        .thread_name(|i| format!("lowprio-{i}"))
        .start_handler(move |_| {
            let p = lower_thread_priority();
            *set_priority.lock().unwrap() = p.to_string();
        })
        .build()
        .unwrap();
    pool.broadcast(|_| ()); // every thread has started, so its priority is set
    let masters: Vec<Vec<u16>> = rows.iter().enumerate().map(|(i, &r)| cells(r, 7000 + i as u64)).collect();
    let mut out = std::io::stdout().lock();
    writeln!(
        out,
        "{{\"ready\":true,\"rows\":{:?},\"logical_cpus\":{cpus},\"global_threads\":{},\"pool_threads\":{pool_threads},\"pool_priority\":\"{}\"}}",
        rows,
        rayon::current_num_threads(),
        priority.lock().unwrap()
    )
    .unwrap();
    out.flush().unwrap();
    let mut n = 0usize;
    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap();
        let mode = match line.trim() {
            "quit" | "" => break,
            l if l.starts_with("go ") => l[3..].trim().to_string(),
            other => {
                eprintln!("block_worker: unknown command {other:?}");
                std::process::exit(2);
            }
        };
        let start = Instant::now();
        let tables: Vec<String> = match mode.as_str() {
            "seq" => rows.iter().zip(&masters).map(|(&r, m)| one(r, m)).collect(),
            "pool" => pool.install(|| rows.par_iter().zip(masters.par_iter()).map(|(&r, m)| one(r, m)).collect()),
            "pool-seq" => pool.install(|| rows.iter().zip(&masters).map(|(&r, m)| one(r, m)).collect()),
            other => {
                eprintln!("block_worker: unknown mode {other:?}");
                std::process::exit(2);
            }
        };
        let wall = start.elapsed();
        writeln!(out, "{{\"block\":{n},\"mode\":\"{mode}\",\"wall_ms\":{:.2},\"tables\":[{}]}}", ms(wall), tables.join(","))
            .unwrap();
        out.flush().unwrap();
        n += 1;
    }
}
