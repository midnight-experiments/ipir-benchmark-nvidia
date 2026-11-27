//! One chain block's work with CUDA kernels only, while the same GPU answers calls: the CUDA counterpart of
//! `bench/budget.ts` in its mode `gpu`.
//!
//!   cuda_budget <ROWSxCOLS,...> [--set NAME] [--rate 2] [--blocks 30] [--interval-ms 6000] [--baseline-s 20]
//!
//! The tables named (six for a set of `bench/gpu-build-sets.ts`) stay resident on the device (`cuda_fetch_native`)
//! and answer calls of 32 requests at `--rate` calls a second, round robin over the tables, open loop: call k is due
//! at k / rate seconds. First `--baseline-s` seconds of calls alone. Then every `--interval-ms` a chain block
//! arrives and changes one column block of every table; its work is:
//!
//! 1. compile: each table's new cells copied to the device and its block built there (`cuda_build`), every table's
//!    chains in one launch;
//! 2. install: each built block (the cells it read and the matrix it compiled, both on the device) laid out in its
//!    table (`write_block_device`), which goes on answering.
//!
//! The build runs on one stream and the answers on another, so calls are answered while a block is built; a block
//! is installed between two calls. The tables' cells and first matrices are synthetic (the time does not depend on
//! them): what the answers equal is checked by `cuda_fetch_native`, not here.
//!
//! Prints one JSON line in the shape of `budget.ts`'s result: `runs` holds the baseline (`mode: "none"`) and the
//! run with block work (`mode: "cuda"`), with the block work's times (`blockWorkMs`, `compileMs`, `uploadMs`: the
//! install) and the calls' latency from the moment a call is due to its answer's bytes, during block work and
//! outside it. `gpuPeakMiB`: what this run's buffers hold on the device.
#[path = "../cuda_build.rs"]
#[allow(dead_code)]
mod cuda_build;
#[path = "../cuda_fetch_native.rs"]
#[allow(dead_code)]
mod cuda_fetch_native;
#[path = "../gpu_prep.rs"]
#[allow(dead_code)]
mod gpu_prep;
#[path = "../rns32.rs"]
mod rns32;

use cuda_build::{launch_many, CudaBuilder, CudaDevice};
use cuda_fetch_native::{CudaFetch, FetchDevice, FetchShape, Tuning, KEY_WORDS};
use ipir_bench::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

const GOAL_MS: f64 = 1000.0;
const BUDGET_MS: f64 = 3000.0;

struct Table {
    fetch: Mutex<CudaFetch>,
    shape: FetchShape,
    requests: Vec<u8>,
}

struct Call {
    due: f64,
    end: f64,
}

/// `{n, min, p50, p90, p99, max}` of `xs`, as `stats` in `bench/lib.ts`.
fn stats(xs: &[f64]) -> String {
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let at = |p: f64| -> String {
        if s.is_empty() {
            return "null".into();
        }
        let rank = p / 100.0 * (s.len() - 1) as f64;
        let (lo, hi) = (rank.floor() as usize, rank.ceil() as usize);
        format!("{:.3}", s[lo] + (s[hi] - s[lo]) * (rank - lo as f64))
    };
    format!("{{\"n\":{},\"min\":{},\"p50\":{},\"p90\":{},\"p99\":{},\"max\":{}}}", s.len(), at(0.0), at(50.0), at(90.0), at(99.0), at(100.0))
}

/// Answers calls of 32 at `rate` a second, round robin over the tables, until `stop`. Times are milliseconds since
/// `origin`. Returns the calls, how many were issued, and the first failure.
fn answer_calls(tables: &[Table], rate: f64, origin: Instant, stop: &AtomicBool) -> (Vec<Call>, usize, Option<String>) {
    let t0 = origin.elapsed();
    let mut done = Vec::new();
    let mut k = 0usize;
    loop {
        let due = t0 + Duration::from_secs_f64(k as f64 / rate);
        while origin.elapsed() < due && !stop.load(Ordering::Relaxed) {
            std::thread::sleep((due - origin.elapsed()).min(Duration::from_millis(20)));
        }
        if stop.load(Ordering::Relaxed) {
            return (done, k, None);
        }
        let t = &tables[k % tables.len()];
        k += 1;
        let answered = t.shape.prepare(&t.requests, 32).and_then(|p| {
            let (coefficients, _, digests) = t.fetch.lock().unwrap().run(&p, || t.shape.digests(&t.requests))?;
            Ok(t.shape.answers(&coefficients, &digests))
        });
        match answered {
            Ok(answers) if answers.len() == 32 => done.push(Call { due: ms(due), end: ms(origin.elapsed()) }),
            Ok(_) => return (done, k, Some("a call gave fewer than 32 answers".into())),
            Err(e) => return (done, k, Some(e)),
        }
    }
}

struct Options {
    shapes: Vec<(usize, usize)>,
    set: String,
    rate: f64,
    blocks: usize,
    interval_ms: f64,
    baseline_s: f64,
}

fn run(o: &Options) -> Result<String, String> {
    let build_dev = CudaDevice::new(0)?;
    let dev = FetchDevice::with_context(build_dev.context())?;
    let (used_at_start, total) = dev.memory()?;
    let tuning = Tuning::from_env("IPIR_CUDA_FETCH")?;
    let mut rng = Xorshift::new(0x6b10c);
    // One synthetic compiled matrix for every block of every table: 26-bit signed entries.
    let matrix: Vec<i32> = (0..BLOCK_COLS * KEY_WORDS).map(|_| (rng.next() as i32) >> 6).collect();
    let mut tables = Vec::new();
    let mut builders = Vec::new();
    let mut new_cells = Vec::new();
    let mut resident = 0usize;
    for (i, &(rows, cols)) in o.shapes.iter().enumerate() {
        let id = setup_cols(rows, cols).id();
        let block = cells(rows, 1 + i as u64);
        let mut fetch = CudaFetch::empty(&dev, rows, cols, id, tuning)?;
        for k in 0..cols / BLOCK_COLS {
            fetch.write_block(k, &block, &matrix)?;
        }
        resident += fetch.resident_bytes();
        let requests = synthetic_requests(request_bytes(rows), 32, &id, &mut rng);
        tables.push(Table { shape: fetch.shape, fetch: Mutex::new(fetch), requests });
        builders.push(CudaBuilder::with_device(&build_dev, &setup(rows))?);
        new_cells.push(cells(rows, 100 + i as u64));
    }

    // One chain block's work; `b` picks the column block each table replaces.
    let block_work = |builders: &mut [CudaBuilder], b: usize| -> Result<(f64, f64), String> {
        let t = Instant::now();
        for (builder, c) in builders.iter_mut().zip(&new_cells) {
            builder.upload(c)?;
        }
        launch_many(&builders.iter().collect::<Vec<_>>())?;
        let compile = ms(t.elapsed());
        let t = Instant::now();
        for (table, builder) in tables.iter().zip(builders.iter()) {
            let k = b % (table.shape.cols / BLOCK_COLS);
            table.fetch.lock().unwrap().write_block_device(k, builder.cells_buffer(), builder.matrix_buffer())?;
        }
        Ok((compile, ms(t.elapsed())))
    };
    // Before the clock: the kernels' first run, and every table's call buffers.
    block_work(&mut builders, 0)?;
    for t in &tables {
        let p = t.shape.prepare(&t.requests, 32)?;
        t.fetch.lock().unwrap().run(&p, || ())?;
    }
    let (used, _) = dev.memory()?;
    let call_bytes: usize = tables.iter().map(|t| t.fetch.lock().unwrap().call_bytes()).sum();

    let origin = Instant::now();
    // The baseline: calls alone.
    let stop = AtomicBool::new(false);
    let (base, base_issued, base_failure) = std::thread::scope(|sc| {
        let caller = sc.spawn(|| answer_calls(&tables, o.rate, origin, &stop));
        std::thread::sleep(Duration::from_secs_f64(o.baseline_s));
        stop.store(true, Ordering::Relaxed);
        caller.join().unwrap()
    });
    if let Some(e) = base_failure {
        return Err(e);
    }

    // Calls and a chain block every interval.
    let stop = AtomicBool::new(false);
    let mut work: Vec<(f64, f64, f64, f64)> = Vec::new(); // start, end, compile, install
    let (calls, issued, failure) = std::thread::scope(|sc| -> Result<_, String> {
        let caller = sc.spawn(|| answer_calls(&tables, o.rate, origin, &stop));
        let phase = Instant::now();
        let mut outcome = Ok(());
        for b in 0..o.blocks {
            let due = Duration::from_secs_f64(b as f64 * o.interval_ms / 1e3);
            if phase.elapsed() < due {
                std::thread::sleep(due - phase.elapsed());
            }
            let start = ms(origin.elapsed());
            match block_work(&mut builders, b + 1) {
                Ok((compile, install)) => work.push((start, ms(origin.elapsed()), compile, install)),
                Err(e) => {
                    outcome = Err(e);
                    break;
                }
            }
        }
        if outcome.is_ok() {
            std::thread::sleep(Duration::from_secs_f64((o.interval_ms / 1e3).min(2.0)));
        }
        stop.store(true, Ordering::Relaxed);
        let r = caller.join().unwrap();
        outcome.map(|_| r)
    })?;
    if let Some(e) = failure {
        return Err(e);
    }

    let latency = |cs: &[&Call]| stats(&cs.iter().map(|c| c.end - c.due).collect::<Vec<_>>());
    let overlaps = |c: &Call| work.iter().any(|w| c.due < w.1 && c.end > w.0);
    let all: Vec<&Call> = calls.iter().collect();
    let during: Vec<&Call> = calls.iter().filter(|c| overlaps(c)).collect();
    let outside: Vec<&Call> = calls.iter().filter(|c| !overlaps(c)).collect();
    let work_ms: Vec<f64> = work.iter().map(|w| w.1 - w.0).collect();
    let shapes: Vec<String> = o.shapes.iter().map(|(r, c)| format!("{{\"rows\":{r},\"cols\":{c}}}")).collect();
    Ok(format!(
        "{{\"ok\":true,\"backend\":\"cuda-native\",\"device\":{:?},\"set\":{:?},\"tables\":[{}],\"rates\":[{rate}],\"modes\":[\"cuda\"],\"blocks\":{},\
         \"intervalMs\":{},\"baselineS\":{},\"budgetMs\":{BUDGET_MS},\"goalMs\":{GOAL_MS},\"residentBytes\":{resident},\"callBytes\":{call_bytes},\
         \"stagingBytes\":{},\"deviceUsedAtStartBytes\":{used_at_start},\"deviceUsedBytes\":{used},\"deviceTotalBytes\":{total},\"gpuPeakMiB\":{},\
         \"tuning\":{:?},\"runs\":[{{\"rate\":{rate},\"mode\":\"none\",\"issued\":{base_issued},\"calls\":{},\"answerLatencyMs\":{}}},\
         {{\"rate\":{rate},\"mode\":\"cuda\",\"blocks\":{},\"blockWorkMs\":{},\"compileMs\":{},\"uploadMs\":{},\"overBudget\":{},\"overGoal\":{},\
         \"issued\":{issued},\"calls\":{},\"answerLatencyMs\":{{\"all\":{},\"duringBlockWork\":{},\"outsideBlockWork\":{}}}}}]}}",
        dev.name(),
        o.set,
        shapes.join(","),
        o.blocks,
        o.interval_ms,
        o.baseline_s,
        dev.staging_bytes(),
        (used - used_at_start).div_ceil(1 << 20),
        tuning.label(),
        base.len(),
        latency(&base.iter().collect::<Vec<_>>()),
        work.len(),
        stats(&work_ms),
        stats(&work.iter().map(|w| w.2).collect::<Vec<_>>()),
        stats(&work.iter().map(|w| w.3).collect::<Vec<_>>()),
        work_ms.iter().filter(|&&x| x > BUDGET_MS).count(),
        work_ms.iter().filter(|&&x| x > GOAL_MS).count(),
        calls.len(),
        latency(&all),
        latency(&during),
        latency(&outside),
        rate = o.rate,
    ))
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut take = |name: &str| -> Option<String> {
        let i = args.iter().position(|a| a == name)?;
        let v = args.get(i + 1).cloned();
        args.drain(i..(i + 2).min(args.len()));
        v
    };
    let number = |v: Option<String>, default: f64| v.map_or(default, |x| x.parse().expect("a number"));
    let set = take("--set").unwrap_or_else(|| "custom".into());
    let rate = number(take("--rate"), 2.0);
    let blocks = number(take("--blocks"), 30.0) as usize;
    let interval_ms = number(take("--interval-ms"), 6000.0);
    let baseline_s = number(take("--baseline-s"), 20.0);
    if args.len() != 1 {
        eprintln!("usage: cuda_budget <ROWSxCOLS,...> [--set NAME] [--rate 2] [--blocks 30] [--interval-ms 6000] [--baseline-s 20]");
        std::process::exit(2);
    }
    let shapes: Vec<(usize, usize)> = args[0]
        .split(',')
        .map(|x| {
            let (r, c) = x.to_lowercase().split_once('x').map(|(r, c)| (r.parse().unwrap(), c.parse().unwrap())).expect("ROWSxCOLS");
            (r, c)
        })
        .collect();
    let o = Options { shapes, set, rate, blocks, interval_ms, baseline_s };
    match run(&o) {
        Ok(line) => println!("{line}"),
        Err(e) => {
            let does_not_fit = e.contains("OUT_OF_MEMORY") || e.contains("out of memory");
            println!("{{\"ok\":false,\"backend\":\"cuda-native\",\"set\":{:?},\"doesNotFit\":{does_not_fit},\"error\":{e:?}}}", o.set);
            std::process::exit(1);
        }
    }
}
