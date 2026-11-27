//! Task 1 on the CPU: `fetch` through the vendored crates' server path, for one table shape.
//!
//!   cpu_fetch <rows> <cols> [runs] [--dump DIR] [--throughput SECONDS]
//!
//! Builds the table with `NativeServer::build_with_concurrency(setup, cells, 1)`, as the add-on's `ServerTable::build`
//! does, from the same xorshift cells as `bench/fetch-shape.ts`. Then answers a call of 1 and a call of 32 requests:
//! `NativeServer::respond` for each request of the call, one after another, as the add-on answers a call on the CPU.
//! Each request's work is parallel over the table's columns on Rayon's global pool (all logical cores). Median of
//! `runs` (default 9) calls after one warm-up call.
//!
//! Prints one JSON line in the shape of `fetch-shape.ts`'s: `uploadMs` is null (nothing is uploaded), `buildMs` is the
//! table's CPU build, and per call size the wall time (min, median, max) and the medians of the server's own stage
//! times summed over the call's requests (`cpuStagesMedianMs`: deserialize, scan, pack, serialize). `execMedianMs` is
//! all null: there is no GPU time.
//!
//! `--throughput SECONDS` adds the calls answered per second (`throughput`): for calls of 1 and of 32, callers that
//! each send their next call when their last one is answered, for SECONDS, with 1, 2, 4, 8 and 16 callers (stopping at a number that is a
//! tenth slower than the best). Every caller's requests share Rayon's pool.
//!
//! `--dump DIR` also writes what `bench/check-cpu-answer.ts` needs to compare these answers with the GPU's, byte for
//! byte: `setup-id.bin`, `requests.bin` (32 requests), `answers.bin` (this server's 32 answers) and `matrices.bin`
//! (each block's compiled packing matrix, 2,048 × 4,096 signed 32-bit words, row-major, as `GpuIpirTable` takes
//! them). The matrices are computed again through the public calls that `build_with_concurrency` makes
//! (`for_packing_blocks` in `packing.rs`) and read with patch 0002's `NativePreprocessed::matrix()`.
#[path = "../packing.rs"]
mod packing;

use ipir_bench::*;
use packing::{for_packing_blocks, matrix_words};
use ipir_sp::native::NativeServer;
use std::time::{Duration, Instant};

fn median(xs: &[f64]) -> f64 {
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n % 2 == 1 { s[n / 2] } else { (s[n / 2 - 1] + s[n / 2]) / 2.0 }
}

fn seed(rows: usize, cols: usize) -> u32 {
    0x5eed ^ rows as u32 ^ cols as u32
}

fn dump(dir: &str, rows: usize, cols: usize, server: &NativeServer, requests: &[u8], each: usize) {
    std::fs::create_dir_all(dir).unwrap();
    let setup = setup_cols(rows, cols);
    std::fs::write(format!("{dir}/setup-id.bin"), setup.id()).unwrap();
    std::fs::write(format!("{dir}/requests.bin"), requests).unwrap();
    let mut answers = Vec::new();
    for r in requests.chunks_exact(each) {
        answers.extend(server.respond(r).unwrap().0);
    }
    std::fs::write(format!("{dir}/answers.bin"), answers).unwrap();

    let cells = xorshift_cells(rows, cols, &mut Xorshift::new(seed(rows, cols)));
    let q = setup.profile().packing().q();
    let mut out = Vec::with_capacity(cols / BLOCK_COLS * 2048 * 4096 * 4);
    for_packing_blocks(&setup, &cells, |_, pre| {
        for w in matrix_words(&pre, q) {
            out.extend_from_slice(&w.to_le_bytes());
        }
    });
    std::fs::write(format!("{dir}/matrices.bin"), out).unwrap();
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let dump_dir = args.iter().position(|a| a == "--dump").map(|i| {
        let d = args[i + 1].clone();
        args.drain(i..i + 2);
        d
    });
    let throughput_s: Option<f64> = args.iter().position(|a| a == "--throughput").map(|i| {
        let v = args[i + 1].parse().unwrap();
        args.drain(i..i + 2);
        v
    });
    if args.len() < 2 {
        eprintln!("usage: cpu_fetch <rows> <cols> [runs] [--dump DIR] [--throughput SECONDS]");
        std::process::exit(2);
    }
    let rows: usize = args[0].parse().unwrap();
    let cols: usize = args[1].parse().unwrap();
    let runs: usize = args.get(2).map_or(9, |x| x.parse().unwrap());
    let mut rng = Xorshift::new(seed(rows, cols));
    let cells = xorshift_cells(rows, cols, &mut rng);
    let setup = setup_cols(rows, cols);
    let id = setup.id();
    let t = Instant::now();
    let server = match NativeServer::build_with_concurrency(setup, cells, 1) {
        Ok(s) => s,
        Err(e) => {
            println!("{{\"ok\":false,\"rows\":{rows},\"cols\":{cols},\"stage\":\"building the table\",\"error\":\"{e:?}\"}}");
            std::process::exit(1);
        }
    };
    let build_ms = ms(t.elapsed());
    let each = request_bytes(rows);
    let requests = synthetic_requests(each, 32, &id, &mut rng);

    let mut timing = Vec::new();
    for count in [1usize, 32] {
        let call = &requests[..count * each];
        let answer = |stages: &mut [Duration; 4]| {
            for r in call.chunks_exact(each) {
                let (_, t) = server.respond(r).unwrap();
                stages[0] += t.deserialize;
                stages[1] += t.matrix_vector;
                stages[2] += t.packing;
                stages[3] += t.serialization;
            }
        };
        answer(&mut [Duration::ZERO; 4]);
        let mut wall = Vec::new();
        let mut stages: [Vec<f64>; 4] = Default::default();
        for _ in 0..runs {
            let mut s = [Duration::ZERO; 4];
            let t0 = Instant::now();
            answer(&mut s);
            wall.push(ms(t0.elapsed()));
            for (v, d) in stages.iter_mut().zip(s) {
                v.push(ms(d));
            }
        }
        let min = wall.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = wall.iter().cloned().fold(0.0, f64::max);
        timing.push(format!(
            "\"{count}\":{{\"wallMs\":{{\"min\":{min:.3},\"median\":{:.3},\"max\":{max:.3}}},\
             \"execMedianMs\":{{\"unpack\":null,\"scan\":null,\"pack\":null,\"finish\":null,\"total\":null}},\"execMeasured\":0,\
             \"cpuStagesMedianMs\":{{\"deserialize\":{:.3},\"scan\":{:.3},\"pack\":{:.3},\"serialize\":{:.3}}}}}",
            median(&wall),
            median(&stages[0]),
            median(&stages[1]),
            median(&stages[2]),
            median(&stages[3]),
        ));
    }
    if let Some(dir) = &dump_dir {
        dump(dir, rows, cols, &server, &requests, each);
    }
    let throughput = throughput_s.map_or(String::new(), |seconds| {
        let line = throughput_json(seconds, &CALLERS, &mut |count, callers| {
            calls_per_second(callers, seconds, &|| {
                for r in requests[..count * each].chunks_exact(each) {
                    server.respond(r).map_err(|e| format!("{e:?}"))?;
                }
                Ok(())
            })
        });
        format!("{},", line.unwrap())
    });
    println!(
        "{{\"ok\":true,\"backend\":\"cpu\",\"rows\":{rows},\"cols\":{cols},\"cells\":{},\"colBlocks\":{},\"residentBytes\":{},\
         \"databaseBytes\":{},\"coefficientBytes\":{},\"matrixStorage\":\"{}\",\"cpu\":\"rayon {} threads\",\"rayonThreads\":{},\
         \"uploadMs\":null,\"buildMs\":{build_ms:.2},\"runs\":{runs},{throughput}\"timing\":{{{}}}}}",
        rows * cols,
        cols / BLOCK_COLS,
        server.database_bytes() + server.coefficient_bytes(),
        server.database_bytes(),
        server.coefficient_bytes(),
        server.matrix_storage().first().map_or("?", |s| s.label()),
        rayon::current_num_threads(),
        rayon::current_num_threads(),
        timing.join(",")
    );
}
