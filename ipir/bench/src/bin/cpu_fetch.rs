//! Task 1 on the CPU: `fetch` through the vendored crates' server path, for one table shape.
//!
//!   cpu_fetch <rows> <cols> [runs] [--dump DIR]
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
//! `--dump DIR` also writes what `bench/check-cpu-answer.ts` needs to compare these answers with the GPU's, byte for
//! byte: `setup-id.bin`, `requests.bin` (32 requests), `answers.bin` (this server's 32 answers) and `matrices.bin`
//! (each block's compiled packing matrix, 2,048 × 4,096 signed 32-bit words, row-major, as `GpuIpirTable` takes
//! them). The matrices are computed again through the public calls that `build_with_concurrency` makes
//! (`LiftContext::public_dot` per column, then `NativePreprocessed::build_two_mask`) and read with patch 0002's
//! `NativePreprocessed::matrix()`.
use ipir_bench::*;
use ipir_sp::native::NativeServer;
use rayon::prelude::*;
use reinspiring::{lift_ntt::LiftContext, native::NativePreprocessed};
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
    let p = setup.profile();
    let (d, q) = (p.packing().d(), p.packing().q());
    let lift = LiftContext::new(d, q).unwrap();
    let public = lift.prepare_public_dot(setup.query_masks(), (p.packing().p() - 1).min(q / 2)).unwrap();
    let mut out = Vec::with_capacity(cols / d * 2048 * 4096 * 4);
    for block in 0..cols / d {
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
        let pre = NativePreprocessed::build_two_mask(setup.packing_setup(), &masks).unwrap();
        let m = pre.matrix().to_compiled();
        assert_eq!((m.rows, m.cols), (2048, 4096), "a compiled packing matrix is 2,048 × 4,096");
        for &x in &m.data {
            // Canonical residues modulo q, centred: the signed entries the GPU takes.
            let v = if x >= q / 2 { x as i128 - q as i128 } else { x as i128 };
            let w = i32::try_from(v).expect("a compiled matrix entry fits in 32 bits");
            out.extend_from_slice(&w.to_le_bytes());
        }
    }
    std::fs::write(format!("{dir}/matrices.bin"), out).unwrap();
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let dump_dir = args.iter().position(|a| a == "--dump").map(|i| {
        let d = args[i + 1].clone();
        args.drain(i..i + 2);
        d
    });
    if args.len() < 2 {
        eprintln!("usage: cpu_fetch <rows> <cols> [runs] [--dump DIR]");
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
    println!(
        "{{\"ok\":true,\"backend\":\"cpu\",\"rows\":{rows},\"cols\":{cols},\"cells\":{},\"colBlocks\":{},\"residentBytes\":{},\
         \"databaseBytes\":{},\"coefficientBytes\":{},\"matrixStorage\":\"{}\",\"cpu\":\"rayon {} threads\",\"rayonThreads\":{},\
         \"uploadMs\":null,\"buildMs\":{build_ms:.2},\"runs\":{runs},\"timing\":{{{}}}}}",
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
