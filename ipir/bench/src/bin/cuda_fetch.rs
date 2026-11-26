//! A whole answer with upstream's CUDA kernel: the first pass (the table times the request) on the GPU through
//! `simplepir_kernel::cuda::CudaKernel`, one request at a time as the kernel allows, and everything else of
//! `NativeServer::respond` (parsing, the packing of every column block, serializing) on this host's CPU.
//!
//!   cuda_fetch <rows> <cols> [runs] [--expect DIR]
//!
//! The table and the 32 requests are those of `cpu_fetch` for the shape (the same xorshift cells, setup and
//! requests). The answers go through `NativeServer::respond_with_first_pass` (patch 0004) and are compared, byte for
//! byte, with `NativeServer::respond`'s; `--expect DIR` also compares them with `DIR/answers.bin`, which
//! `cpu_fetch --dump DIR` wrote. A difference ends the program with exit code 1.
//!
//! Prints one JSON line: the kernel's start (`initMs`: context and NVRTC compile), the table's upload (`uploadMs`),
//! the same upload again over a resident table (`reuploadMedianMs`: what a changed column block costs, since the
//! kernel takes a table whole), and for calls of 1 and 32 requests the wall time of the whole answer (min, median,
//! max of `runs`, default 9, after one warm-up call) with the medians of its parts summed over the call's requests:
//! `kernelMs` (the device's own time, from CUDA events), `firstPassMs` (the first pass as the host sees it: the
//! request's copy to the device, the two kernels, the result's copy back), `deserializeMs`, `packMs`, `serializeMs`.
//! `cpu` holds the same call answered by `NativeServer::respond` alone, for the same table in the same process.
//!
//! Built only with `--features cuda`; at run time it needs the CUDA driver library and NVRTC.
use ipir_bench::*;
use ipir_sp::native::NativeServer;
use reinspiring::ReinspiringError;
use simplepir_kernel::{cuda::CudaKernel, FirstDimKernel};
use std::cell::Cell;
use std::time::{Duration, Instant};

fn median(xs: &[f64]) -> f64 {
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n % 2 == 1 { s[n / 2] } else { (s[n / 2 - 1] + s[n / 2]) / 2.0 }
}

fn fail(rows: usize, cols: usize, stage: &str, error: &str) -> ! {
    println!(
        "{{\"ok\":false,\"backend\":\"cuda\",\"rows\":{rows},\"cols\":{cols},\"stage\":\"{stage}\",\"error\":{:?}}}",
        error
    );
    std::process::exit(1);
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let expect = args.iter().position(|a| a == "--expect").map(|i| {
        let d = args[i + 1].clone();
        args.drain(i..i + 2);
        d
    });
    if args.len() < 2 {
        eprintln!("usage: cuda_fetch <rows> <cols> [runs] [--expect DIR]");
        std::process::exit(2);
    }
    let rows: usize = args[0].parse().unwrap();
    let cols: usize = args[1].parse().unwrap();
    let runs: usize = args.get(2).map_or(9, |x| x.parse().unwrap());
    let mut rng = Xorshift::new(0x5eed ^ rows as u32 ^ cols as u32);
    let cells = xorshift_cells(rows, cols, &mut rng);
    let setup = setup_cols(rows, cols);
    let id = setup.id();
    let t = Instant::now();
    let server = match NativeServer::build_with_concurrency(setup, cells.clone(), 1) {
        Ok(s) => s,
        Err(e) => fail(rows, cols, "building the table", &format!("{e:?}")),
    };
    let build_ms = ms(t.elapsed());
    let each = request_bytes(rows);
    let requests = synthetic_requests(each, 32, &id, &mut rng);

    // The CPU's answers: what the CUDA path has to equal.
    let expected: Vec<Vec<u8>> = requests.chunks_exact(each).map(|r| server.respond(r).unwrap().0).collect();
    if let Some(dir) = &expect {
        let dumped = std::fs::read(format!("{dir}/answers.bin")).unwrap();
        if dumped != expected.concat() {
            fail(rows, cols, "comparing with the dumped CPU answers", "the answers differ from cpu_fetch's");
        }
    }

    // The kernel: context, NVRTC compile, then the table (column-major u16, as the cells are).
    let t = Instant::now();
    let mut gpu = match CudaKernel::new(0) {
        Ok(k) => k,
        Err(e) => fail(rows, cols, "starting the CUDA kernel", &e.to_string()),
    };
    let init_ms = ms(t.elapsed());
    let t = Instant::now();
    if let Err(e) = gpu.try_prepare(&cells, rows, cols) {
        fail(rows, cols, "uploading the table", &e.to_string());
    }
    let upload_ms = ms(t.elapsed());

    let kernel_ms = Cell::new(0f64);
    // One call: every request through the CUDA first pass and the CPU packing. Returns the answers and the parts.
    let call = |count: usize| -> (Vec<Vec<u8>>, [Duration; 4], f64) {
        kernel_ms.set(0.0);
        let mut stages = [Duration::ZERO; 4];
        let mut answers = Vec::with_capacity(count);
        for r in requests[..count * each].chunks_exact(each) {
            let (answer, t) = server
                .respond_with_first_pass(r, |query, q| {
                    let mut out = vec![0u64; cols];
                    let k = gpu
                        .multiply_power_of_two_timed(q, &cells, rows, cols, query, &mut out)
                        .map_err(|e| ReinspiringError::InvalidParams(e.to_string()))?;
                    kernel_ms.set(kernel_ms.get() + k as f64);
                    Ok(out)
                })
                .unwrap_or_else(|e| fail(rows, cols, "answering", &format!("{e:?}")));
            stages[0] += t.deserialize;
            stages[1] += t.matrix_vector;
            stages[2] += t.packing;
            stages[3] += t.serialization;
            answers.push(answer);
        }
        (answers, stages, kernel_ms.get())
    };
    // The same call on the CPU alone.
    let cpu_call = |count: usize| -> [Duration; 4] {
        let mut stages = [Duration::ZERO; 4];
        for r in requests[..count * each].chunks_exact(each) {
            let (_, t) = server.respond(r).unwrap();
            stages[0] += t.deserialize;
            stages[1] += t.matrix_vector;
            stages[2] += t.packing;
            stages[3] += t.serialization;
        }
        stages
    };

    // Equality first, on all 32 requests (this is also the warm-up).
    let (answers, _, _) = call(32);
    let equal = answers.iter().zip(&expected).filter(|(a, b)| a == b).count();
    if equal != 32 {
        fail(rows, cols, "comparing with the CPU's answers", &format!("{equal} of 32 answers equal"));
    }

    let mut cuda = Vec::new();
    let mut cpu = Vec::new();
    for count in [1usize, 32] {
        call(count);
        let mut wall = Vec::new();
        let mut kernel = Vec::new();
        let mut parts: [Vec<f64>; 4] = Default::default();
        for _ in 0..runs {
            let t0 = Instant::now();
            let (_, s, k) = call(count);
            wall.push(ms(t0.elapsed()));
            kernel.push(k);
            for (v, d) in parts.iter_mut().zip(s) {
                v.push(ms(d));
            }
        }
        let min = wall.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = wall.iter().cloned().fold(0.0, f64::max);
        cuda.push(format!(
            "\"{count}\":{{\"wallMs\":{{\"min\":{min:.3},\"median\":{:.3},\"max\":{max:.3}}},\"kernelMs\":{:.3},\
             \"firstPassMs\":{:.3},\"deserializeMs\":{:.3},\"packMs\":{:.3},\"serializeMs\":{:.3}}}",
            median(&wall),
            median(&kernel),
            median(&parts[1]),
            median(&parts[0]),
            median(&parts[2]),
            median(&parts[3]),
        ));

        cpu_call(count);
        let mut wall = Vec::new();
        let mut parts: [Vec<f64>; 4] = Default::default();
        for _ in 0..runs {
            let t0 = Instant::now();
            let s = cpu_call(count);
            wall.push(ms(t0.elapsed()));
            for (v, d) in parts.iter_mut().zip(s) {
                v.push(ms(d));
            }
        }
        let min = wall.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = wall.iter().cloned().fold(0.0, f64::max);
        cpu.push(format!(
            "\"{count}\":{{\"wallMs\":{{\"min\":{min:.3},\"median\":{:.3},\"max\":{max:.3}}},\
             \"firstPassMs\":{:.3},\"deserializeMs\":{:.3},\"packMs\":{:.3},\"serializeMs\":{:.3}}}",
            median(&wall),
            median(&parts[1]),
            median(&parts[0]),
            median(&parts[2]),
            median(&parts[3]),
        ));
    }

    // A changed column block: the kernel has no call for one block, so the table goes up whole again.
    let mut again = Vec::new();
    for _ in 0..3 {
        let t = Instant::now();
        if let Err(e) = gpu.try_prepare(&cells, rows, cols) {
            fail(rows, cols, "uploading the table again", &e.to_string());
        }
        again.push(ms(t.elapsed()));
    }

    println!(
        "{{\"ok\":true,\"backend\":\"cuda\",\"rows\":{rows},\"cols\":{cols},\"cells\":{},\"colBlocks\":{},\"tableBytes\":{},\
         \"matrixStorage\":\"{}\",\"rayonThreads\":{},\"buildMs\":{build_ms:.2},\"initMs\":{init_ms:.2},\"uploadMs\":{upload_ms:.2},\
         \"reuploadMedianMs\":{:.2},\"runs\":{runs},\"answersEqual\":{equal},\"comparedWithDump\":{},\"cuda\":{{{}}},\"cpu\":{{{}}}}}",
        rows * cols,
        cols / BLOCK_COLS,
        rows * cols * 2,
        server.matrix_storage().first().map_or("?", |s| s.label()),
        rayon::current_num_threads(),
        median(&again),
        expect.is_some(),
        cuda.join(","),
        cpu.join(","),
    );
}
