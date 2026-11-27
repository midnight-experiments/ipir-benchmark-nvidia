//! A whole answer on an NVIDIA GPU with this benchmark's own CUDA kernels (`cuda_fetch_native`): the first pass for
//! all the requests of a call in one launch, the packing of every column block on the GPU, and the rounding. Only
//! the parsing of the requests, their SHA-256 and the packing of the answer's bits stay on the host.
//!
//!   cuda_fetch_native <rows> <cols> [runs] [--expect DIR] [--sweep] [--throughput SECONDS]
//!
//! The table and the 32 requests are those of `cpu_fetch` for the shape (the same xorshift cells, setup and
//! requests). Checked before any timing, each against the CPU's result; a difference ends the program with exit
//! code 1:
//!
//! - the answers to a call of 32 and to a call of 1, byte for byte with `NativeServer::respond`'s (and with
//!   `DIR/answers.bin`, which `cpu_fetch --dump DIR` wrote, when `--expect DIR` is given);
//! - the first pass alone and the packing product alone, unrounded, with upstream's `PreparedU16Query::dot` and
//!   `NativePreprocessed::prepare_pack`;
//! - after `write_block` of a changed column block (its cells and the matrix the CPU compiled), the answers of a table
//!   built whole on the CPU with those cells; after the original block is written back, the first answers again;
//! - the same changed block built on the device by the CUDA block build (`cuda_build`) and installed from its
//!   buffers (`write_block_device`): the same answers again.
//!
//! Prints one JSON line in the shape of `cuda_fetch`'s, with `backend: "cuda-native"`: the start (`initMs`: context
//! and NVRTC compile), the table's upload (`uploadMs`), one changed column block written from the host
//! (`writeBlockMs`: copy and layout) and installed from the block build's buffers (`installMs`), the bytes resident
//! on the device and those of a call's buffers, and for calls of 1 and 32 requests the wall time of the whole answer
//! (min, median, max of `runs`, default 9, after a warm-up call) with the medians of its parts: `kernelMs` (the
//! device's own time, from CUDA events) and in it `scanMs`, `packMs`, `finishMs`; on the host `parseMs`, `copyInMs`,
//! `hashMs`, `copyOutMs` (the wait for the device after the hashing, and the copy back), `serializeMs`.
//!
//! `--throughput SECONDS` adds the calls answered per second (`throughput`): for calls of 1 and of 32, callers that
//! each send their next call when their last one is answered, for SECONDS, with 1, 2, 4, 8 and 16 callers (stopping
//! at a number that is a tenth slower than the best). The host's part of a call (parsing, hashing, the answer's
//! bytes) runs in the caller; the device's part of one call at a time.
//!
//! `--sweep` times the layouts a call of 32 and a call of 1 can have on this card (columns, matrix rows and requests
//! per thread, segment length, block order), checks every one's answers, lists them on standard error and reports
//! the fastest as `best` (a value for `IPIR_CUDA_FETCH`).
//!
//! Built only with `--features cuda-13` (or `cuda-12`); at run time it needs the CUDA driver library and NVRTC.
#[path = "../cuda_build.rs"]
#[allow(dead_code)]
mod cuda_build;
#[path = "../cuda_fetch_native.rs"]
#[allow(dead_code)]
mod cuda_fetch_native;
#[path = "../gpu_prep.rs"]
#[allow(dead_code)]
mod gpu_prep;
#[path = "../packing.rs"]
mod packing;
#[path = "../rns32.rs"]
mod rns32;

use cuda_build::{CudaBuilder, CudaDevice};
use cuda_fetch_native::{CallTiming, CudaFetch, FetchDevice, Tuning, KEY_WORDS};
use ipir_bench::*;
use packing::{for_packing_blocks, matrix_words};
use ipir_sp::native::NativeServer;
use rayon::prelude::*;
use reinspiring::native::NativeKeys;
use reinspiring::native_kernel::PreparedU16Query;
use std::sync::Mutex;
use std::time::Instant;

fn median(xs: &[f64]) -> f64 {
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n % 2 == 1 { s[n / 2] } else { (s[n / 2 - 1] + s[n / 2]) / 2.0 }
}

struct Shape {
    rows: usize,
    cols: usize,
}

fn fail(s: &Shape, stage: &str, error: &str) -> ! {
    let does_not_fit = error.contains("OUT_OF_MEMORY") || error.contains("out of memory");
    println!(
        "{{\"ok\":false,\"backend\":\"cuda-native\",\"rows\":{},\"cols\":{},\"stage\":\"{stage}\",\"doesNotFit\":{does_not_fit},\"error\":{:?}}}",
        s.rows, s.cols, error
    );
    std::process::exit(1);
}

/// `bits` bits of `bytes` from bit `offset`, one bit at a time: the reference reader, apart from the fast one the
/// fetch uses.
fn slow_bits(bytes: &[u8], offset: usize, bits: usize) -> u64 {
    (0..bits).fold(0u64, |v, i| v | (u64::from((bytes[(offset + i) >> 3] >> ((offset + i) & 7)) & 1) << i))
}

/// The answers of a table built whole on the CPU, one per request.
fn cpu_answers(s: &Shape, cells: &[u16], requests: &[u8], each: usize) -> Vec<Vec<u8>> {
    let server = match NativeServer::build_with_concurrency(setup_cols(s.rows, s.cols), cells.to_vec(), 1) {
        Ok(x) => x,
        Err(e) => fail(s, "building the table", &format!("{e:?}")),
    };
    requests.chunks_exact(each).map(|r| server.respond(r).unwrap().0).collect()
}

fn equal_answers(got: &[Vec<u8>], want: &[Vec<u8>]) -> usize {
    got.iter().zip(want).filter(|(a, b)| a == b).count()
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let expect = args.iter().position(|a| a == "--expect").map(|i| {
        let d = args[i + 1].clone();
        args.drain(i..i + 2);
        d
    });
    let sweep = args.iter().position(|a| a == "--sweep").map(|i| args.remove(i)).is_some();
    let throughput_s: Option<f64> = args.iter().position(|a| a == "--throughput").map(|i| {
        let v = args[i + 1].parse().unwrap();
        args.drain(i..i + 2);
        v
    });
    if args.len() < 2 {
        eprintln!("usage: cuda_fetch_native <rows> <cols> [runs] [--expect DIR] [--sweep] [--throughput SECONDS]");
        std::process::exit(2);
    }
    let rows: usize = args[0].parse().unwrap();
    let cols: usize = args[1].parse().unwrap();
    let runs: usize = args.get(2).map_or(9, |x| x.parse().unwrap());
    let s = Shape { rows, cols };
    let blocks = cols / BLOCK_COLS;
    let tuning = Tuning::from_env().unwrap_or_else(|e| fail(&s, "reading IPIR_CUDA_FETCH", &e));

    let mut rng = Xorshift::new(0x5eed ^ rows as u32 ^ cols as u32);
    let cells = xorshift_cells(rows, cols, &mut rng);
    let setup = setup_cols(rows, cols);
    let id = setup.id();
    let q = setup.profile().packing().q();
    let each = request_bytes(rows);
    let requests = synthetic_requests(each, 32, &id, &mut rng);

    // The CPU's answers: what this path has to equal.
    let t = Instant::now();
    let expected = cpu_answers(&s, &cells, &requests, each);
    let build_ms = ms(t.elapsed());
    if let Some(dir) = &expect {
        let dumped = std::fs::read(format!("{dir}/answers.bin")).unwrap();
        if dumped != expected.concat() {
            fail(&s, "comparing with the dumped CPU answers", "the answers differ from cpu_fetch's");
        }
    }

    // The requests' words, read apart from the fetch's own reader.
    let key_words: Vec<Vec<u64>> =
        requests.chunks_exact(each).map(|r| (0..KEY_WORDS).map(|i| slow_bits(r, 36 * 8 + i * 54, 54)).collect()).collect();
    let selection = 36 * 8 + KEY_WORDS * 54;
    let query_words: Vec<Vec<u64>> =
        requests.chunks_exact(each).map(|r| (0..rows).map(|i| slow_bits(r, selection + i * 49, 49) << 5).collect()).collect();

    // Each block's compiled matrix, and the CPU's packing product of every request with it.
    let keys: Vec<NativeKeys> = key_words.iter().map(|w| NativeKeys::from_kg_words(setup.packing_setup(), w).unwrap()).collect();
    let mut matrices: Vec<Vec<i32>> = Vec::with_capacity(blocks);
    let mut cpu_packing = vec![0u64; 32 * cols];
    let zeros = vec![0u64; BLOCK_COLS];
    for_packing_blocks(&setup, &cells, |k, pre| {
        matrices.push(matrix_words(&pre, q));
        for (g, key) in keys.iter().enumerate() {
            let prepared = pre.prepare_keys(key).unwrap();
            let body = pre.prepare_pack(&prepared).unwrap().finish_two_mask(&zeros).unwrap();
            cpu_packing[g * cols + k * BLOCK_COLS..g * cols + (k + 1) * BLOCK_COLS].copy_from_slice(body.rows().2);
        }
    });
    // The CPU's first pass of every request.
    let mut cpu_first = vec![0u64; 32 * cols];
    for (g, words) in query_words.iter().enumerate() {
        let query = PreparedU16Query::new(words, q).unwrap();
        cpu_first[g * cols..(g + 1) * cols].par_iter_mut().zip(cells.par_chunks_exact(rows)).for_each(|(out, col)| *out = query.dot(col));
    }

    // The device: the block build's context first, the fetch on a stream of its own in it.
    let build_dev = CudaDevice::new(0).unwrap_or_else(|e| fail(&s, "opening the device", &e));
    let dev = FetchDevice::with_context(build_dev.context()).unwrap_or_else(|e| fail(&s, "compiling the fetch kernels", &e));
    let init_ms = ms(build_dev.init + dev.init);
    let (used_before, total) = dev.memory().unwrap_or((0, 0));
    let (mut table, upload) = CudaFetch::new(&dev, rows, cols, id, tuning, &cells, &matrices).unwrap_or_else(|e| fail(&s, "uploading the table", &e));
    let upload_ms = ms(upload);

    // Equality, before any timing.
    let answer = |table: &mut CudaFetch, count: usize, stage: &str| -> (Vec<Vec<u8>>, CallTiming) {
        table.respond(&requests[..count * each], count).unwrap_or_else(|e| fail(&s, stage, &e))
    };
    let must = |got: &[Vec<u8>], want: &[Vec<u8>], stage: &str| {
        let equal = equal_answers(got, want);
        if equal != want.len() || got.len() != want.len() {
            fail(&s, stage, &format!("{equal} of {} answers equal", want.len()));
        }
    };
    must(&answer(&mut table, 32, "answering").0, &expected, "comparing a call of 32 with the CPU's answers");
    must(&answer(&mut table, 1, "answering").0, &expected[..1], "comparing a call of 1 with the CPU's answers");
    for count in [32usize, 1] {
        let first = table.first_pass(&requests[..count * each], count).unwrap_or_else(|e| fail(&s, "the first pass alone", &e));
        if first != cpu_first[..count * cols] {
            fail(&s, "comparing the first pass with the CPU's", &format!("a call of {count}: the unrounded sums differ"));
        }
        let packing = table.packing_product(&requests[..count * each], count).unwrap_or_else(|e| fail(&s, "the packing alone", &e));
        if packing != cpu_packing[..count * cols] {
            fail(&s, "comparing the packing product with the CPU's", &format!("a call of {count}: the unrounded products differ"));
        }
    }

    // A changed column block: new cells for one block, the CPU's answers for the table with them.
    let kb = blocks / 2;
    let block_range = kb * BLOCK_COLS * rows..(kb + 1) * BLOCK_COLS * rows;
    let mut changed_cells = cells.clone();
    for c in &mut changed_cells[block_range.clone()] {
        *c = rng.next() as u16;
    }
    let expected_changed = cpu_answers(&s, &changed_cells, &requests, each);
    if expected_changed == expected {
        fail(&s, "changing a column block", "the changed block left the answers the same");
    }
    let mut changed_matrix = Vec::new();
    for_packing_blocks(&setup, &changed_cells[block_range.clone()], |_, pre| changed_matrix = matrix_words(&pre, q));
    table.write_block(kb, &changed_cells[block_range.clone()], &changed_matrix).unwrap_or_else(|e| fail(&s, "writing a column block", &e));
    must(&answer(&mut table, 32, "answering").0, &expected_changed, "comparing the answers after write_block with the CPU's");
    let mut write = Vec::new();
    for _ in 0..3 {
        let w = table.write_block(kb, &cells[block_range.clone()], &matrices[kb]).unwrap_or_else(|e| fail(&s, "writing a column block", &e));
        write.push((w.copy + w.layout, w.copy, w.layout));
    }
    must(&answer(&mut table, 32, "answering").0, &expected, "comparing the answers after the block was written back");

    // The same changed block built on the device by the CUDA block build, installed from its buffers.
    let mut builder = CudaBuilder::with_device(&build_dev, &ipir_bench::setup(rows)).unwrap_or_else(|e| fail(&s, "preparing the block build", &e));
    let built = builder.build_fused(&changed_cells[block_range.clone()]).unwrap_or_else(|e| fail(&s, "building a column block", &e));
    let install = table
        .write_block_device(kb, builder.cells_buffer(), builder.matrix_buffer())
        .unwrap_or_else(|e| fail(&s, "installing the built block", &e));
    must(&answer(&mut table, 32, "answering").0, &expected_changed, "comparing the answers after the CUDA build's block with the CPU's");
    table.write_block(kb, &cells[block_range], &matrices[kb]).unwrap_or_else(|e| fail(&s, "writing a column block", &e));
    drop(builder);

    let mut best = String::new();
    if sweep {
        best = run_sweep(&s, &mut table, &requests, each, &expected);
    }

    // The times: a call of 1 and a call of 32.
    let mut calls = Vec::new();
    for count in [1usize, 32] {
        answer(&mut table, count, "answering");
        let mut t: Vec<CallTiming> = Vec::new();
        for _ in 0..runs {
            let (got, timing) = answer(&mut table, count, "answering");
            must(&got, &expected[..count], "comparing a timed call with the CPU's answers");
            t.push(timing);
        }
        let of = |f: fn(&CallTiming) -> f64| median(&t.iter().map(f).collect::<Vec<_>>());
        let wall: Vec<f64> = t.iter().map(|x| x.wall).collect();
        let plan = table.plan(count);
        calls.push(format!(
            "\"{count}\":{{\"wallMs\":{{\"min\":{:.3},\"median\":{:.3},\"max\":{:.3}}},\"kernelMs\":{:.3},\"scanMs\":{:.3},\"packMs\":{:.3},\
             \"finishMs\":{:.3},\"parseMs\":{:.3},\"copyInMs\":{:.3},\"hashMs\":{:.3},\"copyOutMs\":{:.3},\"serializeMs\":{:.3},\
             \"layout\":{{\"requests\":{},\"columnsPerThread\":{},\"requestsPerThread\":{},\"matrixRowsPerThread\":{},\"packRequestsPerThread\":{},\
             \"segmentRows\":{},\"scanSegments\":{},\"packSegments\":{}}}}}",
            wall.iter().cloned().fold(f64::INFINITY, f64::min),
            median(&wall),
            wall.iter().cloned().fold(0.0, f64::max),
            of(|x| x.device),
            of(|x| x.scan),
            of(|x| x.pack),
            of(|x| x.finish),
            of(|x| x.parse),
            of(|x| x.copy_in),
            of(|x| x.hash),
            of(|x| x.copy_out),
            of(|x| x.serialize),
            plan.requests,
            plan.ct,
            plan.gt,
            plan.it,
            plan.pgt,
            plan.seg_rows,
            plan.scan_segs,
            plan.pack_segs,
        ));
    }
    let (used_after, _) = dev.memory().unwrap_or((0, 0));
    let (resident_bytes, call_bytes, tuning_label) = (table.resident_bytes(), table.call_bytes(), table.shape.tuning.label());
    let throughput = throughput_s.map_or(String::new(), |seconds| run_throughput(&s, table, &requests, each, &expected, seconds));
    let mid = |f: fn(&(f64, f64, f64)) -> f64| median(&write.iter().map(f).collect::<Vec<_>>());
    println!(
        "{{\"ok\":true,\"backend\":\"cuda-native\",\"device\":{:?},\"rows\":{rows},\"cols\":{cols},\"cells\":{},\"colBlocks\":{blocks},\
         \"tableBytes\":{},\"residentBytes\":{},\"callBytes\":{},\"deviceUsedBeforeBytes\":{used_before},\"deviceUsedAfterBytes\":{used_after},\
         \"deviceTotalBytes\":{total},\"rayonThreads\":{},\"tuning\":{:?},\"buildMs\":{build_ms:.2},\"initMs\":{init_ms:.2},\"uploadMs\":{upload_ms:.2},\
         \"writeBlockMs\":{:.3},\"writeBlockCopyMs\":{:.3},\"writeBlockLayoutMs\":{:.3},\"blockBuildMs\":{:.3},\"installMs\":{:.3},\
         \"runs\":{runs},\"answersEqual\":32,\"comparedWithDump\":{},\"checked\":[\"answers of a call of 32 and of 1\",\"first pass unrounded\",\
         \"packing product unrounded\",\"write_block of a changed block\",\"block written back\",\"block built on the device and installed\"],\
         {}{}\"cuda\":{{{}}}}}",
        dev.name(),
        rows * cols,
        rows * cols * 2,
        resident_bytes,
        call_bytes,
        rayon::current_num_threads(),
        tuning_label,
        mid(|x| x.0),
        mid(|x| x.1),
        mid(|x| x.2),
        ms(built.fused.wall),
        install.layout,
        expect.is_some(),
        best,
        throughput,
        calls.join(","),
    );
}

/// Calls answered per second with 1 to 16 callers, for calls of 1 and of 32: `"throughput":{...},` for the result
/// line. The host's part of a call runs in its caller; the device's part of one call at a time. Before the measure,
/// an answer made this way is compared with the CPU's.
fn run_throughput(s: &Shape, table: CudaFetch, requests: &[u8], each: usize, expected: &[Vec<u8>], seconds: f64) -> String {
    let shape = table.shape;
    let table = Mutex::new(table);
    let answer = |count: usize| -> Result<Vec<Vec<u8>>, String> {
        let call = &requests[..count * each];
        let p = shape.prepare(call, count)?;
        let (coefficients, _, _) = table.lock().unwrap().run(&p, || ())?;
        Ok(shape.answers(&coefficients, &shape.digests(call)))
    };
    for count in [1usize, 32] {
        let got = answer(count).unwrap_or_else(|e| fail(s, "measuring the calls per second", &e));
        if equal_answers(&got, &expected[..count]) != count {
            fail(s, "measuring the calls per second", "an answer differs from the CPU's");
        }
    }
    let line = throughput_json(seconds, &CALLERS, &mut |count, callers| calls_per_second(callers, seconds, &|| answer(count).map(|_| ())));
    format!("{},", line.unwrap_or_else(|e| fail(s, "measuring the calls per second", &e)))
}

/// Times the layouts of a call of 32 and of a call of 1; every one's answers are checked. Leaves the table on the
/// fastest and returns `"best":"...","sweep":{...},` for the result line.
fn run_sweep(s: &Shape, table: &mut CudaFetch, requests: &[u8], each: usize, expected: &[Vec<u8>]) -> String {
    let base = table.shape.tuning;
    // The device's time of one stage for a layout: the median of three calls after a warm-up.
    let mut time = |t: Tuning, count: usize, stage: fn(&CallTiming) -> f64| -> f64 {
        table.shape.tuning = t;
        let mut xs = Vec::new();
        for i in 0..4 {
            let (got, timing) = table.respond(&requests[..count * each], count).unwrap_or_else(|e| fail(s, "the sweep", &format!("{}: {e}", t.label())));
            if equal_answers(&got, &expected[..count]) != count {
                fail(s, "the sweep", &format!("{}: the answers differ from the CPU's", t.label()));
            }
            if i > 0 {
                xs.push(stage(&timing));
            }
        }
        median(&xs)
    };
    let targets = [65_536u32, 262_144, 1_048_576];
    let mut best = base;
    // A call of 32: the first pass, then the packing.
    let mut scan = (f64::INFINITY, base);
    for ct in [1u32, 2, 4, 8] {
        for gt in [1u32, 2, 4, 8, 16] {
            if ct * gt > 32 {
                continue;
            }
            for so in 0..3u32 {
                for target in targets {
                    let t = Tuning { ct, gt, scan_order: so, target, ..base };
                    let x = time(t, 32, |c| c.scan);
                    eprintln!("sweep scan 32: ct={ct} gt={gt} so={so} target={target}: {x:.3} ms");
                    if x < scan.0 {
                        scan = (x, t);
                    }
                }
            }
        }
    }
    let scan_target = scan.1.target;
    best.ct = scan.1.ct;
    best.gt = scan.1.gt;
    best.scan_order = scan.1.scan_order;
    let mut pack = (f64::INFINITY, base);
    for it in [1u32, 2, 4, 8] {
        for pgt in [1u32, 2, 4, 8, 16] {
            if it * pgt > 32 {
                continue;
            }
            for po in 0..3u32 {
                for pseg in [1u32, 2, 4, 8, 16] {
                    let t = Tuning { it, pgt, pack_order: po, pack_segs: pseg, ..base };
                    let x = time(t, 32, |c| c.pack);
                    eprintln!("sweep pack 32: it={it} pgt={pgt} po={po} pseg={pseg}: {x:.3} ms");
                    if x < pack.0 {
                        pack = (x, t);
                    }
                }
            }
        }
    }
    best.it = pack.1.it;
    best.pgt = pack.1.pgt;
    best.pack_order = pack.1.pack_order;
    best.pack_segs = pack.1.pack_segs;
    best.target = scan_target;
    // A call of 1: the widths of a thread that answers one request.
    let mut scan1 = (f64::INFINITY, best);
    for ct1 in [1u32, 2, 4, 8] {
        for target in targets {
            let t = Tuning { ct1, target, ..best };
            let x = time(t, 1, |c| c.scan);
            eprintln!("sweep scan 1: ct1={ct1} target={target}: {x:.3} ms");
            if x < scan1.0 {
                scan1 = (x, t);
            }
        }
    }
    let mut pack1 = (f64::INFINITY, best);
    for it1 in [1u32, 2, 4, 8] {
        let t = Tuning { it1, ..best };
        let x = time(t, 1, |c| c.pack);
        eprintln!("sweep pack 1: it1={it1}: {x:.3} ms");
        if x < pack1.0 {
            pack1 = (x, t);
        }
    }
    best.ct1 = scan1.1.ct1;
    best.it1 = pack1.1.it1;
    table.shape.tuning = best;
    format!(
        "\"best\":{:?},\"sweep\":{{\"scan32Ms\":{:.3},\"pack32Ms\":{:.3},\"scan1Ms\":{:.3},\"scan1Target\":{},\"pack1Ms\":{:.3}}},",
        best.label(),
        scan.0,
        pack.0,
        scan1.0,
        scan1.1.target,
        pack1.0
    )
}
