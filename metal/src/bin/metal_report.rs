//! The report's Metal figures for one size (`bench/report-run.ts` on macOS): the computing of column blocks on the
//! Apple GPU with native Metal kernels. Computing only: there is no native Metal fetch here, so nothing is installed
//! into a table that answers, and no calls are answered meanwhile.
//!
//!   metal_report <rows of the matrix> <its column blocks> <rows of the six tables, r1,...,r6> [reps]
//!
//! - `update1`: one column block of the matrix: wall time from the copy of its cells to the end of the build, and the
//!   command buffer's GPU time; median of `reps` (default 5) after a warm-up build.
//! - `build`: the matrix from scratch, its blocks four at a time (`encode_many`), cells copied for every round.
//! - `update6`: one column block of each of the six tables in one command buffer, back to back, and with the GPU left
//!   idle for 6 s before each build (`afterIdle`), as between chain blocks.
//! - `digests`: `pack_digest`'s SHA-256 over the build's output words for seeds 1 to 3, in `pack_digest --parts`'
//!   line format, for the comparison with the CPU build.
//!
//! Prints one JSON line.
use mbb::{common::*, gpu::*, gpu_digest};
use std::time::{Duration, Instant};

fn median(xs: &[f64]) -> f64 {
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n % 2 == 1 { s[n / 2] } else { (s[n / 2 - 1] + s[n / 2]) / 2.0 }
}

/// The builds of several tables in one command buffer: wall time from the first cell copy, and the GPU time.
fn at_once(builders: &[GpuBuilder], cells: &[Vec<u16>]) -> (f64, f64) {
    let t = Instant::now();
    for (b, c) in builders.iter().zip(cells) {
        b.upload(c);
    }
    let cmd = builders[0].queue().new_command_buffer();
    let refs: Vec<&GpuBuilder> = builders.iter().collect();
    encode_many(&refs, cmd);
    cmd.commit();
    cmd.wait_until_completed();
    assert_eq!(cmd.status(), metal::MTLCommandBufferStatus::Completed);
    (ms(t.elapsed()), gpu_seconds_pub(cmd) * 1e3)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        eprintln!("usage: metal_report <rows> <column blocks> <r1,...,r6> [reps]");
        std::process::exit(2);
    }
    let rows: usize = args[0].parse().unwrap();
    let blocks: usize = args[1].parse().unwrap();
    let six: Vec<usize> = args[2].split(',').map(|x| x.parse().unwrap()).collect();
    let reps: usize = args.get(3).map_or(5, |x| x.parse().unwrap());
    let (tpb, arith) = (1024, Arith::U32);
    let device = metal::Device::system_default().unwrap();

    // One column block of the matrix, and its digests.
    let s = setup(rows);
    let q = s.profile().packing().q();
    let one = GpuBuilder::new(&s, tpb, arith);
    let mut digests = Vec::new();
    for seed in 1..=3u64 {
        one.build_staged(&cells(rows, seed));
        let (all, h, pk, overflow) = gpu_digest(&one, q);
        digests.push(format!("\"{rows} {seed} {all} hints={h} packing={pk}{}\"", if overflow { " (matrix outside i32)" } else { "" }));
    }
    let c1 = cells(rows, 1);
    one.build_fused(&c1);
    let (mut wall, mut gpu) = (Vec::new(), Vec::new());
    for _ in 0..reps {
        let t = Instant::now();
        let f = one.build_fused(&c1);
        wall.push(ms(t.elapsed()));
        gpu.push(ms(f.fused.gpu));
    }
    let update1 = format!("{{\"wallMs\":{:.3},\"gpuMs\":{:.3},\"reps\":{reps}}}", median(&wall), median(&gpu));

    // The matrix from scratch: its blocks four at a time.
    let n = blocks.min(4);
    let builders: Vec<GpuBuilder> = (0..n).map(|_| GpuBuilder::new(&s, tpb, arith)).collect();
    let block_cells: Vec<Vec<u16>> = (0..n).map(|t| cells(rows, 10 + t as u64)).collect();
    at_once(&builders, &block_cells);
    let mut whole = Vec::new();
    for _ in 0..3 {
        let t = Instant::now();
        let mut done = 0;
        while done < blocks {
            let k = n.min(blocks - done);
            at_once(&builders[..k], &block_cells[..k]);
            done += k;
        }
        whole.push(ms(t.elapsed()));
    }
    drop(builders);
    let build = format!("{{\"blocks\":{blocks},\"atOnce\":{n},\"wallMs\":{:.3},\"reps\":3}}", median(&whole));

    // One chain block: a column block of each of the six tables.
    let builders: Vec<GpuBuilder> = six.iter().map(|&r| GpuBuilder::new(&setup(r), tpb, arith)).collect();
    let six_cells: Vec<Vec<u16>> = six.iter().enumerate().map(|(i, &r)| cells(r, 100 + i as u64)).collect();
    at_once(&builders, &six_cells);
    let (mut wall, mut gpu) = (Vec::new(), Vec::new());
    for _ in 0..reps.max(5) {
        let (w, g) = at_once(&builders, &six_cells);
        wall.push(w);
        gpu.push(g);
    }
    let (mut idle_wall, mut idle_gpu) = (Vec::new(), Vec::new());
    for _ in 0..reps {
        std::thread::sleep(Duration::from_millis(6000));
        let (w, g) = at_once(&builders, &six_cells);
        idle_wall.push(w);
        idle_gpu.push(g);
    }
    println!(
        "{{\"ok\":true,\"device\":{:?},\"arith\":\"{arith:?}\",\"rows\":{rows},\"digests\":[{}],\"update1\":{update1},\"build\":{build},\
         \"update6\":{{\"rows\":{six:?},\"wallMs\":{:.3},\"gpuMs\":{:.3},\"afterIdle\":{{\"idleMs\":6000,\"wallMs\":{:.3},\"gpuMs\":{:.3}}}}}}}",
        device.name(),
        digests.join(","),
        median(&wall),
        median(&gpu),
        median(&idle_wall),
        median(&idle_gpu),
    );
}
