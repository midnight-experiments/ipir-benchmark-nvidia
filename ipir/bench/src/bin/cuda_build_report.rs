//! The report's CUDA figures for one size: the computing of column blocks on an NVIDIA GPU with native CUDA kernels
//! (`cuda_build`). The command line and the JSON line are those of the Metal harness (`metal/src/bin/metal_report.rs`).
//! Computing only: nothing is installed into a table that answers, and no calls are answered meanwhile.
//!
//!   cuda_build_report <rows of the matrix> <its column blocks> <rows of the six tables, r1,...,r6> [reps]
//!
//! - `update1`: one column block of the matrix: wall time from the copy of its cells to the device to the end of the
//!   build, and the device's own time for the build (between two CUDA events); median of `reps` (default 5) after a
//!   warm-up build.
//! - `build`: the matrix from scratch, its blocks four at a time (`launch_many`), cells copied for every round.
//! - `update6`: one column block of each of the six tables, launched together, back to back, and with the GPU left
//!   idle for 6 s before each build (`afterIdle`), as between chain blocks.
//! - `digests`: `pack_digest`'s SHA-256 over the build's output words for seeds 1 to 3, in `pack_digest --parts`'
//!   line format (without the storage suffix), for the comparison with the CPU build.
//!
//! Every wall time is the host's and includes the copy of the cells from host memory to the device.
//!
//! Prints one JSON line. Built only with `--features cuda-13` (or `cuda-12`); at run time it needs the CUDA driver
//! library and NVRTC.
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
use std::time::{Duration, Instant};

fn median(xs: &[f64]) -> f64 {
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n % 2 == 1 { s[n / 2] } else { (s[n / 2 - 1] + s[n / 2]) / 2.0 }
}

/// The builds of several tables launched together: wall time from the first cell copy, and the device's time.
fn at_once(builders: &mut [CudaBuilder], cells: &[Vec<u16>]) -> Result<(f64, f64), String> {
    let t = Instant::now();
    for (b, c) in builders.iter_mut().zip(cells) {
        b.upload(c)?;
    }
    let refs: Vec<&CudaBuilder> = builders.iter().collect();
    let gpu = launch_many(&refs)?;
    Ok((ms(t.elapsed()), gpu))
}

fn run(rows: usize, blocks: usize, six: &[usize], reps: usize) -> Result<String, String> {
    let device = CudaDevice::new(0)?;

    // One column block of the matrix, and its digests.
    let s = setup(rows);
    let mut one = CudaBuilder::with_device(&device, &s)?;
    let mut digests = Vec::new();
    for seed in 1..=3u64 {
        one.build_staged(&cells(rows, seed))?;
        let (all, h, pk) = digest(&one)?;
        digests.push(format!("\"{rows} {seed} {all} hints={h} packing={pk}\""));
    }
    let c1 = cells(rows, 1);
    one.build_fused(&c1)?;
    let (mut wall, mut gpu) = (Vec::new(), Vec::new());
    for _ in 0..reps {
        let t = Instant::now();
        let f = one.build_fused(&c1)?;
        wall.push(ms(t.elapsed()));
        gpu.push(ms(f.fused.gpu));
    }
    let update1 = format!("{{\"wallMs\":{:.3},\"gpuMs\":{:.3},\"reps\":{reps}}}", median(&wall), median(&gpu));

    // The matrix from scratch: its blocks four at a time.
    let n = blocks.min(4);
    let mut builders = (0..n).map(|_| CudaBuilder::with_device(&device, &s)).collect::<Result<Vec<_>, _>>()?;
    let block_cells: Vec<Vec<u16>> = (0..n).map(|t| cells(rows, 10 + t as u64)).collect();
    at_once(&mut builders, &block_cells)?;
    let mut whole = Vec::new();
    for _ in 0..3 {
        let t = Instant::now();
        let mut done = 0;
        while done < blocks {
            let k = n.min(blocks - done);
            at_once(&mut builders[..k], &block_cells[..k])?;
            done += k;
        }
        whole.push(ms(t.elapsed()));
    }
    drop(builders);
    let build = format!("{{\"blocks\":{blocks},\"atOnce\":{n},\"wallMs\":{:.3},\"reps\":3}}", median(&whole));

    // One chain block: a column block of each of the six tables.
    let mut builders =
        six.iter().map(|&r| CudaBuilder::with_device(&device, &setup(r))).collect::<Result<Vec<_>, _>>()?;
    let six_cells: Vec<Vec<u16>> = six.iter().enumerate().map(|(i, &r)| cells(r, 100 + i as u64)).collect();
    at_once(&mut builders, &six_cells)?;
    let (mut wall, mut gpu) = (Vec::new(), Vec::new());
    for _ in 0..reps.max(5) {
        let (w, g) = at_once(&mut builders, &six_cells)?;
        wall.push(w);
        gpu.push(g);
    }
    let (mut idle_wall, mut idle_gpu) = (Vec::new(), Vec::new());
    for _ in 0..reps {
        std::thread::sleep(Duration::from_millis(6000));
        let (w, g) = at_once(&mut builders, &six_cells)?;
        idle_wall.push(w);
        idle_gpu.push(g);
    }
    Ok(format!(
        "{{\"ok\":true,\"device\":{:?},\"arith\":\"U32\",\"rows\":{rows},\"digests\":[{}],\"update1\":{update1},\"build\":{build},\
         \"update6\":{{\"rows\":{six:?},\"wallMs\":{:.3},\"gpuMs\":{:.3},\"afterIdle\":{{\"idleMs\":6000,\"wallMs\":{:.3},\"gpuMs\":{:.3}}}}}}}",
        device.name(),
        digests.join(","),
        median(&wall),
        median(&gpu),
        median(&idle_wall),
        median(&idle_gpu),
    ))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        eprintln!("usage: cuda_build_report <rows> <column blocks> <r1,...,r6> [reps]");
        std::process::exit(2);
    }
    let rows: usize = args[0].parse().unwrap();
    let blocks: usize = args[1].parse().unwrap();
    let six: Vec<usize> = args[2].split(',').map(|x| x.parse().unwrap()).collect();
    let reps: usize = args.get(3).map_or(5, |x| x.parse().unwrap());
    match run(rows, blocks, &six, reps) {
        Ok(line) => println!("{line}"),
        Err(e) => {
            println!("{{\"ok\":false,\"backend\":\"cuda\",\"rows\":{rows},\"error\":{e:?}}}");
            std::process::exit(1);
        }
    }
}
