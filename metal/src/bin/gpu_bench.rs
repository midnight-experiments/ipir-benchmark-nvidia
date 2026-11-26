//! Timing of the GPU build of one column block (2,048 columns).
//!
//!   gpu_bench <rows,...> <reps> <tables at once,...> [--tpb N] [--arith u32|u32emu|wgsl|u64 (default u32)]
//!   gpu_bench --set today|10m <reps> [--tpb N] [--arith u32|u32emu|wgsl|u64 (default u32)]
//!
//! First form, per row count:
//! - `staged`: each stage in its own command buffer, its GPU time (Metal's GPUStartTime..GPUEndTime) and wall time;
//! - `fused`: the whole build in one command buffer;
//! - for each "at once" N > 1: N tables of that shape, each with its own buffers, in one command buffer (`encode_many`:
//!   one concurrent encoder, level by level across the tables); the wall time until all are done.
//! Second form: one table per row count of the benchmark's six-table set (`bench/budget.ts`), all at once: the GPU
//! side of one chain block's compile.
//! Setups, kernels and cells are made before the clock starts; one warm-up build per builder is not reported. One
//! JSON line per repetition.
use mbb::{common::*, gpu::*};
use std::time::Instant;

fn st(s: &StageTime) -> String {
    format!("{{\"gpu_ms\":{:.3},\"wall_ms\":{:.3}}}", ms(s.gpu), ms(s.wall))
}

/// N tables' builds in one command buffer (`encode_many`): wall time from the first cell copy to completion, the
/// cell copies, and the command buffer's GPU time.
fn at_once(builders: &[GpuBuilder], cells: &[Vec<u16>]) -> (f64, f64, Vec<f64>) {
    let t = Instant::now();
    let mut upload = 0.0;
    for (b, c) in builders.iter().zip(cells) {
        upload += ms(b.upload(c));
    }
    let cmd = builders[0].queue().new_command_buffer();
    let refs: Vec<&GpuBuilder> = builders.iter().collect();
    encode_many(&refs, cmd);
    cmd.commit();
    cmd.wait_until_completed();
    assert_eq!(cmd.status(), metal::MTLCommandBufferStatus::Completed);
    let wall = ms(t.elapsed());
    (wall, upload, vec![gpu_seconds_pub(cmd) * 1e3])
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut tpb = 1024;
    let mut arith = Arith::U32;
    if let Some(i) = args.iter().position(|a| a == "--arith") {
        arith = match args[i + 1].as_str() {
            "u64" => Arith::U64,
            "u32" => Arith::U32,
            "u32emu" => Arith::U32EmulatedMulhi,
            "wgsl" => Arith::WgslProxy,
            s => panic!("unknown arithmetic {s}"),
        };
        args.drain(i..i + 2);
    }
    if let Some(i) = args.iter().position(|a| a == "--tpb") {
        tpb = args[i + 1].parse().unwrap();
        args.drain(i..i + 2);
    }
    let device = metal::Device::system_default().unwrap();
    eprintln!("device: {}", device.name());
    if args[0] == "--set" {
        let rows: Vec<usize> = match args[1].as_str() {
            "today" => vec![2048, 2048, 2048, 6144, 2048, 2048],
            "10m" => vec![14336, 16384, 8192, 6144, 2048, 2048],
            s => panic!("unknown set {s}"),
        };
        let reps: usize = args[2].parse().unwrap();
        let builders: Vec<GpuBuilder> = rows.iter().map(|&r| GpuBuilder::new(&setup(r), tpb, arith)).collect();
        let cells: Vec<Vec<u16>> = rows.iter().enumerate().map(|(i, &r)| cells(r, 100 + i as u64)).collect();
        at_once(&builders, &cells);
        for rep in 0..reps {
            let (wall, upload, gpu) = at_once(&builders, &cells);
            println!(
                "{{\"set\":\"{}\",\"rows\":{:?},\"rep\":{rep},\"tpb\":{tpb},\"arith\":\"{arith:?}\",\"wall_ms\":{wall:.3},\"upload_cells_ms\":{upload:.3},\"gpu_ms\":{gpu:.3?}}}",
                args[1], rows
            );
        }
        return;
    }
    let all_rows: Vec<usize> = args[0].split(',').map(|x| x.parse().unwrap()).collect();
    let reps: usize = args[1].parse().unwrap();
    let all_at_once: Vec<usize> = args[2].split(',').map(|x| x.parse().unwrap()).collect();
    for &rows in &all_rows {
        let setup = setup(rows);
        let max = *all_at_once.iter().max().unwrap();
        let builders: Vec<GpuBuilder> = (0..max).map(|_| GpuBuilder::new(&setup, tpb, arith)).collect();
        let cells: Vec<Vec<u16>> = (0..max).map(|t| cells(rows, 1 + t as u64)).collect();
        for b in &builders {
            b.build_fused(&cells[0]);
        }
        for &n in &all_at_once {
            for rep in 0..reps {
                if n == 1 {
                    let s = builders[0].build_staged(&cells[0]);
                    let f = builders[0].build_fused(&cells[0]);
                    println!(
                        "{{\"rows\":{rows},\"tables_at_once\":1,\"rep\":{rep},\"tpb\":{tpb},\"arith\":\"{arith:?}\",\"upload_cells_ms\":{:.3},\"hints\":{},\"aggregation\":{},\"chains\":{},\"compile\":{},\"fused\":{}}}",
                        ms(s.upload_cells),
                        st(&s.hints),
                        st(&s.aggregation),
                        st(&s.chains),
                        st(&s.compile),
                        st(&f.fused)
                    );
                } else {
                    let (wall, upload, gpu) = at_once(&builders[..n], &cells[..n]);
                    println!(
                        "{{\"rows\":{rows},\"tables_at_once\":{n},\"rep\":{rep},\"tpb\":{tpb},\"arith\":\"{arith:?}\",\"wall_ms\":{wall:.3},\"upload_cells_ms\":{upload:.3},\"gpu_ms\":{gpu:.3?}}}"
                    );
                }
            }
        }
    }
}
