//! Equality check of the GPU build: `pack_digest`'s SHA-256 over every output word, computed from the GPU's output.
//!
//!   gpu_digest [--stages] [--corrupt] [--tpb N] [--arith u32|u32emu|wgsl|u64] [rows,...] [seeds,...]   (default: 2048,8192,16384,32768 and 1,2,3)
//!
//! Prints `rows seed digest hints=.. packing=..`, the same line format as `pack_digest --parts`, so the two outputs
//! compare with `diff`. If the matrix had an entry outside i32, the label would be wrong for an i64 matrix and the
//! line says so. `--stages` also compares each stage's output with the CPU replica (`cpu_ref`) and names the first
//! that differs.
use mbb::{common::*, cpu_ref::*, gpu::{Arith, GpuBuilder}, gpu_digest};
use reinspiring::lift_ntt::LiftContext;

fn list(s: &str) -> Vec<u64> {
    s.split(',').map(|x| x.trim().parse().expect("a number")).collect()
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let stages = args.iter().any(|a| a == "--stages");
    let corrupt = args.iter().any(|a| a == "--corrupt");
    args.retain(|a| a != "--stages" && a != "--corrupt");
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
    let all_rows = list(args.first().map_or("2048,8192,16384,32768", String::as_str));
    let seeds = list(args.get(1).map_or("1,2,3", String::as_str));
    for &rows in &all_rows {
        let rows = rows as usize;
        let setup = setup(rows);
        let p = setup.profile();
        let (d, q) = (p.packing().d(), p.packing().q());
        let g = GpuBuilder::new(&setup, tpb, arith);
        for &seed in &seeds {
            let cells = cells(rows, seed);
            g.build_staged(&cells);
            let (all, h, pk, overflow) = gpu_digest(&g, q);
            println!("{rows} {seed} {all} hints={h} packing={pk}{}", if overflow { " (matrix outside i32)" } else { "" });
            if corrupt {
                // Negative control: +1 on one matrix word, then +1 on one hint word, each must change the digest.
                unsafe { *(g.matrix.contents() as *mut i32).add(12345) += 1 };
                let (a2, _, p2, _) = gpu_digest(&g, q);
                unsafe { *(g.matrix.contents() as *mut i32).add(12345) -= 1 };
                unsafe { *(g.masks.contents() as *mut u64).add(777) += 1 };
                let (a3, h3, _, _) = gpu_digest(&g, q);
                unsafe { *(g.masks.contents() as *mut u64).add(777) -= 1 };
                let (a4, _, _, _) = gpu_digest(&g, q);
                eprintln!(
                    "  control: matrix +1 {}, hint +1 {}, restored {}",
                    if a2 != all && p2 != pk { "changes the digest" } else { "DOES NOT CHANGE IT" },
                    if a3 != all && h3 != h { "changes the digest" } else { "DOES NOT CHANGE IT" },
                    if a4 == all { "equal again" } else { "DIFFERS" }
                );
            }
            if stages {
                let lift = LiftContext::new(d, q).unwrap();
                let public = lift.prepare_public_dot(setup.query_masks(), (p.packing().p() - 1).min(q / 2)).unwrap();
                let masks = hint_masks_cpu(&lift, &public, &cells, rows, d);
                let gm = g.hint_masks();
                let bad = (0..d).filter(|&c| gm[c] != masks[c]).count();
                eprintln!("  hints: {bad} of {d} columns differ");
                // Stage by stage from the CPU's hint masks: aggregation + reorder, then chains, then compile.
                let (out, _) = build_two_mask_replica(setup.packing_setup(), &masks);
                let slots = g.slot_rows();
                eprintln!(
                    "  after chains: mask {} other mask {}",
                    if slots[0] == out.mask { "equal" } else { "DIFFERS" },
                    if slots[d / 2] == out.other_mask { "equal" } else { "DIFFERS" }
                );
                let cpu = out.matrix.to_compiled();
                let (m, _) = g.matrix_i32();
                let diff = m.iter().zip(&cpu.data).filter(|(&x, &y)| (x as i128).rem_euclid(q as i128) as u64 != y).count();
                eprintln!("  matrix: {diff} of {} words differ (storage {})", cpu.data.len(), out.matrix.storage().label());
            }
        }
    }
}
