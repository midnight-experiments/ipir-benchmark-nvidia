//! A step-by-step CPU replica of `NativePreprocessed::build_two_mask` (patched crates) built from the crates' public
//! and patch-0003 calls, timed per step. It is the profile of one build and the stage-by-stage reference the GPU build
//! is checked against.
use crate::common::ms;
use rayon::prelude::*;
use reinspiring::{
    compile::{collapse_kg_exponents, compile_fast, tau_coeffs},
    lift_ntt::{LiftContext, PreparedPublicDot},
    native::{aggregate_masks_reference, decompose, NativeParams, NativeSetup},
    native_matrix::NativeMatrix,
};
use std::time::{Duration, Instant};

/// The slot order of the collapse chains: odd exponents as left powers of 5, then their negatives. `order[s]` is
/// the aggregation row that becomes slot s.
pub fn slot_order(d: usize) -> Vec<usize> {
    let mut order = Vec::with_capacity(d);
    let mut g = 1usize;
    for _ in 0..d / 2 {
        order.push((g - 1) / 2);
        g = g * 5 % (2 * d);
    }
    g = 1;
    for _ in 0..d / 2 {
        let e = (2 * d - g) % (2 * d);
        order.push((e - 1) / 2);
        g = g * 5 % (2 * d);
    }
    order
}

#[derive(Default, Debug, Clone)]
pub struct ChainProfile {
    pub steps: usize,
    pub decompose: Duration,
    pub tau_in: Duration,
    pub dot: Duration,
    pub tau_out_add: Duration,
    pub total: Duration,
}

#[derive(Default, Debug, Clone)]
pub struct ReplicaProfile {
    pub aggregation: Duration,
    pub slots: Duration,
    /// Both chains one after another (each chain's own profile below).
    pub chains: [ChainProfile; 2],
    pub compile_fft: Duration,
    pub from_blocks: Duration,
}

pub struct ReplicaOutput {
    pub mask: Vec<u64>,
    pub other_mask: Vec<u64>,
    pub matrix: NativeMatrix,
    /// slots after aggregation and reorder, before the chains (for checking the GPU stage by stage).
    pub slots_before: Vec<Vec<u64>>,
    /// The digits of every chain step, limb-major: digits[limb][global step].
    pub digits: Vec<Vec<Vec<u64>>>,
}

fn run_chain(
    lift: &LiftContext,
    public_w: &PreparedPublicDot,
    p: &NativeParams,
    half: &mut [Vec<u64>],
    exps: &[u64],
    prof: &mut ChainProfile,
) -> Vec<Vec<Vec<u64>>> {
    let (d, q) = (p.d(), p.q());
    let start = Instant::now();
    let mut digits = Vec::with_capacity(half.len() - 1);
    for (step, idx) in (1..half.len()).rev().enumerate() {
        let t = Instant::now();
        let ds = decompose(&half[idx], p).unwrap();
        prof.decompose += t.elapsed();
        let t = Instant::now();
        let inverse = spiral_rs::number_theory::invert_uint_mod(exps[step], (2 * d) as u64).unwrap();
        let unrotated: Vec<_> = ds.iter().map(|x| tau_coeffs(x, inverse, q)).collect();
        prof.tau_in += t.elapsed();
        let t = Instant::now();
        let dot = lift.public_dot_parallel_primes_profiled(public_w, &unrotated).unwrap();
        prof.dot += t.elapsed();
        let t = Instant::now();
        let update = tau_coeffs(&dot, exps[step], q);
        for (a, x) in half[idx - 1].iter_mut().zip(update) {
            *a = (*a + x) & (q - 1);
        }
        prof.tau_out_add += t.elapsed();
        digits.push(ds);
        prof.steps += 1;
    }
    prof.total += start.elapsed();
    digits
}

/// The packing state of one column block from its hint masks, step by step.
pub fn build_two_mask_replica(setup: &NativeSetup, masks: &[Vec<u64>]) -> (ReplicaOutput, ReplicaProfile) {
    let p = setup.params();
    let (d, q) = (p.d(), p.q());
    let mut prof = ReplicaProfile::default();
    let lift = LiftContext::new(d, q).unwrap();
    let t = Instant::now();
    let mut agg = aggregate_masks_reference(masks, q).unwrap();
    prof.aggregation = t.elapsed();
    let t = Instant::now();
    let mut slots: Vec<Vec<u64>> = slot_order(d).into_iter().map(|r| std::mem::take(&mut agg[r])).collect();
    prof.slots = t.elapsed();
    let slots_before = slots.clone();
    let exponents = collapse_kg_exponents(d);
    let public_w = lift.prepare_public_dot(setup.gadget_masks(), (1u64 << (p.gadget_bits() - 1)).min(q / 2)).unwrap();
    let (left, right) = slots.split_at_mut(d / 2);
    let mut chain_digits = Vec::new();
    for (c, (half, exps)) in [(left, &exponents[..d / 2 - 1]), (right, &exponents[d / 2 - 1..])].into_iter().enumerate() {
        let mut cp = ChainProfile::default();
        chain_digits.extend(run_chain(&lift, &public_w, p, half, exps, &mut cp));
        prof.chains[c] = cp;
    }
    let mut digits: Vec<Vec<Vec<u64>>> = (0..p.ell()).map(|_| Vec::with_capacity(chain_digits.len())).collect();
    for ds in chain_digits {
        for (limb, x) in digits.iter_mut().zip(ds) {
            limb.push(x);
        }
    }
    let t = Instant::now();
    let blocks: Vec<_> = digits.par_iter().map(|ts| compile_fast(ts, &exponents, q).unwrap()).collect();
    prof.compile_fft = t.elapsed();
    let t = Instant::now();
    let matrix = NativeMatrix::from_blocks(blocks).unwrap();
    prof.from_blocks = t.elapsed();
    (
        ReplicaOutput { mask: slots[0].clone(), other_mask: slots[d / 2].clone(), matrix, slots_before, digits },
        prof,
    )
}

/// The hint masks of one column block on the CPU, as `build_with_concurrency` computes them.
pub fn hint_masks_cpu(lift: &LiftContext, public: &PreparedPublicDot, cells: &[u16], rows: usize, d: usize) -> Vec<Vec<u64>> {
    (0..d)
        .into_par_iter()
        .map(|col| {
            let coeffs: Vec<Vec<u64>> =
                cells[col * rows..(col + 1) * rows].chunks_exact(d).map(|poly| poly.iter().map(|&x| x.into()).collect()).collect();
            lift.public_dot(public, &coeffs).unwrap()
        })
        .collect()
}

impl ReplicaProfile {
    pub fn json(&self) -> String {
        let c = |c: &ChainProfile| {
            format!(
                "{{\"steps\":{},\"total_ms\":{:.2},\"decompose_ms\":{:.2},\"tau_in_ms\":{:.2},\"dot_ms\":{:.2},\"tau_out_add_ms\":{:.2}}}",
                c.steps,
                ms(c.total),
                ms(c.decompose),
                ms(c.tau_in),
                ms(c.dot),
                ms(c.tau_out_add)
            )
        };
        format!(
            "\"aggregation_ms\":{:.2},\"slots_ms\":{:.2},\"chain_left\":{},\"chain_right\":{},\"compile_fft_ms\":{:.2},\"from_blocks_ms\":{:.2}",
            ms(self.aggregation),
            ms(self.slots),
            c(&self.chains[0]),
            c(&self.chains[1]),
            ms(self.compile_fft),
            ms(self.from_blocks)
        )
    }
}
