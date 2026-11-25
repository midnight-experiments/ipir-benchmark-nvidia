//! The host side of the GPU block build (`gpu/block-build.ts`): the constant tables of one table shape, and the CPU's
//! outputs of one block's build, stage by stage, to check the GPU's against.
//!
//! The GPU build computes the same output words as `NativePublicSetup`'s CPU build of one column block (hint masks,
//! both public masks, the compiled matrix). Its exact dot products run over three 30-bit primes (`rns32`) instead of
//! the crates' 42-bit ones: the exact integer is the same, so the words are the same. The tables here come from the
//! crates' own data through the read-only accessors of patch 0003.
use crate::rns32::Rns32;
use ipir_sp::native::NativePublicSetup;
use rayon::prelude::*;
use reinspiring::{
    compile::{collapse_kg_exponents, compile_fast, tau_coeffs},
    lift_ntt::LiftContext,
    native::{aggregate_masks_reference, decompose, NativeSetup},
    native_matrix::NativeMatrix,
};
use std::io::Write;

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

/// g^-1 modulo 2d (a power of two; g odd): g^(d - 1), since the unit group has order d.
fn invert_mod_2d(g: u64, d: usize) -> u64 {
    let m = 2 * d as u64;
    let (mut r, mut b, mut e) = (1u64, g % m, d as u64 - 1);
    while e > 0 {
        if e & 1 == 1 {
            r = r * b % m;
        }
        b = b * b % m;
        e >>= 1;
    }
    assert_eq!(r * (g % m) % m, 1);
    r
}

/// The tables of one table shape: a JSON header and named sections of u32 words.
pub struct GpuPrep {
    pub header: String,
    pub sections: Vec<(&'static str, Vec<u32>)>,
}

impl GpuPrep {
    pub fn new(setup: &NativePublicSetup) -> Self {
        let p = setup.profile();
        let packing = p.packing();
        let (d, q) = (packing.d(), packing.q());
        let rows = p.rows();
        assert_eq!(d, crate::rns32::D, "the kernels are written for d = 2048");
        assert_eq!(p.cols(), d, "one column block");
        assert!(q.is_power_of_two() && q.trailing_zeros() > 32 && q.trailing_zeros() < 64);
        assert!(p.is_two_mask(), "two-mask output");
        let ps = setup.packing_setup();
        let np = ps.params();
        assert_eq!(np.ell(), 2, "two gadget limbs");
        let (bits, dropped) = (np.gadget_bits(), np.dropped_bits());
        assert!((1..32).contains(&bits) && (1..32).contains(&dropped));
        assert_eq!(dropped + 2 * bits, q.trailing_zeros(), "the digits cover q");
        // The compile runs in i32: every value of its transform is at most d * 2^(bits-1) in magnitude.
        assert!((d as u64) << (bits - 1) < 1 << 31, "compile values exceed i32");

        let rns = Rns32::new(q);
        let flat = |x: &[Vec<Vec<u32>>]| x.iter().flatten().flatten().copied().collect::<Vec<u32>>();
        let mut tables = Vec::with_capacity(3 * 4 * d);
        for pr in &rns.primes {
            for t in [&pr.fwd, &pr.fwdq, &pr.inv, &pr.invq] {
                tables.extend_from_slice(t);
            }
        }
        // Hints: the query masks, as `build_with_concurrency` prepares them. Chains: the gadget masks, as
        // `build_internal_screened` prepares them. `prepare` checks the three primes' capacity for each.
        let hp = rns.prepare(setup.query_masks(), (packing.p() - 1).min(q / 2));
        let wp = rns.prepare(ps.gadget_masks(), (1u64 << (bits - 1)).min(q / 2));

        let exponents = collapse_kg_exponents(d);
        assert_eq!(exponents.len(), d - 2);
        let bitrev = |x: usize| x.reverse_bits() >> (usize::BITS - d.trailing_zeros());
        let mut rows_seen = vec![false; d];
        let phat_rows: Vec<u32> = exponents
            .iter()
            .map(|&g| {
                let r = ((g - 1) / 2) as usize;
                assert!(!rows_seen[r], "distinct exponents: the compile adds none");
                rows_seen[r] = true;
                bitrev(r) as u32
            })
            .collect();
        let inv_exps: Vec<u32> = exponents.iter().map(|&g| invert_mod_2d(g, d) as u32).collect();
        let exps: Vec<u32> = exponents.iter().map(|&g| g as u32).collect();
        let mut slot_of_row = vec![0u32; d];
        for (s, r) in slot_order(d).into_iter().enumerate() {
            slot_of_row[r] = s as u32;
        }
        let missing_rows: Vec<u32> = (0..d).filter(|&r| !rows_seen[r]).map(|r| bitrev(r) as u32).collect();
        // Aggregation stages exact in i64: the crate's own public bound len * (q - 1) <= i64::MAX. The kernels have
        // one widening stage and the last stage in 128 bits after them.
        let mut i64_stages = 0u32;
        while (2u128 << i64_stages) * (q as u128 - 1) <= i64::MAX as u128 && (2usize << i64_stages) <= d {
            i64_stages += 1;
        }
        assert_eq!(i64_stages + 2, d.trailing_zeros(), "the kernels expect all but two stages in i64");

        let c = rns.crt;
        let half = c.product / 2;
        let w = |x: u128| format!("[{},{},{}]", x as u32, (x >> 32) as u32, (x >> 64) as u32);
        assert!(c.product >> 96 == 0);
        let header = format!(
            concat!(
                "{{\"format\":\"ipir-gpu-build-prep/1\",\"d\":{},\"logd\":{},\"rows\":{},\"polys\":{},\"qBits\":{},",
                "\"gadgetBits\":{},\"droppedBits\":{},\"steps\":{},\"i64Stages\":{},\"missingRows\":{},",
                "\"primes\":[{},{},{}],\"inv01\":{},\"inv01q\":{},\"inv012\":{},\"inv012q\":{},\"p0ModP2\":{},",
                "\"p0ModP2q\":{},\"p01\":[{},{}],\"product\":{},\"half\":{}}}"
            ),
            d,
            d.trailing_zeros(),
            rows,
            rows / d,
            q.trailing_zeros(),
            bits,
            dropped,
            d / 2 - 1,
            i64_stages,
            missing_rows.len(),
            c.p[0],
            c.p[1],
            c.p[2],
            c.inv01,
            c.inv01q,
            c.inv012,
            c.inv012q,
            c.p0_mod_p2,
            c.p0_mod_p2q,
            c.p01 as u32,
            (c.p01 >> 32) as u32,
            w(c.product),
            w(half),
        );
        GpuPrep {
            header,
            sections: vec![
                ("tables", tables),
                ("hintT", flat(&hp.t)),
                ("hintQ", flat(&hp.tq)),
                ("gadgetT", flat(&wp.t)),
                ("gadgetQ", flat(&wp.tq)),
                ("exps", exps),
                ("invExps", inv_exps),
                ("phatRows", phat_rows),
                ("slotOfRow", slot_of_row),
                ("missingRows", missing_rows),
            ],
        }
    }

    /// One file: the magic `IPIRGBP1`, the length of the JSON header, the header (with the sections' names and
    /// lengths in words added) padded with spaces to a multiple of 4 bytes, then the sections, little-endian u32.
    pub fn write(&self, path: &str) -> std::io::Result<()> {
        let sections: Vec<String> =
            self.sections.iter().map(|(n, v)| format!("{{\"name\":\"{n}\",\"words\":{}}}", v.len())).collect();
        let mut header = format!("{},\"sections\":[{}]}}", &self.header[..self.header.len() - 1], sections.join(","));
        while header.len() % 4 != 0 {
            header.push(' ');
        }
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        f.write_all(b"IPIRGBP1")?;
        f.write_all(&(header.len() as u32).to_le_bytes())?;
        f.write_all(header.as_bytes())?;
        for (_, v) in &self.sections {
            f.write_all(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
        }
        f.flush()
    }
}

/// The CPU's outputs of one block's build, stage by stage (the crates' public calls and patch 0003's).
pub struct Reference {
    /// The hint mask of each column, d words each.
    pub hint_masks: Vec<Vec<u64>>,
    /// The slots after aggregation and reorder, before the chains.
    pub slots_before: Vec<Vec<u64>>,
    /// The slots after the chains: slot 0 is the mask, slot d/2 the other mask.
    pub slots_after: Vec<Vec<u64>>,
    /// The compile input: the centered gadget digits of every chain step, [limb][row][d], row = bitrev((g - 1) / 2).
    pub digits: Vec<i32>,
    /// The compiled matrix's canonical words, d rows of 2d, row-major.
    pub matrix: Vec<u64>,
    pub storage_label: &'static str,
}

/// The hint masks of one column block, as `build_with_concurrency` computes them.
pub fn hint_masks(setup: &NativePublicSetup, cells: &[u16]) -> Vec<Vec<u64>> {
    let p = setup.profile();
    let (d, q, rows) = (p.packing().d(), p.packing().q(), p.rows());
    let lift = LiftContext::new(d, q).unwrap();
    let public = lift.prepare_public_dot(setup.query_masks(), (p.packing().p() - 1).min(q / 2)).unwrap();
    (0..d)
        .into_par_iter()
        .map(|col| {
            let coeffs: Vec<Vec<u64>> =
                cells[col * rows..(col + 1) * rows].chunks_exact(d).map(|poly| poly.iter().map(|&x| x.into()).collect()).collect();
            lift.public_dot(&public, &coeffs).unwrap()
        })
        .collect()
}

/// A step-by-step replica of `NativePreprocessed::build_two_mask`. `chain_steps`: how many steps of each chain to
/// run (all of them: d/2 - 1), for checking a GPU chain kernel after its first steps.
pub fn reference(setup: &NativePublicSetup, cells: &[u16], chain_steps: usize) -> Reference {
    let ps: &NativeSetup = setup.packing_setup();
    let p = ps.params();
    let (d, q) = (p.d(), p.q());
    let masks = hint_masks(setup, cells);
    let lift = LiftContext::new(d, q).unwrap();
    let mut agg = aggregate_masks_reference(&masks, q).unwrap();
    let mut slots: Vec<Vec<u64>> = slot_order(d).into_iter().map(|r| std::mem::take(&mut agg[r])).collect();
    let slots_before = slots.clone();
    let exponents = collapse_kg_exponents(d);
    let public_w = lift.prepare_public_dot(ps.gadget_masks(), (1u64 << (p.gadget_bits() - 1)).min(q / 2)).unwrap();
    let bitrev = |x: usize| x.reverse_bits() >> (usize::BITS - d.trailing_zeros());
    let center = |x: u64| if x > q / 2 { -((q - x) as i64) as i32 } else { x as i32 };
    let mut digits_i32 = vec![0i32; 2 * d * d];
    let mut all_digits: Vec<Vec<Vec<u64>>> = vec![Vec::new(), Vec::new()];
    let (left, right) = slots.split_at_mut(d / 2);
    for (half, exps) in [(left, &exponents[..d / 2 - 1]), (right, &exponents[d / 2 - 1..])] {
        for (step, idx) in (1..half.len()).rev().enumerate() {
            if step >= chain_steps {
                break;
            }
            let ds = decompose(&half[idx], p).unwrap();
            let inverse = invert_mod_2d(exps[step], d);
            let unrotated: Vec<_> = ds.iter().map(|x| tau_coeffs(x, inverse, q)).collect();
            let dot = lift.public_dot(&public_w, &unrotated).unwrap();
            let update = tau_coeffs(&dot, exps[step], q);
            for (a, x) in half[idx - 1].iter_mut().zip(update) {
                *a = (*a + x) & (q - 1);
            }
            let row = bitrev(((exps[step] - 1) / 2) as usize);
            for (limb, x) in ds.iter().enumerate() {
                for (i, &v) in x.iter().enumerate() {
                    digits_i32[(limb * d + row) * d + i] = center(v);
                }
            }
            for (limb, x) in all_digits.iter_mut().zip(ds) {
                limb.push(x);
            }
        }
    }
    let (matrix, storage_label) = if chain_steps >= d / 2 - 1 {
        let blocks: Vec<_> = all_digits.par_iter().map(|ts| compile_fast(ts, &exponents, q).unwrap()).collect();
        let m = NativeMatrix::from_blocks(blocks).unwrap();
        (m.to_compiled().data, m.storage().label())
    } else {
        (Vec::new(), "")
    };
    Reference { hint_masks: masks, slots_before, slots_after: slots, digits: digits_i32, matrix, storage_label }
}
