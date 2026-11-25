//! Spiral's single-prime NTT with lazy Shoup products, and the Shoup helpers
//! that the exact public dot product and its CRT reconstruction share.
use spiral_rs::params::Params;

// Shoup multiplication by a fixed w < p < 2^62 with quotient = floor(w * 2^64 / p):
// for any x < 2^64 the result is congruent to x * w and lies in [0, 2p), so the
// wrapping u64 arithmetic is exact (the bound spiral's `multiply_root` uses).
#[inline(always)]
pub(crate) fn mul_shoup_lazy(x: u64, w: u64, quotient: u64, p: u64) -> u64 {
    let estimate = ((u128::from(x) * u128::from(quotient)) >> 64) as u64;
    x.wrapping_mul(w).wrapping_sub(estimate.wrapping_mul(p))
}

pub(crate) fn shoup_quotient(w: u64, p: u64) -> u64 {
    ((u128::from(w) << 64) / u128::from(p)) as u64
}

// Branch-free: on uniformly distributed residues a compare-and-jump mispredicts
// about half the time, which costs more than the arithmetic around it.
#[inline(always)]
pub(crate) fn reduce_once(x: u64, p: u64) -> u64 {
    std::hint::select_unpredictable(x >= p, x.wrapping_sub(p), x)
}

// Spiral's single-prime `ntt_forward_alt` on its own tables, with lazy Shoup
// products in [0, 2p) instead of fully reduced ones. Every stage keeps values in
// [0, 4p) and congruent to spiral's; the final pass returns the same canonical
// residues word for word. `params` holds one prime.
pub(crate) fn ntt_forward(params: &Params, a: &mut [u64]) {
    let p = params.modulus;
    let two_p = 2 * p;
    let roots = params.get_ntt_forward_table(0);
    let quotients = params.get_ntt_forward_prime_table(0);
    let n = a.len();
    let (mut m, mut t) = (1, n / 2);
    while m < n {
        for (i, block) in a.chunks_exact_mut(2 * t).enumerate() {
            let (w, w_quotient) = (roots[m + i], quotients[m + i]);
            let (xs, ys) = block.split_at_mut(t);
            for (x, y) in xs.iter_mut().zip(ys.iter_mut()) {
                let cx = reduce_once(*x, two_p);
                let v = mul_shoup_lazy(*y, w, w_quotient, p);
                *x = cx + v;
                *y = cx + two_p - v;
            }
        }
        m *= 2;
        t /= 2;
    }
    for x in a.iter_mut() {
        *x = reduce_once(reduce_once(*x, two_p), p);
    }
}

// Spiral's single-prime `ntt_inverse_alt` (its tables fold in the 1/2 per
// stage), with lazy Shoup products. Inputs are canonical; values stay in
// [0, 2p) and congruent to spiral's, and the output is canonical.
pub(crate) fn ntt_inverse(params: &Params, a: &mut [u64]) {
    let p = params.modulus;
    let two_p = 2 * p;
    let roots = params.get_ntt_inverse_table(0);
    let quotients = params.get_ntt_inverse_prime_table(0);
    let n = a.len();
    let (mut h, mut t) = (n / 2, 1);
    while h >= 1 {
        for (i, block) in a.chunks_exact_mut(2 * t).enumerate() {
            let (w, w_quotient) = (roots[h + i], quotients[h + i]);
            let (xs, ys) = block.split_at_mut(t);
            for (x, y) in xs.iter_mut().zip(ys.iter_mut()) {
                let (vx, vy) = (*x, *y);
                let difference = two_p - vy + vx;
                let sum =
                    vx + vy - std::hint::select_unpredictable(vx << 1 >= difference, two_p, 0);
                *x = (sum + (p & (difference & 1).wrapping_neg())) >> 1;
                *y = mul_shoup_lazy(difference, w, w_quotient, p);
            }
        }
        h /= 2;
        t *= 2;
    }
    for x in a.iter_mut() {
        *x = reduce_once(reduce_once(*x, two_p), p);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lift_ntt::LiftContext;
    use rand::{Rng, SeedableRng};

    #[test]
    fn transforms_equal_spiral_word_for_word() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x0002);
        for d in [2, 8, 64, 2048, 4096] {
            let ctx = LiftContext::new(d, 1 << 54).unwrap();
            for params in &ctx.params {
                let p = params.modulus;
                for pattern in 0..8 {
                    let input: Vec<u64> = (0..d)
                        .map(|i| match pattern {
                            0 => 0,
                            1 => p - 1,
                            2 => u64::from(i == 0),
                            3 => [p - 1, 0][i % 2],
                            _ => rng.gen::<u64>() % p,
                        })
                        .collect();
                    let (mut ours, mut theirs) = (input.clone(), input.clone());
                    ntt_forward(params, &mut ours);
                    spiral_rs::ntt::ntt_forward(params, &mut theirs);
                    assert_eq!(ours, theirs, "forward d={d} p={p} pattern={pattern}");
                    let (mut ours, mut theirs) = (input.clone(), input);
                    ntt_inverse(params, &mut ours);
                    spiral_rs::ntt::ntt_inverse(params, &mut theirs);
                    assert_eq!(ours, theirs, "inverse d={d} p={p} pattern={pattern}");
                }
            }
        }
    }
}
