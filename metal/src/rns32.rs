//! An exact public dot product over three 30-bit NTT primes, for GPUs whose integer units are 32 bits wide (Apple
//! GPUs, and WGSL, which has no 64-bit integers).
//!
//! `LiftContext::public_dot` computes the exact integer sum of negacyclic products sum_k a_k * b_k (centered
//! operands), then reduces it modulo q. The crate does it over two or three 42-bit auxiliary primes. Any set of
//! primes whose product exceeds twice the bound on |sum| gives the same integer, so the same output words. Here the
//! primes are below 2^30, so lazy Shoup products keep every value in [0, 4p) within a u32 and the quotient is a
//! 32-bit `mulhi`. The capacity check is the crate's: product > 2 * d * right_bound * sum_k max|a_k|.
//!
//! The NTT is the textbook negacyclic one (Cooley-Tukey forward, natural to bit-reversed order; Gentleman-Sande
//! inverse, bit-reversed to natural), with psi^bitrev tables. The inverse's 1/d is folded into the prepared operand.

pub const D: usize = 2048;

fn pow_mod(mut b: u64, mut e: u64, m: u64) -> u64 {
    let mut r = 1u64;
    b %= m;
    while e > 0 {
        if e & 1 == 1 {
            r = (r as u128 * b as u128 % m as u128) as u64;
        }
        b = (b as u128 * b as u128 % m as u128) as u64;
        e >>= 1;
    }
    r
}

fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    // Deterministic Miller-Rabin for n < 2^64.
    let mut d = n - 1;
    let mut s = 0;
    while d % 2 == 0 {
        d /= 2;
        s += 1;
    }
    'outer: for a in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if a % n == 0 {
            continue;
        }
        let mut x = pow_mod(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 0..s - 1 {
            x = (x as u128 * x as u128 % n as u128) as u64;
            if x == n - 1 {
                continue 'outer;
            }
        }
        return false;
    }
    true
}

/// The three largest primes below 2^30 that are 1 mod 2d.
pub fn primes() -> [u32; 3] {
    let step = 2 * D as u64;
    let mut out = Vec::new();
    let mut p = ((1u64 << 30) - 1) / step * step + 1;
    while out.len() < 3 {
        if p < (1 << 30) && is_prime(p) {
            out.push(p as u32);
        }
        p -= step;
    }
    [out[0], out[1], out[2]]
}

fn bitrev(x: usize, bits: u32) -> usize {
    x.reverse_bits() >> (usize::BITS - bits)
}

#[inline(always)]
pub fn shoup32(w: u32, p: u32) -> u32 {
    (((w as u64) << 32) / p as u64) as u32
}

#[inline(always)]
pub fn mul_shoup32(x: u32, w: u32, wq: u32, p: u32) -> u32 {
    let est = ((x as u64 * wq as u64) >> 32) as u32;
    x.wrapping_mul(w).wrapping_sub(est.wrapping_mul(p))
}

#[inline(always)]
fn reduce(x: u32, m: u32) -> u32 {
    if x >= m {
        x - m
    } else {
        x
    }
}

/// One prime's tables: forward psi^bitrev(k) and quotients, inverse psi^-bitrev(k) and quotients (D words each).
pub struct Prime {
    pub p: u32,
    pub fwd: Vec<u32>,
    pub fwdq: Vec<u32>,
    pub inv: Vec<u32>,
    pub invq: Vec<u32>,
    pub d_inv: u32,
}

impl Prime {
    pub fn new(p: u32) -> Self {
        let (p64, n2) = (p as u64, 2 * D as u64);
        // A primitive 2D-th root of unity: psi^D = -1.
        let psi = (2..p64)
            .map(|g| pow_mod(g, (p64 - 1) / n2, p64))
            .find(|&r| pow_mod(r, D as u64, p64) == p64 - 1)
            .unwrap();
        let psi_inv = pow_mod(psi, p64 - 2, p64);
        let bits = D.trailing_zeros();
        let fwd: Vec<u32> = (0..D).map(|k| pow_mod(psi, bitrev(k, bits) as u64, p64) as u32).collect();
        let inv: Vec<u32> = (0..D).map(|k| pow_mod(psi_inv, bitrev(k, bits) as u64, p64) as u32).collect();
        Prime {
            p,
            fwdq: fwd.iter().map(|&w| shoup32(w, p)).collect(),
            invq: inv.iter().map(|&w| shoup32(w, p)).collect(),
            fwd,
            inv,
            d_inv: pow_mod(D as u64, p64 - 2, p64) as u32,
        }
    }

    /// Forward NTT, canonical output (lazy values in [0, 4p) inside).
    pub fn forward(&self, a: &mut [u32]) {
        let (p, two_p) = (self.p, 2 * self.p);
        let (mut m, mut t) = (1, D / 2);
        while m < D {
            for i in 0..m {
                let (w, wq) = (self.fwd[m + i], self.fwdq[m + i]);
                for j in 2 * i * t..2 * i * t + t {
                    let cx = reduce(a[j], two_p);
                    let v = mul_shoup32(a[j + t], w, wq, p);
                    a[j] = cx + v;
                    a[j + t] = cx + two_p - v;
                }
            }
            m *= 2;
            t /= 2;
        }
        for x in a.iter_mut() {
            *x = reduce(reduce(*x, two_p), p);
        }
    }

    /// Inverse NTT without the 1/D factor, canonical input and output (lazy values in [0, 2p) inside).
    pub fn inverse_unscaled(&self, a: &mut [u32]) {
        let (p, two_p) = (self.p, 2 * self.p);
        let (mut h, mut t) = (D / 2, 1);
        while h >= 1 {
            for i in 0..h {
                let (w, wq) = (self.inv[h + i], self.invq[h + i]);
                for j in 2 * i * t..2 * i * t + t {
                    let (u, v) = (a[j], a[j + t]);
                    a[j] = reduce(u + v, two_p);
                    a[j + t] = mul_shoup32(u + two_p - v, w, wq, p);
                }
            }
            h /= 2;
            t *= 2;
        }
        for x in a.iter_mut() {
            *x = reduce(*x, p);
        }
    }
}

/// Garner CRT constants for three primes, and the centered reduction modulo q = 2^k.
#[derive(Clone, Copy, Debug)]
pub struct Crt3 {
    pub p: [u32; 3],
    /// p0^-1 mod p1, (p0 p1)^-1 mod p2, p0 mod p2, and their Shoup quotients.
    pub inv01: u32,
    pub inv01q: u32,
    pub inv012: u32,
    pub inv012q: u32,
    pub p0_mod_p2: u32,
    pub p0_mod_p2q: u32,
    pub p01: u64,
    pub product: u128,
}

impl Crt3 {
    pub fn new(p: [u32; 3]) -> Self {
        let [p0, p1, p2] = p.map(|x| x as u64);
        let inv01 = pow_mod(p0 % p1, p1 - 2, p1) as u32;
        let inv012 = pow_mod(p0 * p1 % p2, p2 - 2, p2) as u32;
        let p0_mod_p2 = (p0 % p2) as u32;
        Crt3 {
            p,
            inv01,
            inv01q: shoup32(inv01, p[1]),
            inv012,
            inv012q: shoup32(inv012, p[2]),
            p0_mod_p2,
            p0_mod_p2q: shoup32(p0_mod_p2, p[2]),
            p01: p0 * p1,
            product: p0 as u128 * p1 as u128 * p2 as u128,
        }
    }

    /// The integer of magnitude below product/2 with these residues, modulo q (a power of two).
    pub fn reconstruct(&self, r: [u32; 3], q: u64) -> u64 {
        let [p0, p1, p2] = self.p;
        let t1 = reduce(mul_shoup32(reduce(r[1] + p1 - reduce(r[0], p1), p1), self.inv01, self.inv01q, p1), p1);
        // x mod p2 = r0 + p0 t1 mod p2
        let a = reduce(mul_shoup32(t1, self.p0_mod_p2, self.p0_mod_p2q, p2), p2);
        let x2 = reduce(reduce(r[0], p2) + a, p2);
        let t2 = reduce(mul_shoup32(reduce(r[2] + p2 - x2, p2), self.inv012, self.inv012q, p2), p2);
        let x = r[0] as u128 + p0 as u128 * t1 as u128 + self.p01 as u128 * t2 as u128;
        let signed = if x > self.product / 2 { x.wrapping_sub(self.product) } else { x };
        (signed as u64) & (q - 1)
    }
}

/// Prepared public operands over the three primes: NTT(a_k) * D^-1 and Shoup quotients, [prime][k][D].
pub struct Prepared32 {
    pub t: Vec<Vec<Vec<u32>>>,
    pub tq: Vec<Vec<Vec<u32>>>,
}

pub struct Rns32 {
    pub primes: Vec<Prime>,
    pub crt: Crt3,
    pub q: u64,
}

/// centered(x, q) modulo p, canonical.
pub fn lift_centered(x: u64, q: u64, p: u32) -> u32 {
    let p = p as u64;
    let r = if x > q / 2 { (p - (q - x) % p) % p } else { x % p };
    r as u32
}

fn centered_abs(x: u64, q: u64) -> u128 {
    (if x > q / 2 { q - x } else { x }) as u128
}

impl Rns32 {
    pub fn new(q: u64) -> Self {
        let p = primes();
        Rns32 { primes: p.iter().map(|&x| Prime::new(x)).collect(), crt: Crt3::new(p), q }
    }

    /// Prepares the public operands; panics unless the three primes hold the sum for every right operand with
    /// coefficients of magnitude at most `right_bound` (the crate's own capacity rule).
    pub fn prepare(&self, a: &[Vec<u64>], right_bound: u64) -> Prepared32 {
        let q = self.q;
        let maxima: u128 = a.iter().map(|poly| poly.iter().map(|&x| centered_abs(x, q)).max().unwrap()).sum();
        let need = maxima * D as u128 * right_bound as u128 * 2;
        assert!(need < self.crt.product, "three 30-bit primes lack capacity: need {need}");
        let mut t = Vec::new();
        let mut tq = Vec::new();
        for pr in &self.primes {
            let p = pr.p as u64;
            let mut tp = Vec::new();
            let mut tqp = Vec::new();
            for poly in a {
                let mut v: Vec<u32> = poly.iter().map(|&x| lift_centered(x, q, p as u32)).collect();
                pr.forward(&mut v);
                let v: Vec<u32> = v.iter().map(|&x| (x as u64 * pr.d_inv as u64 % p) as u32).collect();
                tqp.push(v.iter().map(|&w| shoup32(w, pr.p)).collect());
                tp.push(v);
            }
            t.push(tp);
            tq.push(tqp);
        }
        Prepared32 { t, tq }
    }

    /// The exact sum of negacyclic products of the prepared operands and `b` (centered coefficients), modulo q.
    pub fn dot(&self, a: &Prepared32, b: &[Vec<u64>]) -> Vec<u64> {
        let q = self.q;
        let mut res = vec![vec![0u32; D]; 3];
        for (j, pr) in self.primes.iter().enumerate() {
            let p = pr.p;
            let mut sum = vec![0u32; D];
            for (k, poly) in b.iter().enumerate() {
                let mut v: Vec<u32> = poly.iter().map(|&x| lift_centered(x, q, p)).collect();
                pr.forward(&mut v);
                for i in 0..D {
                    let prod = reduce(mul_shoup32(v[i], a.t[j][k][i], a.tq[j][k][i], p), p);
                    sum[i] = reduce(sum[i] + prod, p);
                }
            }
            pr.inverse_unscaled(&mut sum);
            res[j] = sum;
        }
        (0..D).map(|i| self.crt.reconstruct([res[0][i], res[1][i], res[2][i]], q)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use reinspiring::lift_ntt::LiftContext;

    #[test]
    fn dot_equals_lift_context() {
        let q = 1u64 << 54;
        let lift = LiftContext::new(D, q).unwrap();
        let rns = Rns32::new(q);
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(7);
        for (polys, bound) in [(2usize, 1u64 << 18), (1, 65535), (16, 65535)] {
            let a: Vec<Vec<u64>> = (0..polys).map(|_| (0..D).map(|_| rng.gen::<u64>() % q).collect()).collect();
            let b: Vec<Vec<u64>> = (0..polys)
                .map(|_| {
                    (0..D)
                        .map(|_| {
                            let v = rng.gen_range(0..=bound);
                            if rng.gen::<bool>() { (q - v) % q } else { v }
                        })
                        .collect()
                })
                .collect();
            let want = lift.public_dot(&lift.prepare_public_dot(&a, bound).unwrap(), &b).unwrap();
            let got = rns.dot(&rns.prepare(&a, bound), &b);
            assert_eq!(got, want, "polys={polys}");
        }
    }
}
