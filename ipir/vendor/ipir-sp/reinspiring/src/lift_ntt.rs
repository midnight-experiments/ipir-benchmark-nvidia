//! Exact negacyclic products via pinned auxiliary NTT primes and signed CRT.
//!
//! Spiral supplies the NTT and pointwise multiplication. Generic products and
//! online leftovers use per-product reconstruction unless a checked bound
//! proves that the entire signed sum fits two primes. Public preprocessing
//! checks full-sum capacity with `prepare_public_dot` before accumulation.
use crate::error::ReinspiringError;
use crate::lazy_ntt::{mul_shoup_lazy, ntt_forward, ntt_inverse, reduce_once, shoup_quotient};
use rayon::prelude::*;
use spiral_rs::{
    params::Params,
    poly::{from_ntt_alloc, multiply, to_ntt_alloc, PolyMatrix, PolyMatrixNTT, PolyMatrixRaw},
};
use zeroize::{Zeroize, Zeroizing};

// Spiral matrix allocations do not erase their contents on drop. These guards
// protect decoder scratch through normal return or unwinding.
struct SecretRaw<'a, 'b>(&'b mut PolyMatrixRaw<'a>);
impl Drop for SecretRaw<'_, '_> {
    fn drop(&mut self) {
        self.0.as_mut_slice().zeroize();
    }
}
struct SecretNtt<'a, 'b>(&'b mut PolyMatrixNTT<'a>);
impl Drop for SecretNtt<'_, '_> {
    fn drop(&mut self) {
        self.0.as_mut_slice().zeroize();
    }
}

// Distinct primes, each 1 mod 8192. Their product fits i128 and exceeds
// d*q^2 for every supported degree/modulus. Keep separate Spiral Params:
// Spiral's combined `modulus` field is only u64.
const PRIMES: [u64; 3] = [4398046568449, 4398046666753, 4398046781441];

/// Validated, reusable auxiliary NTT context. No native-q NTT is required.
pub struct LiftContext {
    d: usize,
    q: u64,
    pub(crate) params: Vec<Params>,
    product: u128,
    inverses: [u64; 2],
    // Shoup quotients for division-free CRT: inverses[0] mod p1, inverses[1] mod p2, p0 mod p2.
    crt_quotients: [u64; 3],
}

/// Offline transforms of a public left operand. The prime count is selected
/// from its public coefficient bound, never from a client secret.
pub struct PreparedLiftOperand {
    pub(crate) d: usize,
    pub(crate) q: u64,
    // limb, prime, NTT coefficient; immutable and independent of request data.
    pub(crate) transforms: Vec<Vec<Vec<u64>>>,
    pub(crate) sum_fits_two: bool,
}

/// Request-local NTT transforms of key bodies or a private decoder secret.
/// Erased on drop; reusable only within the request and matching profile.
pub struct PreparedLiftRight {
    d: usize,
    q: u64,
    // limb, prime, coefficient; up to three primes, checked by sum_cached.
    transforms: Vec<Vec<Vec<u64>>>,
}
/// Cached public operands for an exact integer sum of products. Unlike
/// `PreparedLiftOperand`, capacity covers the entire sum before reconstruction.
/// The right-hand coefficient bound is checked on every evaluation. This API
/// is for public preprocessing data; validation is not constant-time.
pub struct PreparedPublicDot {
    d: usize,
    q: u64,
    right_bound: u64,
    // prime, operand, NTT coefficient
    transforms: Vec<Vec<Vec<u64>>>,
    // Shoup quotients of `transforms`, same layout.
    quotients: Vec<Vec<Vec<u64>>>,
}

impl Drop for PreparedLiftRight {
    fn drop(&mut self) {
        self.clear_secret();
    }
}

impl PreparedLiftRight {
    pub(crate) fn clear_secret(&mut self) {
        use zeroize::Zeroize;
        for poly in self.transforms.iter_mut().flatten() {
            poly.zeroize();
        }
    }
}

impl PreparedLiftOperand {
    /// Heap bytes retained by the public NTT coefficients.
    pub fn storage_bytes(&self) -> usize {
        self.transforms.iter().flatten().map(|x| x.len() * 8).sum()
    }
}

impl LiftContext {
    // Only native decoder may use this: its sampled secret has the public
    // finite support bound supplied here. No secret-dependent capacity choice.
    pub(crate) fn prepare_decode_masks(
        &self,
        masks: &[Vec<u64>],
        support: u64,
    ) -> Result<PreparedLiftOperand, ReinspiringError> {
        let transforms = self.public_dot_transforms(masks, support)?;
        if transforms.len() != 2 {
            return Err(ReinspiringError::InvalidParams(
                "decoder requires two-prime capacity".into(),
            ));
        }
        Ok(PreparedLiftOperand {
            d: self.d,
            q: self.q,
            sum_fits_two: true,
            transforms: (0..masks.len())
                .map(|i| transforms.iter().map(|p| p[i].clone()).collect())
                .collect(),
        })
    }

    pub(crate) fn prepare_decode_secret(
        &self,
        secret: &[u64],
        conjugate: bool,
    ) -> Result<PreparedLiftRight, ReinspiringError> {
        let mut out = PreparedLiftRight {
            d: self.d,
            q: self.q,
            transforms: vec![Vec::with_capacity(2)],
        };
        for params in &self.params[..2] {
            let mut raw_storage = PolyMatrixRaw::zero(params, 1, 1);
            let raw = SecretRaw(&mut raw_storage);
            for (dst, &x) in raw.0.as_mut_slice().iter_mut().zip(secret) {
                *dst = centered(x, self.q).rem_euclid(params.modulus as i128) as u64;
            }
            let mut transformed_storage = PolyMatrixNTT::zero(params, 1, 1);
            let transformed = SecretNtt(&mut transformed_storage);
            spiral_rs::poly::to_ntt(transformed.0, raw.0);
            out.transforms[0].push(transformed.0.as_slice().to_vec());
        }
        if conjugate {
            // Spiral stores evaluations in bit-reversed order. Negating an
            // odd evaluation exponent complements all index bits, which also
            // complements the bit-reversed index.
            out.transforms.push(
                out.transforms[0]
                    .iter()
                    .map(|p| p.iter().rev().copied().collect())
                    .collect(),
            );
        }
        Ok(out)
    }

    /// Transform uploaded bodies once per request, independently of database
    /// scan output. Canonical validation is performed before retaining data.
    pub fn prepare_right(&self, b: &[Vec<u64>]) -> Result<PreparedLiftRight, ReinspiringError> {
        self.prepare_right_primes(b, 3)
    }

    /// Native packing digits have a profile-wide public bound. Use it to
    /// prepare enough primes for every block, independent of the first block.
    pub(crate) fn prepare_bounded_right(
        &self,
        b: &[Vec<u64>],
        left_bound: u64,
    ) -> Result<PreparedLiftRight, ReinspiringError> {
        if left_bound > self.q / 2 {
            return Err(ReinspiringError::InvalidParams(
                "invalid public left bound".into(),
            ));
        }
        let bound = 2 * self.d as u128 * left_bound as u128 * (self.q / 2) as u128;
        let count = if bound < PRIMES[0] as u128 * PRIMES[1] as u128 {
            2
        } else {
            3
        };
        self.prepare_right_primes(b, count)
    }

    fn prepare_right_primes(
        &self,
        b: &[Vec<u64>],
        count: usize,
    ) -> Result<PreparedLiftRight, ReinspiringError> {
        if b.is_empty() {
            return Err(ReinspiringError::LweShape(
                "empty prepared right operand".into(),
            ));
        }
        let mut transforms = Vec::with_capacity(b.len());
        for poly in b {
            self.validate(poly)?;
            let mut primes = Vec::with_capacity(count);
            for params in &self.params[..count] {
                let mut raw = PolyMatrixRaw::zero(params, 1, 1);
                for (dst, &x) in raw.as_mut_slice().iter_mut().zip(poly) {
                    *dst = centered(x, self.q).rem_euclid(params.modulus as i128) as u64;
                }
                primes.push(to_ntt_alloc(&raw).as_slice().to_vec());
            }
            transforms.push(primes);
        }
        Ok(PreparedLiftRight {
            d: self.d,
            q: self.q,
            transforms,
        })
    }

    /// Exact sum with both operands transformed. Preserve the existing public
    /// per-product capacity checks: reconstruct each limb before summing modulo q.
    pub fn sum_cached(
        &self,
        a: &PreparedLiftOperand,
        b: &PreparedLiftRight,
    ) -> Result<Vec<u64>, ReinspiringError> {
        if a.d != self.d
            || b.d != self.d
            || a.q != self.q
            || b.q != self.q
            || a.transforms.len() != b.transforms.len()
            || a.transforms
                .iter()
                .zip(&b.transforms)
                .any(|(left, right)| left.len() > right.len())
        {
            return Err(ReinspiringError::LweShape("cached operand mismatch".into()));
        }
        if a.sum_fits_two {
            return self.sum_cached_two(a, b);
        }
        let mut result = vec![0u64; self.d];
        for (left_primes, right_primes) in a.transforms.iter().zip(&b.transforms) {
            let mut residues = Vec::with_capacity(left_primes.len());
            for ((params, lhs), rhs) in self.params.iter().zip(left_primes).zip(right_primes) {
                let mut left = PolyMatrixNTT::zero(params, 1, 1);
                let mut right = PolyMatrixNTT::zero(params, 1, 1);
                left.as_mut_slice().copy_from_slice(lhs);
                right.as_mut_slice().copy_from_slice(rhs);
                let mut out = PolyMatrixNTT::zero(params, 1, 1);
                multiply(&mut out, &left, &right);
                residues.push(from_ntt_alloc(&out).as_slice().to_vec());
            }
            let product: u128 = PRIMES[..residues.len()]
                .iter()
                .map(|&x| x as u128)
                .product();
            for (i, dst) in result.iter_mut().enumerate() {
                let mut x = residues[0][i] as u128;
                let mut m = PRIMES[0] as u128;
                for j in 1..residues.len() {
                    let p = PRIMES[j] as u128;
                    let delta = (residues[j][i] as u128 + p - x % p) % p;
                    x += m * (delta * self.inverses[j - 1] as u128 % p);
                    m *= p;
                }
                let signed = if x > product / 2 {
                    x as i128 - product as i128
                } else {
                    x as i128
                };
                *dst = (*dst + signed.rem_euclid(self.q as i128) as u64) % self.q;
            }
        }
        Ok(result)
    }

    // Only reachable with a checked bound for the full integer sum, unlike
    // the generic per-limb path. Two inverse NTTs per block regardless of ell.
    fn sum_cached_two(
        &self,
        a: &PreparedLiftOperand,
        b: &PreparedLiftRight,
    ) -> Result<Vec<u64>, ReinspiringError> {
        let mut residues = Zeroizing::new(vec![vec![0u64; self.d]; 2]);
        for (j, params) in self.params[..2].iter().enumerate() {
            let mut sum_storage = PolyMatrixNTT::zero(params, 1, 1);
            let sum = SecretNtt(&mut sum_storage);
            let mut left = PolyMatrixNTT::zero(params, 1, 1);
            let mut right_storage = PolyMatrixNTT::zero(params, 1, 1);
            let right = SecretNtt(&mut right_storage);
            let mut out_storage = PolyMatrixNTT::zero(params, 1, 1);
            let out = SecretNtt(&mut out_storage);
            for (lhs, rhs) in a.transforms.iter().zip(&b.transforms) {
                left.as_mut_slice().copy_from_slice(&lhs[j]);
                right.0.as_mut_slice().copy_from_slice(&rhs[j]);
                multiply(out.0, &left, right.0);
                for (dst, &x) in sum.0.as_mut_slice().iter_mut().zip(out.0.as_slice()) {
                    let value = *dst + x;
                    *dst = if value >= params.modulus {
                        value - params.modulus
                    } else {
                        value
                    };
                }
            }
            // Each context has exactly one CRT prime. Invert in our guarded
            // allocation; from_ntt_alloc would copy secrets to Spiral's TLS.
            spiral_rs::ntt::ntt_inverse(params, sum.0.as_mut_slice());
            for (i, dst) in residues[j].iter_mut().enumerate() {
                *dst = params.crt_compose(sum.0.as_slice(), i);
            }
        }
        Ok((0..self.d)
            .map(|i| self.reconstruct_two(residues[0][i], residues[1][i]))
            .collect())
    }

    // Both inputs are canonical auxiliary-prime residues, p0 < p1. The
    // caller has already proven signed reconstruction capacity for the sum.
    fn reconstruct_two(&self, r0: u64, r1: u64) -> u64 {
        let p0 = PRIMES[0] as u128;
        let p1 = PRIMES[1] as u128;
        let product = p0 * p1;
        let delta = r1 + PRIMES[1] - r0;
        let delta = if delta >= PRIMES[1] {
            delta - PRIMES[1]
        } else {
            delta
        };
        let t = reduce_once(
            mul_shoup_lazy(delta, self.inverses[0], self.crt_quotients[0], PRIMES[1]),
            PRIMES[1],
        );
        let x = r0 as u128 + p0 * t as u128;
        let signed = if x > product / 2 {
            x as i128 - product as i128
        } else {
            x as i128
        };
        if self.q.is_power_of_two() {
            signed as u64 & (self.q - 1)
        } else {
            signed.rem_euclid(self.q as i128) as u64
        }
    }
    /// Construct for d in 2..=4096 and q in 2..=2^56.
    pub fn new(d: usize, q: u64) -> Result<Self, ReinspiringError> {
        if !(2..=4096).contains(&d) || !d.is_power_of_two() || !(2..=1 << 56).contains(&q) {
            return Err(ReinspiringError::InvalidParams(
                "unsupported lifted product parameters".into(),
            ));
        }
        let product = PRIMES.iter().map(|&x| x as u128).product::<u128>();
        let need = (d as u128)
            .checked_mul(q as u128)
            .and_then(|x| x.checked_mul(q as u128))
            .ok_or_else(|| ReinspiringError::InvalidParams("lift bound overflow".into()))?;
        if product <= need {
            return Err(ReinspiringError::InvalidParams(
                "insufficient CRT reconstruction capacity".into(),
            ));
        }
        let inverses = [
            spiral_rs::number_theory::invert_uint_mod(PRIMES[0] % PRIMES[1], PRIMES[1]).unwrap(),
            spiral_rs::number_theory::invert_uint_mod(
                ((PRIMES[0] as u128 * PRIMES[1] as u128) % PRIMES[2] as u128) as u64,
                PRIMES[2],
            )
            .unwrap(),
        ];
        let params = PRIMES
            .iter()
            .map(|&p| Params::init(d, &[p], 6.4, 1, 2, 20, 1, 1, 1, 1, false, 0, 0, 1, 1, 0))
            .collect();
        let crt_quotients = [
            shoup_quotient(inverses[0], PRIMES[1]),
            shoup_quotient(inverses[1], PRIMES[2]),
            shoup_quotient(PRIMES[0], PRIMES[2]),
        ];
        Ok(Self {
            d,
            q,
            params,
            product,
            inverses,
            crt_quotients,
        })
    }

    fn validate(&self, a: &[u64]) -> Result<(), ReinspiringError> {
        if a.len() != self.d || a.iter().any(|&x| x >= self.q) {
            return Err(ReinspiringError::LweShape(
                "noncanonical lifted polynomial or wrong degree".into(),
            ));
        }
        Ok(())
    }

    /// Exact product of two canonical polynomials modulo q.
    pub fn multiply(&self, a: &[u64], b: &[u64]) -> Result<Vec<u64>, ReinspiringError> {
        self.validate(a)?;
        self.validate(b)?;
        let mut residues = Vec::with_capacity(3);
        for params in &self.params {
            let mut ar = PolyMatrixRaw::zero(params, 1, 1);
            let mut br = PolyMatrixRaw::zero(params, 1, 1);
            for i in 0..self.d {
                ar.get_poly_mut(0, 0)[i] =
                    centered(a[i], self.q).rem_euclid(params.modulus as i128) as u64;
                br.get_poly_mut(0, 0)[i] =
                    centered(b[i], self.q).rem_euclid(params.modulus as i128) as u64;
            }
            let mut out = PolyMatrixNTT::zero(params, 1, 1);
            multiply(&mut out, &to_ntt_alloc(&ar), &to_ntt_alloc(&br));
            residues.push(from_ntt_alloc(&out).get_poly(0, 0).to_vec());
        }
        Ok((0..self.d)
            .map(|i| {
                let mut x = residues[0][i] as u128;
                let mut m = PRIMES[0] as u128;
                for j in 1..3 {
                    let p = PRIMES[j] as u128;
                    let delta = (residues[j][i] as u128 + p - x % p) % p;
                    x += m * (delta * self.inverses[j - 1] as u128 % p);
                    m *= p;
                }
                let signed = if x > self.product / 2 {
                    x as i128 - self.product as i128
                } else {
                    x as i128
                };
                signed.rem_euclid(self.q as i128) as u64
            })
            .collect())
    }

    /// Cache transforms of public polynomials. For each product, signed integer
    /// coefficients are bounded by d*max_abs(left)*floor(q/2). Choose two primes
    /// only when their product strictly exceeds twice that bound. Three primes
    /// retain the generic exact-product bound. Reconstruct limbs separately.
    pub fn prepare_public(&self, a: &[Vec<u64>]) -> Result<PreparedLiftOperand, ReinspiringError> {
        if a.is_empty() {
            return Err(ReinspiringError::LweShape("empty prepared operand".into()));
        }
        let mut transforms = Vec::with_capacity(a.len());
        let mut sum_max = Some(0u128);
        for poly in a {
            self.validate(poly)?;
            let max = poly
                .iter()
                .map(|&x| centered(x, self.q).unsigned_abs())
                .max()
                .unwrap();
            sum_max = sum_max.and_then(|sum| sum.checked_add(max));
            let bound = 2 * self.d as u128 * max * (self.q / 2) as u128;
            let count = if (PRIMES[0] as u128) * (PRIMES[1] as u128) > bound {
                2
            } else {
                3
            };
            let mut primes = Vec::with_capacity(count);
            for params in &self.params[..count] {
                let mut raw = PolyMatrixRaw::zero(params, 1, 1);
                for (dst, &x) in raw.as_mut_slice().iter_mut().zip(poly) {
                    *dst = centered(x, self.q).rem_euclid(params.modulus as i128) as u64;
                }
                primes.push(to_ntt_alloc(&raw).as_slice().to_vec());
            }
            transforms.push(primes);
        }
        Ok(PreparedLiftOperand {
            d: self.d,
            q: self.q,
            transforms,
            sum_fits_two: sum_max
                .and_then(|sum| sum.checked_mul(2 * self.d as u128 * (self.q / 2) as u128))
                .is_some_and(|bound| bound < PRIMES[0] as u128 * PRIMES[1] as u128),
        })
    }

    /// Exact sum with offline public transforms and capacity-checked CRT. Uploaded
    /// bodies remain full-width; no approximate arithmetic or modulus switch.
    pub fn sum_prepared(
        &self,
        a: &PreparedLiftOperand,
        b: &[Vec<u64>],
    ) -> Result<Vec<u64>, ReinspiringError> {
        if a.d != self.d || a.q != self.q || a.transforms.len() != b.len() {
            return Err(ReinspiringError::LweShape(
                "prepared operand mismatch".into(),
            ));
        }
        let mut result = vec![0u64; self.d];
        for (primes, poly) in a.transforms.iter().zip(b) {
            self.validate(poly)?;
            let mut residues = Vec::with_capacity(primes.len());
            for (params, lhs) in self.params.iter().zip(primes) {
                let mut left = PolyMatrixNTT::zero(params, 1, 1);
                left.as_mut_slice().copy_from_slice(lhs);
                let mut right = PolyMatrixRaw::zero(params, 1, 1);
                for (dst, &x) in right.as_mut_slice().iter_mut().zip(poly) {
                    *dst = centered(x, self.q).rem_euclid(params.modulus as i128) as u64;
                }
                let mut out = PolyMatrixNTT::zero(params, 1, 1);
                multiply(&mut out, &left, &to_ntt_alloc(&right));
                residues.push(from_ntt_alloc(&out).as_slice().to_vec());
            }
            let product = PRIMES[..primes.len()]
                .iter()
                .map(|&p| p as u128)
                .product::<u128>();
            for (i, dst) in result.iter_mut().enumerate() {
                let mut x = residues[0][i] as u128;
                let mut m = PRIMES[0] as u128;
                for j in 1..primes.len() {
                    let p = PRIMES[j] as u128;
                    let delta = (residues[j][i] as u128 + p - x % p) % p;
                    x += m * (delta * self.inverses[j - 1] as u128 % p);
                    m *= p;
                }
                let signed = if x > product / 2 {
                    x as i128 - product as i128
                } else {
                    x as i128
                };
                *dst = if self.q.is_power_of_two() {
                    dst.wrapping_add(signed as u64) & (self.q - 1)
                } else {
                    (*dst + signed.rem_euclid(self.q as i128) as u64) % self.q
                };
            }
        }
        Ok(result)
    }

    /// Cache a public polynomial vector for repeated dot products. If A_i is
    /// the largest absolute centered coefficient of operand i and B is the
    /// supplied right bound, every integer output coefficient has magnitude
    /// at most d*B*sum_i(A_i). Require CRT capacity strictly greater than twice
    /// this bound, including the sum across all operands.
    pub fn prepare_public_dot(
        &self,
        a: &[Vec<u64>],
        right_bound: u64,
    ) -> Result<PreparedPublicDot, ReinspiringError> {
        let transforms = self.public_dot_transforms(a, right_bound)?;
        let quotients = transforms
            .iter()
            .zip(&self.params)
            .map(|(polys, params)| {
                polys
                    .iter()
                    .map(|poly| {
                        poly.iter()
                            .map(|&w| shoup_quotient(w, params.modulus))
                            .collect()
                    })
                    .collect()
            })
            .collect();
        Ok(PreparedPublicDot {
            d: self.d,
            q: self.q,
            right_bound,
            transforms,
            quotients,
        })
    }

    // The capacity checks and transforms of `prepare_public_dot` (prime,
    // operand, NTT coefficient), without the Shoup quotients `public_dot` uses.
    fn public_dot_transforms(
        &self,
        a: &[Vec<u64>],
        right_bound: u64,
    ) -> Result<Vec<Vec<Vec<u64>>>, ReinspiringError> {
        if a.is_empty() || right_bound > self.q / 2 {
            return Err(ReinspiringError::LweShape(
                "invalid public dot bound or shape".into(),
            ));
        }
        let mut maxima = 0u128;
        for poly in a {
            self.validate(poly)?;
            let max = poly
                .iter()
                .map(|&x| centered(x, self.q).unsigned_abs())
                .max()
                .unwrap();
            maxima = maxima.checked_add(max).ok_or_else(|| {
                ReinspiringError::InvalidParams("public dot bound overflow".into())
            })?;
        }
        let need = maxima
            .checked_mul(self.d as u128)
            .and_then(|x| x.checked_mul(right_bound as u128))
            .and_then(|x| x.checked_mul(2))
            .ok_or_else(|| ReinspiringError::InvalidParams("public dot bound overflow".into()))?;
        let count = if need < PRIMES[0] as u128 * PRIMES[1] as u128 {
            2
        } else if need < self.product {
            3
        } else {
            return Err(ReinspiringError::InvalidParams(
                "insufficient public dot CRT capacity".into(),
            ));
        };
        Ok(self.params[..count]
            .iter()
            .map(|params| {
                a.iter()
                    .map(|poly| {
                        let mut raw = PolyMatrixRaw::zero(params, 1, 1);
                        for (dst, &x) in raw.as_mut_slice().iter_mut().zip(poly) {
                            *dst = centered(x, self.q).rem_euclid(params.modulus as i128) as u64;
                        }
                        to_ntt_alloc(&raw).as_slice().to_vec()
                    })
                    .collect()
            })
            .collect())
    }

    /// Evaluate a capacity-checked public dot product. Accumulate in the
    /// auxiliary NTT rings and perform one inverse transform per prime, then
    /// reconstruct the signed integer sum before reducing modulo q.
    pub fn public_dot(
        &self,
        a: &PreparedPublicDot,
        b: &[Vec<u64>],
    ) -> Result<Vec<u64>, ReinspiringError> {
        self.dot(a, b, false)
    }

    /// `public_dot` with the auxiliary primes evaluated in parallel on the
    /// current Rayon pool, for a caller on a sequential critical path. Same words.
    pub(crate) fn public_dot_parallel_primes(
        &self,
        a: &PreparedPublicDot,
        b: &[Vec<u64>],
    ) -> Result<Vec<u64>, ReinspiringError> {
        self.dot(a, b, true)
    }

    fn dot(
        &self,
        a: &PreparedPublicDot,
        b: &[Vec<u64>],
        parallel_primes: bool,
    ) -> Result<Vec<u64>, ReinspiringError> {
        if a.d != self.d || a.q != self.q || a.transforms[0].len() != b.len() {
            return Err(ReinspiringError::LweShape(
                "public dot context or shape mismatch".into(),
            ));
        }
        let (q, half) = (self.q, self.q / 2);
        for poly in b {
            self.validate(poly)?;
            // |centered(x, q)| > right_bound, for canonical x.
            if poly
                .iter()
                .any(|&x| std::hint::select_unpredictable(x > half, q - x, x) > a.right_bound)
            {
                return Err(ReinspiringError::LweShape(
                    "public dot coefficient exceeds bound".into(),
                ));
            }
        }
        let residue = |j: usize| {
            let params = &self.params[j];
            let p = params.modulus;
            let mut operand = vec![0u64; self.d];
            let mut sum = vec![0u64; self.d];
            for ((lhs, lhs_quotients), rhs) in a.transforms[j].iter().zip(&a.quotients[j]).zip(b) {
                if a.right_bound < p {
                    // centered(x, q) + p when negative; |centered| <= right_bound < p.
                    for (dst, &x) in operand.iter_mut().zip(rhs) {
                        *dst = std::hint::select_unpredictable(x > half, p.wrapping_sub(q - x), x);
                    }
                } else {
                    for (dst, &x) in operand.iter_mut().zip(rhs) {
                        *dst = centered(x, q).rem_euclid(p as i128) as u64;
                    }
                }
                ntt_forward(params, &mut operand);
                for (((dst, &x), &w), &w_quotient) in
                    sum.iter_mut().zip(&operand).zip(lhs).zip(lhs_quotients)
                {
                    let product = reduce_once(mul_shoup_lazy(x, w, w_quotient, p), p);
                    *dst = reduce_once(*dst + product, p);
                }
            }
            ntt_inverse(params, &mut sum);
            sum
        };
        let residues: Vec<Vec<u64>> = if parallel_primes {
            (0..a.transforms.len())
                .into_par_iter()
                .map(residue)
                .collect()
        } else {
            (0..a.transforms.len()).map(residue).collect()
        };
        Ok(if residues.len() == 2 {
            (0..self.d)
                .map(|i| self.reconstruct_two(residues[0][i], residues[1][i]))
                .collect()
        } else {
            (0..self.d)
                .map(|i| self.reconstruct_three(residues[0][i], residues[1][i], residues[2][i]))
                .collect()
        })
    }

    // Canonical residues modulo all three primes. Mixed-radix (Garner) CRT
    // with Shoup products; the signed value is reduced modulo q.
    fn reconstruct_three(&self, r0: u64, r1: u64, r2: u64) -> u64 {
        let [p0, p1, p2] = PRIMES;
        // r0 < p0 < p1, so r0 is already reduced modulo p1.
        let t1 = reduce_once(
            mul_shoup_lazy(
                reduce_once(r1 + p1 - r0, p1),
                self.inverses[0],
                self.crt_quotients[0],
                p1,
            ),
            p1,
        );
        // (r0 + p0 * t1) mod p2, with r0 < p0 < p2.
        let x_mod_p2 = reduce_once(
            r0 + reduce_once(mul_shoup_lazy(t1, p0, self.crt_quotients[2], p2), p2),
            p2,
        );
        let t2 = reduce_once(
            mul_shoup_lazy(
                reduce_once(r2 + p2 - x_mod_p2, p2),
                self.inverses[1],
                self.crt_quotients[1],
                p2,
            ),
            p2,
        );
        let x = r0 as u128 + p0 as u128 * t1 as u128 + p0 as u128 * p1 as u128 * t2 as u128;
        let signed = if x > self.product / 2 {
            x as i128 - self.product as i128
        } else {
            x as i128
        };
        if self.q.is_power_of_two() {
            signed as u64 & (self.q - 1)
        } else {
            signed.rem_euclid(self.q as i128) as u64
        }
    }

    /// Sum independent exact products, reducing each one before accumulation.
    pub fn sum(&self, a: &[Vec<u64>], b: &[Vec<u64>]) -> Result<Vec<u64>, ReinspiringError> {
        if a.is_empty() || a.len() != b.len() {
            return Err(ReinspiringError::LweShape(
                "lifted limb count mismatch".into(),
            ));
        }
        let mut out = vec![0; self.d];
        for (a, b) in a.iter().zip(b) {
            for (dst, x) in out.iter_mut().zip(self.multiply(a, b)?) {
                *dst = (*dst + x) % self.q;
            }
        }
        Ok(out)
    }
}

pub(crate) fn centered(x: u64, q: u64) -> i128 {
    if x > q / 2 {
        x as i128 - q as i128
    } else {
        x as i128
    }
}

/// Exact modular product using a checked auxiliary context.
pub fn mul_mod_q(a: &[u64], b: &[u64], q: u64) -> Vec<u64> {
    LiftContext::new(a.len(), q)
        .and_then(|ctx| ctx.multiply(a, b))
        .expect("valid lifted product")
}

/// Checked convenience sum; long-lived callers should retain a LiftContext.
pub fn leftover_sum(a: &[Vec<u64>], b: &[Vec<u64>], q: u64) -> Result<Vec<u64>, ReinspiringError> {
    let d = a
        .first()
        .ok_or_else(|| ReinspiringError::LweShape("empty leftover".into()))?
        .len();
    LiftContext::new(d, q)?.sum(a, b)
}

/// Legacy single-prime entry point. Reject insufficient capacity explicitly.
/// The actual multiplication uses the validated multi-prime engine.
pub fn lifted_mul_mod_q(
    a: &[u64],
    b: &[u64],
    q: u64,
    lift_q: u64,
) -> Result<Vec<u64>, ReinspiringError> {
    let need = (a.len() as u128)
        .checked_mul(q as u128)
        .and_then(|x| x.checked_mul(q as u128));
    if need.is_none() || lift_q as u128 <= need.unwrap() {
        return Err(ReinspiringError::InvalidParams(
            "single lift modulus cannot reconstruct product".into(),
        ));
    }
    LiftContext::new(a.len(), q)?.multiply(a, b)
}

#[cfg(test)]
mod exact_fast_path_tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    // Reference: the same dot product through spiral's to_ntt, multiply and
    // from_ntt, reconstructed with u128 remainders.
    fn public_dot_reference(ctx: &LiftContext, a: &PreparedPublicDot, b: &[Vec<u64>]) -> Vec<u64> {
        let mut residues = Vec::with_capacity(a.transforms.len());
        for (params, operands) in ctx.params.iter().zip(&a.transforms) {
            let mut sum = PolyMatrixNTT::zero(params, 1, 1);
            let mut left = PolyMatrixNTT::zero(params, 1, 1);
            let mut raw = PolyMatrixRaw::zero(params, 1, 1);
            let mut product = PolyMatrixNTT::zero(params, 1, 1);
            for (lhs, rhs) in operands.iter().zip(b) {
                left.as_mut_slice().copy_from_slice(lhs);
                for (dst, &x) in raw.as_mut_slice().iter_mut().zip(rhs) {
                    let signed = centered(x, ctx.q);
                    *dst = if a.right_bound < params.modulus {
                        if signed < 0 {
                            (signed + params.modulus as i128) as u64
                        } else {
                            signed as u64
                        }
                    } else {
                        signed.rem_euclid(params.modulus as i128) as u64
                    };
                }
                multiply(&mut product, &left, &to_ntt_alloc(&raw));
                for (dst, &x) in sum.as_mut_slice().iter_mut().zip(product.as_slice()) {
                    let value = *dst + x;
                    *dst = if value >= params.modulus {
                        value - params.modulus
                    } else {
                        value
                    };
                }
            }
            residues.push(from_ntt_alloc(&sum).as_slice().to_vec());
        }
        let product: u128 = PRIMES[..residues.len()]
            .iter()
            .map(|&x| x as u128)
            .product();
        (0..ctx.d)
            .map(|i| {
                let mut x = residues[0][i] as u128;
                let mut m = PRIMES[0] as u128;
                for j in 1..residues.len() {
                    let p = PRIMES[j] as u128;
                    let delta = (residues[j][i] as u128 + p - x % p) % p;
                    x += m * (delta * ctx.inverses[j - 1] as u128 % p);
                    m *= p;
                }
                let signed = if x > product / 2 {
                    x as i128 - product as i128
                } else {
                    x as i128
                };
                signed.rem_euclid(ctx.q as i128) as u64
            })
            .collect()
    }

    #[test]
    fn reconstruction_equals_division_formula() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x0003);
        let [p0, p1, p2] = PRIMES;
        for q in [12289, 1 << 54, 1 << 56] {
            let ctx = LiftContext::new(16, q).unwrap();
            let edges = |p: u64| [0, 1, p / 2, p / 2 + 1, p - 1];
            let mut cases: Vec<[u64; 3]> = Vec::new();
            for r0 in edges(p0) {
                for r1 in edges(p1) {
                    for r2 in edges(p2) {
                        cases.push([r0, r1, r2]);
                    }
                }
            }
            cases.extend((0..200_000).map(|_| {
                [
                    rng.gen::<u64>() % p0,
                    rng.gen::<u64>() % p1,
                    rng.gen::<u64>() % p2,
                ]
            }));
            for [r0, r1, r2] in cases {
                for count in [2, 3] {
                    let residues = [r0, r1, r2];
                    let product: u128 = PRIMES[..count].iter().map(|&x| x as u128).product();
                    let mut x = r0 as u128;
                    let mut m = p0 as u128;
                    for j in 1..count {
                        let p = PRIMES[j] as u128;
                        let delta = (residues[j] as u128 + p - x % p) % p;
                        x += m * (delta * ctx.inverses[j - 1] as u128 % p);
                        m *= p;
                    }
                    let signed = if x > product / 2 {
                        x as i128 - product as i128
                    } else {
                        x as i128
                    };
                    let expected = signed.rem_euclid(q as i128) as u64;
                    let actual = if count == 2 {
                        ctx.reconstruct_two(r0, r1)
                    } else {
                        ctx.reconstruct_three(r0, r1, r2)
                    };
                    assert_eq!(
                        actual, expected,
                        "q={q} residues={residues:?} count={count}"
                    );
                }
            }
        }
    }

    #[test]
    fn public_dot_equals_reference_at_production_shapes() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x0004);
        for (d, q) in [(2048, 1u64 << 54), (64, 12289), (64, 1 << 56), (2, 1 << 54)] {
            let ctx = LiftContext::new(d, q).unwrap();
            for operands in [1, 2, 4, 16] {
                let a: Vec<Vec<u64>> = (0..operands)
                    .map(|_| (0..d).map(|_| rng.gen::<u64>() % q).collect())
                    .collect();
                // Small bounds (the packing digits, the 16-bit hints) and the
                // largest admitted bound, which selects the generic residue path.
                for bound in [0, 1, (1 << 18).min(q / 2), 65535.min(q / 2), q / 2] {
                    let Ok(cached) = ctx.prepare_public_dot(&a, bound) else {
                        continue;
                    };
                    for pattern in 0..3 {
                        let b: Vec<Vec<u64>> = (0..operands)
                            .map(|_| {
                                (0..d)
                                    .map(|i| match pattern {
                                        0 => [bound, (q - bound) % q, 0][i % 3],
                                        1 => (q - bound) % q,
                                        _ => {
                                            let v = rng.gen_range(0..=bound);
                                            if rng.gen::<bool>() {
                                                v
                                            } else {
                                                (q - v) % q
                                            }
                                        }
                                    })
                                    .collect()
                            })
                            .collect();
                        assert_eq!(
                            ctx.public_dot(&cached, &b).unwrap(),
                            public_dot_reference(&ctx, &cached, &b),
                            "d={d} q={q} operands={operands} bound={bound} pattern={pattern} primes={}",
                            cached.transforms.len()
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod secret_scratch_tests {
    use super::*;
    #[test]
    fn matrix_guards_erase_on_return_error_and_unwind() {
        let context = LiftContext::new(8, 1 << 54).unwrap();
        let params = &context.params[0];
        for exit in 0..3 {
            let mut raw = PolyMatrixRaw::zero(params, 1, 1);
            let mut ntt = PolyMatrixNTT::zero(params, 1, 1);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let raw = SecretRaw(&mut raw);
                let ntt = SecretNtt(&mut ntt);
                raw.0.as_mut_slice().fill(17);
                ntt.0.as_mut_slice().fill(23);
                match exit {
                    0 => Ok(()),
                    1 => Err(()),
                    _ => panic!("exercise secret scratch unwinding"),
                }
            }));
            assert!(raw.as_slice().iter().all(|&x| x == 0));
            assert!(ntt.as_slice().iter().all(|&x| x == 0));
            match exit {
                0 => assert_eq!(result.unwrap(), Ok(())),
                1 => assert_eq!(result.unwrap(), Err(())),
                _ => assert!(result.is_err()),
            }
        }
    }
}
