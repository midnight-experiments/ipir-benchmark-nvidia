// The per-block build of an IPIR+SP table on the GPU: hint columns, mask aggregation, the two collapse chains and
// matrix compilation, with the same output words as the CPU build of the patched crates (reinspiring with patches
// 0001 and 0002).
//
// The host prepends `#define TPB <threads per threadgroup>` and `#define D <ring degree>` / `#define LOGD`.
// Arithmetic:
// - Auxiliary NTT primes p < 2^43: Shoup products with a precomputed quotient per fixed multiplier, values kept in
//   [0, 2p) or [0, 4p) as reinspiring's `lazy_ntt.rs` does, on spiral-rs's own root tables.
// - q = 2^54: reductions are masks.
// - Mask aggregation: exact integers in two's complement 128 bits (two ulongs).
#include <metal_stdlib>
using namespace metal;

#define PER (D / TPB)

// ---------------------------------------------------------------------------------------------------------------
// Scalar helpers

inline ulong mul_shoup_lazy(ulong x, ulong w, ulong wq, ulong p) {
    ulong estimate = mulhi(x, wq);
    return x * w - estimate * p;
}

inline ulong reduce_once(ulong x, ulong p) {
    return x >= p ? x - p : x;
}

struct u128 {
    ulong lo;
    ulong hi;
};

inline u128 u128_add(u128 a, u128 b) {
    u128 r;
    r.lo = a.lo + b.lo;
    r.hi = a.hi + b.hi + (r.lo < a.lo ? 1ul : 0ul);
    return r;
}

inline u128 u128_sub(u128 a, u128 b) {
    u128 r;
    r.lo = a.lo - b.lo;
    r.hi = a.hi - b.hi - (a.lo < b.lo ? 1ul : 0ul);
    return r;
}

inline u128 u128_neg(u128 a) {
    u128 z = {0ul, 0ul};
    return u128_sub(z, a);
}

inline u128 mul_wide(ulong a, ulong b) {
    u128 r;
    r.lo = a * b;
    r.hi = mulhi(a, b);
    return r;
}

inline bool u128_gt(u128 a, u128 b) {
    return a.hi > b.hi || (a.hi == b.hi && a.lo > b.lo);
}

// Constants of the CRT reconstruction (`LiftContext`), filled by the host.
struct CrtConst {
    ulong p[3];
    ulong inv[2];       // p0^-1 mod p1, (p0 p1)^-1 mod p2
    ulong cq[3];        // Shoup quotients of inv[0] mod p1, inv[1] mod p2, p0 mod p2
    ulong p01_lo, p01_hi;     // p0 * p1
    ulong half2_lo, half2_hi; // floor(p0 p1 / 2)
    ulong prod2_lo;           // low word of p0 p1
    ulong half3_lo, half3_hi; // floor(p0 p1 p2 / 2)
    ulong prod3_lo;           // low word of p0 p1 p2
    ulong qmask;              // q - 1 (q a power of two)
};

// x < product is the residue of a signed integer of magnitude below product / 2: the integer modulo q.
inline ulong center_mod_q(u128 x, ulong half_lo, ulong half_hi, ulong prod_lo, ulong qmask) {
    u128 bound = {half_lo, half_hi};
    ulong lo = u128_gt(x, bound) ? x.lo - prod_lo : x.lo;
    return lo & qmask;
}

inline ulong reconstruct_two(ulong r0, ulong r1, constant CrtConst& c) {
    ulong p1 = c.p[1];
    ulong delta = r1 + p1 - r0;
    delta = delta >= p1 ? delta - p1 : delta;
    ulong t = reduce_once(mul_shoup_lazy(delta, c.inv[0], c.cq[0], p1), p1);
    u128 x = u128_add(mul_wide(c.p[0], t), u128{r0, 0ul});
    return center_mod_q(x, c.half2_lo, c.half2_hi, c.prod2_lo, c.qmask);
}

inline ulong reconstruct_three(ulong r0, ulong r1, ulong r2, constant CrtConst& c) {
    ulong p0 = c.p[0], p1 = c.p[1], p2 = c.p[2];
    ulong t1 = reduce_once(mul_shoup_lazy(reduce_once(r1 + p1 - r0, p1), c.inv[0], c.cq[0], p1), p1);
    ulong x_mod_p2 = reduce_once(r0 + reduce_once(mul_shoup_lazy(t1, p0, c.cq[2], p2), p2), p2);
    ulong t2 = reduce_once(mul_shoup_lazy(reduce_once(r2 + p2 - x_mod_p2, p2), c.inv[1], c.cq[1], p2), p2);
    // x = r0 + p0 t1 + (p0 p1) t2, below 2^127.
    u128 x = u128_add(u128{r0, 0ul}, mul_wide(p0, t1));
    u128 pp_t2 = mul_wide(c.p01_lo, t2);
    pp_t2.hi += c.p01_hi * t2;
    x = u128_add(x, pp_t2);
    return center_mod_q(x, c.half3_lo, c.half3_hi, c.prod3_lo, c.qmask);
}

// ---------------------------------------------------------------------------------------------------------------
// NTTs of length D in threadgroup memory: spiral's single-prime `ntt_forward_alt` / `ntt_inverse_alt` with lazy
// Shoup products (reinspiring's lazy_ntt.rs), the D/2 butterflies of each stage spread over TPB threads.
// `tab` holds, for one prime: forward roots, their quotients, inverse roots, their quotients (D words each).

#define TAB_FWD 0
#define TAB_FWDQ (D)
#define TAB_INV (2 * D)
#define TAB_INVQ (3 * D)

// Forward transforms of `count` buffers at once (count 1 or 2), output canonical.
inline void ntt_forward_n(threadgroup ulong* a0, threadgroup ulong* a1, uint count, device const ulong* tab, ulong p,
                          uint tid) {
    ulong two_p = 2 * p;
    uint t = D / 2;
    uint lt = LOGD - 1;
    for (uint m = 1; m < D; m <<= 1) {
        for (uint b = tid; b < D / 2; b += TPB) {
            uint i = b >> lt;
            uint jj = b & (t - 1);
            uint ix = (i << (lt + 1)) + jj;
            uint iy = ix + t;
            ulong w = tab[TAB_FWD + m + i];
            ulong wq = tab[TAB_FWDQ + m + i];
            {
                ulong cx = reduce_once(a0[ix], two_p);
                ulong v = mul_shoup_lazy(a0[iy], w, wq, p);
                a0[ix] = cx + v;
                a0[iy] = cx + two_p - v;
            }
            if (count > 1) {
                ulong cx = reduce_once(a1[ix], two_p);
                ulong v = mul_shoup_lazy(a1[iy], w, wq, p);
                a1[ix] = cx + v;
                a1[iy] = cx + two_p - v;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        t >>= 1;
        lt -= 1;
    }
    for (uint k = 0; k < PER; k++) {
        uint i = tid + k * TPB;
        a0[i] = reduce_once(reduce_once(a0[i], two_p), p);
        if (count > 1) {
            a1[i] = reduce_once(reduce_once(a1[i], two_p), p);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void ntt_inverse(threadgroup ulong* a, device const ulong* tab, ulong p, uint tid) {
    ulong two_p = 2 * p;
    uint t = 1;
    uint lt = 0;
    for (uint h = D / 2; h >= 1; h >>= 1) {
        for (uint b = tid; b < D / 2; b += TPB) {
            uint i = b >> lt;
            uint jj = b & (t - 1);
            uint ix = (i << (lt + 1)) + jj;
            uint iy = ix + t;
            ulong w = tab[TAB_INV + h + i];
            ulong wq = tab[TAB_INVQ + h + i];
            ulong vx = a[ix];
            ulong vy = a[iy];
            ulong difference = two_p - vy + vx;
            ulong sum = vx + vy - ((vx << 1) >= difference ? two_p : 0ul);
            a[ix] = (sum + (p & (0ul - (difference & 1ul)))) >> 1;
            a[iy] = mul_shoup_lazy(difference, w, wq, p);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        t <<= 1;
        lt += 1;
    }
    for (uint k = 0; k < PER; k++) {
        uint i = tid + k * TPB;
        a[i] = reduce_once(reduce_once(a[i], two_p), p);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ---------------------------------------------------------------------------------------------------------------
// 1. Hint columns: for column `col`, mask[col] = sum_k public_k * cells_k (exact integer, CRT over the auxiliary
// primes, reduced mod q). One threadgroup per column; the primes one after another.

struct HintConst {
    uint rows;
    uint polys;   // rows / D
    uint primes;  // 2 or 3
    uint col0;    // first column of this dispatch
};

kernel void hint_columns(device const ushort* cells [[buffer(0)]],   // column-major, col * rows + row
                         device const ulong* mt [[buffer(1)]],       // [prime][poly][D] transforms of the masks
                         device const ulong* mq [[buffer(2)]],       // their Shoup quotients
                         device const ulong* tables [[buffer(3)]],   // [prime][4][D]
                         device ulong* out [[buffer(4)]],            // [col][D]
                         constant HintConst& h [[buffer(5)]],
                         constant CrtConst& c [[buffer(6)]],
                         uint tg [[threadgroup_position_in_grid]],
                         uint tid [[thread_position_in_threadgroup]]) {
    threadgroup ulong buf[D];
    uint col = h.col0 + tg;
    ulong res[3][PER];
    for (uint j = 0; j < h.primes; j++) {
        ulong p = c.p[j];
        device const ulong* tab = tables + j * 4 * D;
        ulong acc[PER];
        for (uint k = 0; k < PER; k++) acc[k] = 0;
        for (uint poly = 0; poly < h.polys; poly++) {
            device const ushort* src = cells + (ulong)col * h.rows + poly * D;
            for (uint k = 0; k < PER; k++) {
                uint i = tid + k * TPB;
                buf[i] = (ulong)src[i];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            ntt_forward_n(buf, buf, 1, tab, p, tid);
            device const ulong* w = mt + ((ulong)j * h.polys + poly) * D;
            device const ulong* wq = mq + ((ulong)j * h.polys + poly) * D;
            for (uint k = 0; k < PER; k++) {
                uint i = tid + k * TPB;
                ulong product = reduce_once(mul_shoup_lazy(buf[i], w[i], wq[i], p), p);
                acc[k] = reduce_once(acc[k] + product, p);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint k = 0; k < PER; k++) buf[tid + k * TPB] = acc[k];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ntt_inverse(buf, tab, p, tid);
        for (uint k = 0; k < PER; k++) res[j][k] = buf[tid + k * TPB];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device ulong* dst = out + (ulong)col * D;
    for (uint k = 0; k < PER; k++) {
        uint i = tid + k * TPB;
        dst[i] = h.primes == 2 ? reconstruct_two(res[0][k], res[1][k], c)
                               : reconstruct_three(res[0][k], res[1][k], res[2][k], c);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// 2. Mask aggregation: the d x d integer ring FFT of `aggregate_masks`, exact in 128 bits.

inline uint bitrev(uint x) {
    return reverse_bits(x) >> (32 - LOGD);
}

// polys[k][(j + k) mod D] = +-(k == 0 ? m_j[0] : -m_j[D - k]), stored at row bitrev(k), as i64.
kernel void agg_gather(device const ulong* masks [[buffer(0)]],  // [col][D]
                       device long* a [[buffer(1)]],             // [D][D]
                       uint2 gid [[thread_position_in_grid]]) {   // (m, k)
    uint m = gid.x, k = gid.y;
    uint j = (m - k) & (D - 1);
    long c = k == 0 ? (long)masks[(ulong)j * D] : -(long)masks[(ulong)j * D + (D - k)];
    a[(ulong)bitrev(k) * D + m] = (j + k >= D) ? -c : c;
}

// The butterfly of one length-`len` stage at coefficient i of pair `pair`: rows u = block + kk, v = u + len/2,
// rot = 2 (D/len) kk; u' = u + X^rot v, v' = u - X^rot v (negacyclic). Returns the source index of v's term and
// whether it is negated.
inline void stage_rows(uint len, uint pair, uint i, thread uint& ru, thread uint& rv, thread uint& vi,
                       thread bool& negate) {
    uint half_len = len >> 1;
    uint block = (pair / half_len) * len;
    uint kk = pair % half_len;
    ru = block + kk;
    rv = ru + half_len;
    uint rot = 2 * (D / len) * kk;
    negate = i < rot;
    vi = negate ? D - rot + i : i - rot;
}

// Stages whose public bound len * (q - 1) fits i64 (len <= 512 for q = 2^54), exact in i64 as the CPU does.
kernel void agg_stage64(device const long* src [[buffer(0)]],
                        device long* dst [[buffer(1)]],
                        constant uint& len [[buffer(2)]],
                        uint2 gid [[thread_position_in_grid]]) {   // (i, pair)
    uint ru, rv, vi;
    bool negate;
    stage_rows(len, gid.y, gid.x, ru, rv, vi, negate);
    long u = src[(ulong)ru * D + gid.x];
    long t = src[(ulong)rv * D + vi];
    t = negate ? -t : t;
    dst[(ulong)ru * D + gid.x] = u + t;
    dst[(ulong)rv * D + gid.x] = u - t;
}

// The first stage past the i64 bound: reads i64, writes i128.
kernel void agg_stage_widen(device const long* src [[buffer(0)]],
                            device ulong2* dst [[buffer(1)]],
                            constant uint& len [[buffer(2)]],
                            uint2 gid [[thread_position_in_grid]]) {
    uint ru, rv, vi;
    bool negate;
    stage_rows(len, gid.y, gid.x, ru, rv, vi, negate);
    long u = src[(ulong)ru * D + gid.x];
    long v = src[(ulong)rv * D + vi];
    u128 uw = {(ulong)u, u < 0 ? ~0ul : 0ul};
    u128 t = {(ulong)v, v < 0 ? ~0ul : 0ul};
    if (negate) t = u128_neg(t);
    u128 nu = u128_add(uw, t), nv = u128_sub(uw, t);
    dst[(ulong)ru * D + gid.x] = ulong2(nu.lo, nu.hi);
    dst[(ulong)rv * D + gid.x] = ulong2(nv.lo, nv.hi);
}

inline ulong agg_final(u128 x, ulong qmask) {
    return ((x.lo >> LOGD) | (x.hi << (64 - LOGD))) & qmask;
}

// A stage in i128 (len < D).
kernel void agg_stage128(device const ulong2* src [[buffer(0)]],
                         device ulong2* dst [[buffer(1)]],
                         constant uint& len [[buffer(2)]],
                         uint2 gid [[thread_position_in_grid]]) {
    uint ru, rv, vi;
    bool negate;
    stage_rows(len, gid.y, gid.x, ru, rv, vi, negate);
    ulong2 uu = src[(ulong)ru * D + gid.x];
    ulong2 vv = src[(ulong)rv * D + vi];
    u128 u = {uu.x, uu.y}, t = {vv.x, vv.y};
    if (negate) t = u128_neg(t);
    u128 nu = u128_add(u, t), nv = u128_sub(u, t);
    dst[(ulong)ru * D + gid.x] = ulong2(nu.lo, nu.hi);
    dst[(ulong)rv * D + gid.x] = ulong2(nv.lo, nv.hi);
}

// The last stage (len = D) in i128, with the final shift and mask, written straight to each row's slot.
kernel void agg_stage_last(device const ulong2* src [[buffer(0)]],
                           device ulong* slots [[buffer(1)]],
                           device const uint* slot_of_row [[buffer(2)]],
                           constant ulong& qmask [[buffer(3)]],
                           uint2 gid [[thread_position_in_grid]]) {
    uint ru, rv, vi;
    bool negate;
    stage_rows(D, gid.y, gid.x, ru, rv, vi, negate);
    ulong2 uu = src[(ulong)ru * D + gid.x];
    ulong2 vv = src[(ulong)rv * D + vi];
    u128 u = {uu.x, uu.y}, t = {vv.x, vv.y};
    if (negate) t = u128_neg(t);
    slots[(ulong)slot_of_row[ru] * D + gid.x] = agg_final(u128_add(u, t), qmask);
    slots[(ulong)slot_of_row[rv] * D + gid.x] = agg_final(u128_sub(u, t), qmask);
}

// ---------------------------------------------------------------------------------------------------------------
// 3. The two collapse chains, one threadgroup each, all steps in one dispatch. Step s of chain c works on the
// half's slot idx = D/2 - 1 - s: its gadget digits (decompose), their automorphism by g^-1, the exact dot product
// with the prepared gadget masks w over the auxiliary primes, the automorphism g of the result, added into slot
// idx - 1. The digits go straight into the compile input (row bitrev((g - 1) / 2) of each limb).

struct TraceConst {
    uint primes;        // primes of w's prepared transforms (2 or 3)
    uint bits;          // gadget bits
    uint dropped;       // dropped bits
    uint steps;         // D/2 - 1
    ulong qmask;
    ulong half_q;       // q / 2
};

kernel void trace_chains(device ulong* slots [[buffer(0)]],          // [D][D]
                         device int* phat [[buffer(1)]],             // [2][D][D]
                         device const ulong* wt [[buffer(2)]],       // [prime][limb][D]
                         device const ulong* wq [[buffer(3)]],
                         device const ulong* tables [[buffer(4)]],
                         device const uint* exps [[buffer(5)]],      // [2 * steps]
                         device const uint* inv_exps [[buffer(6)]],
                         device const uint* phat_rows [[buffer(7)]],
                         constant TraceConst& tc [[buffer(8)]],
                         constant CrtConst& c [[buffer(9)]],
                         uint chain [[threadgroup_position_in_grid]],
                         uint tid [[thread_position_in_threadgroup]]) {
    threadgroup ulong buf0[D];
    threadgroup ulong buf1[D];
    uint base = chain * (D / 2);
    long z = 1l << tc.bits;
    long rounding = tc.dropped == 0 ? 0 : (1l << (tc.dropped - 1));
    for (uint step = 0; step < tc.steps; step++) {
        uint idx = D / 2 - 1 - step;
        uint gstep = chain * tc.steps + step;
        uint g = exps[gstep];
        uint ginv = inv_exps[gstep];
        uint prow = phat_rows[gstep];
        device const ulong* src = slots + (ulong)(base + idx) * D;
        device ulong* dst = slots + (ulong)(base + idx - 1) * D;
        long dig[2][PER];
        for (uint k = 0; k < PER; k++) {
            uint i = tid + k * TPB;
            long x = ((long)src[i] + rounding) >> tc.dropped;
            for (uint l = 0; l < 2; l++) {
                long digit = ((x + z / 2) & (z - 1)) - z / 2;
                dig[l][k] = digit;
                phat[((ulong)l * D + prow) * D + i] = (int)digit;
                x = (x - digit) >> tc.bits;
            }
        }
        ulong res[3][PER];
        for (uint j = 0; j < tc.primes; j++) {
            ulong p = c.p[j];
            device const ulong* tab = tables + j * 4 * D;
            for (uint k = 0; k < PER; k++) {
                uint i = tid + k * TPB;
                uint e = (i * ginv) & (2 * D - 1);
                uint pos = e & (D - 1);
                bool neg = e >= D;
                long v0 = neg ? -dig[0][k] : dig[0][k];
                long v1 = neg ? -dig[1][k] : dig[1][k];
                buf0[pos] = v0 < 0 ? (ulong)((long)p + v0) : (ulong)v0;
                buf1[pos] = v1 < 0 ? (ulong)((long)p + v1) : (ulong)v1;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            ntt_forward_n(buf0, buf1, 2, tab, p, tid);
            device const ulong* w0 = wt + ((ulong)j * 2 + 0) * D;
            device const ulong* w1 = wt + ((ulong)j * 2 + 1) * D;
            device const ulong* q0 = wq + ((ulong)j * 2 + 0) * D;
            device const ulong* q1 = wq + ((ulong)j * 2 + 1) * D;
            for (uint k = 0; k < PER; k++) {
                uint i = tid + k * TPB;
                ulong s = reduce_once(mul_shoup_lazy(buf0[i], w0[i], q0[i], p), p);
                ulong product = reduce_once(mul_shoup_lazy(buf1[i], w1[i], q1[i], p), p);
                buf0[i] = reduce_once(s + product, p);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            ntt_inverse(buf0, tab, p, tid);
            for (uint k = 0; k < PER; k++) res[j][k] = buf0[tid + k * TPB];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint k = 0; k < PER; k++) {
            uint i = tid + k * TPB;
            ulong val = tc.primes == 2 ? reconstruct_two(res[0][k], res[1][k], c)
                                       : reconstruct_three(res[0][k], res[1][k], res[2][k], c);
            uint e = (i * g) & (2 * D - 1);
            uint pos = e & (D - 1);
            ulong upd = e >= D ? ((0ul - val) & tc.qmask) : val;
            dst[pos] = (dst[pos] + upd) & tc.qmask;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// 4. Compilation: per limb, the ring FFT at powers of X^2 over the generating polynomial (rows already in
// bit-reversed order), then M[r][j] = +-a[j][(r - j) mod D], limbs side by side. The digits are below 2^(bits-1) in
// magnitude and the D-point transform grows them by at most D, so for this profile (2^18 * 2^11 = 2^29) every
// value is an exact integer in i32: the same integer the CPU holds modulo q, and the centered word it stores. The
// host checks the bound.

kernel void cmp_stage(device const int* src [[buffer(0)]],   // [limb][D][D]
                      device int* dst [[buffer(1)]],
                      constant uint& len [[buffer(2)]],
                      uint3 gid [[thread_position_in_grid]]) {  // (i, pair, limb)
    ulong off = (ulong)gid.z * D * D;
    uint ru, rv, vi;
    bool negate;
    stage_rows(len, gid.y, gid.x, ru, rv, vi, negate);
    int u = src[off + (ulong)ru * D + gid.x];
    int t = src[off + (ulong)rv * D + vi];
    t = negate ? -t : t;
    dst[off + (ulong)ru * D + gid.x] = u + t;
    dst[off + (ulong)rv * D + gid.x] = u - t;
}

// out[r][limb * D + j] = (r < j ? -x : x), x = a_limb[j][(r - j) mod D]. A 32 x 32 tile goes through threadgroup
// memory so reads (along a row of a) and writes (along a row of out) are both contiguous.
kernel void cmp_matrix(device const int* a [[buffer(0)]],     // [limb][D][D]
                       device int* out [[buffer(1)]],         // [D][2D]
                       uint3 tgp [[threadgroup_position_in_grid]],     // (j tile, r tile, limb)
                       uint3 lid [[thread_position_in_threadgroup]]) {  // 32 x 8
    threadgroup int tile[32][33];
    uint j0 = tgp.x * 32, r0 = tgp.y * 32;
    ulong off = (ulong)tgp.z * D * D;
    for (uint s = 0; s < 4; s++) {
        uint jl = lid.y + 8 * s;
        uint j = j0 + jl, r = r0 + lid.x;
        int x = a[off + (ulong)j * D + ((r - j) & (D - 1))];
        tile[jl][lid.x] = r < j ? -x : x;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 0; s < 4; s++) {
        uint rl = lid.y + 8 * s;
        out[(ulong)(r0 + rl) * (2 * D) + tgp.z * D + j0 + lid.x] = tile[lid.x][rl];
    }
}

// Rows of the compile input that no chain step writes (exponents the chains do not produce), set to zero.
kernel void zero_rows(device int* phat [[buffer(0)]],            // [limb][D][D]
                      device const uint* rows [[buffer(1)]],
                      uint3 gid [[thread_position_in_grid]]) {   // (i, missing row, limb)
    phat[((ulong)gid.z * D + rows[gid.y]) * D + gid.x] = 0;
}
