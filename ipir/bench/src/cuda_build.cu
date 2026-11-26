// The per-block build of an IPIR+SP table on an NVIDIA GPU: hint columns, mask aggregation, the two collapse chains
// and matrix compilation, with the same output words as the CPU build of the patched crates. A port of the native
// Metal kernels (metal/shaders/build.metal and build32.metal, the `Arith::U32` variant): the exact public dot
// products run over three 30-bit NTT primes (bench/src/rns32.rs) with native 32-bit Shoup products.
//
// The host (bench/src/cuda_build.rs) prepends `#define D <ring degree>`, `#define LOGD` and `#define WARP_SYNC 0|1`,
// and compiles this with NVRTC.
// Arithmetic:
// - NTT primes p < 2^30: lazy Shoup products with a precomputed quotient per fixed multiplier, values in [0, 2p) or
//   [0, 4p) within a u32.
// - q = 2^54: reductions are masks.
// - Mask aggregation: exact integers, in i64 while the public bound allows, then in two's complement 128 bits.
//
// Metal's threadgroup memory is `__shared__` memory here (at most 4 * D * 4 = 32 KB per block, below the 48 KB a
// block has without opting in to more), its thread groups of D / 2 = 1,024 threads are blocks of 1,024 threads, and
// its threadgroup barrier is `__syncthreads()`. Where the Metal kernels use a SIMD-group barrier (NTT stages whose
// butterflies stay inside the 64 elements one group of 32 threads owns), this uses `__syncthreads()` as well, or
// `__syncwarp()` with WARP_SYNC 1: a warp is 32 consecutive threads, the same ownership.

typedef unsigned short u16;
typedef unsigned int u32;
typedef int i32;
typedef unsigned long long u64;
typedef long long i64;

#if D != (1 << LOGD)
#error "D must be 2^LOGD"
#endif

// ---------------------------------------------------------------------------------------------------------------
// Scalar helpers

struct u128 {
    u64 lo;
    u64 hi;
};

__device__ __forceinline__ u128 u128_add(u128 a, u128 b) {
    u128 r;
    r.lo = a.lo + b.lo;
    r.hi = a.hi + b.hi + (r.lo < a.lo ? 1ull : 0ull);
    return r;
}

__device__ __forceinline__ u128 u128_sub(u128 a, u128 b) {
    u128 r;
    r.lo = a.lo - b.lo;
    r.hi = a.hi - b.hi - (a.lo < b.lo ? 1ull : 0ull);
    return r;
}

__device__ __forceinline__ u128 u128_neg(u128 a) {
    u128 z = {0ull, 0ull};
    return u128_sub(z, a);
}

__device__ __forceinline__ u128 mul_wide(u64 a, u64 b) {
    u128 r;
    r.lo = a * b;
    r.hi = __umul64hi(a, b);
    return r;
}

__device__ __forceinline__ bool u128_gt(u128 a, u128 b) {
    return a.hi > b.hi || (a.hi == b.hi && a.lo > b.lo);
}

// Garner CRT constants of the three primes (`rns32::Crt3`), filled by the host. The same layout as the host's
// `Crt32` (repr(C)): nine u32, four bytes of padding, six u64.
struct Crt32 {
    u32 p[3];
    u32 inv01, inv01q, inv012, inv012q, p0_mod_p2, p0_mod_p2q;
    u64 p01;
    u64 prod_lo, prod_hi;   // p0 p1 p2
    u64 half_lo, half_hi;   // floor(p0 p1 p2 / 2)
    u64 qmask;              // q - 1 (q a power of two)
};

__device__ __forceinline__ u32 mul_shoup32(u32 x, u32 w, u32 wq, u32 p) {
    return x * w - __umulhi(x, wq) * p;
}

__device__ __forceinline__ u32 red32(u32 x, u32 m) {
    return x >= m ? x - m : x;
}

// The integer of magnitude below p0 p1 p2 / 2 with residues r0, r1, r2, modulo q.
__device__ __forceinline__ u64 crt3(u32 r0, u32 r1, u32 r2, const Crt32& c) {
    u32 p0 = c.p[0], p1 = c.p[1], p2 = c.p[2];
    u32 t1 = red32(mul_shoup32(red32(r1 + p1 - red32(r0, p1), p1), c.inv01, c.inv01q, p1), p1);
    u32 a = red32(mul_shoup32(t1, c.p0_mod_p2, c.p0_mod_p2q, p2), p2);
    u32 x2 = red32(red32(r0, p2) + a, p2);
    u32 t2 = red32(mul_shoup32(red32(r2 + p2 - x2, p2), c.inv012, c.inv012q, p2), p2);
    // x = r0 + p0 t1 + p01 t2 < 2^90
    u128 x = {(u64)r0 + (u64)p0 * (u64)t1, 0ull};
    x = u128_add(x, mul_wide(c.p01, (u64)t2));
    u128 bound = {c.half_lo, c.half_hi};
    u64 lo = u128_gt(x, bound) ? x.lo - c.prod_lo : x.lo;
    return lo & c.qmask;
}

// ---------------------------------------------------------------------------------------------------------------
// NTTs of length D in shared memory, one butterfly per thread and stage (D / 2 threads). Tables per prime: fwd,
// fwdq, inv, invq (D words each).
//
// Ownership: in a forward NTT's last stage thread b writes elements 2b and 2b + 1, and an inverse NTT's first stage
// reads them; its last stage writes b and b + D/2. Kernels keep those elements per thread between transforms, so the
// pointwise products and the CRT need no barrier. Stages whose butterflies stay inside one 64-element chunk
// (half-size t <= 32, preceded by a stage that also did) only need the 32 threads that own the chunk, one warp, to
// be synchronised.

#define T32_FWD 0
#define T32_FWDQ (D)
#define T32_INV (2 * D)
#define T32_INVQ (3 * D)

__device__ __forceinline__ void fwd_bfly(u32* a, u32 b, u32 lt, const u32* tab, u32 p) {
    u32 t = 1u << lt;
    u32 i = b >> lt;
    u32 ix = (i << (lt + 1)) + (b & (t - 1));
    u32 iy = ix + t;
    u32 m = (D / 2) >> lt;            // stage with half-size t has m = D / (2t) blocks
    u32 w = tab[T32_FWD + m + i], wq = tab[T32_FWDQ + m + i];
    u32 two_p = 2 * p;
    u32 cx = red32(a[ix], two_p);
    u32 v = mul_shoup32(a[iy], w, wq, p);
    a[ix] = cx + v;
    a[iy] = cx + two_p - v;
}

__device__ __forceinline__ void inv_bfly(u32* a, u32 b, u32 lt, const u32* tab, u32 p) {
    u32 t = 1u << lt;
    u32 i = b >> lt;
    u32 ix = (i << (lt + 1)) + (b & (t - 1));
    u32 iy = ix + t;
    u32 h = (D / 2) >> lt;
    u32 w = tab[T32_INV + h + i], wq = tab[T32_INVQ + h + i];
    u32 two_p = 2 * p;
    u32 u = a[ix], v = a[iy];
    a[ix] = red32(u + v, two_p);
    a[iy] = mul_shoup32(u + two_p - v, w, wq, p);
}

#if WARP_SYNC
#define CHUNK_SYNC() __syncwarp()
#else
#define CHUNK_SYNC() __syncthreads()
#endif

// Barrier before a forward stage of half-size 2^lt (the previous stage had 2^(lt+1)).
__device__ __forceinline__ void fwd_sync(u32 lt) {
    if (lt + 1 <= 5) CHUNK_SYNC();
    else __syncthreads();
}
// Barrier before an inverse stage of half-size 2^lt (the previous stage had 2^(lt-1)).
__device__ __forceinline__ void inv_sync(u32 lt) {
    if (lt <= 5) CHUNK_SYNC();
    else __syncthreads();
}

// ---------------------------------------------------------------------------------------------------------------
// 1. Hint columns: for column `col`, mask[col] = sum_k public_k * cells_k (exact integer, CRT over the three
// primes, reduced mod q). One block of D / 2 threads per column, the three primes side by side in three buffers.

extern "C" __global__ void hint_columns32(const u16* cells,     // column-major, col * rows + row
                                          const u32* mt,        // [prime][poly][D] transforms of the masks / D
                                          const u32* mq,        // their Shoup quotients
                                          const u32* tables,    // [prime][4][D]
                                          u64* out,             // [col][D]
                                          u32 rows,
                                          u32 polys,            // rows / D
                                          Crt32 c) {
    __shared__ u32 buf[3][D];
    u32 col = blockIdx.x;
    u32 tid = threadIdx.x;
    u32 acc[3][2] = {{0, 0}, {0, 0}, {0, 0}};
    u32 e0 = 2 * tid, e1 = 2 * tid + 1;
    for (u32 poly = 0; poly < polys; poly++) {
        const u16* src = cells + (u64)col * rows + (u64)poly * D;
        u32 x0 = src[tid], x1 = src[tid + D / 2];
        __syncthreads();   // the previous products have read their elements
        for (u32 j = 0; j < 3; j++) {
            buf[j][tid] = x0;
            buf[j][tid + D / 2] = x1;
        }
        for (int lt = LOGD - 1; lt >= 0; lt--) {
            if (lt == LOGD - 1) __syncthreads();
            else fwd_sync(lt);
            for (u32 j = 0; j < 3; j++) fwd_bfly(buf[j], tid, lt, tables + j * 4 * D, c.p[j]);
        }
        for (u32 j = 0; j < 3; j++) {
            u32 p = c.p[j];
            const u32* w = mt + ((u64)j * polys + poly) * D;
            const u32* wq = mq + ((u64)j * polys + poly) * D;
            u32 v0 = red32(red32(buf[j][e0], 2 * p), p), v1 = red32(red32(buf[j][e1], 2 * p), p);
            acc[j][0] = red32(acc[j][0] + red32(mul_shoup32(v0, w[e0], wq[e0], p), p), p);
            acc[j][1] = red32(acc[j][1] + red32(mul_shoup32(v1, w[e1], wq[e1], p), p), p);
        }
    }
    for (u32 j = 0; j < 3; j++) {
        buf[j][e0] = acc[j][0];
        buf[j][e1] = acc[j][1];
    }
    for (u32 lt = 0; lt < LOGD; lt++) {
        if (lt > 0) inv_sync(lt);
        for (u32 j = 0; j < 3; j++) inv_bfly(buf[j], tid, lt, tables + j * 4 * D, c.p[j]);
    }
    u64* dst = out + (u64)col * D;
    for (u32 k = 0; k < 2; k++) {
        u32 i = tid + k * (D / 2);
        dst[i] = crt3(red32(buf[0][i], c.p[0]), red32(buf[1][i], c.p[1]), red32(buf[2][i], c.p[2]), c);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// 2. Mask aggregation: the d x d integer ring FFT of `aggregate_masks`, exact. Blocks of 256 threads along a row:
// thread x = blockIdx.x * 256 + threadIdx.x, y = blockIdx.y.

__device__ __forceinline__ u32 bitrev(u32 x) {
    return __brev(x) >> (32 - LOGD);
}

// polys[k][(j + k) mod D] = +-(k == 0 ? m_j[0] : -m_j[D - k]), stored at row bitrev(k), as i64.
extern "C" __global__ void agg_gather(const u64* masks,   // [col][D]
                                      i64* a) {           // [D][D]
    u32 m = blockIdx.x * blockDim.x + threadIdx.x, k = blockIdx.y;
    u32 j = (m - k) & (D - 1);
    i64 c = k == 0 ? (i64)masks[(u64)j * D] : -(i64)masks[(u64)j * D + (D - k)];
    a[(u64)bitrev(k) * D + m] = (j + k >= D) ? -c : c;
}

// The butterfly of one length-`len` stage at coefficient i of pair `pair`: rows u = block + kk, v = u + len/2,
// rot = 2 (D/len) kk; u' = u + X^rot v, v' = u - X^rot v (negacyclic). Gives the source index of v's term and
// whether it is negated.
__device__ __forceinline__ void stage_rows(u32 len, u32 pair, u32 i, u32& ru, u32& rv, u32& vi, bool& negate) {
    u32 half_len = len >> 1;
    u32 block = (pair / half_len) * len;
    u32 kk = pair % half_len;
    ru = block + kk;
    rv = ru + half_len;
    u32 rot = 2 * (D / len) * kk;
    negate = i < rot;
    vi = negate ? D - rot + i : i - rot;
}

// Stages whose public bound len * (q - 1) fits i64 (len <= 512 for q = 2^54), exact in i64 as the CPU does.
extern "C" __global__ void agg_stage64(const i64* src, i64* dst, u32 len) {
    u32 x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y;   // (i, pair)
    u32 ru, rv, vi;
    bool negate;
    stage_rows(len, y, x, ru, rv, vi, negate);
    i64 u = src[(u64)ru * D + x];
    i64 t = src[(u64)rv * D + vi];
    t = negate ? -t : t;
    dst[(u64)ru * D + x] = u + t;
    dst[(u64)rv * D + x] = u - t;
}

// The first stage past the i64 bound: reads i64, writes i128 (two u64, low word first).
extern "C" __global__ void agg_stage_widen(const i64* src, u64* dst, u32 len) {
    u32 x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y;
    u32 ru, rv, vi;
    bool negate;
    stage_rows(len, y, x, ru, rv, vi, negate);
    i64 u = src[(u64)ru * D + x];
    i64 v = src[(u64)rv * D + vi];
    u128 uw = {(u64)u, u < 0 ? ~0ull : 0ull};
    u128 t = {(u64)v, v < 0 ? ~0ull : 0ull};
    if (negate) t = u128_neg(t);
    u128 nu = u128_add(uw, t), nv = u128_sub(uw, t);
    u64 ou = 2 * ((u64)ru * D + x), ov = 2 * ((u64)rv * D + x);
    dst[ou] = nu.lo;
    dst[ou + 1] = nu.hi;
    dst[ov] = nv.lo;
    dst[ov + 1] = nv.hi;
}

__device__ __forceinline__ u64 agg_final(u128 x, u64 qmask) {
    return ((x.lo >> LOGD) | (x.hi << (64 - LOGD))) & qmask;
}

// A stage in i128 (len < D).
extern "C" __global__ void agg_stage128(const u64* src, u64* dst, u32 len) {
    u32 x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y;
    u32 ru, rv, vi;
    bool negate;
    stage_rows(len, y, x, ru, rv, vi, negate);
    u64 iu = 2 * ((u64)ru * D + x), iv = 2 * ((u64)rv * D + vi);
    u128 u = {src[iu], src[iu + 1]}, t = {src[iv], src[iv + 1]};
    if (negate) t = u128_neg(t);
    u128 nu = u128_add(u, t), nv = u128_sub(u, t);
    u64 ou = iu, ov = 2 * ((u64)rv * D + x);
    dst[ou] = nu.lo;
    dst[ou + 1] = nu.hi;
    dst[ov] = nv.lo;
    dst[ov + 1] = nv.hi;
}

// The last stage (len = D) in i128, with the final shift and mask, written straight to each row's slot.
extern "C" __global__ void agg_stage_last(const u64* src, u64* slots, const u32* slot_of_row, u64 qmask) {
    u32 x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y;
    u32 ru, rv, vi;
    bool negate;
    stage_rows(D, y, x, ru, rv, vi, negate);
    u64 iu = 2 * ((u64)ru * D + x), iv = 2 * ((u64)rv * D + vi);
    u128 u = {src[iu], src[iu + 1]}, t = {src[iv], src[iv + 1]};
    if (negate) t = u128_neg(t);
    slots[(u64)slot_of_row[ru] * D + x] = agg_final(u128_add(u, t), qmask);
    slots[(u64)slot_of_row[rv] * D + x] = agg_final(u128_sub(u, t), qmask);
}

// ---------------------------------------------------------------------------------------------------------------
// 3. The collapse chains, one block of D / 2 threads each, all steps in one launch. Step s of chain c works on the
// half's slot idx = D/2 - 1 - s: its gadget digits (decompose), their automorphism by g^-1, the exact dot product
// with the prepared gadget masks w over the three primes, the automorphism g of the result, added into slot
// idx - 1. The digits go straight into the compile input (row bitrev((g - 1) / 2) of each limb).
//
// One launch runs the chains of several tables: block b is chain b % 2 of table b / 2, and `chains` holds each
// table's buffers.
//
// Four buffers (32 KB): B0, B1 = prime 0's limbs, B2, B3 = prime 1's. Per step:
//   A. scatter both limbs' rotated digits for primes 0 and 1; forward NTT of B0..B3; products summed into B0 (prime
//      0) and B2 (prime 1) at each thread's own elements.
//   B. scatter prime 2's digits into B1, B3; inverse NTT of B0, B2 alongside the forward NTT of B1, B3; prime 2's
//      products into B1.
//   C. inverse NTT of B1; CRT; automorphism g; add into the next slot.

// One table's buffers (device addresses), as the host lays them out: eight u64.
struct ChainTable {
    u64* slots;             // [D][D]
    i32* phat;              // [2][D][D]
    const u32* wt;          // [prime][limb][D], scaled by D^-1
    const u32* wq;
    const u32* tables;      // [prime][4][D]
    const u32* exps;        // [2 * steps]
    const u32* inv_exps;
    const u32* phat_rows;
};

extern "C" __global__ void trace_chains32(const ChainTable* chains,
                                          u32 bits,        // gadget bits
                                          u32 dropped,     // dropped bits
                                          u32 steps,       // D/2 - 1
                                          u64 qmask,
                                          Crt32 c) {
    __shared__ u32 buf[4][D];
    const ChainTable ct = chains[blockIdx.x >> 1];
    u32 chain = blockIdx.x & 1;
    u32 tid = threadIdx.x;
    u64* slots = ct.slots;
    i32* phat = ct.phat;
    const u32* wt = ct.wt;
    const u32* wq = ct.wq;
    u32 base = chain * (D / 2);
    i64 z = 1ll << bits;
    i64 rounding = dropped == 0 ? 0 : (1ll << (dropped - 1));
    u32 p0 = c.p[0], p1 = c.p[1], p2 = c.p[2];
    const u32* tab0 = ct.tables;
    const u32* tab1 = ct.tables + 4 * D;
    const u32* tab2 = ct.tables + 8 * D;
    u32 e0 = 2 * tid, e1 = 2 * tid + 1;
    for (u32 step = 0; step < steps; step++) {
        u32 idx = D / 2 - 1 - step;
        u32 gstep = chain * steps + step;
        u32 g = ct.exps[gstep], ginv = ct.inv_exps[gstep], prow = ct.phat_rows[gstep];
        const u64* src = slots + (u64)(base + idx) * D;
        u64* dst = slots + (u64)(base + idx - 1) * D;
        i32 dig[2][2];
        u32 pos[2];
        bool neg[2];
        for (u32 k = 0; k < 2; k++) {
            u32 i = tid + k * (D / 2);
            i64 x = ((i64)src[i] + rounding) >> dropped;
            for (u32 l = 0; l < 2; l++) {
                i64 digit = ((x + z / 2) & (z - 1)) - z / 2;
                dig[l][k] = (i32)digit;
                phat[((u64)l * D + prow) * D + i] = (i32)digit;
                x = (x - digit) >> bits;
            }
            u32 e = (i * ginv) & (2 * D - 1);
            pos[k] = e & (D - 1);
            neg[k] = e >= D;
        }
        // A.
        for (u32 k = 0; k < 2; k++) {
            i32 v0 = neg[k] ? -dig[0][k] : dig[0][k];
            i32 v1 = neg[k] ? -dig[1][k] : dig[1][k];
            buf[0][pos[k]] = v0 < 0 ? (u32)((i32)p0 + v0) : (u32)v0;
            buf[1][pos[k]] = v1 < 0 ? (u32)((i32)p0 + v1) : (u32)v1;
            buf[2][pos[k]] = v0 < 0 ? (u32)((i32)p1 + v0) : (u32)v0;
            buf[3][pos[k]] = v1 < 0 ? (u32)((i32)p1 + v1) : (u32)v1;
        }
        for (int lt = LOGD - 1; lt >= 0; lt--) {
            if (lt == LOGD - 1) __syncthreads();
            else fwd_sync(lt);
            fwd_bfly(buf[0], tid, lt, tab0, p0);
            fwd_bfly(buf[1], tid, lt, tab0, p0);
            fwd_bfly(buf[2], tid, lt, tab1, p1);
            fwd_bfly(buf[3], tid, lt, tab1, p1);
        }
        for (u32 k = 0; k < 2; k++) {
            u32 e = k == 0 ? e0 : e1;
            u32 a0 = red32(red32(buf[0][e], 2 * p0), p0), a1 = red32(red32(buf[1][e], 2 * p0), p0);
            u32 s = red32(mul_shoup32(a0, wt[0 * D + e], wq[0 * D + e], p0), p0);
            s = red32(s + red32(mul_shoup32(a1, wt[1 * D + e], wq[1 * D + e], p0), p0), p0);
            buf[0][e] = s;
            u32 b0 = red32(red32(buf[2][e], 2 * p1), p1), b1 = red32(red32(buf[3][e], 2 * p1), p1);
            u32 s1 = red32(mul_shoup32(b0, wt[2 * D + e], wq[2 * D + e], p1), p1);
            s1 = red32(s1 + red32(mul_shoup32(b1, wt[3 * D + e], wq[3 * D + e], p1), p1), p1);
            buf[2][e] = s1;
        }
        // B. Inverse stage 0 of B0, B2 touches only own elements; B1, B3 are free once every thread has read them.
        inv_bfly(buf[0], tid, 0, tab0, p0);
        inv_bfly(buf[2], tid, 0, tab1, p1);
        __syncthreads();
        for (u32 k = 0; k < 2; k++) {
            i32 v0 = neg[k] ? -dig[0][k] : dig[0][k];
            i32 v1 = neg[k] ? -dig[1][k] : dig[1][k];
            buf[1][pos[k]] = v0 < 0 ? (u32)((i32)p2 + v0) : (u32)v0;
            buf[3][pos[k]] = v1 < 0 ? (u32)((i32)p2 + v1) : (u32)v1;
        }
        for (u32 s = 0; s < LOGD; s++) {
            __syncthreads();
            if (s + 1 < LOGD) {
                inv_bfly(buf[0], tid, s + 1, tab0, p0);
                inv_bfly(buf[2], tid, s + 1, tab1, p1);
            }
            fwd_bfly(buf[1], tid, LOGD - 1 - s, tab2, p2);
            fwd_bfly(buf[3], tid, LOGD - 1 - s, tab2, p2);
        }
        for (u32 k = 0; k < 2; k++) {
            u32 e = k == 0 ? e0 : e1;
            u32 a0 = red32(red32(buf[1][e], 2 * p2), p2), a1 = red32(red32(buf[3][e], 2 * p2), p2);
            u32 s = red32(mul_shoup32(a0, wt[4 * D + e], wq[4 * D + e], p2), p2);
            s = red32(s + red32(mul_shoup32(a1, wt[5 * D + e], wq[5 * D + e], p2), p2), p2);
            buf[1][e] = s;
        }
        // C.
        for (u32 lt = 0; lt < LOGD; lt++) {
            if (lt > 0) inv_sync(lt);
            inv_bfly(buf[1], tid, lt, tab2, p2);
        }
        for (u32 k = 0; k < 2; k++) {
            u32 i = tid + k * (D / 2);
            u64 val = crt3(red32(buf[0][i], p0), red32(buf[2][i], p1), red32(buf[1][i], p2), c);
            u32 e = (i * g) & (2 * D - 1);
            u32 at = e & (D - 1);
            u64 upd = e >= D ? ((0ull - val) & qmask) : val;
            dst[at] = (dst[at] + upd) & qmask;
        }
        // The next step reads this step's slot (device memory) and reuses the buffers.
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------------------------------------------
// 4. Compilation: per limb, the ring FFT at powers of X^2 over the generating polynomial (rows already in
// bit-reversed order), then M[r][j] = +-a[j][(r - j) mod D], limbs side by side. The digits are below 2^(bits-1) in
// magnitude and the D-point transform grows them by at most D, so for this profile (2^18 * 2^11 = 2^29) every
// value is an exact integer in i32: the same integer the CPU holds modulo q, and the centered word it stores. The
// host checks the bound.

extern "C" __global__ void cmp_stage(const i32* src,   // [limb][D][D]
                                     i32* dst,
                                     u32 len) {
    u32 x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y;   // (i, pair), limb = blockIdx.z
    u64 off = (u64)blockIdx.z * D * D;
    u32 ru, rv, vi;
    bool negate;
    stage_rows(len, y, x, ru, rv, vi, negate);
    i32 u = src[off + (u64)ru * D + x];
    i32 t = src[off + (u64)rv * D + vi];
    t = negate ? -t : t;
    dst[off + (u64)ru * D + x] = u + t;
    dst[off + (u64)rv * D + x] = u - t;
}

// out[r][limb * D + j] = (r < j ? -x : x), x = a_limb[j][(r - j) mod D]. A 32 x 32 tile goes through shared memory
// so reads (along a row of a) and writes (along a row of out) are both contiguous. Blocks of 32 x 8 threads, grid
// (j tile, r tile, limb).
extern "C" __global__ void cmp_matrix(const i32* a,     // [limb][D][D]
                                      i32* out) {       // [D][2D]
    __shared__ i32 tile[32][33];
    u32 lx = threadIdx.x, ly = threadIdx.y;
    u32 j0 = blockIdx.x * 32, r0 = blockIdx.y * 32;
    u64 off = (u64)blockIdx.z * D * D;
    for (u32 s = 0; s < 4; s++) {
        u32 jl = ly + 8 * s;
        u32 j = j0 + jl, r = r0 + lx;
        i32 x = a[off + (u64)j * D + ((r - j) & (D - 1))];
        tile[jl][lx] = r < j ? -x : x;
    }
    __syncthreads();
    for (u32 s = 0; s < 4; s++) {
        u32 rl = ly + 8 * s;
        out[(u64)(r0 + rl) * (2 * D) + blockIdx.z * D + j0 + lx] = tile[lx][rl];
    }
}

// Rows of the compile input that no chain step writes (exponents the chains do not produce), set to zero. Grid
// (i / 256, missing row, limb).
extern "C" __global__ void zero_rows(i32* phat,            // [limb][D][D]
                                     const u32* rows) {
    u32 x = blockIdx.x * blockDim.x + threadIdx.x;
    phat[((u64)blockIdx.z * D + rows[blockIdx.y]) * D + x] = 0;
}
