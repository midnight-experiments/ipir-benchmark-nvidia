// 32-bit variants of the hint and chain kernels: the exact public dot products over three 30-bit NTT primes
// (mbb/src/rns32.rs), with native 32-bit Shoup products. Same output words as the crates (the exact integer is the
// same; only the auxiliary primes differ). Appended to build.metal; needs TPB == D / 2 == 1024 (one butterfly per
// thread per buffer and stage).
//
// Ownership: in a forward NTT's last stage thread b writes elements 2b and 2b + 1, and an inverse NTT's first stage
// reads them; its last stage writes b and b + D/2. Kernels keep those elements per thread between transforms, so the
// pointwise products and the CRT need no barrier. Stages whose butterflies stay inside one 64-element chunk
// (half-size t <= 32, preceded by a stage that also did) are synchronised with a SIMD-group barrier: the 32 threads
// of a SIMD group own exactly one such chunk.

#if TPB != D / 2
#error "32-bit kernels need TPB == D / 2"
#endif

struct Crt32 {
    uint p[3];
    uint inv01, inv01q, inv012, inv012q, p0_mod_p2, p0_mod_p2q;
    ulong p01;
    ulong prod_lo, prod_hi;   // p0 p1 p2
    ulong half_lo, half_hi;   // floor(p0 p1 p2 / 2)
    ulong qmask;
};

// WGSL has 32-bit integers but no mulhi; EMULATE_MULHI32 computes the high word from 16-bit halves (four 16 x 16
// products), as a WGSL port must, to measure that cost on this GPU.
inline uint mulhi32(uint a, uint b) {
#ifdef EMULATE_MULHI32
    uint al = a & 0xffffu, ah = a >> 16, bl = b & 0xffffu, bh = b >> 16;
    uint ll = al * bl, lh = al * bh, hl = ah * bl, hh = ah * bh;
    uint mid = (ll >> 16) + (lh & 0xffffu) + (hl & 0xffffu);
    return hh + (lh >> 16) + (hl >> 16) + (mid >> 16);
#else
    return mulhi(a, b);
#endif
}

inline uint mul_shoup32(uint x, uint w, uint wq, uint p) {
    return x * w - mulhi32(x, wq) * p;
}

inline uint red32(uint x, uint m) {
    return x >= m ? x - m : x;
}

inline ulong crt3(uint r0, uint r1, uint r2, constant Crt32& c) {
    uint p0 = c.p[0], p1 = c.p[1], p2 = c.p[2];
    uint t1 = red32(mul_shoup32(red32(r1 + p1 - red32(r0, p1), p1), c.inv01, c.inv01q, p1), p1);
    uint a = red32(mul_shoup32(t1, c.p0_mod_p2, c.p0_mod_p2q, p2), p2);
    uint x2 = red32(red32(r0, p2) + a, p2);
    uint t2 = red32(mul_shoup32(red32(r2 + p2 - x2, p2), c.inv012, c.inv012q, p2), p2);
    // x = r0 + p0 t1 + p01 t2 < 2^90
    u128 x = u128{(ulong)r0 + (ulong)p0 * (ulong)t1, 0ul};
    x = u128_add(x, mul_wide(c.p01, (ulong)t2));
    u128 bound = {c.half_lo, c.half_hi};
    ulong lo = u128_gt(x, bound) ? x.lo - c.prod_lo : x.lo;
    return lo & c.qmask;
}

// Tables per prime: fwd, fwdq, inv, invq (D words each).
#define T32_FWD 0
#define T32_FWDQ (D)
#define T32_INV (2 * D)
#define T32_INVQ (3 * D)

inline void fwd_bfly(threadgroup uint* a, uint b, uint lt, device const uint* tab, uint p) {
    uint t = 1u << lt;
    uint i = b >> lt;
    uint ix = (i << (lt + 1)) + (b & (t - 1));
    uint iy = ix + t;
    uint m = (D / 2) >> lt;            // stage with half-size t has m = D / (2t) blocks
    uint w = tab[T32_FWD + m + i], wq = tab[T32_FWDQ + m + i];
    uint two_p = 2 * p;
    uint cx = red32(a[ix], two_p);
    uint v = mul_shoup32(a[iy], w, wq, p);
    a[ix] = cx + v;
    a[iy] = cx + two_p - v;
}

inline void inv_bfly(threadgroup uint* a, uint b, uint lt, device const uint* tab, uint p) {
    uint t = 1u << lt;
    uint i = b >> lt;
    uint ix = (i << (lt + 1)) + (b & (t - 1));
    uint iy = ix + t;
    uint h = (D / 2) >> lt;
    uint w = tab[T32_INV + h + i], wq = tab[T32_INVQ + h + i];
    uint two_p = 2 * p;
    uint u = a[ix], v = a[iy];
    a[ix] = red32(u + v, two_p);
    a[iy] = mul_shoup32(u + two_p - v, w, wq, p);
}

// WGSL has workgroupBarrier but no subgroup-scope memory barrier: WORKGROUP_BARRIERS_ONLY uses the threadgroup
// barrier everywhere, as a WGSL port must.
#ifdef WORKGROUP_BARRIERS_ONLY
#define SIMD_SYNC() threadgroup_barrier(mem_flags::mem_threadgroup)
#else
#define SIMD_SYNC() simdgroup_barrier(mem_flags::mem_threadgroup)
#endif

// Barrier before a forward stage of half-size 2^lt (the previous stage had 2^(lt+1)).
inline void fwd_sync(uint lt) {
    if (lt + 1 <= 5) SIMD_SYNC();
    else threadgroup_barrier(mem_flags::mem_threadgroup);
}
// Barrier before an inverse stage of half-size 2^lt (the previous stage had 2^(lt-1)).
inline void inv_sync(uint lt) {
    if (lt <= 5) SIMD_SYNC();
    else threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ---------------------------------------------------------------------------------------------------------------
// Hint columns, three primes side by side in three buffers.

kernel void hint_columns32(device const ushort* cells [[buffer(0)]],
                           device const uint* mt [[buffer(1)]],      // [prime][poly][D], scaled by D^-1
                           device const uint* mq [[buffer(2)]],
                           device const uint* tables [[buffer(3)]],  // [prime][4][D]
                           device ulong* out [[buffer(4)]],          // [col][D]
                           constant HintConst& h [[buffer(5)]],
                           constant Crt32& c [[buffer(6)]],
                           uint tg [[threadgroup_position_in_grid]],
                           uint tid [[thread_position_in_threadgroup]]) {
    threadgroup uint buf[3][D];
    uint col = h.col0 + tg;
    uint acc[3][2] = {{0, 0}, {0, 0}, {0, 0}};
    uint e0 = 2 * tid, e1 = 2 * tid + 1;
    for (uint poly = 0; poly < h.polys; poly++) {
        device const ushort* src = cells + (ulong)col * h.rows + poly * D;
        uint x0 = src[tid], x1 = src[tid + D / 2];
        threadgroup_barrier(mem_flags::mem_threadgroup);   // the previous products have read their elements
        for (uint j = 0; j < 3; j++) {
            buf[j][tid] = x0;
            buf[j][tid + D / 2] = x1;
        }
        for (int lt = LOGD - 1; lt >= 0; lt--) {
            if (lt == LOGD - 1) threadgroup_barrier(mem_flags::mem_threadgroup);
            else fwd_sync(lt);
            for (uint j = 0; j < 3; j++) fwd_bfly(buf[j], tid, lt, tables + j * 4 * D, c.p[j]);
        }
        for (uint j = 0; j < 3; j++) {
            uint p = c.p[j];
            device const uint* w = mt + ((ulong)j * h.polys + poly) * D;
            device const uint* wq = mq + ((ulong)j * h.polys + poly) * D;
            uint v0 = red32(red32(buf[j][e0], 2 * p), p), v1 = red32(red32(buf[j][e1], 2 * p), p);
            acc[j][0] = red32(acc[j][0] + red32(mul_shoup32(v0, w[e0], wq[e0], p), p), p);
            acc[j][1] = red32(acc[j][1] + red32(mul_shoup32(v1, w[e1], wq[e1], p), p), p);
        }
    }
    for (uint j = 0; j < 3; j++) {
        buf[j][e0] = acc[j][0];
        buf[j][e1] = acc[j][1];
    }
    for (uint lt = 0; lt < LOGD; lt++) {
        if (lt > 0) inv_sync(lt);
        for (uint j = 0; j < 3; j++) inv_bfly(buf[j], tid, lt, tables + j * 4 * D, c.p[j]);
    }
    device ulong* dst = out + (ulong)col * D;
    for (uint k = 0; k < 2; k++) {
        uint i = tid + k * (D / 2);
        dst[i] = crt3(red32(buf[0][i], c.p[0]), red32(buf[1][i], c.p[1]), red32(buf[2][i], c.p[2]), c);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// The two collapse chains with three 30-bit primes. Four buffers (32 KB): B0, B1 = prime 0's limbs, B2, B3 = prime
// 1's. Per step:
//   A. scatter both limbs' rotated digits for primes 0 and 1; forward NTT of B0..B3; products summed into B0 (prime
//      0) and B2 (prime 1) at each thread's own elements.
//   B. scatter prime 2's digits into B1, B3; inverse NTT of B0, B2 alongside the forward NTT of B1, B3; prime 2's
//      products into B1.
//   C. inverse NTT of B1; CRT; automorphism g; add into the next slot.

kernel void trace_chains32(device ulong* slots [[buffer(0)]],
                           device int* phat [[buffer(1)]],
                           device const uint* wt [[buffer(2)]],       // [prime][limb][D], scaled by D^-1
                           device const uint* wq [[buffer(3)]],
                           device const uint* tables [[buffer(4)]],
                           device const uint* exps [[buffer(5)]],
                           device const uint* inv_exps [[buffer(6)]],
                           device const uint* phat_rows [[buffer(7)]],
                           constant TraceConst& tc [[buffer(8)]],
                           constant Crt32& c [[buffer(9)]],
                           uint chain [[threadgroup_position_in_grid]],
                           uint tid [[thread_position_in_threadgroup]]) {
    threadgroup uint buf[4][D];
    uint base = chain * (D / 2);
    long z = 1l << tc.bits;
    long rounding = tc.dropped == 0 ? 0 : (1l << (tc.dropped - 1));
    uint p0 = c.p[0], p1 = c.p[1], p2 = c.p[2];
    device const uint* tab0 = tables;
    device const uint* tab1 = tables + 4 * D;
    device const uint* tab2 = tables + 8 * D;
    uint e0 = 2 * tid, e1 = 2 * tid + 1;
    for (uint step = 0; step < tc.steps; step++) {
        uint idx = D / 2 - 1 - step;
        uint gstep = chain * tc.steps + step;
        uint g = exps[gstep], ginv = inv_exps[gstep], prow = phat_rows[gstep];
        device const ulong* src = slots + (ulong)(base + idx) * D;
        device ulong* dst = slots + (ulong)(base + idx - 1) * D;
        int dig[2][2];
        uint pos[2];
        bool neg[2];
        for (uint k = 0; k < 2; k++) {
            uint i = tid + k * (D / 2);
            long x = ((long)src[i] + rounding) >> tc.dropped;
            for (uint l = 0; l < 2; l++) {
                long digit = ((x + z / 2) & (z - 1)) - z / 2;
                dig[l][k] = (int)digit;
                phat[((ulong)l * D + prow) * D + i] = (int)digit;
                x = (x - digit) >> tc.bits;
            }
            uint e = (i * ginv) & (2 * D - 1);
            pos[k] = e & (D - 1);
            neg[k] = e >= D;
        }
        // A.
        for (uint k = 0; k < 2; k++) {
            int v0 = neg[k] ? -dig[0][k] : dig[0][k];
            int v1 = neg[k] ? -dig[1][k] : dig[1][k];
            buf[0][pos[k]] = v0 < 0 ? (uint)((int)p0 + v0) : (uint)v0;
            buf[1][pos[k]] = v1 < 0 ? (uint)((int)p0 + v1) : (uint)v1;
            buf[2][pos[k]] = v0 < 0 ? (uint)((int)p1 + v0) : (uint)v0;
            buf[3][pos[k]] = v1 < 0 ? (uint)((int)p1 + v1) : (uint)v1;
        }
        for (int lt = LOGD - 1; lt >= 0; lt--) {
            if (lt == LOGD - 1) threadgroup_barrier(mem_flags::mem_threadgroup);
            else fwd_sync(lt);
            fwd_bfly(buf[0], tid, lt, tab0, p0);
            fwd_bfly(buf[1], tid, lt, tab0, p0);
            fwd_bfly(buf[2], tid, lt, tab1, p1);
            fwd_bfly(buf[3], tid, lt, tab1, p1);
        }
        for (uint k = 0; k < 2; k++) {
            uint e = k == 0 ? e0 : e1;
            uint a0 = red32(red32(buf[0][e], 2 * p0), p0), a1 = red32(red32(buf[1][e], 2 * p0), p0);
            uint s = red32(mul_shoup32(a0, wt[0 * D + e], wq[0 * D + e], p0), p0);
            s = red32(s + red32(mul_shoup32(a1, wt[1 * D + e], wq[1 * D + e], p0), p0), p0);
            buf[0][e] = s;
            uint b0 = red32(red32(buf[2][e], 2 * p1), p1), b1 = red32(red32(buf[3][e], 2 * p1), p1);
            uint s1 = red32(mul_shoup32(b0, wt[2 * D + e], wq[2 * D + e], p1), p1);
            s1 = red32(s1 + red32(mul_shoup32(b1, wt[3 * D + e], wq[3 * D + e], p1), p1), p1);
            buf[2][e] = s1;
        }
        // B. Inverse stage 0 of B0, B2 touches only own elements; B1, B3 are free once every thread has read them.
        inv_bfly(buf[0], tid, 0, tab0, p0);
        inv_bfly(buf[2], tid, 0, tab1, p1);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 2; k++) {
            int v0 = neg[k] ? -dig[0][k] : dig[0][k];
            int v1 = neg[k] ? -dig[1][k] : dig[1][k];
            buf[1][pos[k]] = v0 < 0 ? (uint)((int)p2 + v0) : (uint)v0;
            buf[3][pos[k]] = v1 < 0 ? (uint)((int)p2 + v1) : (uint)v1;
        }
        for (uint s = 0; s < LOGD; s++) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (s + 1 < LOGD) {
                inv_bfly(buf[0], tid, s + 1, tab0, p0);
                inv_bfly(buf[2], tid, s + 1, tab1, p1);
            }
            fwd_bfly(buf[1], tid, LOGD - 1 - s, tab2, p2);
            fwd_bfly(buf[3], tid, LOGD - 1 - s, tab2, p2);
        }
        for (uint k = 0; k < 2; k++) {
            uint e = k == 0 ? e0 : e1;
            uint a0 = red32(red32(buf[1][e], 2 * p2), p2), a1 = red32(red32(buf[3][e], 2 * p2), p2);
            uint s = red32(mul_shoup32(a0, wt[4 * D + e], wq[4 * D + e], p2), p2);
            s = red32(s + red32(mul_shoup32(a1, wt[5 * D + e], wq[5 * D + e], p2), p2), p2);
            buf[1][e] = s;
        }
        // C.
        for (uint lt = 0; lt < LOGD; lt++) {
            if (lt > 0) inv_sync(lt);
            inv_bfly(buf[1], tid, lt, tab2, p2);
        }
        for (uint k = 0; k < 2; k++) {
            uint i = tid + k * (D / 2);
            ulong val = crt3(red32(buf[0][i], p0), red32(buf[2][i], p1), red32(buf[1][i], p2), c);
            uint e = (i * g) & (2 * D - 1);
            uint at = e & (D - 1);
            ulong upd = e >= D ? ((0ul - val) & tc.qmask) : val;
            dst[at] = (dst[at] + upd) & tc.qmask;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Register-blocked chains (v4). Each NTT of 2048 points is split into four groups of stages done in registers,
// 8 elements per thread: half-sizes t = 1024..256 (elements k + 256c), 128..32 (256(k/32) + k%32 + 32c),
// 16..4 (32(k/4) + k%4 + 4c), 2..1 (8k + c), with the threadgroup buffer in between. The last three groups of a
// SIMD group stay inside one 256-element chunk, so only the first boundary needs a threadgroup barrier. 768
// threads: 256 per prime, all three primes at once; the two limbs one after another, products accumulated in
// registers; the inverse runs the groups in reverse.

#if D == 2048
inline void fwd_group(thread uint* e, uint x0, uint stride, int s_hi, int s_lo, device const uint* tab, uint p) {
    uint two_p = 2 * p;
    for (int s = s_hi; s >= s_lo; s--) {
        uint hb = 1u << s;
        uint lt = ctz(stride << s);
        uint m = (D / 2) >> lt;
        for (uint c = 0; c < 8; c++) {
            if (c & hb) continue;
            uint x = x0 + stride * c;
            uint i = x >> (lt + 1);
            uint w = tab[T32_FWD + m + i], wq = tab[T32_FWDQ + m + i];
            uint cx = red32(e[c], two_p);
            uint v = mul_shoup32(e[c + hb], w, wq, p);
            e[c] = cx + v;
            e[c + hb] = cx + two_p - v;
        }
    }
}

inline void inv_group(thread uint* e, uint x0, uint stride, int s_lo, int s_hi, device const uint* tab, uint p) {
    uint two_p = 2 * p;
    for (int s = s_lo; s <= s_hi; s++) {
        uint hb = 1u << s;
        uint lt = ctz(stride << s);
        uint h = (D / 2) >> lt;
        for (uint c = 0; c < 8; c++) {
            if (c & hb) continue;
            uint x = x0 + stride * c;
            uint i = x >> (lt + 1);
            uint w = tab[T32_INV + h + i], wq = tab[T32_INVQ + h + i];
            uint u = e[c], v = e[c + hb];
            e[c] = red32(u + v, two_p);
            e[c + hb] = mul_shoup32(u + two_p - v, w, wq, p);
        }
    }
}

inline void load8(thread uint* e, threadgroup uint* b, uint x0, uint stride) {
    for (uint c = 0; c < 8; c++) e[c] = b[x0 + stride * c];
}
inline void store8(thread const uint* e, threadgroup uint* b, uint x0, uint stride) {
    for (uint c = 0; c < 8; c++) b[x0 + stride * c] = e[c];
}

// Forward NTT of buffer b (canonical input); returns the transform of elements 8k..8k+7 in e, in [0, 4p).
inline void ntt_fwd_blocked(thread uint* e, threadgroup uint* b, uint k, device const uint* tab, uint p) {
    uint x1 = k, x2 = 256 * (k >> 5) + (k & 31), x3 = 32 * (k >> 2) + (k & 3), x4 = 8 * k;
    load8(e, b, x1, 256);
    fwd_group(e, x1, 256, 2, 0, tab, p);
    store8(e, b, x1, 256);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    load8(e, b, x2, 32);
    fwd_group(e, x2, 32, 2, 0, tab, p);
    store8(e, b, x2, 32);
    SIMD_SYNC();
    load8(e, b, x3, 4);
    fwd_group(e, x3, 4, 2, 0, tab, p);
    store8(e, b, x3, 4);
    SIMD_SYNC();
    load8(e, b, x4, 1);
    fwd_group(e, x4, 1, 1, 0, tab, p);
}

// Inverse NTT (unscaled) of the canonical values e of elements 8k..8k+7; leaves elements k + 256c in e, in [0, 2p).
inline void ntt_inv_blocked(thread uint* e, threadgroup uint* b, uint k, device const uint* tab, uint p) {
    uint x1 = k, x2 = 256 * (k >> 5) + (k & 31), x3 = 32 * (k >> 2) + (k & 3), x4 = 8 * k;
    inv_group(e, x4, 1, 0, 1, tab, p);
    store8(e, b, x4, 1);
    SIMD_SYNC();
    load8(e, b, x3, 4);
    inv_group(e, x3, 4, 0, 2, tab, p);
    store8(e, b, x3, 4);
    SIMD_SYNC();
    load8(e, b, x2, 32);
    inv_group(e, x2, 32, 0, 2, tab, p);
    store8(e, b, x2, 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    load8(e, b, x1, 256);
    inv_group(e, x1, 256, 0, 2, tab, p);
}

#define CHAIN_THREADS 768

kernel void trace_chains32r(device ulong* slots [[buffer(0)]],
                            device int* phat [[buffer(1)]],
                            device const uint* wt [[buffer(2)]],       // [prime][limb][D], scaled by D^-1
                            device const uint* wq [[buffer(3)]],
                            device const uint* tables [[buffer(4)]],
                            device const uint* exps [[buffer(5)]],
                            device const uint* inv_exps [[buffer(6)]],
                            device const uint* phat_rows [[buffer(7)]],
                            constant TraceConst& tc [[buffer(8)]],
                            constant Crt32& c [[buffer(9)]],
                            uint chain [[threadgroup_position_in_grid]],
                            uint tid [[thread_position_in_threadgroup]]) {
    threadgroup uint buf[3][D];
    uint base = chain * (D / 2);
    long z = 1l << tc.bits;
    long rounding = tc.dropped == 0 ? 0 : (1l << (tc.dropped - 1));
    uint j = tid >> 8, k = tid & 255;          // this thread's prime and index within it
    uint p = c.p[j];
    device const uint* tab = tables + j * 4 * D;
    threadgroup uint* b = buf[j];
    const uint per = (D + CHAIN_THREADS - 1) / CHAIN_THREADS;   // 3 coefficients per thread for the digits and CRT
    for (uint step = 0; step < tc.steps; step++) {
        uint idx = D / 2 - 1 - step;
        uint gstep = chain * tc.steps + step;
        uint g = exps[gstep], ginv = inv_exps[gstep], prow = phat_rows[gstep];
        device const ulong* src = slots + (ulong)(base + idx) * D;
        device ulong* dst = slots + (ulong)(base + idx - 1) * D;
        int dig[2][per];
        uint pos[per];
        for (uint r = 0; r < per; r++) {
            uint i = tid + r * CHAIN_THREADS;
            if (i >= D) break;
            long x = ((long)src[i] + rounding) >> tc.dropped;
            int d0 = 0, d1 = 0;
            for (uint l = 0; l < 2; l++) {
                long digit = ((x + z / 2) & (z - 1)) - z / 2;
                if (l == 0) d0 = (int)digit; else d1 = (int)digit;
                phat[((ulong)l * D + prow) * D + i] = (int)digit;
                x = (x - digit) >> tc.bits;
            }
            uint e = (i * ginv) & (2 * D - 1);
            bool neg = e >= D;
            pos[r] = e & (D - 1);
            dig[0][r] = neg ? -d0 : d0;
            dig[1][r] = neg ? -d1 : d1;
        }
        uint acc[8];
        uint e[8];
        for (uint l = 0; l < 2; l++) {
            // Scatter this limb's rotated digits into all three primes' buffers.
            for (uint r = 0; r < per; r++) {
                uint i = tid + r * CHAIN_THREADS;
                if (i >= D) break;
                int v = dig[l][r];
                buf[0][pos[r]] = v < 0 ? (uint)((int)c.p[0] + v) : (uint)v;
                buf[1][pos[r]] = v < 0 ? (uint)((int)c.p[1] + v) : (uint)v;
                buf[2][pos[r]] = v < 0 ? (uint)((int)c.p[2] + v) : (uint)v;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            ntt_fwd_blocked(e, b, k, tab, p);
            device const uint* w = wt + ((ulong)j * 2 + l) * D + 8 * k;
            device const uint* q = wq + ((ulong)j * 2 + l) * D + 8 * k;
            for (uint cc = 0; cc < 8; cc++) {
                uint prod = red32(mul_shoup32(red32(red32(e[cc], 2 * p), p), w[cc], q[cc], p), p);
                acc[cc] = l == 0 ? prod : red32(acc[cc] + prod, p);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);   // every thread is done with the buffers
        }
        ntt_inv_blocked(acc, b, k, tab, p);
        for (uint cc = 0; cc < 8; cc++) b[k + 256 * cc] = red32(acc[cc], p);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = 0; r < per; r++) {
            uint i = tid + r * CHAIN_THREADS;
            if (i >= D) break;
            ulong val = crt3(buf[0][i], buf[1][i], buf[2][i], c);
            uint e2 = (i * g) & (2 * D - 1);
            uint at = e2 & (D - 1);
            ulong upd = e2 >= D ? ((0ul - val) & tc.qmask) : val;
            dst[at] = (dst[at] + upd) & tc.qmask;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
}
#endif
