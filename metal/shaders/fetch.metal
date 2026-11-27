// An IPIR+SP fetch on the Apple GPU: the first pass for all the requests of a call in one dispatch, the packing of
// every column block and request in one dispatch, and the rounding of the sums. A port of the CUDA fetch
// (ipir/bench/src/cuda_fetch_native.cu), with the same layouts on the device; gpu/ipir-wire.ts states the
// arithmetic, and the answers are `NativeServer::respond`'s, byte for byte.
//
// The host (src/fetch.rs) prepends `#define CT`, `#define GT` and `#define IT` and compiles this once per
// combination it uses:
//
// - CT: adjacent table columns one thread of `scan` reads (1, 2, 4 or 8).
// - IT: adjacent matrix rows one thread of `pack` reads (1, 2, 4 or 8).
// - GT: requests one thread of `scan` or `pack` answers (1, 2, 4, 8 or 16).
//
// Arithmetic. Every value is needed modulo q = 2^54, and 2^54 divides 2^64, so sums and products wrap in 64 bits and
// are masked once, in the finish. Apple GPUs emulate 64-bit integers, so a 64-bit value is two 32-bit words here
// (low, high), which are also its bytes in a buffer of 64-bit words. In the packing, a product of a signed 32-bit
// matrix entry h and a 54-bit word w = lo + 2^32 hi adds to an accumulator (a_lo, a_hi), with hu the entry's 32
// bits read unsigned and s its sign bit (h = hu - 2^32 s):
//
//     a_lo += hu * lo                                        (the low 32 bits of the product)
//     a_hi += mulhi(hu, lo) + hu * hi - s * lo + carry       (its high 32 bits; a_hi wraps: the sum modulo 2^64)
//
// The first pass multiplies 16-bit cells and needs no high product (see `scan`).
//
// Layout on the device, as in the CUDA fetch. The table is kept per column block (2,048 columns), row-major inside
// the block: cell (r, c) of block k at `(k * rows + r) * 2048 + c`; the kernels read it as 32-bit words of two
// cells. The compiled matrices are kept per block, column-major: entry (i, j) of block k at
// `(k * 4096 + j) * 2048 + i`. The words of a call's requests are laid out per group of GT requests:
// `(group * rows + r) * GT + g`. Every index fits 32 bits for tables up to 2^30 cells.
#include <metal_stdlib>
using namespace metal;

#define BLOCK_COLS 2048u
#define KEY_WORDS 4096u
#define HIGH22 0x3fffffu

#if CT != 1 && CT != 2 && CT != 4 && CT != 8
#error "CT must be 1, 2, 4 or 8"
#endif
#if IT != 1 && IT != 2 && IT != 4 && IT != 8
#error "IT must be 1, 2, 4 or 8"
#endif

struct ScanParams {
    uint rows;
    uint cols;
    uint seg_rows;
    uint requests;
};

// The first pass: partial[seg][request][column] = sum over the segment's rows of cell[r, column] * query[request][r].
// One thread: CT adjacent columns, the GT requests of one group, one segment of rows.
kernel void scan(device const uint* cells [[buffer(0)]],     // [block][row][2048 cells], two cells per word
                 device const uint* query [[buffer(1)]],     // [group][row][GT] words x << 5, as (low, high)
                 device uint* partial [[buffer(2)]],         // [segment][request][column], as (low, high)
                 constant ScanParams& p [[buffer(3)]],
                 uint3 gid [[thread_position_in_grid]]) {    // (column group, request group, segment)
    const uint col = gid.x * CT;
    if (col >= p.cols) return;
    const uint group = gid.y, seg = gid.z;
    const uint row0 = seg * p.seg_rows;
    const uint cell0 = ((col / BLOCK_COLS) * p.rows + row0) * BLOCK_COLS + col % BLOCK_COLS;
    uint c = cell0 / 2;
    uint q = ((group * p.rows + row0) * GT) * 2;
    // A cell is 16 bits, so its product with each 16-bit half of `lo` fits 32 bits and no high product is needed:
    // three accumulators of weights 2^0, 2^16 and 2^32. t0 and t1 only receive 16-bit pieces (two each row for t1),
    // so they are exact over the at most 32,768 rows of a segment; t2 wraps, which is the sum modulo 2^64.
    uint t0[CT][GT];
    uint t1[CT][GT];
    uint t2[CT][GT];
    for (uint i = 0; i < CT; ++i)
        for (uint g = 0; g < GT; ++g) { t0[i][g] = 0; t1[i][g] = 0; t2[i][g] = 0; }
    for (uint r = 0; r < p.seg_rows; ++r, c += BLOCK_COLS / 2, q += 2 * GT) {
        uint v[CT];
#if CT == 1
        v[0] = (cells[c] >> ((cell0 & 1u) * 16u)) & 0xffffu;
#else
        for (uint i = 0; i < CT; i += 2) {
            const uint w = cells[c + i / 2];
            v[i] = w & 0xffffu;
            v[i + 1] = w >> 16;
        }
#endif
        for (uint g = 0; g < GT; ++g) {
            const uint lo = query[q + 2 * g], hi = query[q + 2 * g + 1];
            const uint l0 = lo & 0xffffu, l1 = lo >> 16;
            for (uint i = 0; i < CT; ++i) {
                const uint p0 = v[i] * l0, p1 = v[i] * l1;
                t0[i][g] += p0 & 0xffffu;
                t1[i][g] += (p0 >> 16) + (p1 & 0xffffu);
                t2[i][g] += (p1 >> 16) + v[i] * hi;
            }
        }
    }
    for (uint g = 0; g < GT; ++g) {
        const uint out = ((seg * p.requests + group * GT + g) * p.cols + col) * 2;
        for (uint i = 0; i < CT; ++i) {
            const uint mid = t1[i][g] << 16;
            const uint low = t0[i][g] + mid;
            partial[out + 2 * i] = low;
            partial[out + 2 * i + 1] = t2[i][g] + (t1[i][g] >> 16) + uint(low < mid);
        }
    }
}

struct PackParams {
    uint cols;
    uint seg_words;
    uint requests;
    uint seg0;
};

// The packing: partial[seg0 + seg][request][column] = sum over the segment's key words of
// H[block of the column][column in the block, j] * key[request][j].
// One thread: IT adjacent rows of a block's matrix (answer columns), the GT requests of one group, one segment of
// key words.
kernel void pack(device const int* mats [[buffer(0)]],       // [block][key word][2048]
                 device const uint* keys [[buffer(1)]],      // [group][key word][GT], as (low, high)
                 device uint* partial [[buffer(2)]],         // the scan's buffer
                 constant PackParams& p [[buffer(3)]],
                 uint3 gid [[thread_position_in_grid]]) {    // (answer column group, request group, segment)
    const uint col = gid.x * IT;
    if (col >= p.cols) return;
    const uint group = gid.y, seg = gid.z;
    const uint word0 = seg * p.seg_words;
    uint m = ((col / BLOCK_COLS) * KEY_WORDS + word0) * BLOCK_COLS + col % BLOCK_COLS;
    uint k = ((group * KEY_WORDS + word0) * GT) * 2;
    uint alo[IT][GT];
    uint ahi[IT][GT];
    for (uint i = 0; i < IT; ++i)
        for (uint g = 0; g < GT; ++g) { alo[i][g] = 0; ahi[i][g] = 0; }
    for (uint j = 0; j < p.seg_words; ++j, m += BLOCK_COLS, k += 2 * GT) {
        int h[IT];
        for (uint i = 0; i < IT; ++i) h[i] = mats[m + i];
        for (uint g = 0; g < GT; ++g) {
            const uint lo = keys[k + 2 * g], hi = keys[k + 2 * g + 1];
            for (uint i = 0; i < IT; ++i) {
                const uint hu = uint(h[i]);
                const uint pl = hu * lo;
                const uint low = alo[i][g] + pl;
                ahi[i][g] += mulhi(hu, lo) + hu * hi - (uint(h[i] >> 31) & lo) + uint(low < pl);
                alo[i][g] = low;
            }
        }
    }
    for (uint g = 0; g < GT; ++g) {
        const uint out = (((p.seg0 + seg) * p.requests + group * GT + g) * p.cols + col) * 2;
        for (uint i = 0; i < IT; ++i) {
            partial[out + 2 * i] = alo[i][g];
            partial[out + 2 * i + 1] = ahi[i][g];
        }
    }
}

struct FinishParams {
    uint cols;
    uint requests;
    uint s0;
    uint s1;
};

// Segments s0 to s1 - 1 of `partial` added modulo 2^64, as (low, high).
static inline uint2 sum_segments(device const uint* partial, constant FinishParams& p, uint request, uint col) {
    uint lo = 0, hi = 0;
    for (uint s = p.s0; s < p.s1; ++s) {
        const uint at = ((s * p.requests + request) * p.cols + col) * 2;
        const uint x = partial[at];
        lo += x;
        hi += partial[at + 1] + uint(lo < x);
    }
    return uint2(lo, hi);
}

// The answer's coefficients: the sum modulo 2^54 rounded to 22 bits, ((v + 2^31) >> 32) mod 2^22. With
// v = 2^32 hi + lo that is hi plus the top bit of lo. One thread per (column, request).
kernel void finish_answer(device const uint* partial [[buffer(0)]],
                          device uint* out [[buffer(1)]],              // [request][column]
                          constant FinishParams& p [[buffer(2)]],
                          uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.cols) return;
    const uint2 v = sum_segments(partial, p, gid.y, gid.x);
    out[gid.y * p.cols + gid.x] = ((v.y & HIGH22) + (v.x >> 31)) & HIGH22;
}

// The same sums unrounded, modulo 2^54, as (low, high): for checking the first pass (the scan's segments) and the
// packing product (the packing's segments) on their own.
kernel void finish_words(device const uint* partial [[buffer(0)]],
                         device uint* out [[buffer(1)]],               // [request][column], as (low, high)
                         constant FinishParams& p [[buffer(2)]],
                         uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.cols) return;
    const uint2 v = sum_segments(partial, p, gid.y, gid.x);
    const uint at = (gid.y * p.cols + gid.x) * 2;
    out[at] = v.x;
    out[at + 1] = v.y & HIGH22;
}

struct PutParams {
    uint rows;
    uint block;
};

// One column block's cells from column-major (`col * rows + row`, as the host and the block build keep them) to the
// table's layout. One thread per cell.
kernel void put_cells(device const ushort* src [[buffer(0)]],
                      device ushort* cells [[buffer(1)]],
                      constant PutParams& p [[buffer(2)]],
                      uint3 gid [[thread_position_in_grid]]) {         // (row, column of the block)
    if (gid.x >= p.rows) return;
    cells[(p.block * p.rows + gid.x) * BLOCK_COLS + gid.y] = src[gid.y * p.rows + gid.x];
}

// One block's compiled matrix from row-major (`i * 4096 + j`, as the CPU and the block build give it) to the layout
// the packing reads. One thread per entry.
kernel void put_matrix(device const int* src [[buffer(0)]],
                       device int* mats [[buffer(1)]],
                       constant PutParams& p [[buffer(2)]],
                       uint3 gid [[thread_position_in_grid]]) {        // (key word, row of the matrix)
    mats[(p.block * KEY_WORDS + gid.x) * BLOCK_COLS + gid.y] = src[gid.y * KEY_WORDS + gid.x];
}
