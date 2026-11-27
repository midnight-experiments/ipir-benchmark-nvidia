// An IPIR+SP fetch on an NVIDIA GPU: the first pass for all the requests of a call in one launch, the packing of
// every column block and request in one launch, and the rounding of the sums. `gpu/ipir-wire.ts` states the
// arithmetic; the answers are `NativeServer::respond`'s, byte for byte.
//
// The host (bench/src/cuda_fetch_native.rs) prepends `#define CT`, `#define GT` and `#define IT` and compiles this
// with NVRTC, once per combination it uses:
//
// - CT: adjacent table columns one thread of `scan` reads (1, 2, 4 or 8).
// - IT: adjacent matrix rows one thread of `pack` reads (1, 2, 4 or 8).
// - GT: requests one thread of `scan` or `pack` answers (1, 2, 4, 8 or 16).
//
// Arithmetic. Every value is needed modulo q = 2^54, and 2^54 divides 2^64, so sums and products wrap in 64 bits
// and are masked once, in `finish`. A product of a cell v (16 bits, or a signed 32-bit matrix entry) and a 54-bit
// word w = lo + 2^32 hi is kept in two accumulators, so that no 64-bit multiply is needed:
//
//     a += v * lo      (a 32 x 32 -> 64 bit product, added in 64 bits)
//     b += v * hi      (32 bits: b has weight 2^32, and only its low 32 bits reach the sum modulo 2^64)
//
// and the sum is a + 2^32 b. A signed entry h = hu - 2^32 s (hu its 32 bits read unsigned, s its sign bit) gives
// h * w = hu * lo + 2^32 (hu * hi - s * lo) modulo 2^64.
//
// Layout on the device. The table is kept per column block (2,048 columns), row-major inside the block:
// cell (r, c) of block k at `(k * rows + r) * 2048 + c`. The compiled matrices are kept per block, column-major:
// entry (i, j) of block k (i an answer column of the block, j a key word) at `(k * 4096 + j) * 2048 + i`. So in
// both kernels neighbouring threads read neighbouring words, a column block is one contiguous range, and one can be
// replaced without moving the others. The words of a call's requests are laid out per group of GT requests:
// `(group * rows + r) * GT + g`, so a thread reads its GT words of a row from one place.

typedef unsigned short u16;
typedef unsigned int u32;
typedef int i32;
typedef unsigned long long u64;

#define BLOCK_COLS 2048u
#define KEY_WORDS 4096u
#define MASK54 ((1ull << 54) - 1ull)

#if CT != 1 && CT != 2 && CT != 4 && CT != 8
#error "CT must be 1, 2, 4 or 8"
#endif
#if IT != 1 && IT != 2 && IT != 4 && IT != 8
#error "IT must be 1, 2, 4 or 8"
#endif

// The grid's three dimensions hold the column groups, the request groups and the segments. `order` says which
// dimension the column groups are in (0: x, 1: y, 2: z); the request groups and the segments take the other two, in
// that order. Blocks start roughly in the order of their index with x fastest, so `order` chooses which blocks run
// next to each other: those that read the same request words (0), or those that read the same table words (1, 2).
__device__ __forceinline__ void grid_roles(u32 order, u32* col_block, u32* group, u32* seg) {
    const u32 x = blockIdx.x, y = blockIdx.y, z = blockIdx.z;
    if (order == 0) {
        *col_block = x; *group = y; *seg = z;
    } else if (order == 1) {
        *group = x; *col_block = y; *seg = z;
    } else {
        *group = x; *seg = y; *col_block = z;
    }
}

__device__ __forceinline__ void load_cells(const u16* p, u32* v) {
#if CT == 1
    v[0] = p[0];
#elif CT == 2
    const u32 w = *(const u32*)p;
    v[0] = w & 0xffffu;
    v[1] = w >> 16;
#else
    for (u32 c = 0; c < CT; c += 4) {
        const u64 w = *(const u64*)(p + c);
        v[c] = (u32)w & 0xffffu;
        v[c + 1] = (u32)(w >> 16) & 0xffffu;
        v[c + 2] = (u32)(w >> 32) & 0xffffu;
        v[c + 3] = (u32)(w >> 48);
    }
#endif
}

__device__ __forceinline__ void load_entries(const i32* p, u32* h) {
#if IT == 1
    h[0] = (u32)p[0];
#else
    for (u32 c = 0; c < IT; c += 2) {
        const u64 w = *(const u64*)(p + c);
        h[c] = (u32)w;
        h[c + 1] = (u32)(w >> 32);
    }
#endif
}

// The first pass: partial[seg][request][column] = sum over the segment's rows of cell[r, column] * query[request][r].
// One thread: CT adjacent columns, the GT requests of one group, one segment of rows.
extern "C" __global__ void scan(const u16* cells,   // [block][row][2048]
                                const u64* query,   // [group][row][GT]: the selection words, x << 5
                                u64* partial,       // [segment][request][column]; this kernel's segments are first
                                u32 rows, u32 cols, u32 seg_rows, u32 requests, u32 order) {
    u32 col_block, group, seg;
    grid_roles(order, &col_block, &group, &seg);
    const u32 col = (col_block * blockDim.x + threadIdx.x) * CT;
    if (col >= cols) return;
    const u64 row0 = (u64)seg * seg_rows;
    const u16* p = cells + ((u64)(col / BLOCK_COLS) * rows + row0) * BLOCK_COLS + col % BLOCK_COLS;
    const u64* q = query + ((u64)group * rows + row0) * GT;
    u64 a[CT][GT];
    u32 b[CT][GT];
    for (u32 c = 0; c < CT; ++c)
        for (u32 g = 0; g < GT; ++g) { a[c][g] = 0; b[c][g] = 0; }
    for (u32 r = 0; r < seg_rows; ++r, p += BLOCK_COLS, q += GT) {
        u32 v[CT];
        load_cells(p, v);
        for (u32 g = 0; g < GT; ++g) {
            const u64 w = q[g];
            const u32 lo = (u32)w, hi = (u32)(w >> 32);
            for (u32 c = 0; c < CT; ++c) {
                a[c][g] += (u64)v[c] * lo;
                b[c][g] += v[c] * hi;
            }
        }
    }
    for (u32 g = 0; g < GT; ++g) {
        u64* out = partial + ((u64)seg * requests + group * GT + g) * cols + col;
        for (u32 c = 0; c < CT; ++c) out[c] = a[c][g] + ((u64)b[c][g] << 32);
    }
}

// The packing: partial[seg0 + seg][request][column] = sum over the segment's key words of
// H[block of the column][column in the block, j] * key[request][j].
// One thread: IT adjacent rows of a block's matrix (answer columns), the GT requests of one group, one segment of
// key words.
extern "C" __global__ void pack(const i32* mats,    // [block][key word][2048]
                                const u64* keys,    // [group][key word][GT]
                                u64* partial,       // the scan's buffer
                                u32 cols, u32 seg_words, u32 requests, u32 seg0, u32 order) {
    u32 col_block, group, seg;
    grid_roles(order, &col_block, &group, &seg);
    const u32 col = (col_block * blockDim.x + threadIdx.x) * IT;
    if (col >= cols) return;
    const u64 word0 = (u64)seg * seg_words;
    const i32* p = mats + ((u64)(col / BLOCK_COLS) * KEY_WORDS + word0) * BLOCK_COLS + col % BLOCK_COLS;
    const u64* k = keys + ((u64)group * KEY_WORDS + word0) * GT;
    u64 a[IT][GT];
    u32 b[IT][GT];
    for (u32 c = 0; c < IT; ++c)
        for (u32 g = 0; g < GT; ++g) { a[c][g] = 0; b[c][g] = 0; }
    for (u32 j = 0; j < seg_words; ++j, p += BLOCK_COLS, k += GT) {
        u32 h[IT];
        load_entries(p, h);
        for (u32 g = 0; g < GT; ++g) {
            const u64 w = k[g];
            const u32 lo = (u32)w, hi = (u32)(w >> 32);
            for (u32 c = 0; c < IT; ++c) {
                a[c][g] += (u64)h[c] * lo;
                b[c][g] += h[c] * hi - ((u32)((i32)h[c] >> 31) & lo);
            }
        }
    }
    for (u32 g = 0; g < GT; ++g) {
        u64* out = partial + ((u64)(seg0 + seg) * requests + group * GT + g) * cols + col;
        for (u32 c = 0; c < IT; ++c) out[c] = a[c][g] + ((u64)b[c][g] << 32);
    }
}

// The answer's coefficients: segments s0 to s1 - 1 of `partial` added modulo 2^54 and rounded to 22 bits,
// ((v + 2^31) >> 32) mod 2^22. One thread per (request, column).
extern "C" __global__ void finish_answer(const u64* partial, u32* out,   // [request][column]
                                         u32 cols, u32 requests, u32 s0, u32 s1) {
    const u32 col = blockIdx.x * blockDim.x + threadIdx.x;
    const u32 request = blockIdx.y;
    if (col >= cols) return;
    u64 v = 0;
    for (u32 s = s0; s < s1; ++s) v += partial[((u64)s * requests + request) * cols + col];
    v &= MASK54;
    out[(u64)request * cols + col] = (u32)((v + (1ull << 31)) >> 32) & 0x3fffffu;
}

// The same sums unrounded, modulo 2^54: for checking the first pass (the scan's segments) and the packing product
// (the packing's segments) on their own.
extern "C" __global__ void finish_words(const u64* partial, u64* out,    // [request][column]
                                        u32 cols, u32 requests, u32 s0, u32 s1) {
    const u32 col = blockIdx.x * blockDim.x + threadIdx.x;
    const u32 request = blockIdx.y;
    if (col >= cols) return;
    u64 v = 0;
    for (u32 s = s0; s < s1; ++s) v += partial[((u64)s * requests + request) * cols + col];
    out[(u64)request * cols + col] = v & MASK54;
}

// One column block's cells from column-major (`col * rows + row`, as the host and the block build keep them) to the
// table's layout. One thread per cell; x runs over the rows, so the reads are neighbours.
extern "C" __global__ void put_cells(const u16* src, u16* cells, u32 rows, u32 block) {
    const u32 row = blockIdx.x * blockDim.x + threadIdx.x;
    const u32 col = blockIdx.y;
    if (row >= rows) return;
    cells[((u64)block * rows + row) * BLOCK_COLS + col] = src[(u64)col * rows + row];
}

// One block's compiled matrix from row-major (`i * 4096 + j`, as the CPU and the block build give it) to the
// layout the packing reads. One thread per entry; x runs over the key words.
extern "C" __global__ void put_matrix(const i32* src, i32* mats, u32 block) {
    const u32 j = blockIdx.x * blockDim.x + threadIdx.x;
    const u32 i = blockIdx.y;
    if (j >= KEY_WORDS) return;
    mats[((u64)block * KEY_WORDS + j) * BLOCK_COLS + i] = src[(u64)i * KEY_WORDS + j];
}
