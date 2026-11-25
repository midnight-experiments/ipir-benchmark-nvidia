/**
 * The per-block build of an IPIR+SP table on the GPU, in WebGPU (WGSL): from one column block's cells to its hint
 * masks, both public masks and the compiled packing matrix, with the same output words as the CPU build of the
 * vendored crates (`NativePublicSetup::build_with_concurrency` of one block).
 *
 * It is a port of the Metal build made for spec 00065's question 21 (kernels `hint_columns32`, `agg_*`,
 * `trace_chains32`, `cmp_*`), with what WGSL lacks written out:
 * - no 64-bit integers: the 54-bit words are pairs of u32 (low, high), the aggregation's 128-bit sums four u32, and
 *   the 90-bit CRT sum three;
 * - no `mulhi`: the high half of a 32 × 32 product comes from four 16 × 16 products;
 * - no subgroup barrier: every stage of a transform is separated by `workgroupBarrier()`.
 *
 * The exact dot products (hints and chain steps) run over three 30-bit primes instead of the crates' 42-bit ones.
 * The crates compute the exact integer and only then reduce it modulo q, so any primes whose product exceeds twice the
 * bound on that integer give the same words; `gpu_build_prep` checks the bound when it prepares the operands.
 *
 * Stages, each one compute pass:
 * 1. hints: one workgroup of 1,024 per column; per polynomial of the column (rows / 2,048), its forward transform
 *    modulo the three primes in workgroup memory, the products with the prepared query masks accumulated in
 *    registers; one inverse transform, the CRT, the reduction modulo q.
 * 2. aggregation: the gather of the masks into bit-reversed generating rows, nine ring-FFT stages exact in i64, one
 *    that widens to 128 bits, and the last fused with the shift, the mask and the reorder into the chains' slots.
 * 3. chains: one workgroup of 1,024 per chain, all 1,023 steps in one dispatch, a storage barrier between steps. A
 *    step: the gadget digits of a slot (written straight into the compile input), the automorphism g^-1, the exact
 *    dot product with the prepared gadget masks, the automorphism g, the add into the next slot.
 * 4. compile: per limb, eleven ring-FFT stages in i32, then the negacyclic shift into the matrix through a 32 × 32
 *    tile in workgroup memory. Row-major i32 words (2,048 rows of 4,096), the layout the CPU build stores.
 *
 * The device needs 1,024 invocations per workgroup and 32 KB of workgroup memory (`BLOCK_BUILD_LIMITS`); WebGPU's
 * defaults are 256 and 16 KB. Apple GPUs and NVIDIA GPUs through Vulkan have them.
 */
import { TimestampSpan, writeView } from "./device.ts";

/** Limits a device must be opened with for the build's kernels. */
export const BLOCK_BUILD_LIMITS = {
  maxComputeInvocationsPerWorkgroup: 1024,
  maxComputeWorkgroupSizeX: 1024,
  maxComputeWorkgroupStorageSize: 32768,
} as const;

/** Whether an adapter can run the build. */
export function blockBuildSupported(limits: GPUSupportedLimits): boolean {
  return (Object.keys(BLOCK_BUILD_LIMITS) as (keyof typeof BLOCK_BUILD_LIMITS)[]).every((k) => limits[k] >= BLOCK_BUILD_LIMITS[k]);
}

export interface BlockBuildHeader {
  format: string;
  d: number;
  logd: number;
  rows: number;
  polys: number;
  qBits: number;
  gadgetBits: number;
  droppedBits: number;
  steps: number;
  i64Stages: number;
  missingRows: number;
  primes: [number, number, number];
  inv01: number;
  inv01q: number;
  inv012: number;
  inv012q: number;
  p0ModP2: number;
  p0ModP2q: number;
  p01: [number, number];
  product: [number, number, number];
  half: [number, number, number];
  sections: { name: string; words: number }[];
}

/** The tables `gpu_build_prep` wrote for one table shape. */
export interface BlockBuildPrep {
  header: BlockBuildHeader;
  /** All sections, one after another. */
  words: Uint32Array;
  /** Each section's first word in `words`. */
  offsets: Record<string, number>;
}

export function parseBlockBuildPrep(bytes: Uint8Array): BlockBuildPrep {
  if (new TextDecoder().decode(bytes.subarray(0, 8)) !== "IPIRGBP1") throw new Error("not a gpu_build_prep file");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const headerLen = view.getUint32(8, true);
  const header: BlockBuildHeader = JSON.parse(new TextDecoder().decode(bytes.subarray(12, 12 + headerLen)));
  if (header.format !== "ipir-gpu-build-prep/1") throw new Error(`unknown prep format ${header.format}`);
  const total = header.sections.reduce((n, s) => n + s.words, 0);
  const start = 12 + headerLen;
  if (bytes.byteLength !== start + 4 * total) throw new Error("gpu_build_prep file: wrong length");
  const words = new Uint32Array(total);
  new Uint8Array(words.buffer).set(bytes.subarray(start));
  const offsets: Record<string, number> = {};
  let at = 0;
  for (const s of header.sections) {
    offsets[s.name] = at;
    at += s.words;
  }
  if (header.d !== 2048 || header.logd !== 11) throw new Error("the kernels are written for d = 2,048");
  if (header.i64Stages + 2 !== header.logd) throw new Error("the kernels expect all but two aggregation stages in i64");
  return { header, words, offsets };
}

const u = (x: number) => `${x >>> 0}u`;

/** Constants and scalar helpers shared by the kernels. */
function commonWgsl(prep: BlockBuildPrep): string {
  const h = prep.header;
  const o = (name: string): number => {
    const at = prep.offsets[name];
    if (at === undefined) throw new Error(`gpu_build_prep file: no section ${name}`);
    return at;
  };
  return /* wgsl */ `
const D = ${u(h.d)};
const LOGD = ${u(h.logd)};
const HALF_D = ${u(h.d / 2)};
const ROWS = ${u(h.rows)};
const POLYS = ${u(h.polys)};
const STEPS = ${u(h.steps)};
const BITS = ${u(h.gadgetBits)};
const DROPPED = ${u(h.droppedBits)};
const QMASK_HI = ${u(2 ** (h.qBits - 32) - 1)};
const P0 = ${u(h.primes[0])};
const P1 = ${u(h.primes[1])};
const P2 = ${u(h.primes[2])};
const INV01 = ${u(h.inv01)};
const INV01Q = ${u(h.inv01q)};
const INV012 = ${u(h.inv012)};
const INV012Q = ${u(h.inv012q)};
const P0_MOD_P2 = ${u(h.p0ModP2)};
const P0_MOD_P2Q = ${u(h.p0ModP2q)};
const P01_LO = ${u(h.p01[0])};
const P01_HI = ${u(h.p01[1])};
const PROD0 = ${u(h.product[0])};
const PROD1 = ${u(h.product[1])};
const HALF0 = ${u(h.half[0])};
const HALF1 = ${u(h.half[1])};
const HALF2 = ${u(h.half[2])};
// First words of the sections of K. Per prime, the transform tables are fwd, fwdq, inv, invq (D words each).
const K_TABLES = ${u(o("tables"))};
const K_HINT_T = ${u(o("hintT"))};
const K_HINT_Q = ${u(o("hintQ"))};
const K_GADGET_T = ${u(o("gadgetT"))};
const K_GADGET_Q = ${u(o("gadgetQ"))};
const K_EXPS = ${u(o("exps"))};
const K_INV_EXPS = ${u(o("invExps"))};
const K_PHAT_ROWS = ${u(o("phatRows"))};
const K_SLOT_OF_ROW = ${u(o("slotOfRow"))};
const K_MISSING_ROWS = ${u(o("missingRows"))};

// The high word of a 32 x 32 product, from 16-bit halves.
fn mulhi32(a: u32, b: u32) -> u32 {
  let al = a & 0xffffu; let ah = a >> 16u; let bl = b & 0xffffu; let bh = b >> 16u;
  let ll = al * bl; let lh = al * bh; let hl = ah * bl; let hh = ah * bh;
  let mid = (ll >> 16u) + (lh & 0xffffu) + (hl & 0xffffu);
  return hh + (lh >> 16u) + (hl >> 16u) + (mid >> 16u);
}
// x * w mod p in [0, 2p), wq = floor(w 2^32 / p).
fn mul_shoup32(x: u32, w: u32, wq: u32, p: u32) -> u32 {
  return x * w - mulhi32(x, wq) * p;
}
fn red32(x: u32, m: u32) -> u32 {
  return select(x, x - m, x >= m);
}
fn neg64(a: vec2<u32>) -> vec2<u32> {
  return vec2<u32>(0u - a.x, (0u - a.y) - select(0u, 1u, a.x != 0u));
}
fn add64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  let lo = a.x + b.x;
  return vec2<u32>(lo, a.y + b.y + select(0u, 1u, lo < a.x));
}
fn sub64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  return vec2<u32>(a.x - b.x, a.y - b.y - select(0u, 1u, a.x < b.x));
}
fn bitrev(x: u32) -> u32 {
  return reverseBits(x) >> (32u - LOGD);
}`;
}

/** The CRT of three residues, centered and reduced modulo q: the 54-bit word as (low, high). */
const CRT_WGSL = /* wgsl */ `
fn crt3(r0: u32, r1: u32, r2: u32) -> vec2<u32> {
  let t1 = red32(mul_shoup32(red32(r1 + P1 - red32(r0, P1), P1), INV01, INV01Q, P1), P1);
  let a = red32(mul_shoup32(t1, P0_MOD_P2, P0_MOD_P2Q, P2), P2);
  let x2 = red32(red32(r0, P2) + a, P2);
  let t2 = red32(mul_shoup32(red32(r2 + P2 - x2, P2), INV012, INV012Q, P2), P2);
  // x = r0 + p0 t1 + (p0 p1) t2 < 2^90, in three words (s0, s1, s2).
  let m_lo = P0 * t1;
  let m_hi = mulhi32(P0, t1);
  let w0 = r0 + m_lo;
  let w1 = m_hi + select(0u, 1u, w0 < m_lo);
  let a_lo = P01_LO * t2;
  let a_hi = mulhi32(P01_LO, t2);
  let b_lo = P01_HI * t2;
  let b_hi = mulhi32(P01_HI, t2);
  let s0 = w0 + a_lo;
  let c0 = select(0u, 1u, s0 < a_lo);
  let mid = a_hi + b_lo;
  let cm = select(0u, 1u, mid < a_hi);
  let s1a = w1 + mid;
  let c1a = select(0u, 1u, s1a < mid);
  let s1 = s1a + c0;
  let c1b = select(0u, 1u, s1 < c0);
  let s2 = b_hi + cm + c1a + c1b;
  // Above half the product: the integer is x - product. Only its low 64 bits are needed.
  let above = s2 > HALF2 || (s2 == HALF2 && (s1 > HALF1 || (s1 == HALF1 && s0 > HALF0)));
  let lo = select(s0, s0 - PROD0, above);
  let hi = select(s1, s1 - PROD1 - select(0u, 1u, s0 < PROD0), above);
  return vec2<u32>(lo, hi & QMASK_HI);
}`;

/**
 * One butterfly of a transform in workgroup memory. `o`: the buffer's first word in `buf`; `b`: the butterfly (the
 * invocation); `lt`: log2 of the stage's half-size; `tab`: the prime's tables in K.
 */
const NTT_WGSL = /* wgsl */ `
fn fwd_bfly(o: u32, b: u32, lt: u32, tab: u32, p: u32) {
  let t = 1u << lt;
  let i = b >> lt;
  let ix = o + (i << (lt + 1u)) + (b & (t - 1u));
  let iy = ix + t;
  let m = HALF_D >> lt;
  let w = K[tab + m + i];
  let wq = K[tab + D + m + i];
  let two_p = 2u * p;
  let cx = red32(buf[ix], two_p);
  let v = mul_shoup32(buf[iy], w, wq, p);
  buf[ix] = cx + v;
  buf[iy] = cx + two_p - v;
}
fn inv_bfly(o: u32, b: u32, lt: u32, tab: u32, p: u32) {
  let t = 1u << lt;
  let i = b >> lt;
  let ix = o + (i << (lt + 1u)) + (b & (t - 1u));
  let iy = ix + t;
  let h = HALF_D >> lt;
  let w = K[tab + 2u * D + h + i];
  let wq = K[tab + 3u * D + h + i];
  let two_p = 2u * p;
  let x = buf[ix];
  let y = buf[iy];
  buf[ix] = red32(x + y, two_p);
  buf[iy] = mul_shoup32(x + two_p - y, w, wq, p);
}
fn prime(j: u32) -> u32 {
  return select(select(P2, P1, j == 1u), P0, j == 0u);
}`;

/**
 * Ownership in the transforms (as in the Metal kernels): a forward transform's last stage has invocation b write
 * elements 2b and 2b + 1, and an inverse transform's first stage reads them; its last stage writes b and b + D/2.
 * The kernels keep those elements per invocation between transforms, so the products and the CRT need no barrier.
 */
function hintsWgsl(prep: BlockBuildPrep): string {
  return /* wgsl */ `${commonWgsl(prep)}
@group(0) @binding(0) var<storage, read> cells: array<u32>;        // 16-bit cells, column-major, two per word
@group(0) @binding(1) var<storage, read> K: array<u32>;
@group(0) @binding(2) var<storage, read_write> masks: array<u32>;  // [col][D] words of 54 bits, (low, high)
var<workgroup> buf: array<u32, ${3 * prep.header.d}>;               // three primes
${NTT_WGSL}
${CRT_WGSL}
fn cell(i: u32) -> u32 {
  return (cells[i >> 1u] >> ((i & 1u) * 16u)) & 0xffffu;
}

@compute @workgroup_size(1024)
fn hint_columns(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
  let col = wg.x;
  let e0 = 2u * tid;
  let e1 = e0 + 1u;
  var acc: array<u32, 6>;
  for (var poly = 0u; poly < POLYS; poly++) {
    let src = col * ROWS + poly * D;
    let x0 = cell(src + tid);
    let x1 = cell(src + tid + HALF_D);
    workgroupBarrier();   // the previous products have read their elements
    for (var j = 0u; j < 3u; j++) {
      buf[j * D + tid] = x0;
      buf[j * D + tid + HALF_D] = x1;
    }
    for (var s = 0u; s < LOGD; s++) {
      workgroupBarrier();
      for (var j = 0u; j < 3u; j++) {
        fwd_bfly(j * D, tid, LOGD - 1u - s, K_TABLES + j * 4u * D, prime(j));
      }
    }
    for (var j = 0u; j < 3u; j++) {
      let p = prime(j);
      let w = K_HINT_T + (j * POLYS + poly) * D;
      let wq = K_HINT_Q + (j * POLYS + poly) * D;
      let v0 = red32(red32(buf[j * D + e0], 2u * p), p);
      let v1 = red32(red32(buf[j * D + e1], 2u * p), p);
      acc[2u * j] = red32(acc[2u * j] + red32(mul_shoup32(v0, K[w + e0], K[wq + e0], p), p), p);
      acc[2u * j + 1u] = red32(acc[2u * j + 1u] + red32(mul_shoup32(v1, K[w + e1], K[wq + e1], p), p), p);
    }
  }
  for (var j = 0u; j < 3u; j++) {
    buf[j * D + e0] = acc[2u * j];
    buf[j * D + e1] = acc[2u * j + 1u];
  }
  for (var lt = 0u; lt < LOGD; lt++) {
    if (lt > 0u) { workgroupBarrier(); }
    for (var j = 0u; j < 3u; j++) {
      inv_bfly(j * D, tid, lt, K_TABLES + j * 4u * D, prime(j));
    }
  }
  for (var k = 0u; k < 2u; k++) {
    let i = tid + k * HALF_D;
    let v = crt3(red32(buf[i], P0), red32(buf[D + i], P1), red32(buf[2u * D + i], P2));
    let at = 2u * (col * D + i);
    masks[at] = v.x;
    masks[at + 1u] = v.y;
  }
}`;
}

/** The butterfly rows of one ring-FFT stage (aggregation and compile). */
const STAGE_ROWS_WGSL = /* wgsl */ `
struct Len { len: u32 }
@group(0) @binding(2) var<uniform> stage: Len;
struct StageRows { ru: u32, rv: u32, vi: u32, negate: bool }
// The butterfly of a length-len stage at coefficient i of pair: rows u = block + kk, v = u + len/2,
// rot = 2 (D/len) kk; u' = u + X^rot v, v' = u - X^rot v (negacyclic).
fn stage_rows(len: u32, pair: u32, i: u32) -> StageRows {
  let half_len = len >> 1u;
  let block = (pair / half_len) * len;
  let kk = pair % half_len;
  let ru = block + kk;
  let rot = 2u * (D / len) * kk;
  let negate = i < rot;
  return StageRows(ru, ru + half_len, select(i - rot, D - rot + i, negate), negate);
}`;

function aggregationWgsl(prep: BlockBuildPrep, kernel: "gather" | "stage64" | "widen" | "last"): string {
  const common = commonWgsl(prep);
  if (kernel === "gather") {
    return /* wgsl */ `${common}
@group(0) @binding(0) var<storage, read> masks: array<u32>;     // [col][D] (low, high)
@group(0) @binding(1) var<storage, read_write> a: array<u32>;   // [D][D] i64 as (low, high)
// polys[k][(j + k) mod D] = +-(k == 0 ? m_j[0] : -m_j[D - k]), stored at row bitrev(k).
@compute @workgroup_size(256)
fn agg_gather(@builtin(global_invocation_id) gid: vec3<u32>) {
  let m = gid.x;
  let k = gid.y;
  let j = (m - k) & (D - 1u);
  let at_mask = 2u * (j * D + select(D - k, 0u, k == 0u));
  var c = vec2<u32>(masks[at_mask], masks[at_mask + 1u]);
  if ((k != 0u) != (j + k >= D)) { c = neg64(c); }
  let at = 2u * (bitrev(k) * D + m);
  a[at] = c.x;
  a[at + 1u] = c.y;
}`;
  }
  if (kernel === "stage64") {
    return /* wgsl */ `${common}
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
${STAGE_ROWS_WGSL}
// A stage whose public bound len (q - 1) fits i64, exact in i64 as the CPU does.
@compute @workgroup_size(256)
fn agg_stage64(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = stage_rows(stage.len, gid.y, gid.x);
  let iu = 2u * (r.ru * D + gid.x);
  let it = 2u * (r.rv * D + r.vi);
  let x = vec2<u32>(src[iu], src[iu + 1u]);
  var t = vec2<u32>(src[it], src[it + 1u]);
  if (r.negate) { t = neg64(t); }
  let nu = add64(x, t);
  let nv = sub64(x, t);
  let iv = 2u * (r.rv * D + gid.x);
  dst[iu] = nu.x;
  dst[iu + 1u] = nu.y;
  dst[iv] = nv.x;
  dst[iv + 1u] = nv.y;
}`;
  }
  const wide = /* wgsl */ `
fn neg128(a: vec4<u32>) -> vec4<u32> {
  let x = 0u - a.x;
  let b1 = select(0u, 1u, a.x != 0u);
  let y = (0u - a.y) - b1;
  let b2 = select(0u, 1u, a.y != 0u || b1 == 1u);
  let z = (0u - a.z) - b2;
  let b3 = select(0u, 1u, a.z != 0u || b2 == 1u);
  return vec4<u32>(x, y, z, (0u - a.w) - b3);
}
fn add128(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
  let x = a.x + b.x;
  let c1 = select(0u, 1u, x < a.x);
  let y0 = a.y + b.y;
  let y = y0 + c1;
  let c2 = select(0u, 1u, y0 < a.y) + select(0u, 1u, y < y0);
  let z0 = a.z + b.z;
  let z = z0 + c2;
  let c3 = select(0u, 1u, z0 < a.z) + select(0u, 1u, z < z0);
  return vec4<u32>(x, y, z, a.w + b.w + c3);
}`;
  if (kernel === "widen") {
    return /* wgsl */ `${common}
@group(0) @binding(0) var<storage, read> src: array<u32>;         // i64 as (low, high)
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;   // i128, four words
${STAGE_ROWS_WGSL}
${wide}
fn widen(x: vec2<u32>) -> vec4<u32> {
  let s = select(0u, 0xffffffffu, (x.y >> 31u) == 1u);
  return vec4<u32>(x.x, x.y, s, s);
}
// The first stage past the i64 bound: reads i64, writes i128.
@compute @workgroup_size(256)
fn agg_widen(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = stage_rows(stage.len, gid.y, gid.x);
  let iu = 2u * (r.ru * D + gid.x);
  let it = 2u * (r.rv * D + r.vi);
  let x = widen(vec2<u32>(src[iu], src[iu + 1u]));
  var t = widen(vec2<u32>(src[it], src[it + 1u]));
  if (r.negate) { t = neg128(t); }
  let nu = add128(x, t);
  let nv = add128(x, neg128(t));
  let ou = 4u * (r.ru * D + gid.x);
  let ov = 4u * (r.rv * D + gid.x);
  dst[ou] = nu.x; dst[ou + 1u] = nu.y; dst[ou + 2u] = nu.z; dst[ou + 3u] = nu.w;
  dst[ov] = nv.x; dst[ov + 1u] = nv.y; dst[ov + 2u] = nv.z; dst[ov + 3u] = nv.w;
}`;
  }
  return /* wgsl */ `${common}
@group(0) @binding(0) var<storage, read> src: array<u32>;           // i128, four words
@group(0) @binding(1) var<storage, read_write> slots: array<u32>;   // [slot][D] (low, high)
@group(0) @binding(2) var<storage, read> K: array<u32>;
${wide}
// (x >> LOGD) mod q.
fn agg_final(x: vec4<u32>) -> vec2<u32> {
  return vec2<u32>((x.x >> LOGD) | (x.y << (32u - LOGD)), ((x.y >> LOGD) | (x.z << (32u - LOGD))) & QMASK_HI);
}
// The last stage (len = D) in i128, with the final shift and mask, written straight to each row's slot.
@compute @workgroup_size(256)
fn agg_last(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let kk = gid.y;             // len = D: one block, rows kk and kk + D/2, rot = 2 kk
  let rot = 2u * kk;
  let negate = i < rot;
  let vi = select(i - rot, D - rot + i, negate);
  let iu = 4u * (kk * D + i);
  let it = 4u * ((kk + HALF_D) * D + vi);
  let x = vec4<u32>(src[iu], src[iu + 1u], src[iu + 2u], src[iu + 3u]);
  var t = vec4<u32>(src[it], src[it + 1u], src[it + 2u], src[it + 3u]);
  if (negate) { t = neg128(t); }
  let nu = agg_final(add128(x, t));
  let nv = agg_final(add128(x, neg128(t)));
  let ou = 2u * (K[K_SLOT_OF_ROW + kk] * D + i);
  let ov = 2u * (K[K_SLOT_OF_ROW + kk + HALF_D] * D + i);
  slots[ou] = nu.x; slots[ou + 1u] = nu.y;
  slots[ov] = nv.x; slots[ov + 1u] = nv.y;
}`;
}

/**
 * The two collapse chains. Four buffers of workgroup memory (32 KB): B0, B1 = prime 0's two limbs, B2, B3 = prime
 * 1's. Per step:
 *   A. scatter both limbs' rotated digits for primes 0 and 1; forward transform of B0..B3; the products summed into
 *      B0 (prime 0) and B2 (prime 1) at each invocation's own elements.
 *   B. scatter prime 2's digits into B1, B3; inverse transform of B0, B2 alongside the forward transform of B1, B3;
 *      prime 2's products into B1.
 *   C. inverse transform of B1; CRT; automorphism g; add into the next slot.
 */
function chainsWgsl(prep: BlockBuildPrep): string {
  const h = prep.header;
  return /* wgsl */ `${commonWgsl(prep)}
@group(0) @binding(0) var<storage, read_write> slots: array<u32>;   // [slot][D] (low, high)
@group(0) @binding(1) var<storage, read_write> phat: array<i32>;    // the compile input, [limb][row][D]
@group(0) @binding(2) var<storage, read> K: array<u32>;
var<workgroup> buf: array<u32, ${4 * h.d}>;
${NTT_WGSL}
${CRT_WGSL}
const Z_MASK = ${u(2 ** h.gadgetBits - 1)};
const HALF_Z = ${u(2 ** (h.gadgetBits - 1))};
const ROUNDING = ${u(2 ** (h.droppedBits - 1))};
const TAB0 = K_TABLES;
const TAB1 = K_TABLES + 4u * D;
const TAB2 = K_TABLES + 8u * D;
// A centered digit modulo p.
fn lift(v: i32, p: u32) -> u32 {
  return select(u32(v), u32(i32(p) + v), v < 0);
}

@compute @workgroup_size(1024)
fn trace_chains(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
  let chain = wg.x;
  let base = chain * HALF_D;
  let e0 = 2u * tid;
  let e1 = e0 + 1u;
  for (var step = 0u; step < STEPS; step++) {
    let idx = HALF_D - 1u - step;
    let gstep = chain * STEPS + step;
    let g = K[K_EXPS + gstep];
    let ginv = K[K_INV_EXPS + gstep];
    let prow = K[K_PHAT_ROWS + gstep];
    let src = (base + idx) * D;
    let dst = (base + idx - 1u) * D;
    var dig: array<i32, 4>;     // [limb][k]
    var pos: array<u32, 2>;
    var neg: array<bool, 2>;
    for (var k = 0u; k < 2u; k++) {
      let i = tid + k * HALF_D;
      // x = (slot word + rounding) >> dropped bits, below 2^(qBits + 1 - dropped) (two words).
      let s_lo = slots[2u * (src + i)];
      let s_hi = slots[2u * (src + i) + 1u];
      let r_lo = s_lo + ROUNDING;
      let r_hi = s_hi + select(0u, 1u, r_lo < s_lo);
      let x_lo = (r_lo >> DROPPED) | (r_hi << (32u - DROPPED));
      let x_hi = r_hi >> DROPPED;
      // The centered digit of the low limb, then x = (x - digit) >> bits, which fits one word.
      let d0 = i32((x_lo + HALF_Z) & Z_MASK) - i32(HALF_Z);
      let d0u = u32(d0);
      let y_lo = x_lo - d0u;
      let y_hi = x_hi - select(0u, 0xffffffffu, d0 < 0) - select(0u, 1u, x_lo < d0u);
      let x1 = (y_lo >> BITS) | (y_hi << (32u - BITS));
      let d1 = i32((x1 + HALF_Z) & Z_MASK) - i32(HALF_Z);
      dig[k] = d0;
      dig[2u + k] = d1;
      phat[prow * D + i] = d0;
      phat[(D + prow) * D + i] = d1;
      let e = (i * ginv) & (2u * D - 1u);
      pos[k] = e & (D - 1u);
      neg[k] = e >= D;
    }
    // A.
    for (var k = 0u; k < 2u; k++) {
      let v0 = select(dig[k], -dig[k], neg[k]);
      let v1 = select(dig[2u + k], -dig[2u + k], neg[k]);
      buf[pos[k]] = lift(v0, P0);
      buf[D + pos[k]] = lift(v1, P0);
      buf[2u * D + pos[k]] = lift(v0, P1);
      buf[3u * D + pos[k]] = lift(v1, P1);
    }
    for (var s = 0u; s < LOGD; s++) {
      let lt = LOGD - 1u - s;
      workgroupBarrier();
      fwd_bfly(0u, tid, lt, TAB0, P0);
      fwd_bfly(D, tid, lt, TAB0, P0);
      fwd_bfly(2u * D, tid, lt, TAB1, P1);
      fwd_bfly(3u * D, tid, lt, TAB1, P1);
    }
    for (var k = 0u; k < 2u; k++) {
      let e = e0 + k;
      let a0 = red32(red32(buf[e], 2u * P0), P0);
      let a1 = red32(red32(buf[D + e], 2u * P0), P0);
      var s = red32(mul_shoup32(a0, K[K_GADGET_T + e], K[K_GADGET_Q + e], P0), P0);
      s = red32(s + red32(mul_shoup32(a1, K[K_GADGET_T + D + e], K[K_GADGET_Q + D + e], P0), P0), P0);
      let b0 = red32(red32(buf[2u * D + e], 2u * P1), P1);
      let b1 = red32(red32(buf[3u * D + e], 2u * P1), P1);
      var s1 = red32(mul_shoup32(b0, K[K_GADGET_T + 2u * D + e], K[K_GADGET_Q + 2u * D + e], P1), P1);
      s1 = red32(s1 + red32(mul_shoup32(b1, K[K_GADGET_T + 3u * D + e], K[K_GADGET_Q + 3u * D + e], P1), P1), P1);
      buf[e] = s;
      buf[2u * D + e] = s1;
    }
    // B. The inverse's first stage of B0 and B2 touches only own elements; B1 and B3 are free once every
    // invocation has read them.
    inv_bfly(0u, tid, 0u, TAB0, P0);
    inv_bfly(2u * D, tid, 0u, TAB1, P1);
    workgroupBarrier();
    for (var k = 0u; k < 2u; k++) {
      let v0 = select(dig[k], -dig[k], neg[k]);
      let v1 = select(dig[2u + k], -dig[2u + k], neg[k]);
      buf[D + pos[k]] = lift(v0, P2);
      buf[3u * D + pos[k]] = lift(v1, P2);
    }
    for (var s = 0u; s < LOGD; s++) {
      workgroupBarrier();
      if (s + 1u < LOGD) {
        inv_bfly(0u, tid, s + 1u, TAB0, P0);
        inv_bfly(2u * D, tid, s + 1u, TAB1, P1);
      }
      fwd_bfly(D, tid, LOGD - 1u - s, TAB2, P2);
      fwd_bfly(3u * D, tid, LOGD - 1u - s, TAB2, P2);
    }
    for (var k = 0u; k < 2u; k++) {
      let e = e0 + k;
      let a0 = red32(red32(buf[D + e], 2u * P2), P2);
      let a1 = red32(red32(buf[3u * D + e], 2u * P2), P2);
      var s = red32(mul_shoup32(a0, K[K_GADGET_T + 4u * D + e], K[K_GADGET_Q + 4u * D + e], P2), P2);
      s = red32(s + red32(mul_shoup32(a1, K[K_GADGET_T + 5u * D + e], K[K_GADGET_Q + 5u * D + e], P2), P2), P2);
      buf[D + e] = s;
    }
    // C.
    for (var lt = 0u; lt < LOGD; lt++) {
      if (lt > 0u) { workgroupBarrier(); }
      inv_bfly(D, tid, lt, TAB2, P2);
    }
    for (var k = 0u; k < 2u; k++) {
      let i = tid + k * HALF_D;
      var val = crt3(red32(buf[i], P0), red32(buf[2u * D + i], P1), red32(buf[D + i], P2));
      let e = (i * g) & (2u * D - 1u);
      let at = 2u * (dst + (e & (D - 1u)));
      if (e >= D) {
        val = neg64(val);
        val.y = val.y & QMASK_HI;
      }
      let sum = add64(vec2<u32>(slots[at], slots[at + 1u]), val);
      slots[at] = sum.x;
      slots[at + 1u] = sum.y & QMASK_HI;
    }
    storageBarrier();
    workgroupBarrier();
  }
}`;
}

function compileWgsl(prep: BlockBuildPrep, kernel: "zero" | "stage" | "matrix"): string {
  const common = commonWgsl(prep);
  if (kernel === "zero") {
    return /* wgsl */ `${common}
@group(0) @binding(0) var<storage, read_write> phat: array<i32>;
@group(0) @binding(1) var<storage, read> K: array<u32>;
// Rows of the compile input that no chain step writes (exponents the chains do not produce), set to zero.
@compute @workgroup_size(256)
fn zero_rows(@builtin(global_invocation_id) gid: vec3<u32>) {
  phat[(gid.z * D + K[K_MISSING_ROWS + gid.y]) * D + gid.x] = 0;
}`;
  }
  if (kernel === "stage") {
    return /* wgsl */ `${common}
@group(0) @binding(0) var<storage, read> src: array<i32>;         // [limb][D][D]
@group(0) @binding(1) var<storage, read_write> dst: array<i32>;
${STAGE_ROWS_WGSL}
// One stage of the ring FFT at powers of X^2 over the generating polynomial (rows in bit-reversed order). The
// digits are below 2^(bits - 1) in magnitude and the transform grows them at most D times, so every value is an
// exact i32 (gpu_build_prep checks the bound).
@compute @workgroup_size(256)
fn cmp_stage(@builtin(global_invocation_id) gid: vec3<u32>) {
  let off = gid.z * D * D;
  let r = stage_rows(stage.len, gid.y, gid.x);
  let x = src[off + r.ru * D + gid.x];
  let y = src[off + r.rv * D + r.vi];
  let t = select(y, -y, r.negate);
  dst[off + r.ru * D + gid.x] = x + t;
  dst[off + r.rv * D + gid.x] = x - t;
}`;
  }
  return /* wgsl */ `${common}
@group(0) @binding(0) var<storage, read> a: array<i32>;             // [limb][D][D]
@group(0) @binding(1) var<storage, read_write> matrix: array<i32>;  // [D][2 D], row-major
var<workgroup> tile: array<i32, 1056>;                              // 32 x 33
// matrix[r][limb D + j] = (r < j ? -x : x), x = a_limb[j][(r - j) mod D]. A 32 x 32 tile goes through workgroup
// memory so that the reads (along a row of a) and the writes (along a row of the matrix) are both contiguous.
@compute @workgroup_size(32, 8)
fn cmp_matrix(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let j0 = wg.x * 32u;
  let r0 = wg.y * 32u;
  let off = wg.z * D * D;
  for (var s = 0u; s < 4u; s++) {
    let jl = lid.y + 8u * s;
    let j = j0 + jl;
    let r = r0 + lid.x;
    let x = a[off + j * D + ((r - j) & (D - 1u))];
    tile[jl * 33u + lid.x] = select(x, -x, r < j);
  }
  workgroupBarrier();
  for (var s = 0u; s < 4u; s++) {
    let rl = lid.y + 8u * s;
    matrix[(r0 + rl) * (2u * D) + wg.z * D + j0 + lid.x] = tile[lid.x * 33u + rl];
  }
}`;
}

export type BlockBuildStage = "hints" | "aggregation" | "chains" | "compile";
export const BLOCK_BUILD_STAGES: readonly BlockBuildStage[] = ["hints", "aggregation", "chains", "compile"];

/** The build of one table shape's column blocks: pipelines, constant tables and working buffers. */
export class GpuBlockBuild {
  readonly device: GPUDevice;
  readonly header: BlockBuildHeader;
  /** The block's cells, 16 bits each, column-major (`rows × 2,048`). */
  readonly cells: GPUBuffer;
  /** Output: the hint mask of each column, 2,048 words of 54 bits as (low, high) u32. */
  readonly masks: GPUBuffer;
  /** Output after the chains: slot 0 is the mask, slot 1,024 the other mask; words as (low, high) u32. */
  readonly slots: GPUBuffer;
  /** Output: the compiled matrix, 2,048 rows of 4,096 i32 words, row-major. */
  readonly matrix: GPUBuffer;
  /** The compile input the chains write (digits): `[limb][row][2,048]` i32. Kept for checks. */
  readonly digits: GPUBuffer;
  readonly #passes: Record<BlockBuildStage, { pipeline: GPUComputePipeline; group: GPUBindGroup; workgroups: [number, number, number] }[]>;
  readonly #buffers: GPUBuffer[] = [];

  constructor(device: GPUDevice, prep: BlockBuildPrep) {
    for (const [k, v] of Object.entries(BLOCK_BUILD_LIMITS)) {
      const have = device.limits[k as keyof typeof BLOCK_BUILD_LIMITS];
      if (have < v) throw new Error(`the GPU block build needs ${k} >= ${v}; this device was opened with ${have}`);
    }
    this.device = device;
    const h = this.header = prep.header;
    const d = h.d;
    const dd = d * d;
    const STORAGE = GPUBufferUsage.STORAGE;
    const buffer = (size: number, usage: number) => {
      const b = device.createBuffer({ size, usage });
      this.#buffers.push(b);
      return b;
    };
    const k = buffer(prep.words.byteLength, STORAGE | GPUBufferUsage.COPY_DST);
    writeView(device, k, 0, prep.words);
    this.cells = buffer(h.rows * d * 2, STORAGE | GPUBufferUsage.COPY_DST);
    this.masks = buffer(dd * 8, STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    this.slots = buffer(dd * 8, STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    this.matrix = buffer(2 * dd * 4, STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    const agg64 = [buffer(dd * 8, STORAGE), buffer(dd * 8, STORAGE)];
    const agg128 = buffer(dd * 16, STORAGE);
    const phat = [buffer(2 * dd * 4, STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST), buffer(2 * dd * 4, STORAGE)];
    this.digits = phat[0]!;
    // One uniform word per ring-FFT stage: its length.
    const lens: GPUBuffer[] = [];
    for (let s = 0; s < h.logd; s++) {
      const b = buffer(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      writeView(device, b, 0, new Uint32Array([2 << s, 0, 0, 0]));
      lens.push(b);
    }
    const pipeline = (code: string, entryPoint: string) =>
      device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint } });
    const group = (p: GPUComputePipeline, buffers: GPUBuffer[]) =>
      device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: buffers.map((b, binding) => ({ binding, resource: { buffer: b } })) });
    const pass = (p: GPUComputePipeline, buffers: GPUBuffer[], workgroups: [number, number, number]) => ({ pipeline: p, group: group(p, buffers), workgroups });

    const hint = pipeline(hintsWgsl(prep), "hint_columns");
    const zero = pipeline(compileWgsl(prep, "zero"), "zero_rows");
    const gather = pipeline(aggregationWgsl(prep, "gather"), "agg_gather");
    const stage64 = pipeline(aggregationWgsl(prep, "stage64"), "agg_stage64");
    const widen = pipeline(aggregationWgsl(prep, "widen"), "agg_widen");
    const last = pipeline(aggregationWgsl(prep, "last"), "agg_last");
    const chains = pipeline(chainsWgsl(prep), "trace_chains");
    const cmpStage = pipeline(compileWgsl(prep, "stage"), "cmp_stage");
    const cmpMatrix = pipeline(compileWgsl(prep, "matrix"), "cmp_matrix");

    const x = d / 256;
    const hints = [pass(hint, [this.cells, k, this.masks], [d, 1, 1])];
    if (h.missingRows > 0) hints.push(pass(zero, [phat[0]!, k], [x, h.missingRows, 2]));
    // Aggregation: the gather writes agg64[0]; i64 stage s reads agg64[s % 2].
    const aggregation = [pass(gather, [this.masks, agg64[0]!], [x, d, 1])];
    for (let s = 0; s < h.i64Stages; s++) aggregation.push(pass(stage64, [agg64[s % 2]!, agg64[(s + 1) % 2]!, lens[s]!], [x, d / 2, 1]));
    aggregation.push(pass(widen, [agg64[h.i64Stages % 2]!, agg128, lens[h.i64Stages]!], [x, d / 2, 1]));
    aggregation.push(pass(last, [agg128, this.slots, k], [x, d / 2, 1]));
    // Compile: stage s reads phat[s % 2]; the matrix reads the last stage's output.
    const compile: ReturnType<typeof pass>[] = [];
    for (let s = 0; s < h.logd; s++) compile.push(pass(cmpStage, [phat[s % 2]!, phat[(s + 1) % 2]!, lens[s]!], [x, d / 2, 2]));
    compile.push(pass(cmpMatrix, [phat[h.logd % 2]!, this.matrix], [d / 32, d / 32, 2]));
    this.#passes = { hints, aggregation, chains: [pass(chains, [this.slots, phat[0]!, k], [2, 1, 1])], compile };
  }

  /** Bytes of GPU memory the build holds besides the cells: tables, outputs and scratch. */
  get scratchBytes(): number {
    return this.#buffers.reduce((n, b) => n + b.size, 0) - this.cells.size;
  }

  /** Writes a block's cells (`rows × 2,048`, column-major). */
  uploadCells(cells: Uint16Array): void {
    if (cells.length !== this.header.rows * this.header.d) throw new Error("uploadCells: wrong number of cells");
    writeView(this.device, this.cells, 0, cells);
  }

  /** One stage as one compute pass. */
  encodeStage(enc: GPUCommandEncoder, stage: BlockBuildStage, timestampWrites?: GPUComputePassTimestampWrites): void {
    const pass = enc.beginComputePass(timestampWrites ? { timestampWrites } : {});
    for (const p of this.#passes[stage]) {
      pass.setPipeline(p.pipeline);
      pass.setBindGroup(0, p.group);
      pass.dispatchWorkgroups(...p.workgroups);
    }
    pass.end();
  }

  /** The whole build, four passes; `span` times all of them together. */
  encode(enc: GPUCommandEncoder, span?: TimestampSpan | null): void {
    BLOCK_BUILD_STAGES.forEach((stage, i) => {
      const first = i === 0, lastStage = i === BLOCK_BUILD_STAGES.length - 1;
      this.encodeStage(enc, stage, span ? (first ? span.begin : lastStage ? span.end : undefined) : undefined);
    });
  }

  /** Reads a buffer back (for checks; not part of a build's time). */
  async read(buffer: GPUBuffer, bytes = buffer.size): Promise<Uint8Array> {
    const staging = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(buffer, 0, staging, 0, bytes);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Uint8Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return out;
  }

  destroy(): void {
    for (const b of this.#buffers) b.destroy();
  }
}

/** `2^54`-word helpers for the digest: i32 matrix words as canonical words modulo q, (low, high) u32. */
function canonicalWords(matrix: Int32Array, qBits: number): Uint32Array {
  const out = new Uint32Array(matrix.length * 2);
  const high = 2 ** (qBits - 32) - 1;
  for (let i = 0; i < matrix.length; i++) {
    const x = matrix[i]!;
    out[2 * i] = x >>> 0;
    out[2 * i + 1] = x < 0 ? high : 0;
  }
  return out;
}

const hex = (b: ArrayBuffer) => [...new Uint8Array(b)].map((x) => x.toString(16).padStart(2, "0")).join("");

/**
 * `pack_digest`'s digest of a build's output words (`ipir/bench/src/bin/pack_digest.rs`): SHA-256 over two part
 * digests, one over the hint masks of every column, one over both public masks, the storage label, the dimensions
 * and q, and the compiled matrix's canonical words. Equal digests: every output word equals the CPU build's.
 */
export async function blockBuildDigest(build: GpuBlockBuild): Promise<{ all: string; hints: string; packing: string }> {
  const h = build.header;
  const d = h.d;
  const [masks, slots, matrixBytes] = await Promise.all([build.read(build.masks), build.read(build.slots), build.read(build.matrix)]);
  const matrix = canonicalWords(new Int32Array(matrixBytes.buffer), h.qBits);
  const label = new TextEncoder().encode("i32");
  const dims = new Uint32Array([d, 0, 2 * d, 0, 0, 2 ** (h.qBits - 32)]);
  const row = (slot: number) => slots.subarray(slot * d * 8, (slot + 1) * d * 8);
  const packing = new Uint8Array(2 * d * 8 + label.length + dims.byteLength + matrix.byteLength);
  let at = 0;
  for (const part of [row(0), row(d / 2), label, new Uint8Array(dims.buffer), new Uint8Array(matrix.buffer)]) {
    packing.set(part, at);
    at += part.byteLength;
  }
  const hintsDigest = await crypto.subtle.digest("SHA-256", masks as Uint8Array<ArrayBuffer>);
  const packingDigest = await crypto.subtle.digest("SHA-256", packing);
  const both = new Uint8Array(64);
  both.set(new Uint8Array(hintsDigest), 0);
  both.set(new Uint8Array(packingDigest), 32);
  return { all: hex(await crypto.subtle.digest("SHA-256", both)), hints: hex(hintsDigest), packing: hex(packingDigest) };
}
