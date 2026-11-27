//! The host's part of an IPIR+SP fetch on a GPU, whatever the device: how a call is laid out (`Tuning`, `Plan`),
//! the requests' packed words read into the kernels' layouts (`FetchShape::prepare`), the SHA-256 of each request,
//! and the answers' bytes from the device's 22-bit coefficients (`FetchShape::answers`). `gpu/ipir-wire.ts` states
//! the request and answer formats.
//!
//! Shared by the CUDA fetch (`cuda_fetch_native.rs`) and the Metal fetch (`metal/src/fetch.rs`), each of which
//! includes this file as a module (`#[path]`).
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::time::{Duration, Instant};

/// Columns of a column block, and rows of a block's compiled matrix.
pub const D: usize = 2048;
/// Key words of a request, and columns of a block's compiled matrix.
pub const KEY_WORDS: usize = 4096;
const KEY_BITS: usize = 54;
const QUERY_BITS: usize = 49;
const QUERY_SHIFT: u32 = 5;
const RESPONSE_BITS: usize = 22;
const KEY_OFFSET: usize = 36;
const SELECTION_OFFSET: usize = KEY_OFFSET + KEY_WORDS * KEY_BITS / 8;
const ANSWER_OFFSET: usize = 68;
/// Largest buffer of partial sums a call makes; past it, segments get longer.
const PARTIAL_BUDGET: usize = 256 << 20;

pub fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// Bytes of one request to a table of `rows` rows.
pub fn request_bytes(rows: usize) -> usize {
    SELECTION_OFFSET + (rows * QUERY_BITS).div_ceil(8)
}

/// Bytes of one answer from a table of `cols` columns.
pub fn answer_bytes(cols: usize) -> usize {
    ANSWER_OFFSET + (cols * RESPONSE_BITS).div_ceil(8)
}

/// How a call is laid out on the device. The best values differ between GPUs; `IPIR_CUDA_FETCH` (CUDA) and
/// `IPIR_METAL_FETCH` (Metal) override the defaults
/// (`ct=4,gt=4,it=4,pgt=4,ct1=8,it1=8,seg=0,pseg=0,so=1,po=1,target=262144,threads=256`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tuning {
    /// First pass: adjacent table columns one thread reads (1, 2, 4, 8), and requests one thread answers.
    pub ct: u32,
    pub gt: u32,
    /// Packing: adjacent matrix rows one thread reads, and requests one thread answers.
    pub it: u32,
    pub pgt: u32,
    /// The same two widths when a thread answers one request (a call of 1).
    pub ct1: u32,
    pub it1: u32,
    /// Rows per segment of the first pass (a divisor of the table's rows); 0: chosen per call for `target`.
    pub seg_rows: u32,
    /// Segments of key words in the packing (a power of two up to 16); 0: chosen per call for `target`.
    pub pack_segs: u32,
    /// Which grid dimension holds the column groups (see `grid_roles` in the kernels).
    pub scan_order: u32,
    pub pack_order: u32,
    /// Threads a launch should have at least: segments are split until it has.
    pub target: u32,
    /// Threads of a thread group in the Metal kernels (a CUDA block has 256).
    pub threads: u32,
}

impl Default for Tuning {
    fn default() -> Self {
        Tuning { ct: 4, gt: 4, it: 4, pgt: 4, ct1: 8, it1: 8, seg_rows: 0, pack_segs: 0, scan_order: 1, pack_order: 1, target: 262_144, threads: 256 }
    }
}

impl Tuning {
    /// The defaults with `name=value` pairs applied (`ct=8,gt=2`).
    pub fn parse(spec: &str) -> Result<Self, String> {
        let mut t = Tuning::default();
        for pair in spec.split(',').filter(|p| !p.is_empty()) {
            let (name, value) = pair.split_once('=').ok_or_else(|| format!("tuning: `{pair}` is not name=value"))?;
            let v: u32 = value.parse().map_err(|_| format!("tuning: `{value}` is not a number"))?;
            match name {
                "ct" => t.ct = v,
                "gt" => t.gt = v,
                "it" => t.it = v,
                "pgt" => t.pgt = v,
                "ct1" => t.ct1 = v,
                "it1" => t.it1 = v,
                "seg" => t.seg_rows = v,
                "pseg" => t.pack_segs = v,
                "so" => t.scan_order = v,
                "po" => t.pack_order = v,
                "target" => t.target = v,
                "threads" => t.threads = v,
                _ => return Err(format!("tuning: unknown name `{name}`")),
            }
        }
        let width = |v: u32| matches!(v, 1 | 2 | 4 | 8);
        let group = |v: u32| matches!(v, 1 | 2 | 4 | 8 | 16);
        if !(width(t.ct) && width(t.it) && width(t.ct1) && width(t.it1) && group(t.gt) && group(t.pgt)) {
            return Err("tuning: ct, it, ct1, it1 are 1, 2, 4 or 8; gt and pgt 1, 2, 4, 8 or 16".into());
        }
        if !(t.pack_segs == 0 || (t.pack_segs.is_power_of_two() && t.pack_segs <= 16)) || t.scan_order > 2 || t.pack_order > 2 {
            return Err("tuning: pseg is 0 or a power of two up to 16; so and po are 0, 1 or 2".into());
        }
        Ok(t)
    }

    /// The environment variable `name` (`IPIR_CUDA_FETCH`, `IPIR_METAL_FETCH`), or the defaults.
    pub fn from_env(name: &str) -> Result<Self, String> {
        Self::parse(&std::env::var(name).unwrap_or_default())
    }

    pub fn label(&self) -> String {
        format!(
            "ct={},gt={},it={},pgt={},ct1={},it1={},seg={},pseg={},so={},po={},target={},threads={}",
            self.ct, self.gt, self.it, self.pgt, self.ct1, self.it1, self.seg_rows, self.pack_segs, self.scan_order, self.pack_order, self.target, self.threads
        )
    }
}

/// The times of one call, in milliseconds.
#[derive(Clone, Copy, Debug, Default)]
pub struct CallTiming {
    /// The whole call on the host, from the requests' bytes to the answers' bytes.
    pub wall: f64,
    /// The requests' packed words read into 64-bit words.
    pub parse: f64,
    /// Those words copied to the device.
    pub copy_in: f64,
    /// The device's own time, from CUDA events: the first pass, the packing, the rounding, and all three.
    pub scan: f64,
    pub pack: f64,
    pub finish: f64,
    pub device: f64,
    /// SHA-256 of every request, on the host while the device works.
    pub hash: f64,
    /// The wait for the device after the hashing, and the coefficients copied back.
    pub copy_out: f64,
    /// The answers' bytes: head and the coefficients packed at 22 bits.
    pub serialize: f64,
}

/// One column block written: the copy from the host (zero from device buffers) and the layout kernels.
#[derive(Clone, Copy, Debug, Default)]
pub struct WriteTiming {
    pub copy: f64,
    pub layout: f64,
}

/// How a call was laid out.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Plan {
    pub count: usize,
    /// Requests on the device: `count` rounded up to whole groups (the added ones are zeros).
    pub requests: usize,
    pub ct: u32,
    pub gt: u32,
    pub it: u32,
    pub pgt: u32,
    pub seg_rows: usize,
    pub scan_segs: usize,
    pub pack_segs: usize,
}

/// What a call needs of a table apart from the device: its shape, its setup id and the call layout. A copy lets the
/// host's part of a call (parsing, hashing, the answers' bytes) run without holding the table, so that several
/// callers share one table and only the device's part of each call waits for the others.
#[derive(Clone, Copy, Debug)]
pub struct FetchShape {
    pub rows: usize,
    pub cols: usize,
    pub setup_id: [u8; 32],
    pub tuning: Tuning,
}

/// A call's requests parsed for the device (`FetchShape::prepare`).
pub struct Prepared {
    pub plan: Plan,
    /// The selection words lifted to the ring, `[group][row][gt]`, and the key words, `[group][key word][pgt]`.
    pub query: Vec<u64>,
    pub keys: Vec<u64>,
    /// The parsing's time.
    pub parse: f64,
}

impl FetchShape {
    /// How a call of `count` requests is laid out.
    pub fn plan(&self, count: usize) -> Plan {
        let t = &self.tuning;
        let fit = |mut g: u32| {
            while g as usize > count {
                g /= 2;
            }
            g.max(1)
        };
        let (gt, pgt) = (fit(t.gt), fit(t.pgt));
        let widest = gt.max(pgt) as usize;
        let requests = count.div_ceil(widest) * widest;
        let ct = if gt == 1 { t.ct1 } else { t.ct };
        let it = if pgt == 1 { t.it1 } else { t.it };
        let target = t.target as usize;
        // The first pass: the longest segment that still gives the launch `target` threads.
        let scan_threads = self.cols / ct as usize * (requests / gt as usize);
        let mut seg_rows = if t.seg_rows != 0 && self.rows % t.seg_rows as usize == 0 {
            t.seg_rows as usize
        } else {
            [self.rows, 2048, 1024, 512, 256].into_iter().find(|s| scan_threads * (self.rows / s) >= target).unwrap_or(256)
        };
        // The packing: the fewest segments of key words that do.
        let pack_threads = self.cols / it as usize * (requests / pgt as usize);
        let mut pack_segs =
            if t.pack_segs != 0 { t.pack_segs as usize } else { [1, 2, 4, 8, 16].into_iter().find(|s| pack_threads * s >= target).unwrap_or(16) };
        let bytes = |seg_rows: usize, pack_segs: usize| (self.rows / seg_rows + pack_segs) * requests * self.cols * 8;
        while bytes(seg_rows, pack_segs) > PARTIAL_BUDGET && (pack_segs > 1 || seg_rows < self.rows) {
            if pack_segs > 1 {
                pack_segs /= 2;
            }
            if seg_rows < self.rows {
                seg_rows = if self.rows % (seg_rows * 2) == 0 { seg_rows * 2 } else { self.rows };
            }
        }
        Plan { count, requests, ct, gt, it, pgt, seg_rows, scan_segs: self.rows / seg_rows, pack_segs }
    }

    /// Checks `count` requests laid end to end (the length, each one's magic and setup id) and reads their packed
    /// words into the kernels' layouts; requests past `count` in the last group are zeros.
    pub fn prepare(&self, requests: &[u8], count: usize) -> Result<Prepared, String> {
        let each = request_bytes(self.rows);
        if count == 0 || requests.len() != count * each {
            return Err(format!("{count} requests to this table are {} bytes, got {}", count * each, requests.len()));
        }
        for (i, r) in requests.chunks_exact(each).enumerate() {
            if &r[..4] != b"RNQ3" || r[4..36] != self.setup_id {
                return Err(format!("request {i} is not for this table's setup (magic or setup id)"));
            }
        }
        let t = Instant::now();
        let plan = self.plan(count);
        let lay = |width: usize, per: usize, offset: usize, bits: usize, shift: u32| -> Vec<u64> {
            let mut out = vec![0u64; plan.requests * per];
            out.par_chunks_mut(per * width).enumerate().for_each(|(group, dst)| {
                for g in 0..width {
                    let request = group * width + g;
                    if request >= count {
                        break;
                    }
                    let bytes = &requests[request * each..(request + 1) * each];
                    for i in 0..per {
                        dst[i * width + g] = read_bits(bytes, offset * 8 + i * bits, bits) << shift;
                    }
                }
            });
            out
        };
        let query = lay(plan.gt as usize, self.rows, SELECTION_OFFSET, QUERY_BITS, QUERY_SHIFT);
        let keys = lay(plan.pgt as usize, KEY_WORDS, KEY_OFFSET, KEY_BITS, 0);
        Ok(Prepared { plan, query, keys, parse: ms(t.elapsed()) })
    }

    /// SHA-256 of each of `count` requests: what every answer carries.
    pub fn digests(&self, requests: &[u8]) -> Vec<[u8; 32]> {
        requests.par_chunks_exact(request_bytes(self.rows)).map(|r| Sha256::digest(r).into()).collect()
    }

    /// The answers' bytes from the device's coefficients (`[request][column]`, 22 bits each) and the requests'
    /// digests: the head, then the coefficients packed.
    pub fn answers(&self, coefficients: &[u32], digests: &[[u8; 32]]) -> Vec<Vec<u8>> {
        let (cols, id) = (self.cols, self.setup_id);
        digests
            .par_iter()
            .enumerate()
            .map(|(g, digest)| {
                let mut out = Vec::with_capacity(answer_bytes(cols));
                out.extend_from_slice(b"RNR2");
                out.extend_from_slice(&id);
                out.extend_from_slice(digest);
                pack_bits(&coefficients[g * cols..(g + 1) * cols], RESPONSE_BITS, &mut out);
                out
            })
            .collect()
    }
}

/// The device's part of a call: the copies and the device's own time (CUDA events, or Metal's GPU times).
#[derive(Clone, Copy, Debug, Default)]
pub struct DeviceTiming {
    pub copy_in: f64,
    pub scan: f64,
    pub pack: f64,
    pub finish: f64,
    pub device: f64,
    /// The time of the caller's work while the device ran (`CudaFetch::run`'s closure).
    pub meanwhile: f64,
    /// The wait for the device after that, and the coefficients copied back.
    pub copy_out: f64,
}

/// `bits` bits (at most 56) of `bytes` from bit `bit`, little-endian (`ipir-native-client`'s `read_bits`).
#[inline]
pub fn read_bits(bytes: &[u8], bit: usize, bits: usize) -> u64 {
    let at = bit >> 3;
    let word = if at + 8 <= bytes.len() {
        u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
    } else {
        let mut tail = [0u8; 8];
        tail[..bytes.len() - at].copy_from_slice(&bytes[at..]);
        u64::from_le_bytes(tail)
    };
    (word >> (bit & 7)) & ((1u64 << bits) - 1)
}

/// Appends `values` packed at `bits` bits each (at most 32), little-endian (`u64s_to_contiguous_bytes`).
pub fn pack_bits(values: &[u32], bits: usize, out: &mut Vec<u8>) {
    let (mut acc, mut held) = (0u64, 0usize);
    for &v in values {
        acc |= u64::from(v) << held;
        held += bits;
        while held >= 8 {
            out.push(acc as u8);
            acc >>= 8;
            held -= 8;
        }
    }
    if held > 0 {
        out.push(acc as u8);
    }
}

