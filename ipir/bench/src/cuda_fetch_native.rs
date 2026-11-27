//! An IPIR+SP fetch on an NVIDIA GPU through CUDA: the host side of `cuda_fetch_native.cu`.
//!
//! `FetchDevice` opens the device (or takes an open context) and compiles the kernels with NVRTC. `CudaFetch` is one
//! table that answers: its cells and its compiled packing matrices are copied to the device once and stay there.
//!
//! - `respond(requests, count)`: the answers to a call's requests, the bytes `NativeServer::respond` gives for each.
//!   All the requests go through one launch of the first pass, one of the packing and one of the rounding.
//! - `write_block(k, cells, matrix)`: column block `k` replaced in place (its cells and its compiled matrix), the
//!   rest of the table untouched; `write_block_device` does the same from buffers that are already on the device,
//!   such as the block build's (`cuda_build`).
//! - `first_pass`, `packing_product`: the two sums on their own, unrounded, for checking each against the CPU's.
//! - `FetchShape::prepare`, `CudaFetch::run`, `FetchShape::answers`: the three parts of `respond`, for several
//!   callers on one table: only `run`, the device's part, needs the table to itself.
//!
//! On the host: parsing the requests' packed words (49-bit selection words, 54-bit key words), the SHA-256 of each
//! request (computed while the device works) and the packing of the answer's 22-bit coefficients.
//!
//! A module of the programs that use it (`#[path]`): it needs the `cuda` feature. Each `FetchDevice` has one
//! stream, and cudarc's per-buffer event tracking is off: work on a table is ordered by that stream.
use cudarc::{
    driver::{sys::CUevent_flags, CudaContext, CudaEvent, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::compile_ptx,
};
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const SOURCE: &str = include_str!("cuda_fetch_native.cu");

/// Columns of a column block, and rows of a block's compiled matrix.
const D: usize = 2048;
/// Key words of a request, and columns of a block's compiled matrix.
pub const KEY_WORDS: usize = 4096;
const KEY_BITS: usize = 54;
const QUERY_BITS: usize = 49;
const QUERY_SHIFT: u32 = 5;
const RESPONSE_BITS: usize = 22;
const KEY_OFFSET: usize = 36;
const SELECTION_OFFSET: usize = KEY_OFFSET + KEY_WORDS * KEY_BITS / 8;
const ANSWER_OFFSET: usize = 68;
/// Threads of a block in every kernel.
const THREADS: u32 = 256;
/// Largest buffer of partial sums a call makes; past it, segments get longer.
const PARTIAL_BUDGET: usize = 256 << 20;

fn err(e: impl std::fmt::Display) -> String {
    format!("CUDA: {e}")
}

fn ms(d: Duration) -> f64 {
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

/// How a call is laid out on the device. The best values differ between cards; `IPIR_CUDA_FETCH` overrides the
/// defaults (`ct=4,gt=4,it=4,pgt=4,ct1=8,it1=8,seg=0,pseg=0,so=1,po=1,target=262144`).
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
}

impl Default for Tuning {
    fn default() -> Self {
        Tuning { ct: 4, gt: 4, it: 4, pgt: 4, ct1: 8, it1: 8, seg_rows: 0, pack_segs: 0, scan_order: 1, pack_order: 1, target: 262_144 }
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

    /// `IPIR_CUDA_FETCH`, or the defaults.
    pub fn from_env() -> Result<Self, String> {
        Self::parse(&std::env::var("IPIR_CUDA_FETCH").unwrap_or_default())
    }

    pub fn label(&self) -> String {
        format!(
            "ct={},gt={},it={},pgt={},ct1={},it1={},seg={},pseg={},so={},po={},target={}",
            self.ct, self.gt, self.it, self.pgt, self.ct1, self.it1, self.seg_rows, self.pack_segs, self.scan_order, self.pack_order, self.target
        )
    }
}

/// The device, one stream and the compiled kernels: shared by every table on it.
pub struct FetchDevice {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    finish_answer: CudaFunction,
    finish_words: CudaFunction,
    put_cells: CudaFunction,
    put_matrix: CudaFunction,
    /// `scan` by (CT, GT) and `pack` by (IT, GT), compiled when first used.
    scans: Mutex<HashMap<(u32, u32), CudaFunction>>,
    packs: Mutex<HashMap<(u32, u32), CudaFunction>>,
    /// One column block as the host has it, on its way into a table (cells column-major, the matrix row-major):
    /// shared by the device's tables, as long as the longest one's block.
    staging: Mutex<Option<(CudaSlice<u16>, CudaSlice<i32>)>>,
    /// Opening the device and the first compile.
    pub init: Duration,
}

impl FetchDevice {
    /// Opens device `ordinal`.
    pub fn open(ordinal: usize) -> Result<Arc<Self>, String> {
        let t = Instant::now();
        // SAFETY: these probes only load the vendor libraries.
        if !unsafe { cudarc::driver::sys::is_culib_present() } || !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
            return Err(err("the CUDA driver and NVRTC shared libraries are required"));
        }
        let ctx = CudaContext::new(ordinal).map_err(err)?;
        Self::on(&ctx, t)
    }

    /// A stream of its own on a context that is already open (the block build's, so that its buffers can be read).
    pub fn with_context(ctx: &Arc<CudaContext>) -> Result<Arc<Self>, String> {
        Self::on(ctx, Instant::now())
    }

    fn on(ctx: &Arc<CudaContext>, t: Instant) -> Result<Arc<Self>, String> {
        // SAFETY: every table's work is ordered by this device's one stream, and a buffer of another stream is read
        // only after that stream was waited for (`write_block_device`).
        unsafe { ctx.disable_event_tracking() };
        let module = ctx.load_module(compile_ptx(Self::source(1, 1, 1)).map_err(err)?).map_err(err)?;
        let f = |name: &str| module.load_function(name).map_err(err);
        Ok(Arc::new(FetchDevice {
            stream: ctx.new_stream().map_err(err)?,
            finish_answer: f("finish_answer")?,
            finish_words: f("finish_words")?,
            put_cells: f("put_cells")?,
            put_matrix: f("put_matrix")?,
            scans: Mutex::new(HashMap::new()),
            packs: Mutex::new(HashMap::new()),
            staging: Mutex::new(None),
            ctx: ctx.clone(),
            init: t.elapsed(),
        }))
    }

    fn source(ct: u32, gt: u32, it: u32) -> String {
        format!("#define CT {ct}\n#define GT {gt}\n#define IT {it}\n{SOURCE}")
    }

    fn scan(&self, ct: u32, gt: u32) -> Result<CudaFunction, String> {
        let mut map = self.scans.lock().unwrap();
        if let Some(f) = map.get(&(ct, gt)) {
            return Ok(f.clone());
        }
        let module = self.ctx.load_module(compile_ptx(Self::source(ct, gt, 1)).map_err(err)?).map_err(err)?;
        let f = module.load_function("scan").map_err(err)?;
        map.insert((ct, gt), f.clone());
        Ok(f)
    }

    fn pack(&self, it: u32, gt: u32) -> Result<CudaFunction, String> {
        let mut map = self.packs.lock().unwrap();
        if let Some(f) = map.get(&(it, gt)) {
            return Ok(f.clone());
        }
        let module = self.ctx.load_module(compile_ptx(Self::source(1, gt, it)).map_err(err)?).map_err(err)?;
        let f = module.load_function("pack").map_err(err)?;
        map.insert((it, gt), f.clone());
        Ok(f)
    }

    pub fn name(&self) -> String {
        self.ctx.name().unwrap_or_else(|e| format!("unknown ({e})"))
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Bytes of the staging block the device's tables share.
    pub fn staging_bytes(&self) -> usize {
        self.staging.lock().unwrap().as_ref().map_or(0, |(c, m)| c.len() * 2 + m.len() * 4)
    }

    /// Bytes of the device's memory in use, by every process, and its size.
    pub fn memory(&self) -> Result<(usize, usize), String> {
        self.ctx.bind_to_thread().map_err(err)?;
        let (free, total) = cudarc::driver::result::mem_get_info().map_err(err)?;
        Ok((total - free, total))
    }

    fn event(&self) -> Result<CudaEvent, String> {
        self.stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT)).map_err(err)
    }

    fn sync(&self) -> Result<(), String> {
        self.stream.synchronize().map_err(err)
    }

    /// Uninitialized device memory: every use below writes a buffer before a kernel reads it.
    fn alloc<T: cudarc::driver::DeviceRepr>(&self, len: usize) -> Result<CudaSlice<T>, String> {
        // SAFETY: see above; the element types are plain integers.
        unsafe { self.stream.alloc::<T>(len.max(1)) }.map_err(err)
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

struct CallBuffers {
    query: CudaSlice<u64>,
    keys: CudaSlice<u64>,
    partial: CudaSlice<u64>,
    out: CudaSlice<u32>,
    words: Option<CudaSlice<u64>>,
    bytes: usize,
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Answer,
    FirstPass,
    Packing,
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
    plan: Plan,
    /// The selection words lifted to the ring, `[group][row][gt]`, and the key words, `[group][key word][pgt]`.
    query: Vec<u64>,
    keys: Vec<u64>,
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

/// The device's part of a call: the copies and, from CUDA events, the device's own time.
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

/// One table that answers, resident on the device.
pub struct CudaFetch {
    dev: Arc<FetchDevice>,
    pub shape: FetchShape,
    /// `[block][row][2048]`.
    cells: CudaSlice<u16>,
    /// `[block][key word][2048]`.
    mats: CudaSlice<i32>,
    calls: HashMap<(usize, usize), CallBuffers>,
}

impl CudaFetch {
    /// A table of `rows × cols` with nothing written: every column block needs a `write_block` before a call.
    pub fn empty(dev: &Arc<FetchDevice>, rows: usize, cols: usize, setup_id: [u8; 32], tuning: Tuning) -> Result<Self, String> {
        if rows == 0 || cols == 0 || rows % D != 0 || cols % D != 0 {
            return Err(format!("a table's rows and columns are multiples of {D}, got {rows} x {cols}"));
        }
        Ok(CudaFetch {
            dev: dev.clone(),
            shape: FetchShape { rows, cols, setup_id, tuning },
            cells: dev.alloc(rows * cols)?,
            mats: dev.alloc(cols * KEY_WORDS)?,
            calls: HashMap::new(),
        })
    }

    /// A table from its cells (column-major, `col * rows + row`) and each block's compiled matrix (2,048 × 4,096
    /// signed words, row-major). Returns the table and the time of the copies and their layout.
    pub fn new(
        dev: &Arc<FetchDevice>,
        rows: usize,
        cols: usize,
        setup_id: [u8; 32],
        tuning: Tuning,
        cells: &[u16],
        matrices: &[Vec<i32>],
    ) -> Result<(Self, Duration), String> {
        assert_eq!(cells.len(), rows * cols);
        assert_eq!(matrices.len(), cols / D);
        let t = Instant::now();
        let mut table = Self::empty(dev, rows, cols, setup_id, tuning)?;
        for (k, matrix) in matrices.iter().enumerate() {
            table.write_block(k, &cells[k * D * rows..(k + 1) * D * rows], matrix)?;
        }
        Ok((table, t.elapsed()))
    }

    pub fn device(&self) -> &Arc<FetchDevice> {
        &self.dev
    }

    /// How a call of `count` requests is laid out.
    pub fn plan(&self, count: usize) -> Plan {
        self.shape.plan(count)
    }

    /// Bytes this table keeps on the device between calls: its cells and its matrices.
    pub fn resident_bytes(&self) -> usize {
        let (rows, cols) = (self.shape.rows, self.shape.cols);
        rows * cols * 2 + cols * KEY_WORDS * 4
    }

    /// Bytes of the per-call buffers made so far (they are kept between calls of the same size).
    pub fn call_bytes(&self) -> usize {
        self.calls.values().map(|c| c.bytes).sum()
    }

    fn launch_layout(&self, k: usize, cells: &CudaSlice<u16>, matrix: &CudaSlice<i32>) -> Result<(), String> {
        let dv = &*self.dev;
        let (rows, block) = (self.shape.rows as u32, k as u32);
        let mut a = dv.stream.launch_builder(&dv.put_cells);
        a.arg(cells).arg(&self.cells).arg(&rows).arg(&block);
        // SAFETY: the arguments are the kernel's in order; `cells` holds rows × 2,048 words and block `k` exists.
        unsafe { a.launch(LaunchConfig { grid_dim: (rows.div_ceil(THREADS), D as u32, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: 0 }) }
            .map_err(err)?;
        let mut a = dv.stream.launch_builder(&dv.put_matrix);
        a.arg(matrix).arg(&self.mats).arg(&block);
        // SAFETY: as above; `matrix` holds 2,048 × 4,096 words.
        unsafe { a.launch(LaunchConfig { grid_dim: (KEY_WORDS as u32 / THREADS, D as u32, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: 0 }) }
            .map_err(err)?;
        Ok(())
    }

    /// Column block `k` replaced: its cells (column-major, rows × 2,048) and its compiled matrix (row-major,
    /// 2,048 × 4,096). Waits until the block is in place.
    pub fn write_block(&mut self, k: usize, cells: &[u16], matrix: &[i32]) -> Result<WriteTiming, String> {
        assert!(k < self.shape.cols / D, "column block {k} of {}", self.shape.cols / D);
        assert_eq!(cells.len(), self.shape.rows * D);
        assert_eq!(matrix.len(), D * KEY_WORDS);
        let dev = self.dev.clone();
        // The staging block is the device's: held until this block is in place.
        let mut staging = dev.staging.lock().unwrap();
        if staging.as_ref().is_none_or(|(c, _)| c.len() < cells.len()) {
            *staging = None;
            *staging = Some((dev.alloc(cells.len())?, dev.alloc(D * KEY_WORDS)?));
        }
        let (stage_cells, stage_matrix) = staging.as_mut().unwrap();
        let t = Instant::now();
        dev.stream.memcpy_htod(cells, stage_cells).map_err(err)?;
        dev.stream.memcpy_htod(matrix, stage_matrix).map_err(err)?;
        dev.sync()?;
        let copy = ms(t.elapsed());
        let t = Instant::now();
        self.launch_layout(k, stage_cells, stage_matrix)?;
        dev.sync()?;
        Ok(WriteTiming { copy, layout: ms(t.elapsed()) })
    }

    /// Column block `k` replaced from buffers on the device, in the same layouts as `write_block` takes from the
    /// host (the block build's cells and its compiled matrix). The buffers belong to another stream: the caller has
    /// waited for that stream's work on them. Waits until the block is in place.
    pub fn write_block_device(&mut self, k: usize, cells: &CudaSlice<u16>, matrix: &CudaSlice<i32>) -> Result<WriteTiming, String> {
        assert!(k < self.shape.cols / D, "column block {k} of {}", self.shape.cols / D);
        assert!(cells.len() >= self.shape.rows * D && matrix.len() >= D * KEY_WORDS);
        let t = Instant::now();
        self.launch_layout(k, cells, matrix)?;
        self.dev.sync()?;
        Ok(WriteTiming { copy: 0.0, layout: ms(t.elapsed()) })
    }

    fn buffers(&mut self, plan: &Plan, words: bool) -> Result<(), String> {
        let key = (plan.requests, plan.scan_segs + plan.pack_segs);
        let dev = self.dev.clone();
        let (rows, cols) = (self.shape.rows, self.shape.cols);
        if !self.calls.contains_key(&key) {
            let (r, segs) = key;
            let sizes = [r * rows * 8, r * KEY_WORDS * 8, segs * r * cols * 8, r * cols * 4];
            self.calls.insert(
                key,
                CallBuffers {
                    query: dev.alloc(r * rows)?,
                    keys: dev.alloc(r * KEY_WORDS)?,
                    partial: dev.alloc(segs * r * cols)?,
                    out: dev.alloc(r * cols)?,
                    words: None,
                    bytes: sizes.iter().sum(),
                },
            );
        }
        let c = self.calls.get_mut(&key).unwrap();
        if words && c.words.is_none() {
            c.words = Some(dev.alloc(plan.requests * cols)?);
            c.bytes += plan.requests * cols * 8;
        }
        Ok(())
    }

    fn grid(order: u32, col_blocks: u32, groups: u32, segs: u32) -> (u32, u32, u32) {
        match order {
            0 => (col_blocks, groups, segs),
            1 => (groups, col_blocks, segs),
            _ => (groups, segs, col_blocks),
        }
    }

    /// Launches the kernels of `mode` for a call whose words are on the device; returns the events around them
    /// (before the first pass, after it, after the packing, after the finish).
    fn launch(&self, plan: &Plan, mode: Mode) -> Result<[CudaEvent; 4], String> {
        let dv = &*self.dev;
        let c = &self.calls[&(plan.requests, plan.scan_segs + plan.pack_segs)];
        let (rows, cols, requests) = (self.shape.rows as u32, self.shape.cols as u32, plan.requests as u32);
        let tuning = &self.shape.tuning;
        let e0 = dv.event()?;
        if mode != Mode::Packing {
            let f = dv.scan(plan.ct, plan.gt)?;
            let (seg_rows, order) = (plan.seg_rows as u32, tuning.scan_order);
            let grid = Self::grid(order, (cols / plan.ct).div_ceil(THREADS), requests / plan.gt, plan.scan_segs as u32);
            let mut a = dv.stream.launch_builder(&f);
            a.arg(&self.cells).arg(&c.query).arg(&c.partial).arg(&rows).arg(&cols).arg(&seg_rows).arg(&requests).arg(&order);
            // SAFETY: the arguments are the kernel's in order, and the buffers have the sizes it indexes for this grid.
            unsafe { a.launch(LaunchConfig { grid_dim: grid, block_dim: (THREADS, 1, 1), shared_mem_bytes: 0 }) }.map_err(err)?;
        }
        let e1 = dv.event()?;
        if mode != Mode::FirstPass {
            let f = dv.pack(plan.it, plan.pgt)?;
            let (seg_words, seg0, order) = ((KEY_WORDS / plan.pack_segs) as u32, plan.scan_segs as u32, tuning.pack_order);
            let grid = Self::grid(order, (cols / plan.it).div_ceil(THREADS), requests / plan.pgt, plan.pack_segs as u32);
            let mut a = dv.stream.launch_builder(&f);
            a.arg(&self.mats).arg(&c.keys).arg(&c.partial).arg(&cols).arg(&seg_words).arg(&requests).arg(&seg0).arg(&order);
            // SAFETY: as above.
            unsafe { a.launch(LaunchConfig { grid_dim: grid, block_dim: (THREADS, 1, 1), shared_mem_bytes: 0 }) }.map_err(err)?;
        }
        let e2 = dv.event()?;
        let (s0, s1) = match mode {
            Mode::Answer => (0, plan.scan_segs + plan.pack_segs),
            Mode::FirstPass => (0, plan.scan_segs),
            Mode::Packing => (plan.scan_segs, plan.scan_segs + plan.pack_segs),
        };
        let (s0, s1) = (s0 as u32, s1 as u32);
        let cfg = LaunchConfig { grid_dim: (cols.div_ceil(THREADS), requests, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: 0 };
        if mode == Mode::Answer {
            let mut a = dv.stream.launch_builder(&dv.finish_answer);
            a.arg(&c.partial).arg(&c.out).arg(&cols).arg(&requests).arg(&s0).arg(&s1);
            // SAFETY: as above.
            unsafe { a.launch(cfg) }.map_err(err)?;
        } else {
            let mut a = dv.stream.launch_builder(&dv.finish_words);
            a.arg(&c.partial).arg(c.words.as_ref().unwrap()).arg(&cols).arg(&requests).arg(&s0).arg(&s1);
            // SAFETY: as above.
            unsafe { a.launch(cfg) }.map_err(err)?;
        }
        Ok([e0, e1, e2, dv.event()?])
    }

    fn upload(&mut self, p: &Prepared) -> Result<(), String> {
        let s = self.dev.stream.clone();
        let c = self.calls.get_mut(&(p.plan.requests, p.plan.scan_segs + p.plan.pack_segs)).unwrap();
        s.memcpy_htod(&p.query, &mut c.query).map_err(err)?;
        s.memcpy_htod(&p.keys, &mut c.keys).map_err(err)?;
        Ok(())
    }

    /// The device's part of a call: the prepared words copied in, the three kernels, the coefficients copied back
    /// (`[request][column]`, the plan's requests). `meanwhile` runs on the host while the device works.
    pub fn run<R>(&mut self, p: &Prepared, meanwhile: impl FnOnce() -> R) -> Result<(Vec<u32>, DeviceTiming, R), String> {
        let plan = p.plan;
        self.buffers(&plan, false)?;
        let t = Instant::now();
        self.upload(p)?;
        let copy_in = ms(t.elapsed());
        let events = self.launch(&plan, Mode::Answer)?;
        let t = Instant::now();
        let r = meanwhile();
        let meanwhile = ms(t.elapsed());
        let t = Instant::now();
        let mut coefficients = vec![0u32; plan.requests * self.shape.cols];
        let c = &self.calls[&(plan.requests, plan.scan_segs + plan.pack_segs)];
        self.dev.stream.memcpy_dtoh(&c.out, &mut coefficients).map_err(err)?;
        self.dev.sync()?;
        let copy_out = ms(t.elapsed());
        let between = |a: usize, b: usize| events[a].elapsed_ms(&events[b]).map(f64::from).map_err(err);
        let timing =
            DeviceTiming { copy_in, scan: between(0, 1)?, pack: between(1, 2)?, finish: between(2, 3)?, device: between(0, 3)?, meanwhile, copy_out };
        Ok((coefficients, timing, r))
    }

    /// The answers to `count` requests laid end to end, one `Vec` each, and the call's times. The requests are
    /// hashed while the device works.
    pub fn respond(&mut self, requests: &[u8], count: usize) -> Result<(Vec<Vec<u8>>, CallTiming), String> {
        let wall = Instant::now();
        let shape = self.shape;
        let p = shape.prepare(requests, count)?;
        let (coefficients, d, digests) = self.run(&p, || shape.digests(requests))?;
        let t = Instant::now();
        let answers = shape.answers(&coefficients, &digests);
        let serialize = ms(t.elapsed());
        let timing = CallTiming {
            wall: ms(wall.elapsed()),
            parse: p.parse,
            copy_in: d.copy_in,
            scan: d.scan,
            pack: d.pack,
            finish: d.finish,
            device: d.device,
            hash: d.meanwhile,
            copy_out: d.copy_out,
            serialize,
        };
        Ok((answers, timing))
    }

    fn words(&mut self, requests: &[u8], count: usize, mode: Mode) -> Result<Vec<u64>, String> {
        let p = self.shape.prepare(requests, count)?;
        self.buffers(&p.plan, true)?;
        self.upload(&p)?;
        self.launch(&p.plan, mode)?;
        let c = &self.calls[&(p.plan.requests, p.plan.scan_segs + p.plan.pack_segs)];
        let mut out = self.dev.stream.memcpy_dtov(c.words.as_ref().unwrap()).map_err(err)?;
        self.dev.sync()?;
        out.truncate(count * self.shape.cols);
        Ok(out)
    }

    /// The first pass alone, unrounded: `[request][column]`, each `Σ_r cell[r, column] · (x_r << 5) mod 2^54`.
    pub fn first_pass(&mut self, requests: &[u8], count: usize) -> Result<Vec<u64>, String> {
        self.words(requests, count, Mode::FirstPass)
    }

    /// The packing product alone, unrounded: `[request][column]`, each block's `H_k · K mod 2^54`.
    pub fn packing_product(&mut self, requests: &[u8], count: usize) -> Result<Vec<u64>, String> {
        self.words(requests, count, Mode::Packing)
    }
}

/// `bits` bits (at most 56) of `bytes` from bit `bit`, little-endian (`ipir-native-client`'s `read_bits`).
#[inline]
fn read_bits(bytes: &[u8], bit: usize, bits: usize) -> u64 {
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
fn pack_bits(values: &[u32], bits: usize, out: &mut Vec<u8>) {
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
