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
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const SOURCE: &str = include_str!("cuda_fetch_native.cu");

#[path = "fetch_host.rs"]
mod fetch_host;
pub use fetch_host::*;

/// Threads of a block in every kernel.
const THREADS: u32 = 256;

fn err(e: impl std::fmt::Display) -> String {
    format!("CUDA: {e}")
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
