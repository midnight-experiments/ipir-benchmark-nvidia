//! An IPIR+SP fetch on the Apple GPU through Metal: the host side of `shaders/fetch.metal`, and the counterpart of
//! the CUDA fetch (`ipir/bench/src/cuda_fetch_native.rs`), with which it shares the host's part of a call
//! (`fetch_host`).
//!
//! `MetalFetchDevice` holds the device, one command queue and the compiled kernels. `MetalFetch` is one table that
//! answers: its cells and its compiled packing matrices are laid out on the GPU once and stay there.
//!
//! - `respond(requests, count)`: the answers to a call's requests, the bytes `NativeServer::respond` gives for each.
//!   All the requests go through one dispatch of the first pass, one of the packing and one of the rounding.
//! - `write_block(k, cells, matrix)`: column block `k` replaced in place; `write_block_buffers` does the same from
//!   buffers that are already on the GPU, such as the block build's (`gpu::GpuBuilder`).
//! - `first_pass`, `packing_product`: the two sums on their own, unrounded, for checking each against the CPU's.
//! - `FetchShape::prepare`, `MetalFetch::run`, `FetchShape::answers`: the three parts of `respond`, for several
//!   callers on one table: only `run`, the device's part, needs the table to itself.
//!
//! This Mac's memory is unified: a call's words and its coefficients are in buffers the host reads and writes in
//! place, and the table, the matrices and the partial sums are private to the GPU.
use crate::fetch_host::*;
use metal::objc::rc::autoreleasepool;
use metal::objc::{msg_send, sel, sel_impl};
use metal::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const SOURCE: &str = include_str!("../shaders/fetch.metal");

#[repr(C)]
#[derive(Clone, Copy)]
struct Params([u32; 4]);

fn gpu_times(cmd: &CommandBufferRef) -> (f64, f64) {
    unsafe {
        let start: f64 = msg_send![cmd, GPUStartTime];
        let end: f64 = msg_send![cmd, GPUEndTime];
        (start, end)
    }
}

/// The device, one command queue and the compiled kernels: shared by every table on it.
pub struct MetalFetchDevice {
    pub device: Device,
    queue: CommandQueue,
    finish_answer: ComputePipelineState,
    finish_words: ComputePipelineState,
    put_cells: ComputePipelineState,
    put_matrix: ComputePipelineState,
    /// `scan` by (CT, GT) and `pack` by (IT, GT), compiled when first used.
    scans: Mutex<HashMap<(u32, u32), ComputePipelineState>>,
    packs: Mutex<HashMap<(u32, u32), ComputePipelineState>>,
    /// One column block as the host has it, on its way into a table (cells column-major, the matrix row-major):
    /// shared by the device's tables, as long as the longest one's block.
    staging: Mutex<Option<(Buffer, Buffer)>>,
    /// Opening the device and the first compile.
    pub init: Duration,
}

impl MetalFetchDevice {
    /// The system's default Metal device, with a command queue of its own.
    pub fn open() -> Result<Arc<Self>, String> {
        let t = Instant::now();
        let device = Device::system_default().ok_or("no Metal device")?;
        let library = Self::library(&device, 1, 1, 1)?;
        let f = |name: &str| Self::pipeline(&device, &library, name);
        Ok(Arc::new(MetalFetchDevice {
            queue: device.new_command_queue(),
            finish_answer: f("finish_answer")?,
            finish_words: f("finish_words")?,
            put_cells: f("put_cells")?,
            put_matrix: f("put_matrix")?,
            scans: Mutex::new(HashMap::new()),
            packs: Mutex::new(HashMap::new()),
            staging: Mutex::new(None),
            device,
            init: t.elapsed(),
        }))
    }

    fn library(device: &Device, ct: u32, gt: u32, it: u32) -> Result<Library, String> {
        let src = format!("#define CT {ct}\n#define GT {gt}\n#define IT {it}\n{SOURCE}");
        device.new_library_with_source(&src, &CompileOptions::new()).map_err(|e| format!("Metal compile: {e}"))
    }

    fn pipeline(device: &Device, library: &Library, name: &str) -> Result<ComputePipelineState, String> {
        let f = library.get_function(name, None).map_err(|e| format!("Metal: {name}: {e}"))?;
        device.new_compute_pipeline_state_with_function(&f).map_err(|e| format!("Metal: {name}: {e}"))
    }

    fn scan(&self, ct: u32, gt: u32) -> Result<ComputePipelineState, String> {
        let mut map = self.scans.lock().unwrap();
        if let Some(f) = map.get(&(ct, gt)) {
            return Ok(f.clone());
        }
        let f = Self::pipeline(&self.device, &Self::library(&self.device, ct, gt, 1)?, "scan")?;
        map.insert((ct, gt), f.clone());
        Ok(f)
    }

    fn pack(&self, it: u32, gt: u32) -> Result<ComputePipelineState, String> {
        let mut map = self.packs.lock().unwrap();
        if let Some(f) = map.get(&(it, gt)) {
            return Ok(f.clone());
        }
        let f = Self::pipeline(&self.device, &Self::library(&self.device, 1, gt, it)?, "pack")?;
        map.insert((it, gt), f.clone());
        Ok(f)
    }

    pub fn name(&self) -> String {
        self.device.name().to_string()
    }

    /// Bytes this process has allocated on the device.
    pub fn allocated_bytes(&self) -> usize {
        self.device.current_allocated_size() as usize
    }

    /// Bytes of the staging block the device's tables share.
    pub fn staging_bytes(&self) -> usize {
        self.staging.lock().unwrap().as_ref().map_or(0, |(c, m)| (c.length() + m.length()) as usize)
    }

    fn buffer(&self, bytes: usize, options: MTLResourceOptions) -> Result<Buffer, String> {
        let b = self.device.new_buffer(bytes.max(16) as u64, options);
        // A buffer the device could not make is nil, whose length reads as 0.
        if b.length() == 0 {
            return Err(format!("Metal: out of memory: a buffer of {bytes} bytes was not allocated"));
        }
        Ok(b)
    }

    /// Commits `cmds` in order and waits for the last.
    fn finish(&self, cmds: &[&CommandBufferRef]) -> Result<(), String> {
        for c in cmds {
            c.commit();
        }
        let last = cmds[cmds.len() - 1];
        last.wait_until_completed();
        for c in cmds {
            if c.status() != MTLCommandBufferStatus::Completed {
                return Err("Metal: a command buffer failed".into());
            }
        }
        Ok(())
    }
}

struct CallBuffers {
    query: Buffer,
    keys: Buffer,
    partial: Buffer,
    out: Buffer,
    words: Option<Buffer>,
    bytes: usize,
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Answer,
    FirstPass,
    Packing,
}

/// One table that answers, resident on the GPU.
pub struct MetalFetch {
    dev: Arc<MetalFetchDevice>,
    pub shape: FetchShape,
    /// `[block][row][2048]`.
    cells: Buffer,
    /// `[block][key word][2048]`.
    mats: Buffer,
    calls: HashMap<(usize, usize), CallBuffers>,
}

impl MetalFetch {
    /// A table of `rows × cols` with nothing written: every column block needs a `write_block` before a call.
    pub fn empty(dev: &Arc<MetalFetchDevice>, rows: usize, cols: usize, setup_id: [u8; 32], tuning: Tuning) -> Result<Self, String> {
        if rows == 0 || cols == 0 || rows % D != 0 || cols % D != 0 {
            return Err(format!("a table's rows and columns are multiples of {D}, got {rows} x {cols}"));
        }
        if rows * cols > 1 << 30 || rows > 32_768 {
            return Err(format!("the kernels index up to 2^30 cells and add up to 32,768 rows in 32 bits, got {rows} x {cols}"));
        }
        let private = MTLResourceOptions::StorageModePrivate;
        Ok(MetalFetch {
            dev: dev.clone(),
            shape: FetchShape { rows, cols, setup_id, tuning },
            cells: dev.buffer(rows * cols * 2, private)?,
            mats: dev.buffer(cols * KEY_WORDS * 4, private)?,
            calls: HashMap::new(),
        })
    }

    /// A table from its cells (column-major, `col * rows + row`) and each block's compiled matrix (2,048 × 4,096
    /// signed words, row-major). Returns the table and the time of the copies and their layout.
    pub fn new(
        dev: &Arc<MetalFetchDevice>,
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

    pub fn device(&self) -> &Arc<MetalFetchDevice> {
        &self.dev
    }

    /// How a call of `count` requests is laid out.
    pub fn plan(&self, count: usize) -> Plan {
        self.shape.plan(count)
    }

    /// Bytes this table keeps on the GPU between calls: its cells and its matrices.
    pub fn resident_bytes(&self) -> usize {
        let (rows, cols) = (self.shape.rows, self.shape.cols);
        rows * cols * 2 + cols * KEY_WORDS * 4
    }

    /// Bytes of the per-call buffers made so far (they are kept between calls of the same size).
    pub fn call_bytes(&self) -> usize {
        self.calls.values().map(|c| c.bytes).sum()
    }

    fn threads(&self) -> MTLSize {
        MTLSize::new(u64::from(self.shape.tuning.threads), 1, 1)
    }

    fn encode_layout(&self, cmd: &CommandBufferRef, k: usize, cells: &Buffer, matrix: &Buffer) {
        let p = Params([self.shape.rows as u32, k as u32, 0, 0]);
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.dev.put_cells);
        enc.set_buffer(0, Some(cells), 0);
        enc.set_buffer(1, Some(&self.cells), 0);
        enc.set_bytes(2, 16, &p as *const Params as *const _);
        enc.dispatch_threads(MTLSize::new(self.shape.rows as u64, D as u64, 1), MTLSize::new(256, 1, 1));
        enc.set_compute_pipeline_state(&self.dev.put_matrix);
        enc.set_buffer(0, Some(matrix), 0);
        enc.set_buffer(1, Some(&self.mats), 0);
        enc.set_bytes(2, 16, &p as *const Params as *const _);
        enc.dispatch_threads(MTLSize::new(KEY_WORDS as u64, D as u64, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
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
        if staging.as_ref().is_none_or(|(c, _)| (c.length() as usize) < cells.len() * 2) {
            *staging = None;
            let shared = MTLResourceOptions::StorageModeShared;
            *staging = Some((dev.buffer(cells.len() * 2, shared)?, dev.buffer(D * KEY_WORDS * 4, shared)?));
        }
        let (stage_cells, stage_matrix) = staging.as_ref().unwrap();
        let t = Instant::now();
        // SAFETY: both buffers are shared with the host and at least as long as the slices.
        unsafe {
            std::ptr::copy_nonoverlapping(cells.as_ptr(), stage_cells.contents() as *mut u16, cells.len());
            std::ptr::copy_nonoverlapping(matrix.as_ptr(), stage_matrix.contents() as *mut i32, matrix.len());
        }
        let copy = ms(t.elapsed());
        let t = Instant::now();
        autoreleasepool(|| {
            let cmd = dev.queue.new_command_buffer();
            self.encode_layout(cmd, k, stage_cells, stage_matrix);
            dev.finish(&[cmd])
        })?;
        Ok(WriteTiming { copy, layout: ms(t.elapsed()) })
    }

    /// Column block `k` replaced from buffers on the GPU, in the same layouts as `write_block` takes from the host
    /// (the block build's cells and its compiled matrix). The caller has waited for the work that wrote them. Waits
    /// until the block is in place.
    pub fn write_block_buffers(&mut self, k: usize, cells: &Buffer, matrix: &Buffer) -> Result<WriteTiming, String> {
        assert!(k < self.shape.cols / D, "column block {k} of {}", self.shape.cols / D);
        assert!(cells.length() as usize >= self.shape.rows * D * 2 && matrix.length() as usize >= D * KEY_WORDS * 4);
        let t = Instant::now();
        autoreleasepool(|| {
            let cmd = self.dev.queue.new_command_buffer();
            self.encode_layout(cmd, k, cells, matrix);
            self.dev.finish(&[cmd])
        })?;
        Ok(WriteTiming { copy: 0.0, layout: ms(t.elapsed()) })
    }

    fn buffers(&mut self, plan: &Plan, words: bool) -> Result<(), String> {
        let key = (plan.requests, plan.scan_segs + plan.pack_segs);
        let dev = self.dev.clone();
        let (rows, cols) = (self.shape.rows, self.shape.cols);
        let shared = MTLResourceOptions::StorageModeShared;
        if !self.calls.contains_key(&key) {
            let (r, segs) = key;
            let sizes = [r * rows * 8, r * KEY_WORDS * 8, segs * r * cols * 8, r * cols * 4];
            self.calls.insert(
                key,
                CallBuffers {
                    query: dev.buffer(sizes[0], shared)?,
                    keys: dev.buffer(sizes[1], shared)?,
                    partial: dev.buffer(sizes[2], MTLResourceOptions::StorageModePrivate)?,
                    out: dev.buffer(sizes[3], shared)?,
                    words: None,
                    bytes: sizes.iter().sum(),
                },
            );
        }
        let c = self.calls.get_mut(&key).unwrap();
        if words && c.words.is_none() {
            c.words = Some(dev.buffer(plan.requests * cols * 8, shared)?);
            c.bytes += plan.requests * cols * 8;
        }
        Ok(())
    }

    /// The command buffers of `mode` for a call whose words are in its buffers, in the order they run: the first
    /// pass, the packing, the finish (a stage `mode` leaves out has none).
    fn encode(&self, plan: &Plan, mode: Mode) -> Result<Vec<&CommandBufferRef>, String> {
        let dv = &*self.dev;
        let c = &self.calls[&(plan.requests, plan.scan_segs + plan.pack_segs)];
        let (rows, cols, requests) = (self.shape.rows as u32, self.shape.cols as u32, plan.requests as u32);
        let mut cmds = Vec::new();
        if mode != Mode::Packing {
            let f = dv.scan(plan.ct, plan.gt)?;
            let p = Params([rows, cols, plan.seg_rows as u32, requests]);
            let cmd = dv.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&f);
            enc.set_buffer(0, Some(&self.cells), 0);
            enc.set_buffer(1, Some(&c.query), 0);
            enc.set_buffer(2, Some(&c.partial), 0);
            enc.set_bytes(3, 16, &p as *const Params as *const _);
            enc.dispatch_threads(MTLSize::new(u64::from(cols / plan.ct), u64::from(requests / plan.gt), plan.scan_segs as u64), self.threads());
            enc.end_encoding();
            cmds.push(cmd);
        }
        if mode != Mode::FirstPass {
            let f = dv.pack(plan.it, plan.pgt)?;
            let p = Params([cols, (KEY_WORDS / plan.pack_segs) as u32, requests, plan.scan_segs as u32]);
            let cmd = dv.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&f);
            enc.set_buffer(0, Some(&self.mats), 0);
            enc.set_buffer(1, Some(&c.keys), 0);
            enc.set_buffer(2, Some(&c.partial), 0);
            enc.set_bytes(3, 16, &p as *const Params as *const _);
            enc.dispatch_threads(MTLSize::new(u64::from(cols / plan.it), u64::from(requests / plan.pgt), plan.pack_segs as u64), self.threads());
            enc.end_encoding();
            cmds.push(cmd);
        }
        let (s0, s1) = match mode {
            Mode::Answer => (0, plan.scan_segs + plan.pack_segs),
            Mode::FirstPass => (0, plan.scan_segs),
            Mode::Packing => (plan.scan_segs, plan.scan_segs + plan.pack_segs),
        };
        let p = Params([cols, requests, s0 as u32, s1 as u32]);
        let cmd = dv.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        if mode == Mode::Answer {
            enc.set_compute_pipeline_state(&dv.finish_answer);
            enc.set_buffer(1, Some(&c.out), 0);
        } else {
            enc.set_compute_pipeline_state(&dv.finish_words);
            enc.set_buffer(1, Some(c.words.as_ref().unwrap()), 0);
        }
        enc.set_buffer(0, Some(&c.partial), 0);
        enc.set_bytes(2, 16, &p as *const Params as *const _);
        enc.dispatch_threads(MTLSize::new(u64::from(cols), u64::from(requests), 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
        cmds.push(cmd);
        Ok(cmds)
    }

    fn upload(&self, p: &Prepared) {
        let c = &self.calls[&(p.plan.requests, p.plan.scan_segs + p.plan.pack_segs)];
        // SAFETY: both buffers are shared with the host and were made for this plan's requests.
        unsafe {
            std::ptr::copy_nonoverlapping(p.query.as_ptr(), c.query.contents() as *mut u64, p.query.len());
            std::ptr::copy_nonoverlapping(p.keys.as_ptr(), c.keys.contents() as *mut u64, p.keys.len());
        }
    }

    /// The device's part of a call: the prepared words copied into the call's buffers, the three kernels, the
    /// coefficients copied out (`[request][column]`, the plan's requests). `meanwhile` runs on the host while the
    /// GPU works.
    pub fn run<R>(&mut self, p: &Prepared, meanwhile: impl FnOnce() -> R) -> Result<(Vec<u32>, DeviceTiming, R), String> {
        let plan = p.plan;
        self.buffers(&plan, false)?;
        let t = Instant::now();
        self.upload(p);
        let copy_in = ms(t.elapsed());
        // The command buffers are autoreleased: the pool frees them at the end of the call.
        autoreleasepool(|| {
            let cmds = self.encode(&plan, Mode::Answer)?;
            for c in &cmds {
                c.commit();
            }
            let t = Instant::now();
            let r = meanwhile();
            let meanwhile = ms(t.elapsed());
            let t = Instant::now();
            self.dev.finish_committed(&cmds)?;
            let c = &self.calls[&(plan.requests, plan.scan_segs + plan.pack_segs)];
            let n = plan.requests * self.shape.cols;
            // SAFETY: the buffer is shared with the host and holds `n` coefficients, written by the finish above.
            let coefficients = unsafe { std::slice::from_raw_parts(c.out.contents() as *const u32, n) }.to_vec();
            let copy_out = ms(t.elapsed());
            let times: Vec<(f64, f64)> = cmds.iter().map(|c| gpu_times(c)).collect();
            let span = |i: usize| (times[i].1 - times[i].0) * 1e3;
            let timing = DeviceTiming {
                copy_in,
                scan: span(0),
                pack: span(1),
                finish: span(2),
                device: (times[2].1 - times[0].0) * 1e3,
                meanwhile,
                copy_out,
            };
            Ok((coefficients, timing, r))
        })
    }

    /// The answers to `count` requests laid end to end, one `Vec` each, and the call's times. The requests are
    /// hashed while the GPU works.
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
        self.upload(&p);
        autoreleasepool(|| {
            let cmds = self.encode(&p.plan, mode)?;
            self.dev.finish(&cmds)
        })?;
        let c = &self.calls[&(p.plan.requests, p.plan.scan_segs + p.plan.pack_segs)];
        let n = count * self.shape.cols;
        // SAFETY: the buffer is shared with the host and holds the plan's requests × columns words.
        Ok(unsafe { std::slice::from_raw_parts(c.words.as_ref().unwrap().contents() as *const u64, n) }.to_vec())
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

impl MetalFetchDevice {
    /// Waits for command buffers that are already committed, in order.
    fn finish_committed(&self, cmds: &[&CommandBufferRef]) -> Result<(), String> {
        cmds[cmds.len() - 1].wait_until_completed();
        for c in cmds {
            if c.status() != MTLCommandBufferStatus::Completed {
                return Err("Metal: a command buffer failed".into());
            }
        }
        Ok(())
    }
}
