//! The per-block build on an NVIDIA GPU through CUDA: the host side of `cuda_build.cu`, a port of the native Metal
//! build (`metal/src/gpu.rs`, its `Arith::U32` variant).
//!
//! `CudaDevice::new` opens the device and compiles the kernels with NVRTC. `CudaBuilder::with_device` prepares the
//! constant tables of one table shape (rows) from the crates' own data (`gpu_prep::GpuPrep`: the three 30-bit primes'
//! NTT tables, the prepared transforms of the query masks and of the gadget masks, the chain exponents and the slot
//! order) and allocates the block's working buffers on the device. A build runs one column block from its cells to
//! the compiled matrix in i32 words; its output words are the CPU build's, which `digest` hashes as `pack_digest`
//! does.
//!
//! A module of the programs that use it (`#[path]`), not of the library: it needs patch 0003's accessors and the
//! `cuda` feature. Everything runs on one stream, so cudarc's per-buffer event tracking (which orders work between
//! streams) is turned off.
use crate::gpu_prep::GpuPrep;
use crate::rns32::Rns32;
use cudarc::{
    driver::{
        sys::CUevent_flags, CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, DeviceRepr, LaunchConfig,
        PushKernelArg, ValidAsZeroBits,
    },
    nvrtc::compile_ptx,
};
use ipir_sp::native::NativePublicSetup;
use sha2::{Digest, Sha256};
use std::sync::Arc;
use std::time::{Duration, Instant};

const SOURCE: &str = include_str!("cuda_build.cu");

fn err(e: impl std::fmt::Display) -> String {
    format!("CUDA: {e}")
}

/// `IPIR_CUDA_WARP_SYNC=1` makes the barrier of the NTT stages whose butterflies stay inside the 64 elements one warp
/// owns a `__syncwarp()` (where the Metal kernels use a SIMD-group barrier). By default every barrier is a block-wide
/// `__syncthreads()`.
pub fn warp_sync() -> bool {
    std::env::var("IPIR_CUDA_WARP_SYNC").is_ok_and(|v| v == "1")
}

/// Garner CRT constants of the three primes, as the kernels' `Crt32`.
#[repr(C)]
#[derive(Clone, Copy)]
struct Crt32 {
    p: [u32; 3],
    inv01: u32,
    inv01q: u32,
    inv012: u32,
    inv012q: u32,
    p0_mod_p2: u32,
    p0_mod_p2q: u32,
    _pad: u32,
    p01: u64,
    prod_lo: u64,
    prod_hi: u64,
    half_lo: u64,
    half_hi: u64,
    qmask: u64,
}

// SAFETY: repr(C), integers only, no implicit padding: the layout of the kernels' `Crt32`.
unsafe impl DeviceRepr for Crt32 {}

/// One stage's time: the device's own time (between two CUDA events around its launches) and the host wall time
/// around it (launches and the wait for the stream).
#[derive(Clone, Copy, Debug, Default)]
pub struct StageTime {
    pub gpu: Duration,
    pub wall: Duration,
}

#[derive(Clone, Debug, Default)]
pub struct CudaTiming {
    /// The copy of the block's cells from host memory to the device.
    pub upload_cells: Duration,
    pub hints: StageTime,
    pub aggregation: StageTime,
    pub chains: StageTime,
    pub compile: StageTime,
    /// All four stages launched back to back (the figure a production build would see).
    pub fused: StageTime,
}

/// The device, one stream, and the compiled kernels: shared by the builders of every table.
pub struct CudaDevice {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    hint: CudaFunction,
    agg_gather: CudaFunction,
    agg_stage64: CudaFunction,
    agg_widen: CudaFunction,
    agg_stage128: CudaFunction,
    agg_last: CudaFunction,
    zero_rows: CudaFunction,
    chains: CudaFunction,
    cmp_stage: CudaFunction,
    cmp_matrix: CudaFunction,
    /// Opening the device and compiling the kernels.
    pub init: Duration,
}

impl CudaDevice {
    /// Opens device `ordinal` and compiles the kernels for d = 2,048.
    pub fn new(ordinal: usize) -> Result<Arc<Self>, String> {
        let t = Instant::now();
        // SAFETY: these probes only load the vendor libraries.
        if !unsafe { cudarc::driver::sys::is_culib_present() } || !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
            return Err(err("the CUDA driver and NVRTC shared libraries are required"));
        }
        let ctx = CudaContext::new(ordinal).map_err(err)?;
        // SAFETY: one stream only; every buffer is read back or freed after that stream's work on it.
        unsafe { ctx.disable_event_tracking() };
        let d = crate::rns32::D;
        let src = format!(
            "#define D {d}\n#define LOGD {}\n#define WARP_SYNC {}\n{SOURCE}",
            d.trailing_zeros(),
            warp_sync() as u32
        );
        let module = ctx.load_module(compile_ptx(src).map_err(err)?).map_err(err)?;
        let f = |name: &str| module.load_function(name).map_err(err);
        Ok(Arc::new(CudaDevice {
            stream: ctx.new_stream().map_err(err)?,
            hint: f("hint_columns32")?,
            agg_gather: f("agg_gather")?,
            agg_stage64: f("agg_stage64")?,
            agg_widen: f("agg_stage_widen")?,
            agg_stage128: f("agg_stage128")?,
            agg_last: f("agg_stage_last")?,
            zero_rows: f("zero_rows")?,
            chains: f("trace_chains32")?,
            cmp_stage: f("cmp_stage")?,
            cmp_matrix: f("cmp_matrix")?,
            ctx,
            init: t.elapsed(),
        }))
    }

    /// The GPU's name.
    pub fn name(&self) -> String {
        self.ctx.name().unwrap_or_else(|e| format!("unknown ({e})"))
    }

    /// The device's context: a fetch on the same device opens its own stream in it (`cuda_fetch_native`), so that it
    /// can read a build's buffers.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    fn alloc<T: DeviceRepr + ValidAsZeroBits>(&self, len: usize) -> Result<CudaSlice<T>, String> {
        self.stream.alloc_zeros(len.max(1)).map_err(err)
    }

    fn event(&self) -> Result<cudarc::driver::CudaEvent, String> {
        self.stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT)).map_err(err)
    }
}

pub struct CudaBuilder {
    dev: Arc<CudaDevice>,
    d: usize,
    q: u64,
    rows: usize,
    // kernel parameters
    rows32: u32,
    polys: u32,
    bits: u32,
    dropped: u32,
    steps: u32,
    qmask: u64,
    crt: Crt32,
    n_missing: u32,
    /// Aggregation stages done in i64 (len = 2, 4, ..., 2^i64_stages).
    i64_stages: u32,
    // constant tables
    tables: CudaSlice<u32>,
    mt: CudaSlice<u32>,
    mq: CudaSlice<u32>,
    wt: CudaSlice<u32>,
    wq: CudaSlice<u32>,
    exps: CudaSlice<u32>,
    inv_exps: CudaSlice<u32>,
    phat_rows: CudaSlice<u32>,
    slot_of_row: CudaSlice<u32>,
    missing_rows: CudaSlice<u32>,
    /// This table's row of the chain kernel's table of buffers (`ChainTable`): on the device, and its words.
    chain_table: CudaSlice<u64>,
    chain_row: [u64; 8],
    // working buffers
    cells: CudaSlice<u16>,
    masks: CudaSlice<u64>,
    agg64_a: CudaSlice<i64>,
    agg64_b: CudaSlice<i64>,
    agg128_a: CudaSlice<u64>,
    agg128_b: CudaSlice<u64>,
    slots: CudaSlice<u64>,
    phat_a: CudaSlice<i32>,
    phat_b: CudaSlice<i32>,
    matrix: CudaSlice<i32>,
}

impl CudaBuilder {
    /// Opens device 0, compiles the kernels and prepares one table shape. Several tables share one device through
    /// `CudaDevice::new` and `with_device`.
    pub fn new(setup: &NativePublicSetup) -> Result<Self, String> {
        Self::with_device(&CudaDevice::new(0)?, setup)
    }

    /// Tables and buffers for one table shape on an open device.
    pub fn with_device(dev: &Arc<CudaDevice>, setup: &NativePublicSetup) -> Result<Self, String> {
        let p = setup.profile();
        let packing = p.packing();
        let (d, q) = (packing.d(), packing.q());
        let rows = p.rows();
        let np = setup.packing_setup().params();
        // `GpuPrep::new` checks what the kernels assume: d = 2,048, one column block, q a power of two, two gadget
        // limbs, the compile within i32, all but two aggregation stages within i64.
        let prep = GpuPrep::new(setup);
        let section = |name: &str| -> &[u32] {
            &prep.sections.iter().find(|(n, _)| *n == name).unwrap_or_else(|| panic!("section {name}")).1
        };
        let logd = d.trailing_zeros();
        let mut i64_stages = 0u32;
        while (2u128 << i64_stages) * (q as u128 - 1) <= i64::MAX as u128 && (2usize << i64_stages) <= d {
            i64_stages += 1;
        }
        assert!(i64_stages >= 1 && i64_stages + 1 < logd);
        let c = Rns32::new(q).crt;
        let crt = Crt32 {
            p: c.p,
            inv01: c.inv01,
            inv01q: c.inv01q,
            inv012: c.inv012,
            inv012q: c.inv012q,
            p0_mod_p2: c.p0_mod_p2,
            p0_mod_p2q: c.p0_mod_p2q,
            _pad: 0,
            p01: c.p01,
            prod_lo: c.product as u64,
            prod_hi: (c.product >> 64) as u64,
            half_lo: (c.product / 2) as u64,
            half_hi: ((c.product / 2) >> 64) as u64,
            qmask: q - 1,
        };
        let s = &dev.stream;
        // An empty section still gets a (never read) word, so that every kernel argument is an allocation.
        let up = |name: &str| -> Result<CudaSlice<u32>, String> {
            let words = section(name);
            if words.is_empty() { dev.alloc(1) } else { s.memcpy_stod(words).map_err(err) }
        };
        let dd = d * d;
        // The i128 stages before the last one ping-pong between two buffers; this profile has none.
        let i128_stages = logd - i64_stages - 1;
        let (tables, wt, wq, exps, inv_exps, phat_rows) =
            (up("tables")?, up("gadgetT")?, up("gadgetQ")?, up("exps")?, up("invExps")?, up("phatRows")?);
        let slots: CudaSlice<u64> = dev.alloc(dd)?;
        let phat_a: CudaSlice<i32> = dev.alloc(2 * dd)?;
        let chain_row = [
            slots.device_ptr(s).0,
            phat_a.device_ptr(s).0,
            wt.device_ptr(s).0,
            wq.device_ptr(s).0,
            tables.device_ptr(s).0,
            exps.device_ptr(s).0,
            inv_exps.device_ptr(s).0,
            phat_rows.device_ptr(s).0,
        ];
        let chain_table = s.memcpy_stod(&chain_row).map_err(err)?;
        let b = CudaBuilder {
            dev: dev.clone(),
            d,
            q,
            rows,
            rows32: rows as u32,
            polys: (rows / d) as u32,
            bits: np.gadget_bits(),
            dropped: np.dropped_bits(),
            steps: (d / 2 - 1) as u32,
            qmask: q - 1,
            crt,
            n_missing: section("missingRows").len() as u32,
            i64_stages,
            mt: up("hintT")?,
            mq: up("hintQ")?,
            slot_of_row: up("slotOfRow")?,
            missing_rows: up("missingRows")?,
            tables,
            wt,
            wq,
            exps,
            inv_exps,
            phat_rows,
            chain_table,
            chain_row,
            cells: dev.alloc(rows * d)?,
            masks: dev.alloc(dd)?,
            agg64_a: dev.alloc(dd)?,
            agg64_b: dev.alloc(dd)?,
            agg128_a: dev.alloc(2 * dd)?,
            agg128_b: dev.alloc(if i128_stages > 1 { 2 * dd } else { 1 })?,
            slots,
            phat_a,
            phat_b: dev.alloc(2 * dd)?,
            matrix: dev.alloc(2 * dd)?,
        };
        s.synchronize().map_err(err)?;
        Ok(b)
    }

    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.dev
    }

    pub fn q(&self) -> u64 {
        self.q
    }

    /// The build as a list of dependent levels: each level's launches depend only on earlier levels. Levels: hints
    /// (with zeroing the compile input's unwritten rows), aggregation gather, its stages, the chains, the compile
    /// stages, the matrix.
    pub fn levels(&self) -> usize {
        2 + self.d.trailing_zeros() as usize + 1 + self.d.trailing_zeros() as usize + 1
    }

    /// Level ranges of the four stages: hints, aggregation, chains, compile.
    pub fn stage_levels(&self) -> [std::ops::Range<usize>; 4] {
        let logd = self.d.trailing_zeros() as usize;
        [0..1, 1..2 + logd, 2 + logd..3 + logd, 3 + logd..self.levels()]
    }

    fn chain_level(&self) -> usize {
        2 + self.d.trailing_zeros() as usize
    }

    /// Launches the chain kernel for the `tables` tables whose buffers `chain_tables` lists (two blocks each).
    fn launch_chains(&self, chain_tables: &CudaSlice<u64>, tables: usize) -> Result<(), String> {
        let dv = &*self.dev;
        let mut a = dv.stream.launch_builder(&dv.chains);
        a.arg(chain_tables).arg(&self.bits).arg(&self.dropped).arg(&self.steps).arg(&self.qmask).arg(&self.crt);
        // SAFETY: the arguments are the kernel's parameters in order; every table's buffers are allocations of the
        // sizes the kernel indexes ([d][d] slots, [2][d][d] digits, [3][2][d] masks, [3][4][d] tables, d - 2 steps).
        unsafe {
            a.launch(LaunchConfig {
                grid_dim: (2 * tables as u32, 1, 1),
                block_dim: ((self.d / 2) as u32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(err)?;
        Ok(())
    }

    /// Launches level `level` on the device's stream.
    pub fn launch_level(&self, level: usize) -> Result<(), String> {
        let dv = &*self.dev;
        let s = &dv.stream;
        let d = self.d as u32;
        let logd = self.d.trailing_zeros() as usize;
        let n64 = self.i64_stages as usize;
        // Blocks of 256 threads along x: a grid of (d, y, z) threads.
        let rows_of = |y: u32, z: u32| LaunchConfig { grid_dim: (d / 256, y, z), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        // SAFETY (every launch below): the arguments are the kernel's parameters in order, and each buffer has the
        // size the kernel indexes for this grid (see the kernels' parameter comments).
        if level == 0 {
            let mut a = s.launch_builder(&dv.hint);
            a.arg(&self.cells).arg(&self.mt).arg(&self.mq).arg(&self.tables).arg(&self.masks);
            a.arg(&self.rows32).arg(&self.polys).arg(&self.crt);
            unsafe { a.launch(LaunchConfig { grid_dim: (d, 1, 1), block_dim: (d / 2, 1, 1), shared_mem_bytes: 0 }) }
                .map_err(err)?;
            if self.n_missing > 0 {
                let mut a = s.launch_builder(&dv.zero_rows);
                a.arg(&self.phat_a).arg(&self.missing_rows);
                unsafe { a.launch(rows_of(self.n_missing, 2)) }.map_err(err)?;
            }
        } else if level == 1 {
            let mut a = s.launch_builder(&dv.agg_gather);
            a.arg(&self.masks).arg(&self.agg64_a);
            unsafe { a.launch(rows_of(d, 1)) }.map_err(err)?;
        } else if level < 2 + logd {
            let st = level - 2; // aggregation stage st has len = 2^(st+1)
            let len = 2u32 << st;
            // Ping-pong buffers: gather writes agg64_a; i64 stage st (0-based) reads a if st is even.
            let agg64 = |st: usize| if st % 2 == 0 { (&self.agg64_a, &self.agg64_b) } else { (&self.agg64_b, &self.agg64_a) };
            if st < n64 {
                let (src, dst) = agg64(st);
                let mut a = s.launch_builder(&dv.agg_stage64);
                a.arg(src).arg(dst).arg(&len);
                unsafe { a.launch(rows_of(d / 2, 1)) }.map_err(err)?;
            } else if st == n64 {
                let mut a = s.launch_builder(&dv.agg_widen);
                a.arg(agg64(st).0).arg(&self.agg128_a).arg(&len);
                unsafe { a.launch(rows_of(d / 2, 1)) }.map_err(err)?;
            } else {
                // i128 stage k = st - n64 (k >= 1) reads agg128_a if k is odd.
                let k = st - n64;
                let (src, dst) = if k % 2 == 1 { (&self.agg128_a, &self.agg128_b) } else { (&self.agg128_b, &self.agg128_a) };
                if st + 1 < logd {
                    let mut a = s.launch_builder(&dv.agg_stage128);
                    a.arg(src).arg(dst).arg(&len);
                    unsafe { a.launch(rows_of(d / 2, 1)) }.map_err(err)?;
                } else {
                    let mut a = s.launch_builder(&dv.agg_last);
                    a.arg(src).arg(&self.slots).arg(&self.slot_of_row).arg(&self.qmask);
                    unsafe { a.launch(rows_of(d / 2, 1)) }.map_err(err)?;
                }
            }
        } else if level == 2 + logd {
            self.launch_chains(&self.chain_table, 1)?;
        } else if level < 3 + 2 * logd {
            let st = level - (3 + logd);
            let len = 2u32 << st;
            let (src, dst) = if st % 2 == 0 { (&self.phat_a, &self.phat_b) } else { (&self.phat_b, &self.phat_a) };
            let mut a = s.launch_builder(&dv.cmp_stage);
            a.arg(src).arg(dst).arg(&len);
            unsafe { a.launch(rows_of(d / 2, 2)) }.map_err(err)?;
        } else {
            assert_eq!(level, 3 + 2 * logd);
            let src = if logd % 2 == 0 { &self.phat_a } else { &self.phat_b };
            let mut a = s.launch_builder(&dv.cmp_matrix);
            a.arg(src).arg(&self.matrix);
            unsafe { a.launch(LaunchConfig { grid_dim: (d / 32, d / 32, 2), block_dim: (32, 8, 1), shared_mem_bytes: 0 }) }
                .map_err(err)?;
        }
        Ok(())
    }

    /// Launches `levels` and waits for them.
    pub fn run(&self, levels: std::ops::Range<usize>) -> Result<StageTime, String> {
        let t = Instant::now();
        let start = self.dev.event()?;
        for l in levels {
            self.launch_level(l)?;
        }
        let end = self.dev.event()?;
        self.dev.stream.synchronize().map_err(err)?;
        let gpu = start.elapsed_ms(&end).map_err(err)?;
        Ok(StageTime { gpu: Duration::from_secs_f64(f64::from(gpu) / 1e3), wall: t.elapsed() })
    }

    /// Copies a block's cells from host memory to the device, and waits for the copy.
    pub fn upload(&mut self, cells: &[u16]) -> Result<Duration, String> {
        assert_eq!(cells.len(), self.rows * self.d);
        let t = Instant::now();
        let s = self.dev.stream.clone();
        s.memcpy_htod(cells, &mut self.cells).map_err(err)?;
        s.synchronize().map_err(err)?;
        Ok(t.elapsed())
    }

    /// One block's build, each stage launched and waited for on its own (for the per-stage times).
    pub fn build_staged(&mut self, cells: &[u16]) -> Result<CudaTiming, String> {
        let upload_cells = self.upload(cells)?;
        let [h, a, c, m] = self.stage_levels();
        Ok(CudaTiming {
            upload_cells,
            hints: self.run(h)?,
            aggregation: self.run(a)?,
            chains: self.run(c)?,
            compile: self.run(m)?,
            fused: StageTime::default(),
        })
    }

    /// One block's build, all levels launched back to back.
    pub fn build_fused(&mut self, cells: &[u16]) -> Result<CudaTiming, String> {
        let upload_cells = self.upload(cells)?;
        let fused = self.run(0..self.levels())?;
        Ok(CudaTiming { upload_cells, fused, ..Default::default() })
    }

    fn read<T: DeviceRepr>(&self, buf: &CudaSlice<T>) -> Result<Vec<T>, String> {
        let v = self.dev.stream.memcpy_dtov(buf).map_err(err)?;
        self.dev.stream.synchronize().map_err(err)?;
        Ok(v)
    }

    /// The hint masks, d words per column.
    pub fn hint_masks(&self) -> Result<Vec<Vec<u64>>, String> {
        Ok(self.read(&self.masks)?.chunks(self.d).map(|c| c.to_vec()).collect())
    }

    /// The slots: after the aggregation, its rows in slot order; after the chains, slot 0 is the mask and slot d/2
    /// the other mask.
    pub fn slot_rows(&self) -> Result<Vec<Vec<u64>>, String> {
        Ok(self.read(&self.slots)?.chunks(self.d).map(|c| c.to_vec()).collect())
    }

    /// The compile input, [limb][row][d]: the chains' digits. Only between the chains and the compile, whose stages
    /// reuse the buffer.
    pub fn compile_input(&self) -> Result<Vec<i32>, String> {
        self.read(&self.phat_a)
    }

    /// The compiled matrix as i32 words (d rows, 2d columns). Its entries fit i32 by the bound `GpuPrep::new` checks.
    pub fn matrix_i32(&self) -> Result<Vec<i32>, String> {
        self.read(&self.matrix)
    }

    /// Bytes of the compiled matrix the answer kernel reads (i32 words).
    pub fn matrix_bytes(&self) -> usize {
        2 * self.d * self.d * 4
    }

    /// The last uploaded block's cells on the device (column-major, rows × d), and the last build's compiled matrix
    /// (d rows of 2d words): what a table that answers installs (`CudaFetch::write_block_device`). Read them only
    /// after the build was waited for.
    pub fn cells_buffer(&self) -> &CudaSlice<u16> {
        &self.cells
    }

    pub fn matrix_buffer(&self) -> &CudaSlice<i32> {
        &self.matrix
    }
}

/// Several tables' builds on one stream: each table's hints and aggregation, then every table's chains in one launch
/// (two blocks per table, side by side on the device), then each table's compile. Waits for the end and returns the
/// device's time in milliseconds, from the first launch to the end of the last.
pub fn launch_many(builders: &[&CudaBuilder]) -> Result<f64, String> {
    let first = builders[0];
    let dv = &*first.dev;
    for b in builders {
        assert!(Arc::ptr_eq(&b.dev, &first.dev), "one device");
        assert!(
            (b.d, b.bits, b.dropped, b.steps, b.qmask) == (first.d, first.bits, first.dropped, first.steps, first.qmask),
            "one set of parameters"
        );
    }
    let rows: Vec<u64> = builders.iter().flat_map(|b| b.chain_row).collect();
    let chain_tables = dv.stream.memcpy_stod(&rows).map_err(err)?;
    let chain_level = first.chain_level();
    let start = dv.event()?;
    for b in builders {
        for l in 0..chain_level {
            b.launch_level(l)?;
        }
    }
    first.launch_chains(&chain_tables, builders.len())?;
    for b in builders {
        for l in chain_level + 1..b.levels() {
            b.launch_level(l)?;
        }
    }
    let end = dv.event()?;
    dv.stream.synchronize().map_err(err)?;
    Ok(f64::from(start.elapsed_ms(&end).map_err(err)?))
}

fn hex(d: impl AsRef<[u8]>) -> String {
    d.as_ref().iter().map(|b| format!("{b:02x}")).collect()
}

fn update(h: &mut Sha256, words: &[u64]) {
    for chunk in words.chunks(1 << 12) {
        h.update(chunk.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
    }
}

/// `pack_digest`'s digest of the last build's output words: (all, hints, packing). `pack_digest` hashes the name of
/// the CPU build's matrix storage with the words, and that name depends on the CPU (`i32`, or `packed27` where the
/// CPU has the instructions for it): `IPIR_STORAGE_LABEL` gives the name the CPU reports (`pack_digest --parts`
/// prints it as `storage=`), `i32` when it is not set.
pub fn digest(b: &CudaBuilder) -> Result<(String, String, String), String> {
    let (d, q) = (b.d, b.q);
    let masks = b.hint_masks()?;
    let slots = b.slot_rows()?;
    let matrix: Vec<u64> = b.matrix_i32()?.iter().map(|&x| (x as i128).rem_euclid(q as i128) as u64).collect();
    let (mut hints, mut packing) = (Sha256::new(), Sha256::new());
    for mask in &masks {
        update(&mut hints, mask);
    }
    update(&mut packing, &slots[0]);
    update(&mut packing, &slots[d / 2]);
    packing.update(std::env::var("IPIR_STORAGE_LABEL").unwrap_or_else(|_| "i32".into()));
    update(&mut packing, &[d as u64, 2 * d as u64, q]);
    update(&mut packing, &matrix);
    let (hints, packing) = (hints.finalize(), packing.finalize());
    let mut all = Sha256::new();
    all.update(hints);
    all.update(packing);
    Ok((hex(all.finalize()), hex(hints), hex(packing)))
}
