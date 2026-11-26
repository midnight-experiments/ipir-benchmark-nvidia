//! The per-block build on the Apple GPU through Metal: the host side of `shaders/build.metal`.
//!
//! `GpuBuilder::new` compiles the kernels and prepares the constant tables of one table shape (rows) from the
//! crates' own prepared data: spiral-rs's NTT root tables, the prepared transforms of the query masks (hints) and of
//! the gadget masks w (chains), the CRT constants, the chain exponents and the slot order. `build` runs one column
//! block's build on the GPU, from its cells to the compiled matrix in i32 words, and reports each stage's GPU time.
use crate::cpu_ref::slot_order;
use ipir_sp::native::NativePublicSetup;
use metal::objc::{msg_send, sel, sel_impl};
use metal::*;
use reinspiring::{
    compile::collapse_kg_exponents,
    lift_ntt::{LiftContext, AUX_PRIMES},
};
use std::time::{Duration, Instant};

const SOURCE: &str = include_str!("../shaders/build.metal");
const SOURCE32: &str = include_str!("../shaders/build32.metal");

/// The arithmetic of the hint and chain kernels' NTTs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arith {
    /// The crates' own 42-bit auxiliary primes and spiral's tables, 64-bit Shoup products (emulated on Apple GPUs).
    U64,
    /// Three 30-bit primes (`rns32`), native 32-bit Shoup products.
    U32,
    /// `U32` with the 32-bit high product emulated from 16-bit halves, as WGSL (which has no `mulhi`) needs.
    U32EmulatedMulhi,
    /// A WGSL proxy: `U32EmulatedMulhi` with threadgroup barriers in place of SIMD-group barriers (WGSL has no
    /// subgroup-scope memory barrier). Its 64-bit values (slots, CRT, aggregation) still use Metal's `ulong`; WGSL
    /// would hold them as u32 pairs.
    WgslProxy,
}

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
    p01: u64,
    prod_lo: u64,
    prod_hi: u64,
    half_lo: u64,
    half_hi: u64,
    qmask: u64,
}

/// `MBB_CHAIN_BLOCKED=1` selects the register-blocked 32-bit chain kernel (`trace_chains32r`, 768 threads) instead
/// of the default one (`trace_chains32`, one butterfly per thread and stage). It is correct but slower on the M4 Max.
pub fn chain_v3() -> bool {
    !std::env::var("MBB_CHAIN_BLOCKED").is_ok_and(|v| v == "1")
}

fn as_bytes<T: Copy>(x: &T) -> Vec<u8> {
    unsafe { std::slice::from_raw_parts(x as *const T as *const u8, std::mem::size_of::<T>()) }.to_vec()
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CrtConst {
    p: [u64; 3],
    inv: [u64; 2],
    cq: [u64; 3],
    p01_lo: u64,
    p01_hi: u64,
    half2_lo: u64,
    half2_hi: u64,
    prod2_lo: u64,
    half3_lo: u64,
    half3_hi: u64,
    prod3_lo: u64,
    qmask: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct HintConst {
    rows: u32,
    polys: u32,
    primes: u32,
    col0: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct TraceConst {
    primes: u32,
    bits: u32,
    dropped: u32,
    steps: u32,
    qmask: u64,
    half_q: u64,
}

fn shoup_quotient(w: u64, p: u64) -> u64 {
    ((u128::from(w) << 64) / u128::from(p)) as u64
}

fn crt_const(lift: &LiftContext, q: u64) -> CrtConst {
    let [p0, p1, p2] = AUX_PRIMES;
    let inv0 = spiral_rs::number_theory::invert_uint_mod(p0 % p1, p1).unwrap();
    let inv1 = spiral_rs::number_theory::invert_uint_mod(((p0 as u128 * p1 as u128) % p2 as u128) as u64, p2).unwrap();
    let p01 = p0 as u128 * p1 as u128;
    let p012 = p01 * p2 as u128;
    assert_eq!(lift.aux_params().len(), 3);
    assert!(q.is_power_of_two());
    CrtConst {
        p: AUX_PRIMES,
        inv: [inv0, inv1],
        cq: [shoup_quotient(inv0, p1), shoup_quotient(inv1, p2), shoup_quotient(p0, p2)],
        p01_lo: p01 as u64,
        p01_hi: (p01 >> 64) as u64,
        half2_lo: (p01 / 2) as u64,
        half2_hi: ((p01 / 2) >> 64) as u64,
        prod2_lo: p01 as u64,
        half3_lo: (p012 / 2) as u64,
        half3_hi: ((p012 / 2) >> 64) as u64,
        prod3_lo: p012 as u64,
        qmask: q - 1,
    }
}

/// One stage's time: GPU execution of its command buffer and the host wall time around it (encode, commit, wait).
#[derive(Clone, Copy, Debug, Default)]
pub struct StageTime {
    pub gpu: Duration,
    pub wall: Duration,
}

#[derive(Clone, Debug, Default)]
pub struct GpuTiming {
    /// Copy of the block's cells into the GPU-visible buffer (on this unified-memory Mac a memcpy; on a discrete GPU
    /// the upload).
    pub upload_cells: Duration,
    pub hints: StageTime,
    pub aggregation: StageTime,
    pub chains: StageTime,
    pub compile: StageTime,
    /// All four stages in one command buffer (the figure a production build would see).
    pub fused: StageTime,
}

fn bufs<T>(device: &Device, data: &[T]) -> Buffer {
    device.new_buffer_with_data(
        data.as_ptr() as *const _,
        std::mem::size_of_val(data).max(16) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

pub fn gpu_seconds_pub(cmd: &CommandBufferRef) -> f64 {
    gpu_seconds(cmd)
}

fn gpu_seconds(cmd: &CommandBufferRef) -> f64 {
    unsafe {
        let start: f64 = msg_send![cmd, GPUStartTime];
        let end: f64 = msg_send![cmd, GPUEndTime];
        end - start
    }
}

pub struct GpuBuilder {
    pub device: Device,
    queue: CommandQueue,
    hint: ComputePipelineState,
    agg_gather: ComputePipelineState,
    agg_stage64: ComputePipelineState,
    agg_widen: ComputePipelineState,
    agg_stage128: ComputePipelineState,
    agg_last: ComputePipelineState,
    zero_rows: ComputePipelineState,
    trace: ComputePipelineState,
    cmp_stage: ComputePipelineState,
    cmp_matrix: ComputePipelineState,
    pub tpb: usize,
    trace_threads: usize,
    pub arith: Arith,
    crt_bytes: Vec<u8>,
    d: usize,
    q: u64,
    rows: usize,
    hint_const: HintConst,
    trace_const: TraceConst,
    crt: CrtConst,
    // constant tables
    tables: Buffer,
    mt: Buffer,
    mq: Buffer,
    wt: Buffer,
    wq: Buffer,
    exps: Buffer,
    inv_exps: Buffer,
    phat_rows: Buffer,
    slot_of_row: Buffer,
    missing_rows: Buffer,
    n_missing: u64,
    /// Aggregation stages done in i64 (len = 2, 4, ..., 2^i64_stages).
    i64_stages: u32,
    // working buffers
    pub cells: Buffer,
    pub masks: Buffer,
    agg64_a: Buffer,
    agg64_b: Buffer,
    agg128_a: Buffer,
    agg128_b: Buffer,
    pub slots: Buffer,
    phat_a: Buffer,
    phat_b: Buffer,
    pub matrix: Buffer,
}

impl GpuBuilder {
    /// Kernels and tables for one table shape. `tpb`: threads per threadgroup of the NTT kernels (1024, 512, 256).
    pub fn new(setup: &NativePublicSetup, tpb: usize, arith: Arith) -> Self {
        let device = Device::system_default().expect("a Metal device");
        let p = setup.profile();
        let packing = p.packing();
        let (d, q) = (packing.d(), packing.q());
        let rows = p.rows();
        assert_eq!(p.cols(), d, "one column block");
        assert!(arith == Arith::U64 || tpb == d / 2, "the 32-bit kernels need {} threads", d / 2);
        let src = format!(
            "#define D {d}\n#define LOGD {}\n#define TPB {tpb}\n{}{SOURCE}\n{}",
            d.trailing_zeros(),
            match arith {
                Arith::U32EmulatedMulhi => "#define EMULATE_MULHI32 1\n",
                Arith::WgslProxy => "#define EMULATE_MULHI32 1\n#define WORKGROUP_BARRIERS_ONLY 1\n",
                _ => "",
            },
            if tpb == d / 2 { SOURCE32 } else { "" }
        );
        let options = CompileOptions::new();
        let library = device.new_library_with_source(&src, &options).unwrap_or_else(|e| panic!("Metal compile: {e}"));
        let pipeline = |name: &str| {
            let f = library.get_function(name, None).unwrap();
            device.new_compute_pipeline_state_with_function(&f).unwrap()
        };
        let (hint, trace) = match arith {
            Arith::U64 => (pipeline("hint_columns"), pipeline("trace_chains")),
            Arith::U32 | Arith::U32EmulatedMulhi | Arith::WgslProxy => (
                pipeline("hint_columns32"),
                pipeline(if chain_v3() { "trace_chains32" } else { "trace_chains32r" }),
            ),
        };
        let trace_threads = if arith != Arith::U64 && !chain_v3() { 768 } else { tpb };
        for (name, pl, n) in [("hint_columns", &hint, tpb), ("trace_chains", &trace, trace_threads)] {
            assert!(
                pl.max_total_threads_per_threadgroup() as usize >= n,
                "{name}: at most {} threads per threadgroup",
                pl.max_total_threads_per_threadgroup()
            );
        }
        let lift = LiftContext::new(d, q).unwrap();
        let crt = crt_const(&lift, q);
        let mut tables = Vec::with_capacity(3 * 4 * d);
        for params in lift.aux_params() {
            assert_eq!(params.poly_len, d);
            for t in [
                params.get_ntt_forward_table(0),
                params.get_ntt_forward_prime_table(0),
                params.get_ntt_inverse_table(0),
                params.get_ntt_inverse_prime_table(0),
            ] {
                assert_eq!(t.len(), d);
                tables.extend_from_slice(t);
            }
        }
        // Hints: the query masks' prepared transforms, as `build_with_concurrency` prepares them.
        let public = lift.prepare_public_dot(setup.query_masks(), (packing.p() - 1).min(q / 2)).unwrap();
        let (t, tq, right_bound) = public.gpu_view();
        assert!(right_bound < AUX_PRIMES[0], "cells lift without reduction");
        let flat = |x: &[Vec<Vec<u64>>]| x.iter().flatten().flatten().copied().collect::<Vec<u64>>();
        let hint_const = HintConst { rows: rows as u32, polys: (rows / d) as u32, primes: t.len() as u32, col0: 0 };
        let (mt, mq) = (flat(t), flat(tq));
        // Chains: the gadget masks' prepared transforms, as `build_internal_screened` prepares them.
        let ps = setup.packing_setup();
        let nparams = ps.params();
        assert_eq!(nparams.ell(), 2, "two gadget limbs");
        let public_w = lift.prepare_public_dot(ps.gadget_masks(), (1u64 << (nparams.gadget_bits() - 1)).min(q / 2)).unwrap();
        let (wt, wq, w_bound) = public_w.gpu_view();
        assert!(w_bound < AUX_PRIMES[0]);
        let exponents = collapse_kg_exponents(d);
        assert_eq!(exponents.len(), d - 2);
        let mut rows_seen = vec![false; d];
        let bitrev = |x: usize| x.reverse_bits() >> (usize::BITS - d.trailing_zeros());
        let phat_rows: Vec<u32> = exponents
            .iter()
            .map(|&g| {
                let r = ((g - 1) / 2) as usize;
                assert!(!rows_seen[r], "distinct exponents: the compile adds none");
                rows_seen[r] = true;
                bitrev(r) as u32
            })
            .collect();
        let inv_exps: Vec<u32> = exponents
            .iter()
            .map(|&g| spiral_rs::number_theory::invert_uint_mod(g, 2 * d as u64).unwrap() as u32)
            .collect();
        let exps32: Vec<u32> = exponents.iter().map(|&g| g as u32).collect();
        let mut slot_of_row = vec![0u32; d];
        for (s, r) in slot_order(d).into_iter().enumerate() {
            slot_of_row[r] = s as u32;
        }
        let missing_rows: Vec<u32> = (0..d).filter(|&r| !rows_seen[r]).map(|r| bitrev(r) as u32).collect();
        // The compile runs in i32: every value of its transform is at most d * 2^(bits-1) in magnitude.
        assert!((d as u64) << (nparams.gadget_bits() - 1) < 1 << 31, "compile values exceed i32");
        // Aggregation stages exact in i64: the crate's own public bound len * (q - 1) <= i64::MAX.
        let mut i64_stages = 0u32;
        while (2u128 << i64_stages) * (q as u128 - 1) <= i64::MAX as u128 && (2usize << i64_stages) <= d {
            i64_stages += 1;
        }
        assert!(i64_stages >= 1 && (i64_stages as usize) + 1 < d.trailing_zeros() as usize);
        let trace_const = TraceConst {
            primes: wt.len() as u32,
            bits: nparams.gadget_bits(),
            dropped: nparams.dropped_bits(),
            steps: (d / 2 - 1) as u32,
            qmask: q - 1,
            half_q: q / 2,
        };
        let mut hint_const = hint_const;
        let mut trace_const = trace_const;
        let (tables_buf, mt_buf, mq_buf, wt_buf, wq_buf, crt_bytes) = match arith {
            Arith::U64 => (
                bufs(&device, &tables),
                bufs(&device, &mt),
                bufs(&device, &mq),
                bufs(&device, &flat(wt)),
                bufs(&device, &flat(wq)),
                as_bytes(&crt),
            ),
            Arith::U32 | Arith::U32EmulatedMulhi | Arith::WgslProxy => {
                let rns = crate::rns32::Rns32::new(q);
                let flat32 = |x: &[Vec<Vec<u32>>]| x.iter().flatten().flatten().copied().collect::<Vec<u32>>();
                let mut tables32 = Vec::with_capacity(3 * 4 * d);
                for pr in &rns.primes {
                    for t in [&pr.fwd, &pr.fwdq, &pr.inv, &pr.invq] {
                        tables32.extend_from_slice(t);
                    }
                }
                let hp = rns.prepare(setup.query_masks(), (packing.p() - 1).min(q / 2));
                let wp = rns.prepare(ps.gadget_masks(), (1u64 << (nparams.gadget_bits() - 1)).min(q / 2));
                hint_const.primes = 3;
                trace_const.primes = 3;
                let c = rns.crt;
                let crt32 = Crt32 {
                    p: c.p,
                    inv01: c.inv01,
                    inv01q: c.inv01q,
                    inv012: c.inv012,
                    inv012q: c.inv012q,
                    p0_mod_p2: c.p0_mod_p2,
                    p0_mod_p2q: c.p0_mod_p2q,
                    p01: c.p01,
                    prod_lo: c.product as u64,
                    prod_hi: (c.product >> 64) as u64,
                    half_lo: (c.product / 2) as u64,
                    half_hi: ((c.product / 2) >> 64) as u64,
                    qmask: q - 1,
                };
                (
                    bufs(&device, &tables32),
                    bufs(&device, &flat32(&hp.t)),
                    bufs(&device, &flat32(&hp.tq)),
                    bufs(&device, &flat32(&wp.t)),
                    bufs(&device, &flat32(&wp.tq)),
                    as_bytes(&crt32),
                )
            }
        };
        let shared = MTLResourceOptions::StorageModeShared;
        let private = MTLResourceOptions::StorageModePrivate;
        let dd = (d * d) as u64;
        GpuBuilder {
            queue: device.new_command_queue(),
            agg_gather: pipeline("agg_gather"),
            agg_stage64: pipeline("agg_stage64"),
            agg_widen: pipeline("agg_stage_widen"),
            agg_stage128: pipeline("agg_stage128"),
            agg_last: pipeline("agg_stage_last"),
            zero_rows: pipeline("zero_rows"),
            cmp_stage: pipeline("cmp_stage"),
            cmp_matrix: pipeline("cmp_matrix"),
            hint,
            trace,
            tpb,
            trace_threads,
            d,
            q,
            rows,
            hint_const,
            trace_const,
            crt,
            tables: tables_buf,
            mt: mt_buf,
            mq: mq_buf,
            wt: wt_buf,
            wq: wq_buf,
            arith,
            crt_bytes,
            exps: bufs(&device, &exps32),
            inv_exps: bufs(&device, &inv_exps),
            phat_rows: bufs(&device, &phat_rows),
            slot_of_row: bufs(&device, &slot_of_row),
            n_missing: missing_rows.len() as u64,
            missing_rows: bufs(&device, &missing_rows),
            i64_stages,
            cells: device.new_buffer((rows * d * 2) as u64, shared),
            masks: device.new_buffer(dd * 8, shared),
            agg64_a: device.new_buffer(dd * 8, private),
            agg64_b: device.new_buffer(dd * 8, private),
            agg128_a: device.new_buffer(dd * 16, private),
            agg128_b: device.new_buffer(dd * 16, private),
            slots: device.new_buffer(dd * 8, shared),
            phat_a: device.new_buffer(2 * dd * 4, private),
            phat_b: device.new_buffer(2 * dd * 4, private),
            matrix: device.new_buffer(2 * dd * 4, shared),
            device,
        }
    }

    /// The build as a list of dependent levels: each level's dispatches depend only on earlier levels. Levels:
    /// hints (with zeroing the compile input's unwritten rows), aggregation gather, its stages, the chains, the
    /// compile stages, the matrix.
    pub fn levels(&self) -> usize {
        2 + self.d.trailing_zeros() as usize + 1 + self.d.trailing_zeros() as usize + 1
    }

    /// Level ranges of the four stages: hints, aggregation, chains, compile.
    pub fn stage_levels(&self) -> [std::ops::Range<usize>; 4] {
        let logd = self.d.trailing_zeros() as usize;
        [0..1, 1..2 + logd, 2 + logd..3 + logd, 3 + logd..self.levels()]
    }

    /// Encodes level `level` into `enc`.
    pub fn encode_level(&self, enc: &ComputeCommandEncoderRef, level: usize) {
        let d = self.d as u64;
        let logd = self.d.trailing_zeros() as usize;
        let qmask = self.q - 1;
        let grid2 = MTLSize::new(d, d / 2, 1);
        let tg = MTLSize::new(256, 1, 1);
        // Ping-pong buffers of the aggregation: gather writes agg64_a; i64 stage s (0-based) reads a if s even.
        let agg64 = |s: usize| if s % 2 == 0 { (&self.agg64_a, &self.agg64_b) } else { (&self.agg64_b, &self.agg64_a) };
        let n64 = self.i64_stages as usize;
        match level {
            0 => {
                enc.set_compute_pipeline_state(&self.hint);
                enc.set_buffer(0, Some(&self.cells), 0);
                enc.set_buffer(1, Some(&self.mt), 0);
                enc.set_buffer(2, Some(&self.mq), 0);
                enc.set_buffer(3, Some(&self.tables), 0);
                enc.set_buffer(4, Some(&self.masks), 0);
                enc.set_bytes(5, std::mem::size_of::<HintConst>() as u64, &self.hint_const as *const _ as *const _);
                enc.set_bytes(6, self.crt_bytes.len() as u64, self.crt_bytes.as_ptr() as *const _);
                enc.dispatch_thread_groups(MTLSize::new(d, 1, 1), MTLSize::new(self.tpb as u64, 1, 1));
                if self.n_missing > 0 {
                    enc.set_compute_pipeline_state(&self.zero_rows);
                    enc.set_buffer(0, Some(&self.phat_a), 0);
                    enc.set_buffer(1, Some(&self.missing_rows), 0);
                    enc.dispatch_threads(MTLSize::new(d, self.n_missing, 2), tg);
                }
            }
            1 => {
                enc.set_compute_pipeline_state(&self.agg_gather);
                enc.set_buffer(0, Some(&self.masks), 0);
                enc.set_buffer(1, Some(&self.agg64_a), 0);
                enc.dispatch_threads(MTLSize::new(d, d, 1), tg);
            }
            l if l < 2 + logd => {
                let s = l - 2; // aggregation stage s has len = 2^(s+1)
                let len = 2u32 << s;
                enc.set_bytes(2, 4, &len as *const u32 as *const _);
                if s < n64 {
                    let (src, dst) = agg64(s);
                    enc.set_compute_pipeline_state(&self.agg_stage64);
                    enc.set_buffer(0, Some(src), 0);
                    enc.set_buffer(1, Some(dst), 0);
                } else if s == n64 {
                    enc.set_compute_pipeline_state(&self.agg_widen);
                    enc.set_buffer(0, Some(agg64(s).0), 0);
                    enc.set_buffer(1, Some(&self.agg128_a), 0);
                } else {
                    // i128 stage k = s - n64 (k >= 1) reads agg128_a if k is odd.
                    let k = s - n64;
                    let (src, dst) =
                        if k % 2 == 1 { (&self.agg128_a, &self.agg128_b) } else { (&self.agg128_b, &self.agg128_a) };
                    if s + 1 < logd {
                        enc.set_compute_pipeline_state(&self.agg_stage128);
                        enc.set_buffer(0, Some(src), 0);
                        enc.set_buffer(1, Some(dst), 0);
                    } else {
                        enc.set_compute_pipeline_state(&self.agg_last);
                        enc.set_buffer(0, Some(src), 0);
                        enc.set_buffer(1, Some(&self.slots), 0);
                        enc.set_buffer(2, Some(&self.slot_of_row), 0);
                        enc.set_bytes(3, 8, &qmask as *const u64 as *const _);
                    }
                }
                enc.dispatch_threads(grid2, tg);
            }
            l if l == 2 + logd => {
                enc.set_compute_pipeline_state(&self.trace);
                enc.set_buffer(0, Some(&self.slots), 0);
                enc.set_buffer(1, Some(&self.phat_a), 0);
                enc.set_buffer(2, Some(&self.wt), 0);
                enc.set_buffer(3, Some(&self.wq), 0);
                enc.set_buffer(4, Some(&self.tables), 0);
                enc.set_buffer(5, Some(&self.exps), 0);
                enc.set_buffer(6, Some(&self.inv_exps), 0);
                enc.set_buffer(7, Some(&self.phat_rows), 0);
                enc.set_bytes(8, std::mem::size_of::<TraceConst>() as u64, &self.trace_const as *const _ as *const _);
                enc.set_bytes(9, self.crt_bytes.len() as u64, self.crt_bytes.as_ptr() as *const _);
                enc.dispatch_thread_groups(MTLSize::new(2, 1, 1), MTLSize::new(self.trace_threads as u64, 1, 1));
            }
            l if l < 3 + 2 * logd => {
                let s = l - (3 + logd);
                let len = 2u32 << s;
                let (src, dst) = if s % 2 == 0 { (&self.phat_a, &self.phat_b) } else { (&self.phat_b, &self.phat_a) };
                enc.set_compute_pipeline_state(&self.cmp_stage);
                enc.set_buffer(0, Some(src), 0);
                enc.set_buffer(1, Some(dst), 0);
                enc.set_bytes(2, 4, &len as *const u32 as *const _);
                enc.dispatch_threads(MTLSize::new(d, d / 2, 2), tg);
            }
            l => {
                assert_eq!(l, 3 + 2 * logd);
                let src = if logd % 2 == 0 { &self.phat_a } else { &self.phat_b };
                enc.set_compute_pipeline_state(&self.cmp_matrix);
                enc.set_buffer(0, Some(src), 0);
                enc.set_buffer(1, Some(&self.matrix), 0);
                enc.dispatch_thread_groups(MTLSize::new(d / 32, d / 32, 2), MTLSize::new(32, 8, 1));
            }
        }
    }

    /// Encodes `levels` into one serial compute encoder.
    pub fn encode_levels(&self, cmd: &CommandBufferRef, levels: std::ops::Range<usize>) {
        let enc = cmd.new_compute_command_encoder();
        for l in levels {
            self.encode_level(enc, l);
        }
        enc.end_encoding();
    }

    fn run(&self, f: impl Fn(&CommandBufferRef)) -> StageTime {
        let t = Instant::now();
        let cmd = self.queue.new_command_buffer();
        f(cmd);
        cmd.commit();
        cmd.wait_until_completed();
        assert_eq!(cmd.status(), MTLCommandBufferStatus::Completed, "command buffer failed");
        StageTime { gpu: Duration::from_secs_f64(gpu_seconds(cmd)), wall: t.elapsed() }
    }

    /// Copies a block's cells into the GPU buffer.
    pub fn upload(&self, cells: &[u16]) -> Duration {
        assert_eq!(cells.len(), self.rows * self.d);
        let t = Instant::now();
        unsafe { std::ptr::copy_nonoverlapping(cells.as_ptr(), self.cells.contents() as *mut u16, cells.len()) };
        t.elapsed()
    }

    /// One block's build, each stage in its own command buffer (for the per-stage times).
    pub fn build_staged(&self, cells: &[u16]) -> GpuTiming {
        let upload_cells = self.upload(cells);
        let [h, a, c, m] = self.stage_levels();
        GpuTiming {
            upload_cells,
            hints: self.run(|cmd| self.encode_levels(cmd, h.clone())),
            aggregation: self.run(|cmd| self.encode_levels(cmd, a.clone())),
            chains: self.run(|cmd| self.encode_levels(cmd, c.clone())),
            compile: self.run(|cmd| self.encode_levels(cmd, m.clone())),
            fused: StageTime::default(),
        }
    }

    /// One block's build in one command buffer.
    pub fn build_fused(&self, cells: &[u16]) -> GpuTiming {
        let upload_cells = self.upload(cells);
        let fused = self.run(|cmd| self.encode_all(cmd));
        GpuTiming { upload_cells, fused, ..Default::default() }
    }

    /// Encodes the whole build into `cmd`.
    pub fn encode_all(&self, cmd: &CommandBufferRef) {
        self.encode_levels(cmd, 0..self.levels());
    }

    pub fn queue(&self) -> &CommandQueue {
        &self.queue
    }

    fn read<T: Copy>(&self, buf: &Buffer, len: usize) -> Vec<T> {
        let ptr = buf.contents() as *const T;
        unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
    }

    /// The hint masks, d words per column.
    pub fn hint_masks(&self) -> Vec<Vec<u64>> {
        self.read::<u64>(&self.masks, self.d * self.d).chunks(self.d).map(|c| c.to_vec()).collect()
    }

    /// The slots (after a build: slot 0 is the mask, slot d/2 the other mask).
    pub fn slot_rows(&self) -> Vec<Vec<u64>> {
        self.read::<u64>(&self.slots, self.d * self.d).chunks(self.d).map(|c| c.to_vec()).collect()
    }

    /// The compiled matrix as i32 words (d rows, 2d columns). Its entries fit i32 by the bound `new` checks, so
    /// the second value (an entry outside i32) is always false.
    pub fn matrix_i32(&self) -> (Vec<i32>, bool) {
        (self.read::<i32>(&self.matrix, 2 * self.d * self.d), false)
    }

    /// Bytes of the compiled matrix the answer kernel reads (i32 words).
    pub fn matrix_bytes(&self) -> usize {
        2 * self.d * self.d * 4
    }
}

/// Several tables' builds in one command buffer: one concurrent compute encoder, level by level across the tables,
/// with a buffer memory barrier between levels. Returns the command buffer's GPU time.
pub fn encode_many(builders: &[&GpuBuilder], cmd: &CommandBufferRef) {
    let enc = cmd.compute_command_encoder_with_dispatch_type(MTLDispatchType::Concurrent);
    let levels = builders.iter().map(|b| b.levels()).max().unwrap();
    for l in 0..levels {
        for b in builders {
            if l < b.levels() {
                b.encode_level(enc, l);
            }
        }
        unsafe {
            // MTLBarrierScopeBuffers = 1
            let _: () = msg_send![enc, memoryBarrierWithScope: 1u64];
        }
    }
    enc.end_encoding();
}
