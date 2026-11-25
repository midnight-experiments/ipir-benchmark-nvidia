/**
 * Times the build of column blocks on the GPU (`gpu/block-build.ts`): the same work as step 2's CPU `compile`, from a
 * block's cells to its hint masks, public masks and compiled matrix.
 *
 *   deno run --allow-read --allow-write --allow-run --allow-env bench/gpu-compile.ts --out FILE
 *     [--rows 2048,8192,16384,32768] [--reps 5] [--at-once 4,6] [--sets today,10m] [--set-reps 10]
 *     [--bin ipir/target/release]
 *
 * Per row count:
 * - one table: the wall time from writing the block's cells to the build's end (what a server waits for), the GPU's
 *   own time for the four passes together (timestamp queries), and for each stage alone (a second build, each pass
 *   timed on its own);
 * - N tables of that shape at once (`--at-once`), all their chains in one dispatch: wall and GPU time.
 * Per set (`--sets`): one block of each of the block budget's six tables at once, one chain block's whole compile.
 *
 * A case that fails (most often: its buffers do not fit in the GPU's free memory) is recorded with its error and the
 * run goes on with the next, on a new device if the failure lost the old one; the run is `ok` when at least one case
 * was measured. The sets are measured first and the largest cases last, so a failure costs the least.
 *
 * Kernels, tables and scratch are made before the clock starts (`setupMs`), and one build per set is not reported.
 * The cells are random; the time does not depend on them. Correctness is `bench/check-gpu-build.ts`'s job.
 */
import { BLOCK_BUILD_STAGES, type BlockBuildPrep, GpuBlockBuildSet, parseBlockBuildPrep } from "../gpu/block-build.ts";
import { TimestampSpan } from "../gpu/device.ts";
import { SET_ROWS } from "./gpu-build-sets.ts";
import { median, openBenchGpu, options, scoped, syntheticCells, xorshift } from "./lib.ts";

const opts = options(Deno.args);
const out = opts.get("out");
if (!out) throw new Error("usage: gpu-compile.ts --out FILE [options]");
const rowsList = (opts.get("rows") ?? "2048,8192,16384,32768").split(",").filter(Boolean).map(Number);
const reps = Number(opts.get("reps") ?? 5);
const atOnce = (opts.get("at-once") ?? "4,6").split(",").filter(Boolean).map(Number);
const sets = (opts.get("sets") ?? "today,10m").split(",").filter(Boolean);
const setReps = Number(opts.get("set-reps") ?? 10);
const bin = opts.get("bin") ?? "ipir/target/release";

const summary = (xs: number[]) => (xs.length ? { min: Math.min(...xs), median: median(xs), max: Math.max(...xs), n: xs.length } : null);
const result: Record<string, unknown> = { ok: false, reps, setReps };
let stage = "opening the GPU";
const tmp = await Deno.makeTempDir({ prefix: "ipir-gpu-compile-" });
try {
  const gpu = await openBenchGpu();
  let device = gpu.device;
  result.gpu = gpu.description;
  result.timestampQuery = gpu.adapter.timestampQuery;
  if (!gpu.adapter.blockBuild) throw new Error("this adapter lacks the limits the GPU block build needs (1,024 invocations per workgroup, 32 KB of workgroup memory)");
  let lost: string | null = null;
  const watch = (d: GPUDevice) => d.lost.then((i) => d === device && (lost = `device lost (${i.reason}): ${i.message}`));
  watch(device);

  const preps = new Map<number, BlockBuildPrep>();
  async function prep(rows: number): Promise<BlockBuildPrep> {
    let p = preps.get(rows);
    if (!p) {
      const file = `${tmp}/prep-${rows}.bin`;
      const r = await new Deno.Command(`${bin}/gpu_build_prep`, { args: [String(rows), file], stdout: "piped", stderr: "piped" }).output();
      if (!r.success) throw new Error(`gpu_build_prep ${rows} failed: ${new TextDecoder().decode(r.stderr).slice(-1000)}`);
      preps.set(rows, p = parseBlockBuildPrep(await Deno.readFile(file)));
    }
    return p;
  }

  /** One build of the set from its cells: wall time (cells written to build done) and the GPU's time of the passes. */
  async function build(set: GpuBlockBuildSet, cells: Uint16Array[]): Promise<{ wallMs: number; execMs: number | null }> {
    const span = TimestampSpan.open(device);
    const t = performance.now();
    cells.forEach((c, i) => set.uploadCells(i, c));
    const enc = device.createCommandEncoder();
    set.encode(enc, span);
    span?.resolve(enc);
    device.queue.submit([enc.finish()]);
    await Promise.all([device.queue.onSubmittedWorkDone(), span?.map()]);
    const wallMs = performance.now() - t;
    if (lost) throw new Error(lost);
    return { wallMs, execMs: span ? span.ms() : null };
  }

  /** One build with each stage's pass timed on its own. */
  async function buildStaged(set: GpuBlockBuildSet): Promise<Record<string, number | null>> {
    const spans = BLOCK_BUILD_STAGES.map(() => TimestampSpan.open(device));
    const enc = device.createCommandEncoder();
    BLOCK_BUILD_STAGES.forEach((s, i) => set.encodeStage(enc, s, spans[i]?.both));
    for (const s of spans) s?.resolve(enc);
    device.queue.submit([enc.finish()]);
    await Promise.all([device.queue.onSubmittedWorkDone(), ...spans.map((s) => s?.map())]);
    return Object.fromEntries(BLOCK_BUILD_STAGES.map((s, i) => [s, spans[i] ? spans[i]!.ms() : null]));
  }

  let measured = 0;
  /** `measure`, or the case's failure: recorded, and the run goes on (unless the device is lost). */
  async function time(what: string, rows: number[], n: number): Promise<Record<string, unknown>> {
    let set: GpuBlockBuildSet | null = null;
    try {
      stage = `${what}: making the kernels and tables`;
      const t0 = performance.now();
      set = await scoped(device, stage, async () => {
        const s = new GpuBlockBuildSet(device, await Promise.all(rows.map(prep)));
        await device.queue.onSubmittedWorkDone();
        return s;
      });
      const r = await measure(what, set, rows, n, performance.now() - t0);
      measured++;
      return r;
    } catch (e) {
      const deviceLost = lost;
      const error = [String((e as Error)?.message ?? e), deviceLost].filter(Boolean).join("; ");
      console.error(`gpu compile ${what}: FAILED at ${stage}: ${error}`);
      set?.destroy();
      set = null;
      if (deviceLost) {
        // The failure took the device with it (an allocation the driver could not make): go on with a new one.
        device = (await openBenchGpu()).device;
        lost = null;
        watch(device);
      }
      return { tables: rows.length, failed: true, doesNotFit: /out of memory|not enough memory|OutOfMemory/i.test(error), deviceLost: deviceLost !== null, stage, error };
    } finally {
      set?.destroy();
      await device.queue.onSubmittedWorkDone().catch(() => {});
    }
  }

  async function measure(what: string, set: GpuBlockBuildSet, rows: number[], n: number, setupMs: number) {
    const next = xorshift(0x9b1d ^ rows.reduce((a, r) => a * 31 + r, rows.length));
    const cells = rows.map((r) => syntheticCells(r, 2048, next));
    stage = `${what}: building`;
    await scoped(device, stage, () => build(set, cells));
    const wall: number[] = [], exec: number[] = [];
    for (let i = 0; i < n; i++) {
      const b = await scoped(device, stage, () => build(set, cells));
      wall.push(b.wallMs);
      if (b.execMs !== null) exec.push(b.execMs);
    }
    const stages: Record<string, number[]> = Object.fromEntries(BLOCK_BUILD_STAGES.map((s) => [s, []]));
    if (rows.length === 1) {
      for (let i = 0; i < n; i++) {
        const s = await scoped(device, stage, () => buildStaged(set));
        for (const [k, v] of Object.entries(s)) if (v !== null) stages[k]!.push(v);
      }
    }
    const r = {
      tables: rows.length,
      setupMs,
      scratchBytes: set.scratchBytes,
      cellsBytes: rows.reduce((a, x) => a + x * 2048 * 2, 0),
      wallMs: summary(wall),
      execMs: summary(exec),
      stagesExecMedianMs: rows.length === 1 ? Object.fromEntries(Object.entries(stages).map(([k, v]) => [k, median(v)])) : null,
    };
    console.error(`gpu compile ${what}: wall ${r.wallMs?.median?.toFixed(1)} ms, GPU ${r.execMs?.median?.toFixed(1) ?? "n/a"} ms`);
    return r;
  }

  // One chain block of each set first, then one table per row count, then several tables at once.
  const setResults: Record<string, unknown>[] = [];
  result.sets = setResults;
  for (const name of sets) {
    const rows = SET_ROWS[name];
    if (!rows) throw new Error(`--sets takes today and 10m: ${name}`);
    setResults.push({ set: name, rows, ...(await time(`set ${name}`, rows, setReps)) });
  }
  const shapes: (Record<string, unknown> & { atOnce: Record<string, unknown>[] })[] = [];
  result.shapes = shapes;
  for (const rows of rowsList) shapes.push({ rows, ...(await time(`${rows} rows`, [rows], reps)), atOnce: [] });
  for (const n of atOnce) {
    for (const [i, rows] of rowsList.entries()) shapes[i]!.atOnce.push(await time(`${rows} rows, ${n} tables`, Array(n).fill(rows), reps));
  }
  if (measured === 0) throw new Error("no case could be measured");
  result.ok = true;
  device.destroy();
} catch (e) {
  result.stage = stage;
  result.error = String((e as Error)?.message ?? e);
  console.error(`gpu compile FAILED at ${stage}: ${result.error}`);
} finally {
  await Deno.remove(tmp, { recursive: true }).catch(() => {});
}
await Deno.writeTextFile(out, JSON.stringify(result, null, 1));
Deno.exit(result.ok ? 0 : 1);
