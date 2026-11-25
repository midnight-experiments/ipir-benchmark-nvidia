/**
 * Task 3: the block budget, simulated: how much of the 6 s between two chain blocks a block's table work takes, the
 * time the server does not have for answering at full speed. The goal is under 1 s (`GOAL_MS`); 3 s is the ceiling
 * (`BUDGET_MS`), the most that is tolerable and far from good. Six tables stay resident on the GPU and answer calls of 32 requests at a
 * steady rate (open loop: call k is due at k / rate seconds, whatever happened to the calls before it). Every
 * `--interval-ms` (6 s, the chain's block time) a chain block arrives and changes one column block of each table;
 * its work is:
 *
 * 1. compile: the CPU builds the six changed column blocks (`ipir/bench` `block_worker`, a separate process, so this
 *    process's event loop keeps issuing calls): setup expansion, the block's whole hint columns and its packing
 *    matrix. The hints of a block are recomputed whole: an upper bound for the hint updates of new records, which
 *    are not built yet;
 * 2. upload: the six new blocks (cells and a compiled matrix) copied to the GPU and laid out there, by creating a
 *    one-block `GpuIpirTable` of each table's rows (the same copies and layout kernels as a table's `create`) and
 *    releasing it. The matrices uploaded are synthetic (the content does not change the time).
 *
 * The block's work time (compile + upload) is compared with the 1 s goal and the 3 s ceiling, and the calls' latency (from the moment a call is
 * due to its answer in host memory) with and without block work running.
 *
 * With `--answer cpu`, the CPU backend: the six tables stay resident in a `cpu_server` process instead
 * (`--cpu-server PATH`), which answers the calls on the CPU through the vendored crates' server path, all logical
 * cores at normal priority, while `block_worker` compiles. No GPU is opened. There is no upload (the CPU server would
 * swap the compiled block in place), so `uploadMs` is null and a block's work is its compile. When the CPU cannot keep
 * up with the rate, calls queue; at the end of each run the calls still waiting after `--drain-s` (default 30) are
 * dropped and counted as `cancelled`.
 *
 *   deno run <permissions> bench/budget.ts --worker ipir/target/release/block_worker --out FILE
 *     [--set today|10m | --tables ROWSxCOLS,...] [--rates 2,5,10] [--modes seq,pool] [--blocks 30] [--interval-ms 6000] [--baseline-s 20]
 *     [--pool-threads N] [--answer gpu|cpu] [--cpu-server ipir/target/release/cpu_server] [--drain-s 30]
 *
 * Modes (`block_worker`): `seq`, the blocks one after another on all cores at normal priority; `pool`, all at once on
 * a pool of half the logical cores at the lowest priority; `pool-seq`, one after another on that pool.
 *
 * Mode `gpu` (needs `--gpu-prep PATH`, the `gpu_build_prep` program; GPU answers only): the six changed blocks are
 * built on the GPU instead (`gpu/block-build.ts`), on the device that is answering the calls. A block's work is then:
 *
 * 1. compile: the six blocks' cells written to the GPU and built there in one submission, all chains side by side
 *    (hint columns recomputed whole, as on the CPU);
 * 2. upload: each block laid out for the answer as before, but from the build's GPU buffers (its cells and the matrix
 *    it built), so nothing more crosses from the host.
 *
 * `--worker` is not needed when `gpu` is the only mode.
 */
import { type BlockBuildPrep, GpuBlockBuildSet, parseBlockBuildPrep } from "../gpu/block-build.ts";
import { GpuIpirTable } from "../gpu/ipir.ts";
import { openBenchGpu, options, residentBytes, scoped, stats, syntheticCells, syntheticMatrix, syntheticRequests, syntheticSetupId, xorshift } from "./lib.ts";

const SETS: Record<string, { name: string; rows: number; cols: number }[]> = {
  today: [
    { name: "read", rows: 2048, cols: 2048 },
    { name: "nullifier", rows: 2048, cols: 2048 },
    { name: "shielded groups", rows: 2048, cols: 4096 },
    { name: "Dust groups", rows: 6144, cols: 28672 },
    { name: "first coins", rows: 2048, cols: 2048 },
    { name: "generation groups", rows: 2048, cols: 4096 },
  ],
  "10m": [
    { name: "read", rows: 14336, cols: 34816 },
    { name: "nullifier", rows: 16384, cols: 18432 },
    { name: "shielded groups", rows: 8192, cols: 20480 },
    { name: "Dust groups", rows: 6144, cols: 28672 },
    { name: "first coins", rows: 2048, cols: 2048 },
    { name: "generation groups", rows: 2048, cols: 4096 },
  ],
};

const opts = options(Deno.args);
const out = opts.get("out");
const workerPath = opts.get("worker");
if (!out) throw new Error("--out FILE is required");
// `--tables 4096x12288,4096x8192,...`: the tables' shapes given outright (rows x columns each), for sizes between
// the two sets; `--set` is then only the name in the results.
const custom = opts.get("tables");
const setName = opts.get("set") ?? (custom ? "custom" : "today");
const tables = custom
  ? custom.split(",").map((x, i) => {
    const [rows, cols] = x.toLowerCase().split("x").map(Number);
    if (!rows || !cols) throw new Error(`bad table ${x} (expected ROWSxCOLS)`);
    return { name: `table ${i + 1}`, rows, cols };
  })
  : SETS[setName];
if (!tables) throw new Error(`unknown set ${setName}`);
const rates = (opts.get("rates") ?? "2,5,10").split(",").map(Number);
const modes = (opts.get("modes") ?? "seq,pool").split(",");
const gpuPrepPath = opts.get("gpu-prep");
if (modes.some((m) => m !== "gpu") && !workerPath) throw new Error("--worker PATH is required for the CPU modes");
if (modes.includes("gpu") && !gpuPrepPath) throw new Error("mode gpu needs --gpu-prep PATH (gpu_build_prep)");
const blocks = Number(opts.get("blocks") ?? 30);
const intervalMs = Number(opts.get("interval-ms") ?? 6000);
const baselineS = Number(opts.get("baseline-s") ?? 20);
const answerOn = opts.get("answer") ?? "gpu";
if (answerOn !== "gpu" && answerOn !== "cpu") throw new Error(`--answer must be gpu or cpu: ${answerOn}`);
const cpuServerPath = opts.get("cpu-server");
if (answerOn === "cpu" && !cpuServerPath) throw new Error("--answer cpu needs --cpu-server PATH");
if (answerOn === "cpu" && modes.includes("gpu")) throw new Error("mode gpu builds on the device that answers: it needs --answer gpu");
const drainS = Number(opts.get("drain-s") ?? 30);
const BUDGET_MS = 3000;
const GOAL_MS = 1000;
const log = (s: string) => console.error(`budget ${setName}: ${s}`);

/** Lines from a child process's stdout, one at a time (`block_worker`, `cpu_server`). */
class Worker {
  readonly #child: Deno.ChildProcess;
  readonly #writer: WritableStreamDefaultWriter<Uint8Array>;
  readonly #lines: AsyncIterator<string>;
  constructor(path: string, args: string[]) {
    this.#child = new Deno.Command(path, {
      args,
      stdin: "piped",
      stdout: "piped",
      stderr: "inherit",
    }).spawn();
    this.#writer = this.#child.stdin.getWriter();
    let buffer = "";
    const reader = this.#child.stdout.pipeThrough(new TextDecoderStream()).getReader();
    this.#lines = (async function* () {
      while (true) {
        const nl = buffer.indexOf("\n");
        if (nl >= 0) {
          const line = buffer.slice(0, nl);
          buffer = buffer.slice(nl + 1);
          yield line;
          continue;
        }
        const { value, done } = await reader.read();
        if (done) return;
        buffer += value;
      }
    })();
  }
  async line(): Promise<Record<string, unknown>> {
    const r = await this.#lines.next();
    if (r.done) throw new Error("the child process exited");
    return JSON.parse(r.value);
  }
  async send(cmd: string): Promise<void> {
    await this.#writer.write(new TextEncoder().encode(cmd + "\n"));
  }
  async close(): Promise<void> {
    await this.send("quit").catch(() => {});
    await this.#writer.close().catch(() => {});
    await this.#child.status;
  }
}

/**
 * The CPU backend's resident tables: a `cpu_server` process; `respond(i)` answers a call of 32 to table `i` and
 * resolves when the answer is ready, or rejects with "cancelled" when `cancel()` dropped it before it started.
 */
class CpuTables {
  readonly #worker: Worker;
  readonly #pending = new Map<number, { resolve: () => void; reject: (e: Error) => void }>();
  #next = 0;
  ready: Record<string, unknown> = {};
  constructor(path: string, shapes: { rows: number; cols: number }[]) {
    this.#worker = new Worker(path, [shapes.map((t) => `${t.rows}x${t.cols}`).join(",")]);
  }
  async start(): Promise<void> {
    this.ready = await this.#worker.line();
    (async () => {
      while (true) {
        let msg: Record<string, unknown>;
        try {
          msg = await this.#worker.line();
        } catch {
          for (const p of this.#pending.values()) p.reject(new Error("cpu_server exited"));
          this.#pending.clear();
          return;
        }
        const p = this.#pending.get(msg.done as number);
        if (!p) continue;
        this.#pending.delete(msg.done as number);
        if (msg.cancelled) p.reject(new Error("cancelled"));
        else p.resolve();
      }
    })();
  }
  respond(table: number): Promise<void> {
    const id = this.#next++;
    const done = new Promise<void>((resolve, reject) => this.#pending.set(id, { resolve, reject }));
    this.#worker.send(`call ${id} ${table}`);
    return done;
  }
  cancel(): Promise<void> {
    return this.#worker.send("cancel");
  }
  close(): Promise<void> {
    return this.#worker.close();
  }
}

interface Call {
  table: number;
  due: number;
  end: number;
}

const result: Record<string, unknown> = {
  set: setName,
  tables,
  rates,
  modes,
  blocks,
  intervalMs,
  baselineS,
  budgetMs: BUDGET_MS,
  goalMs: GOAL_MS,
  residentBytes: tables.reduce((a, t) => a + residentBytes(t.rows, t.cols), 0),
  answer: answerOn,
  note: "compile recomputes each changed block's whole hint columns: an upper bound for the hint updates of new records, which are not built yet",
};
let stage = answerOn === "gpu" ? "opening the GPU" : "starting cpu_server";
try {
  let device: GPUDevice | null = null;
  let lost: string | null = null;
  // The resident tables: on the GPU (`GpuIpirTable`), or in `cpu_server` on the CPU.
  const resident: { table: GpuIpirTable; requests: Uint8Array }[] = [];
  const uploads: { rows: number; cells: Uint16Array; setupId: Uint8Array }[] = [];
  let matrix: Int32Array | null = null;
  let cpu: CpuTables | null = null;
  if (answerOn === "gpu") {
    const gpu = await openBenchGpu();
    device = gpu.device;
    device.lost.then((i) => (lost = `device lost (${i.reason}): ${i.message}`));
    result.gpu = gpu.description;
    stage = "compiling the kernels";
    await scoped(device, stage, () => GpuIpirTable.compileKernels(device!));
    for (const [i, t] of tables.entries()) {
      stage = `uploading table ${t.name} ${t.rows}x${t.cols}`;
      log(stage);
      const next = xorshift(0xb10c ^ (i * 7919) ^ t.rows ^ t.cols);
      const cells = syntheticCells(t.rows, t.cols, next);
      const matrices = Array.from({ length: t.cols / 2048 }, () => syntheticMatrix(next));
      matrix ??= matrices[0]!;
      const setupId = syntheticSetupId(next);
      const table = await scoped(device, stage, () => GpuIpirTable.create(device!, { rows: t.rows, cols: t.cols, cells, setupId, matrices }));
      resident.push({ table, requests: syntheticRequests(t.rows, 32, setupId, next) });
      uploads.push({ rows: t.rows, cells: cells.slice(0, t.rows * 2048), setupId });
    }
    stage = "warming up";
    for (const r of resident) await scoped(device, stage, () => r.table.respond(r.requests, 32));
  } else {
    result.gpu = null;
    log(`building the tables on the CPU (${cpuServerPath})`);
    cpu = new CpuTables(cpuServerPath!, tables);
    await cpu.start();
    result.cpuServer = cpu.ready;
    log(`cpu_server ready: ${JSON.stringify(cpu.ready)}`);
    stage = "warming up";
    for (let i = 0; i < tables.length; i++) await cpu.respond(i);
  }
  const tableCount = tables.length;
  const answer = (i: number): Promise<unknown> => (cpu ? cpu.respond(i) : resident[i]!.table.respond(resident[i]!.requests, 32));

  // The GPU build of one block of every table, on the device that answers.
  let gpuBuild: GpuBlockBuildSet | null = null;
  if (modes.includes("gpu")) {
    stage = "making the GPU block build";
    const tmp = await Deno.makeTempDir({ prefix: "ipir-budget-" });
    try {
      const preps: BlockBuildPrep[] = [];
      for (const [i, t] of tables.entries()) {
        const file = `${tmp}/prep-${i}.bin`;
        const r = await new Deno.Command(gpuPrepPath!, { args: [String(t.rows), file], stdout: "piped", stderr: "piped" }).output();
        if (!r.success) throw new Error(`gpu_build_prep ${t.rows} failed: ${new TextDecoder().decode(r.stderr).slice(-1000)}`);
        preps.push(parseBlockBuildPrep(await Deno.readFile(file)));
      }
      const t0 = performance.now();
      gpuBuild = await scoped(device!, stage, async () => {
        // The hint masks are not read here, so the tables share one buffer for them.
        const set = new GpuBlockBuildSet(device!, preps, { sharedMasks: true });
        // One build before the clock: the kernels' first run.
        uploads.forEach((u, i) => set.uploadCells(i, u.cells));
        const enc = device!.createCommandEncoder();
        set.encode(enc);
        device!.queue.submit([enc.finish()]);
        await device!.queue.onSubmittedWorkDone();
        return set;
      });
      result.gpuBuild = { setupMs: performance.now() - t0, scratchBytes: gpuBuild.scratchBytes, cellsBytes: uploads.reduce((a, u) => a + u.cells.byteLength, 0) };
      log(`GPU block build ready: ${JSON.stringify(result.gpuBuild)}`);
    } finally {
      await Deno.remove(tmp, { recursive: true }).catch(() => {});
    }
  }

  let worker: Worker | null = null;
  if (workerPath && modes.some((m) => m !== "gpu")) {
    stage = "starting block_worker";
    const poolThreads = opts.get("pool-threads");
    worker = new Worker(workerPath, [tables.map((t) => t.rows).join(","), ...(poolThreads ? [poolThreads] : [])]);
    result.worker = await worker.line();
    log(`worker ready: ${JSON.stringify(result.worker)}`);
  }

  /** Issues calls at `rate` per second, round robin over the tables, until `stop()`; returns the calls. */
  function calls(rate: number) {
    const done: Call[] = [];
    const failures: string[] = [];
    let cancelled = 0;
    const t0 = performance.now();
    let k = 0;
    let stopped = false;
    let pending = 0;
    let timer: number | undefined;
    const tick = () => {
      if (stopped) return;
      const now = performance.now();
      while (t0 + (k * 1000) / rate <= now) {
        const due = t0 + (k * 1000) / rate;
        const i = k % tableCount;
        k++;
        pending++;
        answer(i).then(
          () => done.push({ table: i, due, end: performance.now() }),
          (e) => (String(e).includes("cancelled") ? cancelled++ : failures.push(String(e))),
        ).finally(() => pending--);
      }
      timer = setTimeout(tick, Math.max(0, t0 + (k * 1000) / rate - performance.now()));
    };
    tick();
    return {
      async stop() {
        stopped = true;
        clearTimeout(timer);
        const until = performance.now() + drainS * 1000;
        while (pending > 0 && (!cpu || performance.now() < until)) await new Promise((r) => setTimeout(r, 5));
        if (pending > 0 && cpu) {
          await cpu.cancel();
          while (pending > 0) await new Promise((r) => setTimeout(r, 5));
        }
        return { done, failures, issued: k, cancelled };
      },
    };
  }

  const latency = (cs: Call[]) => stats(cs.map((c) => c.end - c.due));
  const runs: Record<string, unknown>[] = [];
  for (const rate of rates) {
    stage = `baseline at ${rate} calls/s`;
    log(stage);
    const base = calls(rate);
    await new Promise((r) => setTimeout(r, baselineS * 1000));
    const b = await base.stop();
    if (lost) throw new Error(lost);
    runs.push({ rate, mode: "none", issued: b.issued, calls: b.done.length, cancelled: b.cancelled, failures: b.failures.slice(0, 5), answerLatencyMs: latency(b.done) });
    log(`  answers p50 ${latency(b.done).p50} ms, p99 ${latency(b.done).p99} ms${b.cancelled ? `, ${b.cancelled} of ${b.issued} calls dropped unanswered` : ""}`);

    for (const mode of modes) {
      stage = `${mode} at ${rate} calls/s`;
      log(stage);
      const run = calls(rate);
      const work: { start: number; end: number; compileMs: number; uploadMs: number | null; workMs: number; worker: unknown }[] = [];
      const phase0 = performance.now();
      for (let b = 0; b < blocks; b++) {
        const due = phase0 + b * intervalMs;
        const wait = due - performance.now();
        if (wait > 0) await new Promise((r) => setTimeout(r, wait));
        const start = performance.now();
        let reply: Record<string, unknown>;
        if (mode === "gpu") {
          await scoped(device!, "building the blocks on the GPU", async () => {
            uploads.forEach((u, i) => gpuBuild!.uploadCells(i, u.cells));
            const enc = device!.createCommandEncoder();
            gpuBuild!.encode(enc);
            device!.queue.submit([enc.finish()]);
            await device!.queue.onSubmittedWorkDone();
          });
          reply = { mode: "gpu" };
        } else {
          await worker!.send(`go ${mode}`);
          reply = await worker!.line();
        }
        const compiled = performance.now();
        if (device) {
          for (const [i, u] of uploads.entries()) {
            // After a GPU build: the cells it read and the matrix it built, both already on the GPU. Otherwise the
            // cells and a synthetic matrix from the host.
            const input = mode === "gpu"
              ? { rows: u.rows, cols: 2048, cells: gpuBuild!.cells(i), setupId: u.setupId, matrices: [gpuBuild!.matrix(i)] }
              : { rows: u.rows, cols: 2048, cells: u.cells, setupId: u.setupId, matrices: [matrix!] };
            const one = await scoped(device, "uploading a block", () => GpuIpirTable.create(device!, input));
            one.release();
          }
        }
        const end = performance.now();
        work.push({ start, end, compileMs: compiled - start, uploadMs: device ? end - compiled : null, workMs: end - start, worker: reply });
        if (lost) throw new Error(lost);
      }
      await new Promise((r) => setTimeout(r, Math.min(intervalMs, 2000)));
      const c = await run.stop();
      const overlaps = (x: Call) => work.some((w) => x.due < w.end && x.end > w.start);
      const during = c.done.filter(overlaps);
      const idle = c.done.filter((x) => !overlaps(x));
      const workMs = work.map((w) => w.workMs);
      const row = {
        rate,
        mode,
        blocks: work.length,
        blockWorkMs: stats(workMs),
        compileMs: stats(work.map((w) => w.compileMs)),
        uploadMs: device ? stats(work.map((w) => w.uploadMs!)) : null,
        overBudget: workMs.filter((x) => x > BUDGET_MS).length,
        overGoal: workMs.filter((x) => x > GOAL_MS).length,
        issued: c.issued,
        calls: c.done.length,
        cancelled: c.cancelled,
        failures: c.failures.slice(0, 5),
        answerLatencyMs: { all: latency(c.done), duringBlockWork: latency(during), outsideBlockWork: latency(idle) },
        work: work.map((w) => ({ compileMs: w.compileMs, uploadMs: w.uploadMs, workMs: w.workMs, worker: w.worker })),
      };
      runs.push(row);
      log(`  block work p50 ${row.blockWorkMs.p50} ms (${((row.blockWorkMs.p50 ?? 0) / intervalMs * 100).toFixed(1)} % of the ${intervalMs / 1000} s between blocks), p99 ${row.blockWorkMs.p99} ms (${row.overGoal}/${work.length} over 1 s, ${row.overBudget}/${work.length} over 3 s); answers during work p50 ${row.answerLatencyMs.duringBlockWork.p50} ms, p99 ${row.answerLatencyMs.duringBlockWork.p99} ms${c.cancelled ? `; ${c.cancelled} of ${c.issued} calls dropped unanswered` : ""}`);
    }
  }
  await worker?.close();
  gpuBuild?.destroy();
  for (const r of resident) r.table.release();
  await cpu?.close();
  result.ok = true;
  result.runs = runs;
  await Deno.writeTextFile(out, JSON.stringify(result, null, 1));
  device?.destroy();
} catch (e) {
  result.ok = false;
  result.stage = stage;
  result.error = String((e as Error)?.message ?? e);
  // The tables, or the GPU build's scratch next to them, exceed the GPU's memory: a result, not a fault.
  result.doesNotFit = /out of memory|not enough memory|OutOfMemory/i.test(String(result.error));
  log(`FAILED at ${stage}: ${result.error}`);
  await Deno.writeTextFile(out, JSON.stringify(result, null, 1));
  Deno.exit(1);
}
