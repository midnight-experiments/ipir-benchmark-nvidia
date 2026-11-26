/**
 * A whole table built from scratch, on the CPU and on the GPU: every column block's hint columns and packing matrix
 * computed from the cells. A server does this when it starts, when a table changes its number of rows, and when a
 * table is added. (A table that only gains column blocks keeps the blocks it has: `new_without_columns`.)
 *
 *   deno run <permissions> bench/table-build.ts <rows> <cols> [--at-once 4] [--check-matrices]
 *     [--serving ROWSxCOLS [--rate 2]] [--bin ipir/target/release]
 *
 * 1. CPU: `cpu_fetch <rows> <cols> 1 --dump DIR` builds the table as the server does (its `buildMs`) and leaves its
 *    setup id, 32 requests, the CPU's 32 answers and every block's compiled matrix in DIR.
 * 2. GPU: an empty table (`GpuIpirTable.createEmpty`), then the blocks, `--at-once` of them per submission
 *    (`gpu/block-build.ts`, all their chains in one dispatch), each written into the table from the build's GPU
 *    buffers (`writeBlock`). The cells come from host memory, 2 bytes each; nothing else crosses.
 * 3. Check: the GPU-built table answers the 32 requests, and the answers are compared with the CPU's, byte for byte.
 *    `--check-matrices` also reads every block's matrix back and compares it with the CPU's (the build time then
 *    includes the reads and is not reported).
 *
 * `--serving ROWSxCOLS`: the table is built while another table of that shape, already resident, answers calls of 32
 * at `--rate` calls a second: a table re-dimensioned while the server runs, where the old and the new table are in
 * memory together. The old table's answers during the build are compared with the CPU's too.
 *
 * Prints one JSON line. GPU memory is what the benchmark's buffers hold (`trackGpuMemory`): the peak during the build,
 * and what stays resident. Exits 1 on a difference or a failure (`doesNotFit` when the GPU's memory ran out).
 */
import { GpuBlockBuildSet, parseBlockBuildPrep } from "../gpu/block-build.ts";
import { GpuIpirTable } from "../gpu/ipir.ts";
import { ipirRequestBytes } from "../ipir/sizes.ts";
import { MIB, openBenchGpu, options, parseShapes, scoped, stats, syntheticCells, trackGpuMemory, xorshift } from "./lib.ts";

const D = 2048;
const positional = Deno.args.slice(0, 2);
const rows = Number(positional[0]);
const cols = Number(positional[1]);
if (!rows || !cols) throw new Error("usage: table-build.ts <rows> <cols> [--at-once 4] [--check-matrices] [--serving ROWSxCOLS [--rate 2]]");
const opts = options(Deno.args.slice(2));
const atOnce = Math.max(1, Number(opts.get("at-once") ?? 4));
const checkMatrices = opts.has("check-matrices");
const serving = opts.has("serving") ? parseShapes(opts.get("serving")!)[0]! : null;
const rate = Number(opts.get("rate") ?? 2);
const bin = opts.get("bin") ?? "ipir/target/release";
const blocks = cols / D;
const base: Record<string, unknown> = { rows, cols, blocks, atOnce, serving: serving ? `${serving[0]}x${serving[1]}` : null };
const log = (s: string) => console.error(`table-build ${rows}x${cols}: ${s}`);

async function run(cmd: string, args: string[]): Promise<Record<string, unknown>> {
  const r = await new Deno.Command(cmd, { args, stdout: "piped", stderr: "piped" }).output();
  const lines = new TextDecoder().decode(r.stdout).trim().split("\n");
  let parsed: Record<string, unknown> = {};
  try {
    parsed = JSON.parse(lines[lines.length - 1] ?? "");
  } catch { /* no JSON */ }
  if (!r.success) throw new Error(`${cmd} ${args.join(" ")} failed: ${parsed.error ?? new TextDecoder().decode(r.stderr).slice(-400)}`);
  return parsed;
}

/** What `cpu_fetch --dump` left for a shape, and the table's cells. */
async function dumped(dir: string, r: number, c: number) {
  const cpu = await run(`${bin}/cpu_fetch`, [String(r), String(c), "1", "--dump", dir]);
  const raw = await Deno.readFile(`${dir}/matrices.bin`);
  return {
    cpu,
    setupId: await Deno.readFile(`${dir}/setup-id.bin`),
    requests: await Deno.readFile(`${dir}/requests.bin`),
    answers: await Deno.readFile(`${dir}/answers.bin`),
    matrices: new Int32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4),
    cells: syntheticCells(r, c, xorshift(0x5eed ^ r ^ c)),
  };
}
const same = (a: Uint8Array, b: Uint8Array) => a.length === b.length && a.every((x, i) => x === b[i]);

const tmp = await Deno.makeTempDir({ prefix: "ipir-table-build-" });
let stage = "building the table on the CPU";
try {
  log(stage);
  const t = await dumped(`${tmp}/new`, rows, cols);
  base.cpuBuildMs = t.cpu.buildMs;
  base.cpuMatrixStorage = t.cpu.matrixStorage;
  stage = "preparing the GPU build";
  await run(`${bin}/gpu_build_prep`, [String(rows), `${tmp}/prep.bin`]);
  const prep = parseBlockBuildPrep(await Deno.readFile(`${tmp}/prep.bin`));
  const old = serving ? await dumped(`${tmp}/old`, serving[0], serving[1]) : null;

  stage = "opening the GPU";
  const gpu = await openBenchGpu();
  const { device } = gpu;
  let lost: string | null = null;
  device.lost.then((i) => (lost = `device lost (${i.reason}): ${i.message}`));
  const memory = trackGpuMemory(device);
  await scoped(device, "compiling the kernels", () => GpuIpirTable.compileKernels(device));

  // The table that answers while the new one is built.
  let oldTable: GpuIpirTable | null = null;
  const during: { ms: number; equal: boolean }[] = [];
  let stop = false;
  let caller: Promise<void> | null = null;
  if (old && serving) {
    stage = "uploading the table that answers meanwhile";
    const per = D * 4096;
    const matrices = Array.from({ length: serving[1] / D }, (_, k) => old.matrices.subarray(k * per, (k + 1) * per));
    oldTable = await scoped(device, stage, () => GpuIpirTable.create(device, { rows: serving[0], cols: serving[1], cells: old.cells, setupId: old.setupId, matrices }));
    const first = await scoped(device, stage, () => oldTable!.respond(old.requests, 32));
    if (!same(first, old.answers)) throw new Error("the serving table's answers differ from the CPU's before the build");
  }
  base.gpuResidentBeforeMiB = Math.round(memory.live() / MIB);

  stage = "making the empty table and the build's buffers";
  let t0 = performance.now();
  const table = await scoped(device, stage, () => GpuIpirTable.createEmpty(device, { rows, cols, setupId: t.setupId }));
  const set = await scoped(device, stage, async () => {
    const s = new GpuBlockBuildSet(device, Array.from({ length: Math.min(atOnce, blocks) }, () => prep), { sharedMasks: true });
    await device.queue.onSubmittedWorkDone();
    return s;
  });
  const setupMs = performance.now() - t0;

  if (oldTable && old) {
    // Calls of 32 at a steady rate until the build ends, each compared with the CPU's answers.
    caller = (async () => {
      const start = performance.now();
      for (let i = 0; !stop; i++) {
        const due = start + (i * 1000) / rate;
        const wait = due - performance.now();
        if (wait > 0) await new Promise((r) => setTimeout(r, wait));
        if (stop) break;
        const a = await oldTable!.respond(old.requests, 32);
        during.push({ ms: performance.now() - due, equal: same(a, old.answers) });
      }
    })();
  }

  stage = "building the blocks on the GPU";
  log(`${stage} (${blocks} blocks, ${set.tables} at once)`);
  const per = D * 4096;
  let matricesEqual = 0;
  const rounds: number[] = [];
  t0 = performance.now();
  for (let b0 = 0; b0 < blocks; b0 += set.tables) {
    const n = Math.min(set.tables, blocks - b0);
    const r0 = performance.now();
    await scoped(device, stage, async () => {
      for (let i = 0; i < n; i++) set.uploadCells(i, t.cells.subarray((b0 + i) * D * rows, (b0 + i + 1) * D * rows));
      const enc = device.createCommandEncoder();
      set.encode(enc);
      device.queue.submit([enc.finish()]);
      for (let i = 0; i < n; i++) await table.writeBlock(b0 + i, set.cells(i), set.matrix(i));
    });
    rounds.push(performance.now() - r0);
    if (checkMatrices) {
      for (let i = 0; i < n; i++) {
        const got = await set.readMatrix(i);
        const want = t.matrices.subarray((b0 + i) * per, (b0 + i + 1) * per);
        if (same(got, new Uint8Array(want.buffer, want.byteOffset, want.byteLength))) matricesEqual++;
      }
    }
    if (lost) throw new Error(lost);
  }
  const gpuBuildMs = performance.now() - t0;
  const peak = memory.peak();
  stop = true;
  await caller;
  set.destroy();

  stage = "answering from the table built on the GPU";
  const answers = await scoped(device, stage, () => table.respond(t.requests, 32));
  const answersEqual = same(answers, t.answers);
  const result = {
    ok: answersEqual && (!checkMatrices || matricesEqual === blocks) && during.every((x) => x.equal),
    ...base,
    gpu: gpu.description,
    answersEqual,
    matricesEqual: checkMatrices ? matricesEqual : null,
    requestBytes: ipirRequestBytes(rows),
    answerBytes: t.answers.length / 32,
    gpuSetupMs: Math.round(setupMs * 10) / 10,
    gpuBuildMs: checkMatrices ? null : Math.round(gpuBuildMs * 10) / 10,
    gpuRoundMs: checkMatrices ? null : stats(rounds),
    gpuPeakMiB: Math.round(peak / MIB),
    gpuResidentMiB: Math.round(table.residentBytes / MIB),
    servingDuringBuild: old ? { calls: during.length, equal: during.filter((x) => x.equal).length, latencyMs: stats(during.map((x) => x.ms)) } : null,
  };
  table.release();
  oldTable?.release();
  console.log(JSON.stringify(result));
  device.destroy();
  if (!result.ok) Deno.exit(1);
} catch (e) {
  const error = String((e as Error)?.message ?? e);
  console.log(JSON.stringify({ ok: false, ...base, stage, error, doesNotFit: /out of memory|not enough memory|OutOfMemory/i.test(error) }));
  Deno.exit(1);
} finally {
  await Deno.remove(tmp, { recursive: true }).catch(() => {});
}
