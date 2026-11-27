/**
 * The CPU backend's equality check, once per shape: answers the 32 requests that `cpu_fetch --dump DIR` wrote with
 * `GpuIpirTable` on this host's GPU, for the same table (the same xorshift cells, the setup id and the compiled
 * packing matrices that the CPU server built), and compares the answers with the CPU's, byte for byte.
 *
 *   deno run <permissions> bench/check-cpu-answer.ts <dir> <rows> <cols> [--time RUNS] [--throughput SECONDS]
 *
 * Prints one JSON line: `{"equal":true|false,...}`, with the first differing answer and byte when they differ. Exits
 * 1 when they differ or the check fails.
 *
 * `--time RUNS` also times this table, the one the CPU built, as `bench/fetch-shape.ts` times a synthetic one: the
 * upload (`uploadMs`), calls of 1 and of 32 of the dump's requests (`timing`: wall time and the GPU's own time per
 * stage, medians of RUNS after a warm-up), and the write of one changed column block from host memory
 * (`blockUploadMs`, median of 5). `bench/cuda-compare.ts` reads these next to `cuda_fetch`'s for the same table.
 *
 * `--throughput SECONDS` adds the calls answered per second (`throughput`): for calls of 1 and of 32, callers that
 * each send their next call when their last one is answered, for SECONDS, with 1, 2 and 4 callers. A number of
 * callers that fails (the calls' buffers did not fit together) is recorded as null and ends that call size's levels.
 */
import { GpuIpirTable } from "../gpu/ipir.ts";
import { ipirRequestBytes, ipirResponseBytes } from "../ipir/sizes.ts";
import { median, openBenchGpu, scoped, syntheticCells, xorshift } from "./lib.ts";

const timeAt = Deno.args.indexOf("--time");
const timeRuns = timeAt >= 0 ? Number(Deno.args[timeAt + 1]) : 0;
const throughputAt = Deno.args.indexOf("--throughput");
const throughputS = throughputAt >= 0 ? Number(Deno.args[throughputAt + 1]) : 0;
const [dir, rowsArg, colsArg] = Deno.args;
if (!dir || !rowsArg || !colsArg) throw new Error("usage: check-cpu-answer.ts <dir> <rows> <cols>");
const rows = Number(rowsArg);
const cols = Number(colsArg);
let stage = "reading the dump";
try {
  const setupId = await Deno.readFile(`${dir}/setup-id.bin`);
  const requests = await Deno.readFile(`${dir}/requests.bin`);
  const cpuAnswers = await Deno.readFile(`${dir}/answers.bin`);
  const raw = await Deno.readFile(`${dir}/matrices.bin`);
  const words = new Int32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
  const per = 2048 * 4096;
  const matrices = Array.from({ length: cols / 2048 }, (_, k) => words.subarray(k * per, (k + 1) * per));
  const count = requests.length / ipirRequestBytes(rows);
  stage = "making the cells";
  const cells = syntheticCells(rows, cols, xorshift(0x5eed ^ rows ^ cols));
  stage = "opening the GPU";
  const gpu = await openBenchGpu();
  stage = "uploading the table";
  let t = performance.now();
  const table = await scoped(gpu.device, stage, () => GpuIpirTable.create(gpu.device, { rows, cols, cells, setupId, matrices }, { timestamps: timeRuns > 0 }));
  const uploadMs = performance.now() - t;
  stage = "answering on the GPU";
  const gpuAnswers = await scoped(gpu.device, stage, () => table.respond(requests, count));
  // --time: this table's calls of 1 and of 32, and the write of one changed block.
  const timed: Record<string, unknown> = {};
  if (timeRuns > 0) {
    const eachRequest = requests.length / count;
    const timing: Record<string, unknown> = {};
    for (const n of [1, 32].filter((n) => n <= count)) {
      stage = `timing calls of ${n}`;
      const call = requests.subarray(0, n * eachRequest);
      const stages: Record<string, number[]> = { unpack: [], scan: [], pack: [], finish: [], total: [] };
      const wall: number[] = [];
      await scoped(gpu.device, stage, () => table.respond(call, n));
      for (let r = 0; r < timeRuns; r++) {
        table.lastStagesMs = null;
        const t0 = performance.now();
        await scoped(gpu.device, stage, () => table.respond(call, n));
        wall.push(performance.now() - t0);
        const s = table.lastStagesMs as Record<string, number | null> | null;
        for (const k of Object.keys(stages)) {
          const v = s?.[k];
          if (v !== undefined && v !== null) stages[k]!.push(v);
        }
      }
      timing[String(n)] = {
        wallMs: { min: Math.min(...wall), median: median(wall), max: Math.max(...wall) },
        execMedianMs: Object.fromEntries(Object.entries(stages).map(([k, v]) => [k, median(v)])),
        execMeasured: stages.total!.length,
      };
    }
    stage = "writing one changed block";
    const block = { rows, cols: 2048, cells: cells.subarray(0, rows * 2048), setupId, matrices: [matrices[0]!] };
    const writes: number[] = [];
    for (let r = 0; r < 6; r++) {
      t = performance.now();
      const one = await scoped(gpu.device, stage, () => GpuIpirTable.create(gpu.device, block));
      if (r > 0) writes.push(performance.now() - t);
      one.release();
    }
    Object.assign(timed, { uploadMs, runs: timeRuns, timing, blockUploadMs: median(writes) });
  }
  if (throughputS > 0) {
    const eachRequest = requests.length / count;
    const throughput: Record<string, unknown> = { seconds: throughputS };
    for (const n of [1, 32].filter((n) => n <= count)) {
      const call = requests.subarray(0, n * eachRequest);
      const byCallers: Record<string, number | null> = {};
      let best = { callers: 0, rate: 0 };
      for (const callers of [1, 2, 4]) {
        stage = `calls per second: calls of ${n}, ${callers} caller(s)`;
        let done = 0;
        let failed: unknown = null;
        const t0 = performance.now();
        const until = t0 + throughputS * 1000;
        await Promise.all(Array.from({ length: callers }, async () => {
          while (performance.now() < until && !failed) {
            try {
              await scoped(gpu.device, stage, () => table.respond(call, n));
              done++;
            } catch (e) {
              failed = e;
            }
          }
        }));
        if (failed) {
          byCallers[String(callers)] = null;
          break;
        }
        const rate = done / ((performance.now() - t0) / 1000);
        byCallers[String(callers)] = Math.round(rate * 100) / 100;
        if (rate > best.rate) best = { callers, rate };
      }
      throughput[String(n)] = { callsPerSecond: Math.round(best.rate * 100) / 100, callers: best.callers, byCallers };
    }
    timed.throughput = throughput;
  }
  table.release();
  gpu.device.destroy();
  let firstDiff: { answer: number; byte: number } | null = null;
  const each = ipirResponseBytes(cols);
  if (gpuAnswers.length !== cpuAnswers.length) firstDiff = { answer: -1, byte: -1 };
  else {
    for (let i = 0; i < gpuAnswers.length; i++) {
      if (gpuAnswers[i] !== cpuAnswers[i]) {
        firstDiff = { answer: Math.floor(i / each), byte: i % each };
        break;
      }
    }
  }
  console.log(JSON.stringify({ equal: firstDiff === null, rows, cols, requests: count, answerBytes: cpuAnswers.length, gpu: gpu.description, firstDiff, ...timed }));
  if (firstDiff) Deno.exit(1);
} catch (e) {
  console.log(JSON.stringify({ equal: false, rows, cols, stage, error: String((e as Error)?.message ?? e) }));
  Deno.exit(1);
}
