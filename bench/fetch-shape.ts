/**
 * Task 1 for one table shape, in its own process (so a shape that exhausts the GPU's memory or loses the device
 * cannot spoil the next): `fetch`, the time to answer a call of 32 requests (the padded call every wallet sends) and
 * a call of 1 on the GPU, with the proof of concept's `GpuIpirTable` (`gpu/ipir.ts`, byte for byte as the native
 * add-on's answer, checked in its repository) on a synthetic table.
 *
 *   deno run <permissions> bench/fetch-shape.ts <rows> <cols> [runs]
 *
 * Prints one JSON line: the upload time, and for calls of 1 and 32 the wall time (from the call to the answer in
 * host memory) and the GPU's own execution time per stage (timestamp queries, when the adapter has them): min,
 * median, max of `runs` (default 9) after one warm-up call. Exits 1 with `{"ok":false,...}` on a failure.
 */
import { GpuIpirTable } from "../gpu/ipir.ts";
import { median, openBenchGpu, residentBytes, scoped, syntheticCells, syntheticMatrix, syntheticRequests, syntheticSetupId, xorshift } from "./lib.ts";

const rows = Number(Deno.args[0]);
const cols = Number(Deno.args[1]);
const runs = Number(Deno.args[2] ?? 9);
const base = { rows, cols, cells: rows * cols, colBlocks: cols / 2048, residentBytes: residentBytes(rows, cols) };
let stage = "opening the GPU";
try {
  const gpu = await openBenchGpu();
  const { device } = gpu;
  let lost: string | null = null;
  device.lost.then((i) => (lost = `device lost (${i.reason}): ${i.message}`));
  stage = "compiling the kernels";
  await scoped(device, stage, () => GpuIpirTable.compileKernels(device));
  stage = "making the synthetic table";
  const next = xorshift(0x5eed ^ rows ^ cols);
  const cells = syntheticCells(rows, cols, next);
  const matrices = Array.from({ length: cols / 2048 }, () => syntheticMatrix(next));
  const setupId = syntheticSetupId(next);
  const requests = syntheticRequests(rows, 32, setupId, next);
  const each = requests.length / 32;
  stage = "uploading the table";
  let t = performance.now();
  const table = await scoped(device, stage, () => GpuIpirTable.create(device, { rows, cols, cells, setupId, matrices }, { timestamps: true }));
  const uploadMs = performance.now() - t;
  const timing: Record<string, unknown> = {};
  for (const count of [1, 32]) {
    stage = `answering calls of ${count}`;
    const call = requests.subarray(0, count * each);
    const stages: Record<string, number[]> = { unpack: [], scan: [], pack: [], finish: [], total: [] };
    const wall: number[] = [];
    await scoped(device, stage, () => table.respond(call, count));
    for (let r = 0; r < runs; r++) {
      table.lastStagesMs = null;
      const t0 = performance.now();
      await scoped(device, stage, () => table.respond(call, count));
      wall.push(performance.now() - t0);
      const s = table.lastStagesMs as Record<string, number | null> | null;
      for (const k of Object.keys(stages)) {
        const v = s?.[k];
        if (v !== undefined && v !== null) stages[k]!.push(v);
      }
      if (lost) throw new Error(lost);
    }
    timing[String(count)] = {
      wallMs: { min: Math.min(...wall), median: median(wall), max: Math.max(...wall) },
      execMedianMs: Object.fromEntries(Object.entries(stages).map(([k, v]) => [k, median(v)])),
      execMeasured: stages.total!.length,
      call: table.lastCall,
    };
  }
  table.release();
  console.log(JSON.stringify({ ok: true, ...base, gpu: gpu.description, uploadMs, runs, timing, chunkRows: table.chunkRows }));
  device.destroy();
} catch (e) {
  console.log(JSON.stringify({ ok: false, ...base, stage, error: String((e as Error)?.message ?? e) }));
  Deno.exit(1);
}
