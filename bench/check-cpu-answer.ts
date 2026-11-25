/**
 * The CPU backend's equality check, once per shape: answers the 32 requests that `cpu_fetch --dump DIR` wrote with
 * `GpuIpirTable` on this host's GPU, for the same table (the same xorshift cells, the setup id and the compiled
 * packing matrices that the CPU server built), and compares the answers with the CPU's, byte for byte.
 *
 *   deno run <permissions> bench/check-cpu-answer.ts <dir> <rows> <cols>
 *
 * Prints one JSON line: `{"equal":true|false,...}`, with the first differing answer and byte when they differ. Exits
 * 1 when they differ or the check fails.
 */
import { GpuIpirTable } from "../gpu/ipir.ts";
import { ipirRequestBytes, ipirResponseBytes } from "../ipir/sizes.ts";
import { openBenchGpu, scoped, syntheticCells, xorshift } from "./lib.ts";

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
  const table = await scoped(gpu.device, stage, () => GpuIpirTable.create(gpu.device, { rows, cols, cells, setupId, matrices }));
  stage = "answering on the GPU";
  const gpuAnswers = await scoped(gpu.device, stage, () => table.respond(requests, count));
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
  console.log(JSON.stringify({ equal: firstDiff === null, rows, cols, requests: count, answerBytes: cpuAnswers.length, gpu: gpu.description, firstDiff }));
  if (firstDiff) Deno.exit(1);
} catch (e) {
  console.log(JSON.stringify({ equal: false, rows, cols, stage, error: String((e as Error)?.message ?? e) }));
  Deno.exit(1);
}
