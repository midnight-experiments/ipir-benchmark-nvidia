/**
 * Upstream's CUDA kernel against the WebGPU path, on an NVIDIA GPU: the time of a whole answer to a call of 32
 * requests (and of 1), two ways, for the same table and the same requests.
 *
 * - CUDA: the first pass (the table times the request) on the GPU with `simplepir_kernel::cuda::CudaKernel`, one
 *   request at a time, and the packing of every column block on this host's CPU (`cuda_fetch`). That is all the
 *   kernel offers: it has no packing, and takes one request per call.
 * - WebGPU: both steps on the GPU (`GpuIpirTable`, `gpu/ipir.ts`), the 32 requests in one submission.
 *
 * Per shape, each in its own process so that one that does not fit cannot spoil the next:
 *
 * 1. `cpu_fetch <rows> <cols> 1 --dump DIR`: the table built on the CPU; its setup id, its 32 requests, its compiled
 *    packing matrices and the CPU's 32 answers go to DIR.
 * 2. `cuda_fetch <rows> <cols> <runs> --expect DIR`: the CUDA path's answers, compared byte for byte with the CPU's,
 *    and its times; also the same call on the CPU alone, and the table's upload again (the kernel has no call for
 *    one changed column block).
 * 3. `check-cpu-answer.ts DIR <rows> <cols> --time <runs>`: the WebGPU path's answers for the dumped table, compared
 *    byte for byte with the CPU's, its times, and the write of one changed column block.
 *
 * 4. With `--cuda-fetch-native PATH` (`cuda_fetch_native`): this benchmark's own CUDA kernels, the first pass for
 *    all 32 requests in one launch and the packing on the GPU; its answers compared in the same way (`cudaNative`).
 *
 *   deno run <permissions> bench/cuda-compare.ts --out FILE [--shapes 2048x2048,...] [--runs 9] [--no-cuda]
 *     [--cuda-fetch ipir/target/release/cuda_fetch] [--cpu-fetch ipir/target/release/cpu_fetch] [--timeout-s 1800]
 *     [--cuda-fetch-native ipir/target/release/cuda_fetch_native] [--throughput SECONDS]
 *
 * `--throughput SECONDS`: every path also measures the calls it answers per second, for calls of 1 and of 32 (each
 * program's `--throughput`).
 *
 * `--no-cuda` leaves step 2 out (a host without CUDA): the CPU's and the WebGPU path's times for the same table.
 * Step 1 runs `cpu_fetch` with the same number of runs, so its times of the CPU alone are kept too (`cpu`).
 *
 * A shape whose table does not fit in the GPU's free memory on one path is recorded as such (`ok: false`, with the
 * error) and the other path's figures are kept. Exits 1 when any answer differs from the CPU's or no shape ran.
 */
import { options, parseShapes, residentBytes } from "./lib.ts";

const opts = options(Deno.args);
const out = opts.get("out");
if (!out) throw new Error("--out FILE is required");
// Today's read table, the Dust commitment table (now and before the cap of 8 column blocks on group tables), the
// 10-million-output read table, and the largest shape of step 1.
const shapes = parseShapes(opts.get("shapes") ?? "2048x2048,12288x16384,6144x28672,14336x34816,32768x32768");
const runs = Number(opts.get("runs") ?? 9);
const cudaFetch = opts.get("cuda-fetch") ?? "ipir/target/release/cuda_fetch";
const cpuFetch = opts.get("cpu-fetch") ?? "ipir/target/release/cpu_fetch";
const timeoutS = Number(opts.get("timeout-s") ?? 1800);
const withCuda = !opts.has("no-cuda");
const cudaFetchNative = opts.get("cuda-fetch-native");
const throughput = opts.get("throughput") ? ["--throughput", opts.get("throughput")!] : [];
const here = new URL(".", import.meta.url).pathname;
// deno-lint-ignore no-explicit-any
type Any = any;

/** Runs a command with the time limit; its last stdout line parsed as JSON. */
async function child(cmd: string, args: string[]): Promise<{ success: boolean; parsed: Any; stderr: string }> {
  const proc = new Deno.Command(cmd, { args, stdout: "piped", stderr: "piped" }).spawn();
  const timer = setTimeout(() => {
    try {
      proc.kill("SIGKILL");
    } catch { /* gone */ }
  }, timeoutS * 1000);
  const res = await proc.output();
  clearTimeout(timer);
  const stdout = new TextDecoder().decode(res.stdout).trim().split("\n");
  let parsed: Any = null;
  try {
    parsed = JSON.parse(stdout[stdout.length - 1] ?? "");
  } catch { /* no JSON */ }
  return { success: res.success, parsed, stderr: new TextDecoder().decode(res.stderr).trim() };
}

/** The GPU's name and memory, from nvidia-smi (null when there is none). */
async function nvidia(): Promise<{ name: string; usedMiB: number; totalMiB: number } | null> {
  try {
    const r = await new Deno.Command("nvidia-smi", { args: ["--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"], stdout: "piped", stderr: "null" }).output();
    const [name, used, total] = new TextDecoder().decode(r.stdout).trim().split("\n")[0]!.split(",").map((x) => x.trim());
    return r.success && name ? { name, usedMiB: Number(used), totalMiB: Number(total) } : null;
  } catch {
    return null;
  }
}

const noFit = (e: unknown) => /out of memory|not enough memory|OutOfMemory|OUT_OF_MEMORY|validation error/i.test(String(e));
const results: Any[] = [];
let different = 0;
for (const [rows, cols] of shapes) {
  const before = await nvidia();
  const row: Any = { rows, cols, colBlocks: cols / 2048, tableMiB: (rows * cols * 2) / 2 ** 20, webgpuResidentMiB: residentBytes(rows, cols) / 2 ** 20, gpuUsedMiBBefore: before?.usedMiB ?? null };
  console.error(`cuda-compare ${rows}x${cols}: building the table on the CPU`);
  const dir = await Deno.makeTempDir({ prefix: "ipir-cuda-compare-" });
  try {
    const dumped = await child(cpuFetch, [String(rows), String(cols), String(runs), "--dump", dir, ...throughput]);
    if (!dumped.success) {
      row.error = `cpu_fetch --dump failed: ${dumped.parsed?.error ?? dumped.stderr.slice(-500)}`;
      console.error(`cuda-compare ${rows}x${cols}: ${row.error}`);
      results.push(row);
      continue;
    }
    // The CPU alone (`NativeServer::respond`), from the program that made the dump.
    row.cpu = { buildMs: dumped.parsed?.buildMs ?? null, timing: dumped.parsed?.timing ?? null, throughput: dumped.parsed?.throughput ?? null };
    if (withCuda) {
      console.error(`cuda-compare ${rows}x${cols}: CUDA first pass and CPU packing`);
      const cuda = await child(cudaFetch, [String(rows), String(cols), String(runs), "--expect", dir, ...throughput]);
      row.cuda = cuda.parsed ?? { ok: false, error: `cuda_fetch gave no result: ${cuda.stderr.slice(-500)}` };
    }
    if (!withCuda) {
      // no CUDA on this host
    } else if (!row.cuda.ok) {
      row.cuda.doesNotFit = noFit(row.cuda.error);
      if (/answers|differ/i.test(String(row.cuda.stage))) different++;
      console.error(`cuda-compare ${rows}x${cols}: CUDA ${row.cuda.doesNotFit ? "does not fit" : "FAILED"} at ${row.cuda.stage}: ${row.cuda.error}`);
    } else {
      console.error(`cuda-compare ${rows}x${cols}:   call of 32: ${row.cuda.cuda["32"].wallMs.median} ms (first pass ${row.cuda.cuda["32"].firstPassMs}, of which the device ${row.cuda.cuda["32"].kernelMs}; packing ${row.cuda.cuda["32"].packMs}); CPU alone ${row.cuda.cpu["32"].wallMs.median} ms`);
    }
    if (withCuda && cudaFetchNative) {
      console.error(`cuda-compare ${rows}x${cols}: CUDA, this benchmark's kernels`);
      const native = await child(cudaFetchNative, [String(rows), String(cols), String(runs), "--expect", dir, ...throughput]);
      row.cudaNative = native.parsed ?? { ok: false, error: `cuda_fetch_native gave no result: ${native.stderr.slice(-500)}` };
      if (!row.cudaNative.ok) {
        row.cudaNative.doesNotFit = row.cudaNative.doesNotFit ?? noFit(row.cudaNative.error);
        if (/compar/i.test(String(row.cudaNative.stage))) different++;
        console.error(`cuda-compare ${rows}x${cols}: CUDA (own kernels) ${row.cudaNative.doesNotFit ? "does not fit" : "FAILED"} at ${row.cudaNative.stage}: ${row.cudaNative.error}`);
      } else {
        const c = row.cudaNative.cuda["32"];
        console.error(`cuda-compare ${rows}x${cols}:   call of 32: ${c.wallMs.median} ms (the device ${c.kernelMs}: first pass ${c.scanMs}, packing ${c.packMs})`);
      }
    }
    console.error(`cuda-compare ${rows}x${cols}: WebGPU`);
    const web = await child(Deno.execPath(), ["run", "--allow-read", "--allow-env", `${here}check-cpu-answer.ts`, dir, String(rows), String(cols), "--time", String(runs), ...throughput]);
    const w = web.parsed ?? { equal: null, error: `the WebGPU run gave no result: ${web.stderr.slice(-500)}` };
    row.webgpu = { ok: w.equal === true, ...w };
    if (w.equal === false && !w.error) different++;
    if (!row.webgpu.ok) {
      row.webgpu.doesNotFit = noFit(w.error);
      console.error(`cuda-compare ${rows}x${cols}: WebGPU ${row.webgpu.doesNotFit ? "does not fit" : "FAILED"}${w.stage ? ` at ${w.stage}` : ""}: ${w.error ?? "answers differ from the CPU's"}`);
    } else {
      const t = w.timing["32"];
      console.error(`cuda-compare ${rows}x${cols}:   call of 32: ${t.wallMs.median.toFixed(3)} ms (the device ${t.execMedianMs.total?.toFixed(3)})`);
    }
  } finally {
    await Deno.remove(dir, { recursive: true }).catch(() => {});
  }
  results.push(row);
}

const gpu = await nvidia();
const result = {
  schema: "ipir-benchmark-cuda/1",
  gpu: gpu?.name ?? null,
  gpuTotalMiB: gpu?.totalMiB ?? null,
  runs,
  note: "cuda: first pass on the GPU with upstream's CudaKernel, one request at a time, packing on the CPU; webgpu: both steps on the GPU; cpu: NativeServer::respond. Every path's answers are compared with the CPU's, byte for byte.",
  shapes: results,
};
await Deno.writeTextFile(out, JSON.stringify(result, null, 1));
const ran = results.filter((r) => r.cuda?.ok || r.cudaNative?.ok || r.webgpu?.ok).length;
console.error(`cuda-compare: wrote ${out}; ${ran} of ${results.length} shape(s) ran${different ? `; ${different} DIFFERENT from the CPU's answers` : ""}`);
if (different || ran === 0) Deno.exit(1);
