/**
 * The report's measurements, on one machine: the four operations of a table at 1 million and at 10 million values
 * (a value is one shielded output's read record, 98 bytes), and the GPU memory each needs.
 *
 *   deno run <permissions> bench/report-run.ts --out results/<name>.json [--sizes 1m,10m] [--cuda] [--label TEXT]
 *     [--quick] [--bin ipir/target/release] [--steps LIST] [--merge FILE]
 *
 * Per size (`bench/gpu-build-sets.ts` has the six tables of a size; the first is the read table, the matrix of the
 * values):
 *
 * - `build`: the read table built from scratch, on the CPU and on the GPU (`bench/table-build.ts`).
 * - `redimension`: the same build while a table of the same shape, already resident, answers 2 calls a second: the
 *   old and the new table in memory together.
 * - `update1`: one column block of the read table computed again, on the CPU (`compile_block`) and on the GPU
 *   (`bench/gpu-compile.ts`, which also times the six tables' blocks alone).
 * - `update6`: one chain block, a column block of each of the six tables, while the GPU answers 2 calls a second
 *   (`bench/budget.ts`): with the blocks computed on the GPU, and on the CPU's low-priority pool. When the six tables
 *   and the GPU build do not fit together, the largest smaller size that does is measured (`largest`).
 * - `fetch`: a call of 1 and a call of 32 on the read table (`bench/cuda-compare.ts`): the CPU alone, the WebGPU
 *   path, and with `--cuda` this benchmark's CUDA kernels (`cuda_fetch_native`) and upstream's CUDA kernel with the
 *   packing on the CPU. Every answer is compared with the CPU's. Each path also measures the calls it answers per
 *   second (`throughput`).
 * - `update6.cuda` (with `--cuda`, when `cuda_budget` is built): one chain block with CUDA kernels only, the blocks
 *   built on the device and installed into the six tables while they answer 2 calls a second; `update6.cudaLargest`
 *   when the size does not fit, as for the WebGPU build.
 *
 * - `metal` (macOS, when `metal/target/release/metal_report` is built) and `cudaBuild` (with `--cuda`, when
 *   `cuda_build_report` is built): the computing of column blocks with native Metal or CUDA kernels, for the update
 *   of one block, of the six tables' blocks, and for the whole table. Computing only: these paths have no fetch of
 *   their own, so nothing is installed into a table that answers. Their output's digests are compared with the
 *   CPU's (`pack_digest`).
 *
 * Every step runs in its own process and a step that fails is recorded with its error (`doesNotFit` when the GPU's
 * memory ran out), so the run goes on. `--quick`: the 1M size only, few repetitions, to check that it works.
 *
 * `--steps LIST` runs only the steps named (`build`, `redimension`, `update1`, `update6`, `native`, `fetch`,
 * `update6cuda`, comma-separated); `--merge FILE` starts from an earlier result file of this machine and replaces
 * only those steps in it (`reruns` records when).
 */
import { LADDER, SET_OUTPUTS, SETS } from "./gpu-build-sets.ts";
import { median, options } from "./lib.ts";

// deno-lint-ignore no-explicit-any
type Any = any;
const opts = options(Deno.args);
const out = opts.get("out");
if (!out) throw new Error("--out FILE is required");
const quick = opts.has("quick");
const sizes = (opts.get("sizes") ?? (quick ? "1m" : "1m,10m")).split(",").filter(Boolean);
const withCuda = opts.has("cuda");
const bin = opts.get("bin") ?? "ipir/target/release";
const here = new URL(".", import.meta.url).pathname;
const PERMS = ["--allow-read", "--allow-write", "--allow-run", "--allow-env", "--allow-sys"];
const log = (s: string) => console.error(`report-run: ${s}`);
const only = opts.get("steps")?.split(",").filter(Boolean) ?? null;
const want = (step: string) => !only || only.includes(step);
const mergeFrom = opts.get("merge");
const tmp = await Deno.makeTempDir({ prefix: "ipir-report-" });
const noFit = (e: unknown) => /out of memory|not enough memory|OutOfMemory|OUT_OF_MEMORY|validation error/i.test(String(e));

/** Runs a program; its standard output and whether it succeeded. Its standard error goes to ours. */
async function run(cmd: string, args: string[]): Promise<{ success: boolean; lines: string[] }> {
  const r = await new Deno.Command(cmd, { args, stdout: "piped", stderr: "inherit" }).output();
  return { success: r.success, lines: new TextDecoder().decode(r.stdout).trim().split("\n").filter(Boolean) };
}
const deno = (script: string, args: string[]) => run(Deno.execPath(), ["run", ...PERMS, `${here}${script}`, ...args]);
function lastJson(lines: string[]): Any {
  try {
    return JSON.parse(lines[lines.length - 1] ?? "");
  } catch {
    return null;
  }
}
/** A step whose script prints one JSON line. */
async function jsonStep(script: string, args: string[]): Promise<Any> {
  const r = await deno(script, args);
  const j = lastJson(r.lines) ?? { ok: false, error: "the step printed no result" };
  if (j.ok === false && j.doesNotFit === undefined) j.doesNotFit = noFit(j.error);
  return j;
}
/** A step whose script writes its result to `--out FILE`. */
async function fileStep(script: string, args: string[]): Promise<Any> {
  const file = `${tmp}/step-${crypto.randomUUID()}.json`;
  await deno(script, ["--out", file, ...args]);
  try {
    const j = JSON.parse(await Deno.readTextFile(file));
    if (j.ok === false && j.doesNotFit === undefined) j.doesNotFit = noFit(j.error);
    return j;
  } catch {
    return { ok: false, error: "the step wrote no result" };
  }
}
/** One chain block's work at a set of six tables, by `mode` (gpu, pool), at 2 calls a second. */
function budget(set: string, mode: "gpu" | "pool"): Promise<Any> {
  const args = ["--set", set, "--modes", mode, "--rates", "2", "--blocks", quick ? "4" : "30", "--baseline-s", quick ? "5" : "20"];
  if (mode === "gpu") args.push("--gpu-prep", `${bin}/gpu_build_prep`);
  else args.push("--worker", `${bin}/block_worker`);
  return fileStep("budget.ts", args);
}

/** One chain block's work with CUDA kernels only (`cuda_budget`), at a set of six tables. */
async function cudaBudget(set: string): Promise<Any> {
  const r = await run(`${bin}/cuda_budget`, [SETS[set]!.map((t) => `${t.rows}x${t.cols}`).join(","), "--set", set, "--blocks", quick ? "4" : "30", "--baseline-s", quick ? "5" : "20"]);
  const j = lastJson(r.lines) ?? { ok: false, error: "cuda_budget printed no result" };
  if (j.ok === false && j.doesNotFit === undefined) j.doesNotFit = noFit(j.error);
  return j;
}
const exists = async (path: string) => {
  try {
    await Deno.stat(path);
    return true;
  } catch {
    return false;
  }
};

/** A native build's report (`metal_report`, `cuda_build_report`) for a size, with its digests checked against the CPU's. */
async function nativeBuild(program: string, rows: number, blocks: number, six: number[]): Promise<Any> {
  try {
    await Deno.stat(program);
  } catch {
    return null;
  }
  // The CPU's digests first: they name the CPU's matrix storage, which the digest hashes with the words.
  const cpu = await run(`${bin}/pack_digest`, [String(rows), "1,2,3", "--parts"]);
  const label = cpu.lines[0]?.match(/ storage=(\S+)/)?.[1];
  if (label) Deno.env.set("IPIR_STORAGE_LABEL", label);
  const r = await run(program, [String(rows), String(blocks), six.join(","), quick ? "2" : "5"]);
  const j = lastJson(r.lines) ?? { ok: false, error: `${program} printed no result` };
  if (j.ok && Array.isArray(j.digests)) {
    j.cpuStorage = label ?? null;
    const want = cpu.lines.map((l) => l.replace(/ storage=\S+/, "").trim());
    j.digestsEqual = j.digests.filter((d: string) => want.includes(d.trim())).length;
    j.digestsChecked = j.digests.length;
    if (j.digestsEqual !== j.digestsChecked) j.ok = false, j.error = "the build's output differs from the CPU's";
  }
  return j;
}

const started = new Date().toISOString();
log("the machine");
let sys: Any = {};
try {
  sys = JSON.parse((await deno("sysinfo.ts", [])).lines.join("\n"));
} catch { /* the machine stays unknown */ }
const result: Any = mergeFrom ? JSON.parse(await Deno.readTextFile(mergeFrom)) : {
  schema: "ipir-benchmark-report/1",
  date: started.slice(0, 10),
  started,
  quick,
  label: opts.get("label") ?? null,
  backend: sys.gpuBackend ?? null,
  cuda: withCuda,
  value: { bytes: 98, what: "one shielded output's read record" },
  sysinfo: sys,
  sizes: {},
};
if (mergeFrom) {
  if (result.schema !== "ipir-benchmark-report/1") throw new Error(`${mergeFrom}: not a report result file`);
  const smiNow = sys.nvidiaSmi?.[0];
  (result.reruns ??= []).push({ started, steps: only ?? "all", sizes, loadavg: sys.loadavg ?? null, gpuMemoryUsedMiB: smiNow ? Number(smiNow["memory.used"]) : null });
  if (withCuda) result.cuda = true;
}

// A card that others are using (more than a quarter of its memory at the start): what does not fit says nothing
// about the card, so the largest size that fits is not looked for.
const smi = sys.nvidiaSmi?.[0];
const cardInUse = smi ? Number(smi["memory.used"]) > Number(smi["memory.total"]) / 4 : false;
if (!mergeFrom) result.cardInUse = cardInUse;

for (const size of sizes) {
  const tables = SETS[size];
  if (!tables) throw new Error(`unknown size ${size}`);
  const { rows, cols } = tables[0]!;
  const shape = `${rows}x${cols}`;
  const s: Any = result.sizes[size] ?? { values: SET_OUTPUTS[size], rows, cols, columnBlocks: cols / 2048, tables };
  result.sizes[size] = s;
  const save = () => Deno.writeTextFile(out, JSON.stringify(result, null, 1));

  if (want("build")) {
    log(`${size}: the read table ${shape} built from scratch`);
    s.build = await jsonStep("table-build.ts", [String(rows), String(cols), "--bin", bin]);
    await save();
  }
  if (want("redimension")) {
    log(`${size}: the same build while a table of that shape answers`);
    s.redimension = await jsonStep("table-build.ts", [String(rows), String(cols), "--serving", shape, "--rate", "2", "--bin", bin]);
    await save();
  }

  if (want("update1")) {
    log(`${size}: one column block computed again, on the CPU`);
    const cpu = await run(`${bin}/compile_block`, [String(rows), quick ? "2" : "5", "1"]);
    const walls = cpu.lines.map((l) => {
      try {
        return JSON.parse(l).wall_ms as number;
      } catch {
        return NaN;
      }
    }).filter((x) => Number.isFinite(x));
    s.update1 = { cpu: walls.length ? { ok: true, wallMs: median(walls), reps: walls.length } : { ok: false, error: "compile_block gave no result" } };
    log(`${size}: one column block computed again, on the GPU, and the six tables' blocks alone`);
    s.update1.gpu = await fileStep("gpu-compile.ts", [
      "--rows",
      String(rows),
      "--reps",
      quick ? "2" : "5",
      "--at-once",
      "",
      "--sets",
      size,
      "--set-reps",
      quick ? "3" : "10",
      "--idle-ms",
      quick ? "2000" : "6000",
      "--idle-reps",
      quick ? "2" : "5",
      "--bin",
      bin,
    ]);
    await save();
  }

  if (want("update6")) {
    log(`${size}: one chain block (six tables), blocks computed on the GPU`);
    s.update6 = { ...(s.update6?.cuda ? { cuda: s.update6.cuda, cudaLargest: s.update6.cudaLargest } : {}), gpu: await budget(size, "gpu") };
    if (s.update6.gpu.ok === false && s.update6.gpu.doesNotFit && !cardInUse) {
      // The largest smaller size at which the six tables and the GPU build fit together.
      for (const smaller of LADDER.filter((x) => x.outputs < (SET_OUTPUTS[size] ?? 0))) {
        log(`${size} does not fit with the GPU build: trying ${smaller.set}`);
        const r = await budget(smaller.set, "gpu");
        if (r.ok || !r.doesNotFit) {
          s.update6.largest = r;
          break;
        }
      }
  }
  await save();
  log(`${size}: one chain block (six tables), blocks computed on the CPU's pool`);
  s.update6.cpu = await budget(size, "pool");
  await save();
  }

  if (withCuda && want("update6cuda") && await exists(`${bin}/cuda_budget`)) {
    log(`${size}: one chain block (six tables), built and installed with CUDA kernels`);
    s.update6 ??= {};
    s.update6.cuda = await cudaBudget(size);
    delete s.update6.cudaLargest;
    if (s.update6.cuda.ok === false && s.update6.cuda.doesNotFit && !cardInUse) {
      for (const smaller of LADDER.filter((x) => x.outputs < (SET_OUTPUTS[size] ?? 0))) {
        log(`${size} does not fit with the CUDA build: trying ${smaller.set}`);
        const r = await cudaBudget(smaller.set);
        if (r.ok || !r.doesNotFit) {
          s.update6.cudaLargest = r;
          break;
        }
      }
    }
    await save();
  }

  const six = tables.map((t) => t.rows);
  if (want("native")) {
    if (Deno.build.os === "darwin") {
      log(`${size}: column blocks computed with native Metal kernels`);
      const m = await nativeBuild(`${here}../metal/target/release/metal_report`, rows, cols / 2048, six);
      if (m) s.metal = m;
  }
  if (withCuda) {
    log(`${size}: column blocks computed with CUDA kernels`);
    const c = await nativeBuild(`${bin}/cuda_build_report`, rows, cols / 2048, six);
    if (c) s.cudaBuild = c;
  }
  await save();
  }

  if (want("fetch")) {
    log(`${size}: fetch 1 and fetch 32 on the read table, and the calls answered per second`);
    const native = withCuda && await exists(`${bin}/cuda_fetch_native`) ? ["--cuda-fetch-native", `${bin}/cuda_fetch_native`] : [];
    const f = await fileStep("cuda-compare.ts", ["--shapes", shape, "--runs", quick ? "3" : "9", "--throughput", quick ? "1" : "3", "--cpu-fetch", `${bin}/cpu_fetch`, "--cuda-fetch", `${bin}/cuda_fetch`, ...native, ...(withCuda ? [] : ["--no-cuda"])]);
    s.fetch = f.shapes?.[0] ?? f;
    await save();
  }
}

if (!mergeFrom) {
  result.ended = new Date().toISOString();
  // The load at the end: a shared machine may have become busy during the run.
  try {
    result.loadavgEnd = JSON.parse((await deno("sysinfo.ts", ["--no-adapter"])).lines.join("\n")).loadavg ?? null;
  } catch { /* unknown */ }
} else {
  result.reruns[result.reruns.length - 1].ended = new Date().toISOString();
}
await Deno.writeTextFile(out, JSON.stringify(result, null, 1));
await Deno.remove(tmp, { recursive: true }).catch(() => {});
log(`wrote ${out}`);
