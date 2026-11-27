/**
 * The report: the tables of `REPORT.md`, written from the machines' result files of `bench/report-run.ts`
 * (`./bench.sh --report`). Nothing is typed by hand: a cell is a figure of a result file, or the reason there is none.
 *
 *   deno run --allow-read --allow-write bench/summary.ts [--out REPORT.md] [--readme README.md] [--cpu NAME] FILE=NAME ...
 *
 * `FILE=NAME`: a result file and the machine's short name in the column titles ("M4", "RTX 20"). `--cpu NAME`: the
 * machine whose CPU is the reference column (the first by default). Columns: that CPU; native Metal; WebGPU on every
 * machine; CUDA (this benchmark's kernels) on every machine measured with `--cuda`. The fetch tables add upstream's
 * CUDA kernel on those machines.
 *
 * `--readme FILE`: also writes the figures as one table (a row per operation and size) into FILE, between its
 * `<!-- report-summary:start -->` and `<!-- report-summary:end -->` lines: the README's quick results.
 *
 * A result file may carry a `note` (a sentence added by hand about the run's conditions); section 7 prints it.
 *
 * A cell without a figure says why: `not run`, `does not fit` (the GPU's memory ran out), `failed`, `no code`,
 * `no GPU code` (upstream's CUDA kernel does only the first pass of a fetch).
 */
import { ipirRequestBytes, ipirResponseBytes } from "../ipir/sizes.ts";

// deno-lint-ignore no-explicit-any
type Any = any;
const args = [...Deno.args];
const take = (name: string): string | null => {
  const i = args.indexOf(name);
  if (i < 0) return null;
  const v = args[i + 1] ?? null;
  args.splice(i, 2);
  return v;
};
const out = take("--out") ?? "REPORT.md";
const cpuName = take("--cpu");
const readme = take("--readme");
if (args.length === 0) throw new Error("usage: summary.ts [--out REPORT.md] [--readme README.md] [--cpu NAME] FILE=NAME ...");
const machines: { name: string; file: string; r: Any }[] = [];
for (const a of args) {
  const i = a.lastIndexOf("=");
  if (i < 0) throw new Error(`expected FILE=NAME: ${a}`);
  const r = JSON.parse(await Deno.readTextFile(a.slice(0, i)));
  if (r.schema !== "ipir-benchmark-report/1") throw new Error(`${a.slice(0, i)}: schema ${r.schema}, expected ipir-benchmark-report/1`);
  machines.push({ name: a.slice(i + 1), file: a.slice(0, i), r });
}
const cpu = machines.find((m) => m.name === cpuName) ?? machines[0]!;
const cudas = machines.filter((m) => m.r.cuda);
const SIZES = ["1m", "10m"].filter((s) => machines.some((m) => m.r.sizes?.[s]));
const sizeName = (s: string) => s.toUpperCase();
const MIB = 2 ** 20;

/** A time for a cell: ms below a second, seconds above. */
function time(ms: number | null | undefined): string | null {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) return null;
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)} s`;
  if (ms >= 100) return `${ms.toFixed(0)} ms`;
  return `${ms.toFixed(1)} ms`;
}
const gib = (mib: number | null | undefined) => (mib === null || mib === undefined ? null : mib >= 1024 ? `${(mib / 1024).toFixed(1)} GiB` : `${Math.round(mib)} MiB`);
/** Why a step has no figure. On a card that others were using, running out of memory says nothing about the card. */
let inUse = false;
function why(step: Any): string {
  if (step === null || step === undefined) return "not run";
  // On a card others are using, a lost device is the same shortage of memory as an out-of-memory error.
  if (inUse) return "not run (card in use)";
  if (step.doesNotFit) return "does not fit";
  return "failed";
}
const size = (m: { r: Any }, s: string): Any => m.r.sizes?.[s] ?? null;
const header = ["Values", `CPU ${cpu.name}`, "Metal M4", ...machines.map((m) => `WebGPU ${m.name}`), ...cudas.map((m) => `CUDA ${m.name}`)];
/** The rows of the README's table: every table of the standard columns that has a name, a row per size. */
const summary: string[][] = [];
/**
 * A table of the standard columns: one row per size, from a cell function per kind of column. `name`: its operation in
 * the README's table. `upstream`: a further column per CUDA machine, upstream's kernel (not in the README's table).
 */
function table(
  name: string,
  cells: { cpu: (z: Any) => string; webgpu: (z: Any, m: { name: string; r: Any }) => string; cuda: (z: Any) => string; metal?: (n: Any) => string | null; upstream?: (z: Any) => string },
  rows: string[] = SIZES,
): string[] {
  const head = [...header, ...(cells.upstream ? cudas.map((m) => `CUDA, upstream's kernel ${m.name}`) : [])];
  const lines = [`| ${head.join(" | ")} |`, `|${head.map(() => "---").join("|")}|`];
  for (const s of rows) {
    const on = <T>(m: { r: Any }, f: () => T): T => {
      inUse = m.r.cardInUse === true;
      try {
        return f();
      } finally {
        inUse = false;
      }
    };
    const row = [sizeName(s), on(cpu, () => cells.cpu(size(cpu, s))), cells.metal?.(metalOf(s)) ?? "no code", ...machines.map((m) => on(m, () => cells.webgpu(size(m, s), m))), ...cudas.map((m) => on(m, () => cells.cuda(size(m, s))))];
    lines.push(`| ${[...row, ...(cells.upstream ? cudas.map((m) => on(m, () => cells.upstream!(size(m, s)))) : [])].join(" | ")} |`);
    summary.push([name, ...row]);
  }
  return lines;
}
const NO_GPU = "no GPU code";
/** A figure of a native build (Metal, CUDA): `*` marks "computing only" where the other columns also install. */
const native = (n: Any, f: (n: Any) => string | null, star = true): string | null => {
  if (!n) return null;
  if (!n.ok) return inUse ? "not run (card in use)" : /out of memory/i.test(String(n.error)) ? "does not fit" : "failed";
  const v = f(n);
  return v === null ? null : `${v}${star ? "\\*" : ""}`;
};
const metalOf = (s: string): Any => machines.map((m) => size(m, s)?.metal).find(Boolean) ?? null;
const pct = (ms: number) => `${time(ms)} (${((ms / 6000) * 100).toFixed(1)} %)`;
/** The run of a budget result in `mode` (its block work at 2 calls a second). */
const runOf = (b: Any, mode: string): Any => (b?.ok ? (b.runs ?? []).find((x: Any) => x.mode === mode) : null);
const work = (b: Any, mode: string): string | null => {
  const r = runOf(b, mode);
  if (!r) return null;
  return `${time(r.blockWorkMs.p50)} (${((r.blockWorkMs.p50 / (b.intervalMs ?? 6000)) * 100).toFixed(1)} %)`;
};

/** A path's calls per second for calls of `n` requests: the rate, and for calls of 32 the values that is. */
const rate = (t: Any, n: string): string | null => {
  const v = t?.[n]?.callsPerSecond;
  if (typeof v !== "number" || !Number.isFinite(v)) return null;
  const calls = v >= 100 ? v.toFixed(0) : v.toFixed(1);
  return n === "32" ? `${calls} (${Math.round(v * 32).toLocaleString("en-US")} values)` : calls;
};

const L: string[] = [];
const any = machines.map((m) => m.r).find((r) => SIZES.every((s) => r.sizes?.[s])) ?? machines[0]!.r;
L.push("# Report: a table's four operations at 1 million and 10 million values", "");
L.push(
  "Written by `bench/summary.ts` from the result files named at the end; every figure is a median measured by `./bench.sh --report` on that machine. " +
    "A cell without a figure says why: `not run`, `does not fit` (the GPU's memory ran out), `failed`, `no code` (no native Metal fetch exists here), " +
    "`no GPU code`.",
  "",
);

L.push("## 1. Definitions", "");
L.push(
  "- **Value**: one shielded output's read record, 98 bytes (49 cells of 2 bytes).",
  "- **Matrix**: the table of all values, rows × columns of cells. One fetch reads one row.",
  "- **Column block**: 2,048 columns of a matrix, the unit that is computed again when data is added.",
  `- **CPU ${cpu.name}**: the vendored crates with patch 0002. **Metal**: native Metal kernels for the build of a column block; there is no native Metal fetch. **WebGPU**: Deno's WebGPU, on Metal on the Mac and on Vulkan on the NVIDIA cards.`,
  "- **CUDA**: this benchmark's own CUDA kernels. A fetch answers all the requests of a call in one pass over the matrix and does the packing on the GPU; a column block is built on the GPU and installed into the matrix that answers.",
  "- **CUDA, upstream's kernel**: the CUDA kernel of the IPIR+SP crates as it is. It does the first pass of a fetch for one request per call; the packing runs on that host's CPU. It has no code for a build or an update.",
  "",
);
L.push("| Values | Matrix (rows × columns) | Column blocks | Cells | On the GPU, with the packing matrices | Request | Answer |", "|---|---|---|---|---|---|---|");
for (const s of SIZES) {
  const z = any.sizes[s];
  const cells = (z.rows * z.cols * 2) / MIB;
  L.push(
    `| ${sizeName(s)} | ${z.rows.toLocaleString("en-US")} × ${z.cols.toLocaleString("en-US")} | ${z.columnBlocks} | ${gib(cells)} | ${gib(cells + z.columnBlocks * 32)} | ${(ipirRequestBytes(z.rows) / 1000).toFixed(1)} kB | ${(ipirResponseBytes(z.cols) / 1000).toFixed(1)} kB |`,
  );
}
L.push("");

L.push("## 2. Build a whole matrix from scratch", "");
L.push("At server start, when a matrix changes its number of rows, and when a matrix is added. Every column block's hint columns and packing matrix are computed from the cells.", "");
L.push(...table("Build from scratch", {
  cpu: (z) => time(z?.build?.cpuBuildMs) ?? why(z?.build),
  webgpu: (z) => (z?.build?.ok ? time(z.build.gpuBuildMs)! : why(z?.build)),
  cuda: (z) => native(z?.cudaBuild, (n) => time(n.build?.wallMs)) ?? NO_GPU,
  metal: (n) => native(n, (x) => time(x.build?.wallMs)),
}));
L.push("", "The WebGPU figure is from the cells in host memory to a table that answers; the table built on the GPU answers 32 requests exactly as the CPU-built one.", "");
const STAR = "\\* Computing only, with native kernels: the blocks' hint columns and packing matrices are computed and compared with the CPU's, but nothing is installed into a matrix that answers and no calls are answered meanwhile. Metal has no native fetch to install into; the CUDA build from scratch is not joined to the CUDA fetch.";
if (SIZES.some((s) => metalOf(s) || machines.some((m) => size(m, s)?.cudaBuild))) L.push(STAR, "");

L.push("## 3. Update: data added to one column block", "");
L.push("The cost is the same whether one value or many were added, as long as they fall in one column block.", "");
L.push("**1×: one column block of the matrix computed again**", "");
L.push(...table("Update, 1×: one column block", {
  cpu: (z) => time(z?.update1?.cpu?.wallMs) ?? why(z?.update1?.cpu),
  webgpu: (z) => {
    const g = z?.update1?.gpu;
    const one = g?.shapes?.[0];
    return one && !one.failed ? time(one.wallMs?.median)! : why(one ?? g);
  },
  cuda: (z) => native(z?.cudaBuild, (n) => time(n.update1?.wallMs), false) ?? NO_GPU,
  metal: (n) => native(n, (x) => time(x.update1?.wallMs), false),
}));
L.push("", "**6×: one chain block, a column block in each of the six matrices**, while the GPU answers 2 calls a second: the time, and its share of the 6 s between blocks.", "");
L.push(...table("Update, 6×: one chain block", {
  cpu: (z) => work(z?.update6?.cpu, "pool") ?? why(z?.update6?.cpu),
  webgpu: (z) => {
    const u = z?.update6;
    const w = work(u?.gpu, "gpu");
    if (w) return w;
    const l = u?.largest;
    return l?.ok ? `${why(u?.gpu)}; ${work(l, "gpu")} at ${String(l.set).toUpperCase()}` : why(u?.gpu);
  },
  cuda: (z) => {
    const u = z?.update6;
    if (!u?.cuda) return native(z?.cudaBuild, (n) => (n.update6?.wallMs ? pct(n.update6.wallMs) : null)) ?? NO_GPU;
    const w = work(u.cuda, "cuda");
    if (w) return w;
    const l = u.cudaLargest;
    return l?.ok ? `${why(u.cuda)}; ${work(l, "cuda")} at ${String(l.set).toUpperCase()}` : why(u.cuda);
  },
  metal: (n) => native(n, (x) => (x.update6?.wallMs ? pct(x.update6.wallMs) : null)),
}));
L.push("", `The CPU column computes the six blocks on a low-priority pool of half the cores and writes them to the GPU that answers. The WebGPU and CUDA columns compute them on the GPU that answers and install them into its matrices.`, "");
if (SIZES.some((s) => metalOf(s) || machines.some((m) => size(m, s)?.cudaBuild))) L.push(STAR, "");

for (const [n, title] of [["1", "## 4. Fetch 1 value"], ["32", "## 5. Fetch 32 values (one wallet call)"]] as const) {
  L.push(title, "");
  L.push(n === "1" ? "One request: the wall time from the request's bytes to the answer's bytes." : "The call every wallet sends: 32 requests, padded so that the server cannot tell how many it needed.", "");
  L.push(...table(n === "1" ? "Fetch 1" : "Fetch 32", {
    cpu: (z) => time(z?.fetch?.cpu?.timing?.[n]?.wallMs?.median) ?? why(z?.fetch?.cpu?.timing ? null : z?.fetch),
    webgpu: (z) => (z?.fetch?.webgpu?.ok ? time(z.fetch.webgpu.timing?.[n]?.wallMs?.median)! : why(z?.fetch?.webgpu)),
    cuda: (z) => (z?.fetch?.cudaNative?.ok ? time(z.fetch.cudaNative.cuda?.[n]?.wallMs?.median)! : why(z?.fetch?.cudaNative)),
    upstream: (z) => (z?.fetch?.cuda?.ok ? time(z.fetch.cuda.cuda?.[n]?.wallMs?.median)! : why(z?.fetch?.cuda)),
  }));
  L.push("");
}
L.push("The CUDA cells answer the 32 requests in one pass over the matrix, with the packing on the GPU. Upstream's kernel does the first pass one request at a time and leaves the packing to that NVIDIA host's CPU, so its call of 32 costs about 32 calls of 1. Every WebGPU and CUDA answer equals the CPU's, byte for byte.", "");

L.push("## 6. Answers per second", "");
L.push(
  "The calls a matrix answers per second when calls arrive without a pause: each caller sends its next call as soon as its last one is answered, for a few seconds, with 1, 2 and 4 callers. A cell is the highest rate of the three. For calls of 32 it also gives the values per second (calls × 32). Upstream's CUDA kernel takes one request at a time, so it has one caller.",
  "",
);
for (const [n, title] of [["1", "**Calls of 1 per second**"], ["32", "**Calls of 32 per second**"]] as const) {
  L.push(title, "");
  L.push(...table(n === "1" ? "Calls of 1 per second" : "Calls of 32 per second", {
    cpu: (z) => rate(z?.fetch?.cpu?.throughput, n) ?? "not run",
    webgpu: (z) => (z?.fetch?.webgpu?.ok ? rate(z.fetch.webgpu.throughput, n) ?? "not run" : why(z?.fetch?.webgpu)),
    cuda: (z) => (z?.fetch?.cudaNative?.ok ? rate(z.fetch.cudaNative.throughput, n) ?? "not run" : why(z?.fetch?.cudaNative)),
    upstream: (z) => (z?.fetch?.cuda?.ok ? rate(z.fetch.cuda.throughput, n) ?? "not run" : why(z?.fetch?.cuda)),
  }));
  L.push("");
}

L.push("## 7. Memory: what fits on each GPU", "");
L.push("What the WebGPU path's buffers hold on the GPU, at most, during each operation (the same on every machine). The CUDA path's memory is in section 8.", "");
L.push("| Values | To answer (six matrices) | To answer and update (and one chain block being computed) | To build the matrix from scratch (alone on the card) | To re-dimension the matrix while it serves (old and new together) |", "|---|---|---|---|---|");
const first = (s: string, f: (z: Any) => number | null | undefined): string => {
  for (const m of machines) {
    const v = f(size(m, s));
    if (v !== null && v !== undefined && Number.isFinite(v)) return gib(v)!;
  }
  return "not run";
};
for (const s of SIZES) {
  L.push(
    `| ${sizeName(s)} | ${first(s, (z) => (z?.update6?.gpu?.residentBytes ?? z?.update6?.cpu?.residentBytes) / MIB)} | ${first(s, (z) => (z?.update6?.gpu?.ok ? z.update6.gpu.gpuPeakMiB : null))} | ${first(s, (z) => (z?.build?.ok ? z.build.gpuPeakMiB : null))} | ${first(s, (z) => (z?.redimension?.ok ? z.redimension.gpuPeakMiB : null))} |`,
  );
}
L.push("", `| Values | Operation | ${machines.map((m) => m.name).join(" | ")} |`, `|---|---|${machines.map(() => "---").join("|")}|`);
const fits = (step: Any) => (step?.ok ? "fits" : why(step));
for (const s of SIZES) {
  const row = (name: string, f: (z: Any) => string) =>
    L.push(`| ${sizeName(s)} | ${name} | ${machines.map((m) => {
      inUse = m.r.cardInUse === true;
      const cell = f(size(m, s));
      inUse = false;
      return cell;
    }).join(" | ")} |`);
  row("Answer", (z) => fits(z?.update6?.cpu));
  row("Answer and update", (z) => (z?.update6?.gpu?.ok ? "fits" : z?.update6?.largest?.ok ? `${why(z.update6.gpu)}; largest is ${String(z.update6.largest.set).toUpperCase()}` : why(z?.update6?.gpu)));
  row("Build from scratch", (z) => fits(z?.build));
  row("Re-dimension while serving", (z) => fits(z?.redimension));
}
L.push("");

L.push("## 8. Detail and conditions", "");
for (const m of machines) {
  L.push(`**${m.name}**${m.r.label ? ` (${m.r.label})` : ""}, \`${m.file}\`, ${m.r.date}${m.r.quick ? ", a quick run" : ""}:`, "");
  const smi = m.r.sysinfo?.nvidiaSmi?.[0];
  const load = (l: Any) => (Array.isArray(l) ? Number(l[0]).toFixed(1) : "?");
  const loads = `Load average (1 min) at the start: ${load(m.r.sysinfo?.loadavg)}${m.r.loadavgEnd ? `, at the end: ${load(m.r.loadavgEnd)}` : ""}.`;
  if (smi) L.push(`- GPU memory in use by others at the start: ${gib(Number(smi["memory.used"]))} of ${gib(Number(smi["memory.total"]))}${m.r.cardInUse ? " (a card in use: what did not fit beside that is `not run`)" : ""}. ${loads}`);
  else L.push(`- ${loads}`);
  if (m.r.note) L.push(`- ${m.r.note}`);
  for (const rr of m.r.reruns ?? []) L.push(`- Measured again on ${String(rr.started).slice(0, 10)}: ${Array.isArray(rr.steps) ? rr.steps.join(", ") : rr.steps} (${(rr.sizes ?? []).map(sizeName).join(", ")}); load average ${load(rr.loadavg)} at its start${rr.gpuMemoryUsedMiB !== null && rr.gpuMemoryUsedMiB !== undefined ? `, ${gib(rr.gpuMemoryUsedMiB)} of the GPU's memory in use by others` : ""}.`);
  for (const s of SIZES) {
    const z = size(m, s);
    if (!z) continue;
    const bits: string[] = [];
    if (z.build?.ok) bits.push(`the GPU-built matrix answers as the CPU-built one (${z.build.answersEqual ? "32 of 32" : "NOT equal"})`);
    const sd = z.redimension?.servingDuringBuild;
    if (z.redimension?.ok && sd) bits.push(`during the re-dimension the old matrix answered ${sd.calls} call(s) of 32, ${sd.equal} equal to the CPU's, in ${time(sd.latencyMs?.p50)} (p50)`);
    const g = runOf(z.update6?.gpu, "gpu") ?? runOf(z.update6?.largest, "gpu");
    const base = (z.update6?.gpu?.ok ? z.update6.gpu : z.update6?.largest)?.runs?.find((x: Any) => x.mode === "none");
    if (g) bits.push(`fetch 32 while an update runs ${time(g.answerLatencyMs?.duringBlockWork?.p50)}, against ${time(base?.answerLatencyMs?.p50)} otherwise; no block of ${g.blocks} over 1 s: ${g.overGoal === 0 ? "yes" : `no, ${g.overGoal}`}`);
    const sets = z.update1?.gpu?.sets?.[0];
    if (sets && !sets.failed) bits.push(`the six blocks computed alone on the GPU ${time(sets.wallMs?.median)}, and ${time(sets.afterIdle?.wallMs?.median)} with the GPU idle for 6 s before`);
    for (const [what, n] of [["Metal", z.metal], ["CUDA", z.cudaBuild]] as const) {
      if (n?.digestsChecked) bits.push(`the native ${what} build's output equals the CPU's in ${n.digestsEqual} of ${n.digestsChecked} blocks; its six blocks alone ${time(n.update6?.wallMs)}, and ${time(n.update6?.afterIdle?.wallMs)} after 6 s of idle`);
    }
    const cn = z.fetch?.cudaNative?.ok ? z.fetch.cudaNative : null;
    const c32 = cn?.cuda?.["32"];
    if (c32) {
      bits.push(`CUDA fetch 32: ${time(c32.kernelMs)} on the device (first pass ${time(c32.scanMs)}, packing ${time(c32.packMs)}), and on the host ${time(c32.parseMs + c32.copyInMs)} to read and copy the requests in and ${time(c32.serializeMs)} to write the answers, the SHA-256 of the requests (${time(c32.hashMs)}) running while the device works`);
      bits.push(`CUDA: the matrix holds ${gib(cn.residentBytes / MIB)} on the device and a call's buffers ${gib(cn.callBytes / MIB)}; one changed column block is written from the host in ${time(cn.writeBlockMs)}, and installed from the block build's buffers in ${time(cn.installMs)}`);
    }
    const cu = z.update6?.cuda?.ok ? z.update6.cuda : z.update6?.cudaLargest?.ok ? z.update6.cudaLargest : null;
    const cr = runOf(cu, "cuda");
    if (cr) {
      const cb = cu.runs?.find((x: Any) => x.mode === "none");
      bits.push(`CUDA 6× update${cu === z.update6?.cuda ? "" : ` (at ${String(cu.set).toUpperCase()})`}: ${time(cr.compileMs?.p50)} to build the six blocks and ${time(cr.uploadMs?.p50)} to install them; fetch 32 while it runs ${time(cr.answerLatencyMs?.duringBlockWork?.p50)}, against ${time(cb?.answerLatencyMs?.p50)} otherwise; no block of ${cr.blocks} over 1 s: ${cr.overGoal === 0 ? "yes" : `no, ${cr.overGoal}`}; the six matrices, the calls' buffers and the build hold ${gib(cu.gpuPeakMiB)} on the device`);
    }
    const c = z.fetch?.cuda?.ok ? z.fetch.cuda.cuda?.["32"] : null;
    if (c) bits.push(`upstream's CUDA kernel, fetch 32: first pass ${time(c.kernelMs)} on the device, packing ${time(c.packMs)} on the CPU`);
    const w = z.fetch?.webgpu?.ok ? z.fetch.webgpu.timing?.["32"] : null;
    if (w?.execMedianMs?.total) bits.push(`WebGPU fetch 32: ${time(w.execMedianMs.total)} on the device`);
    if (bits.length) L.push(`- ${sizeName(s)}: ${bits.join("; ")}.`);
  }
  L.push("");
}
L.push("The WebGPU wall time of a fetch has about 12 ms that do not depend on the call (submitting the work and reading the answer back in this harness). A GPU left idle for the 6 s between blocks computes an update more slowly than one that just worked.", "");

await Deno.writeTextFile(out, L.join("\n"));
console.error(`wrote ${out} from ${machines.map((m) => m.file).join(", ")}`);

if (readme) {
  const START = "<!-- report-summary:start -->";
  const END = "<!-- report-summary:end -->";
  const text = await Deno.readTextFile(readme);
  const a = text.indexOf(START);
  const b = text.indexOf(END);
  if (a < 0 || b < a) throw new Error(`${readme}: no ${START} ... ${END} lines to write the table between`);
  const cols = ["Operation", ...header];
  const block = [`| ${cols.join(" | ")} |`, `|${cols.map(() => "---").join("|")}|`, ...summary.map((r) => `| ${r.join(" | ")} |`), ""];
  if (summary.some((r) => r.some((c) => c.endsWith("\\*")))) block.push("\\* Computing only, with native kernels: nothing is installed into a table that answers, and no calls are answered meanwhile.", "");
  // How many cells hold a figure, and why the others do not.
  const cells = summary.flatMap((r) => r.slice(2));
  const reasons = new Map<string, number>();
  for (const c of cells) if (!/^\d/.test(c)) reasons.set(c.split(";")[0]!, (reasons.get(c.split(";")[0]!) ?? 0) + 1);
  const measured = cells.filter((c) => /^\d/.test(c)).length;
  block.push(`${measured} of the ${cells.length} cells hold a figure${reasons.size ? `; the others: ${[...reasons].map(([k, n]) => `${n} \`${k}\``).join(", ")}` : ""}.`, "");
  block.push(`Medians of \`./bench.sh --report\`: ${machines.map((m) => `${m.name}${m.r.label ? ` (${m.r.label})` : ""} on ${m.r.date}`).join(", ")}. The 6× rows give the share of the 6 s between two chain blocks in brackets.`);
  await Deno.writeTextFile(readme, `${text.slice(0, a + START.length)}\n${block.join("\n")}\n${text.slice(b)}`);
  console.error(`wrote the table of ${readme}`);
}
