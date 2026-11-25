/**
 * Turns one backend's results files into the file the overview page reads (`ipir-benchmark-site/1`): the server
 * timings for one backend, `cpu`, `metal`, `nvidia` or `nvidia-no-bar`.
 *
 *   deno run --allow-read --allow-write --allow-run bench/export-site.ts [options] results/<run>.json[:<build>] ...
 *
 * Inputs: results files of `run-all.sh`. A file of schema `ipir-benchmark-nvidia/2` holds the compile and the budget
 * of both builds (`stock`, `patch0002`) and names its backend. A file of schema `/1` (the runs before backends) holds
 * one build, named after a colon: `results/x.json:stock` or `results/x.json:patch0002`. The fetch section comes from the
 * first input that has one. Shapes that did not fit or failed are left out; a section no input has is an empty array.
 *
 * Options:
 *   --backend NAME     the backend (required for `/1` inputs, and for a run whose resizable BAR was not detected
 *                      either way, `nvidia-unknown-bar`; otherwise from the first input)
 *   --label TEXT       the machine's short name (default: from the CPU and GPU names)
 *   --machine JSON     values of `machine` to set, for what the results file did not record (for example
 *                      '{"cpu":"AMD Ryzen 7 2700X","ramGiB":31,"rebar":false,"memoryBudgetLayer":true}')
 *   --date YYYY-MM-DD  the date (default: the latest input's)
 *   --commit SHA       the commit that holds the inputs (default: the last commit that changed them; the inputs must
 *                      be committed and unchanged). `--commit none`: the inputs are not committed yet, and
 *                      `source.commit` is null (`bench.sh` exports this way, on the machine that made the run)
 *   --out FILE         default: site/<host>-<date>-<backend>-site.json
 *
 * Shapes are written `rows × cols` (U+00D7). `execMs` is the GPU's own time from timestamp queries, null on the CPU.
 *
 * A run that built the blocks on the GPU (`gpu/block-build.ts`) also has `gpuCompile` (null otherwise): `check`, how
 * many of the checked blocks equal the CPU's output; `one`, one table per row count, with the GPU's time per stage;
 * `atOnce`, several tables of a row count at once; `sets`, one chain block of the six tables. And `budget.gpu`: the
 * budget cells with the blocks built on the GPU that is answering (`compileP50S` is then the GPU build).
 */
import { options } from "./lib.ts";

const BACKENDS = ["cpu", "metal", "nvidia", "nvidia-no-bar"] as const;
type Backend = (typeof BACKENDS)[number];
const BUILDS = ["stock", "patch0002"] as const;
type Build = (typeof BUILDS)[number];

const flags: string[] = [];
const inputs: { path: string; build: Build | null }[] = [];
for (let i = 0; i < Deno.args.length; i++) {
  const a = Deno.args[i]!;
  if (a.startsWith("--")) flags.push(a, Deno.args[++i] ?? "");
  else {
    const [path, build] = a.split(":") as [string, string | undefined];
    if (build !== undefined && !(BUILDS as readonly string[]).includes(build)) throw new Error(`unknown build ${build} in ${a}`);
    inputs.push({ path, build: (build as Build | undefined) ?? null });
  }
}
const opts = options(flags);
if (inputs.length === 0) throw new Error("usage: export-site.ts [options] results/<run>.json[:stock|:patch0002] ...");

// deno-lint-ignore no-explicit-any
type Any = any;
const files: { path: string; build: Build | null; r: Any }[] = [];
for (const { path, build } of inputs) {
  const r = JSON.parse(await Deno.readTextFile(path));
  if (r.schema === "ipir-benchmark-nvidia/1" && !build) throw new Error(`${path} is a /1 results file: name its build (${path}:stock or ${path}:patch0002)`);
  if (r.schema !== "ipir-benchmark-nvidia/1" && r.schema !== "ipir-benchmark-nvidia/2") throw new Error(`${path}: unknown schema ${r.schema}`);
  if (r.meta?.quick) throw new Error(`${path} is a --quick run`);
  files.push({ path, build, r });
}

const named: string | undefined = opts.get("backend") ?? files.find((f) => f.r.backend?.name)?.r.backend.name;
const backend = named as Backend | undefined;
if (!backend || !(BACKENDS as readonly string[]).includes(backend)) {
  const hint = named === "nvidia-unknown-bar" ? " (resizable BAR was not detected either way: pass --backend nvidia or --backend nvidia-no-bar)" : "";
  throw new Error(`--backend must be one of ${BACKENDS.join(", ")}: ${backend}${hint}`);
}
for (const f of files) {
  // A run names its own backend. `nvidia-unknown-bar` is the one name --backend may settle, as either NVIDIA backend.
  const own = f.r.backend?.name;
  const settles = own === "nvidia-unknown-bar" && (backend === "nvidia" || backend === "nvidia-no-bar");
  if (own && own !== backend && !settles) throw new Error(`${f.path} is backend ${own}, not ${backend}`);
}

async function git(args: string[]): Promise<string> {
  const r = await new Deno.Command("git", { args, stdout: "piped", stderr: "piped" }).output();
  if (!r.success) throw new Error(`git ${args.join(" ")}: ${new TextDecoder().decode(r.stderr)}`);
  return new TextDecoder().decode(r.stdout).trim();
}
let commit: string | null | undefined = opts.get("commit");
if (commit === "none") commit = null;
else if (!commit) {
  const paths = files.map((f) => f.path);
  const dirty = await git(["status", "--porcelain", "--", ...paths]);
  if (dirty) throw new Error(`commit the inputs first, or pass --commit:\n${dirty}`);
  commit = await git(["log", "-1", "--format=%H", "--", ...paths]);
  if (!commit) throw new Error("the inputs are not in any commit: commit them first, or pass --commit (a SHA, or none)");
}

const round = (x: number | null | undefined, d: number): number | null =>
  x === null || x === undefined || Number.isNaN(x) ? null : Math.round(x * 10 ** d) / 10 ** d;
const shape = (rows: number, cols: number) => `${rows} × ${cols}`;

// fetch: the first input that has one.
const fetchFile = files.find((f) => f.r.fetch?.results);
const fetch: Any[] = [];
if (fetchFile) {
  const f = fetchFile.r.fetch;
  const all = [...f.results];
  const fit = f.largest?.fit;
  if (fit && !all.some((x: Any) => x.ok && x.rows === fit.rows && x.cols === fit.cols)) all.push(fit);
  for (const x of all) {
    if (!x?.ok) continue;
    const call = (n: string) => ({ wallMs: round(x.timing[n].wallMs.median, 3)!, execMs: round(x.timing[n].execMedianMs?.total, 3) });
    fetch.push({ shape: shape(x.rows, x.cols), rows: x.rows, cols: x.cols, uploadMs: round(x.uploadMs, 2), call1: call("1"), call32: call("32") });
  }
}

// compile and budget, per build.
const compile: Record<Build, Any[]> = { stock: [], patch0002: [] };
const budget: Record<Build | "gpu", Any[]> = { stock: [], patch0002: [], gpu: [] };
/** A budget set's cells for the page. */
function budgetRows(s: Any, set: string): Any[] {
  return s.runs.filter((x: Any) => x.mode !== "none").map((run: Any) => ({
    shapes: set,
    callsPerSec: run.rate,
    mode: run.mode,
    blockWorkP50S: round(run.blockWorkMs.p50 / 1000, 3),
    blockWorkP99S: round(run.blockWorkMs.p99 / 1000, 3),
    compileP50S: round(run.compileMs.p50 / 1000, 3),
    uploadP50Ms: round(run.uploadMs?.p50, 2),
    over3s: run.overBudget,
    blocks: run.blocks,
  }));
}
for (const f of files) {
  const per: [Build, Any, Any][] = f.r.schema === "ipir-benchmark-nvidia/1"
    ? [[f.build!, f.r.compile, f.r.budget]]
    : BUILDS.filter((b) => !f.build || f.build === b).map((b) => [b, f.r.compile?.[b], f.r.budget?.[b]]);
  for (const [b, c, bud] of per) {
    if (c && compile[b].length) throw new Error(`two inputs have the compile of build ${b}`);
    for (const x of c ?? []) {
      compile[b].push({
        rows: x.rows,
        atOnce: x.tablesAtOnce,
        wallMs: round(x.wallMs, 2),
        hintsMs: round(x.perTableMedianMs.hints, 2),
        packingMs: round(x.perTableMedianMs.packing, 2),
      });
    }
    for (const set of ["today", "10m"] as const) {
      const s = bud?.[set];
      if (!s?.ok) continue;
      if (budget[b].some((x) => x.shapes === set)) throw new Error(`two inputs have the ${set} budget of build ${b}`);
      budget[b].push(...budgetRows(s, set));
    }
  }
}

// The blocks built on the GPU: from the first input that has them.
const gpuFile = files.find((f) => f.r.gpuCompile);
let gpuCompile: Any = null;
if (gpuFile) {
  const g = gpuFile.r.gpuCompile;
  const checks: Any[] = g.check?.checks ?? [];
  const t = g.timing?.ok ? g.timing : null;
  gpuCompile = {
    check: { blocks: checks.length, equal: checks.filter((c) => c.equal === true).length },
    one: (t?.shapes ?? []).filter((s: Any) => !s.failed).map((s: Any) => ({
      rows: s.rows,
      wallMs: round(s.wallMs?.median, 2),
      execMs: round(s.execMs?.median, 2),
      hintsMs: round(s.stagesExecMedianMs?.hints, 2),
      aggregationMs: round(s.stagesExecMedianMs?.aggregation, 2),
      chainsMs: round(s.stagesExecMedianMs?.chains, 2),
      compileMs: round(s.stagesExecMedianMs?.compile, 2),
    })),
    atOnce: (t?.shapes ?? []).flatMap((s: Any) => (s.atOnce ?? []).filter((m: Any) => !m.failed).map((m: Any) => ({ rows: s.rows, tables: m.tables, wallMs: round(m.wallMs?.median, 2), execMs: round(m.execMs?.median, 2) }))),
    sets: (t?.sets ?? []).filter((s: Any) => !s.failed).map((s: Any) => ({ shapes: s.set, wallMs: round(s.wallMs?.median, 2), execMs: round(s.execMs?.median, 2), gpuMemoryMiB: round(s.scratchBytes / 2 ** 20, 0) })),
  };
  for (const set of ["today", "10m"] as const) {
    const s = gpuFile.r.budget?.gpu?.[set];
    if (s?.ok) budget.gpu.push(...budgetRows(s, set));
  }
}

// machine: from the first input, then --machine.
const first = files[0]!.r;
const m = first.machine ?? {};
const cpuOnly = backend === "cpu";
const load = (l: number[] | null | undefined) => (l ? l.map((x) => x.toFixed(2)).join(" ") : null);
const machine: Record<string, unknown> = {
  label: null,
  os: [m.os, m.kernel && !String(m.os ?? "").startsWith("macOS") ? `kernel ${m.kernel}` : null].filter(Boolean).join(", ") || null,
  cpu: m.cpu ?? null,
  cpuThreads: m.logicalCpus ?? null,
  ramGiB: m.ramGiB === null || m.ramGiB === undefined ? null : Math.round(m.ramGiB),
  gpu: cpuOnly ? null : m.gpu ?? null,
  gpuMemoryMiB: cpuOnly ? null : m.vramMiB ?? null,
  driver: cpuOnly ? null : m.driver ?? null,
  rebar: cpuOnly ? null : first.backend?.resizableBar ?? (backend === "nvidia" ? true : backend === "nvidia-no-bar" ? false : null),
  memoryBudgetLayer: first.backend?.memoryBudgetLayer ?? false,
  loadAtStart: load(m.loadavgStart),
};
if (opts.has("machine")) Object.assign(machine, JSON.parse(opts.get("machine")!));
if (!machine.label) {
  const cpu = String(machine.cpu ?? "?").replace(/\(R\)|\(TM\)|\s+\d+-Core Processor|\s+Eight-Core Processor/g, "").replace(/^AMD\s+/, "");
  const gpu = machine.gpu ? String(machine.gpu).replace(/^NVIDIA\s+(GeForce\s+)?/, "") : null;
  machine.label = opts.get("label") ?? (!gpu || gpu === machine.cpu ? cpu : `${gpu} + ${cpu}`);
}
if (opts.has("label")) machine.label = opts.get("label");
const order = ["label", "os", "cpu", "cpuThreads", "ramGiB", "gpu", "gpuMemoryMiB", "driver", "rebar", "memoryBudgetLayer", "loadAtStart"];
for (const k of Object.keys(machine)) if (!order.includes(k)) throw new Error(`--machine: unknown key ${k}`);

const date = opts.get("date") ?? files.map((f) => f.r.meta?.date as string).filter(Boolean).sort().at(-1)!;
if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) throw new Error(`bad date ${date}`);
const site = {
  schema: "ipir-benchmark-site/1",
  backend,
  date,
  source: { repo: "midnight-experiments/ipir-benchmark-nvidia", commit, results: files.map((f) => f.path.replace(/^\.\//, "")) },
  machine: Object.fromEntries(order.map((k) => [k, machine[k] ?? null])),
  fetch,
  compile,
  gpuCompile,
  budget,
};
const host = m.hostname ?? "host";
const out = opts.get("out") ?? `site/${host}-${date}-${backend}-site.json`;
await Deno.writeTextFile(out, JSON.stringify(site, null, 1) + "\n");
console.error(
  `wrote ${out}: ${fetch.length} fetch shapes, compile ${compile.stock.length} stock / ${compile.patch0002.length} patch 0002, ` +
    `budget ${budget.stock.length} stock / ${budget.patch0002.length} patch 0002 / ${budget.gpu.length} GPU build` +
    (gpuCompile ? `; GPU compile: ${gpuCompile.check.equal} of ${gpuCompile.check.blocks} blocks equal, ${gpuCompile.one.length} row counts` : ""),
);
