/**
 * Gathers a run's raw files into one results file and a short Markdown summary.
 *
 *   deno run <permissions> bench/report.ts <raw dir> <out.json> <out.md>
 *
 * The raw dir holds what `run-all.sh` wrote: `meta.json`, `sysinfo-start.json`, `sysinfo-end.json`,
 * `compile.jsonl`, `fetch.json`, `budget-today.json`, `budget-10m.json` (any may be missing if its step failed or
 * was skipped).
 */
import { median } from "./lib.ts";

const [raw, outJson, outMd] = Deno.args;
if (!raw || !outJson || !outMd) throw new Error("usage: report.ts <raw dir> <out.json> <out.md>");

async function json(name: string): Promise<any> {
  try {
    return JSON.parse(await Deno.readTextFile(`${raw}/${name}`));
  } catch {
    return null;
  }
}
async function jsonl(name: string): Promise<any[] | null> {
  try {
    return (await Deno.readTextFile(`${raw}/${name}`)).split("\n").filter((l) => l.trim().startsWith("{")).map((l) => JSON.parse(l));
  } catch {
    return null;
  }
}

const meta = await json("meta.json");
const sysStart = await json("sysinfo-start.json");
const sysEnd = await json("sysinfo-end.json");
const compileLines = await jsonl("compile.jsonl");
const fetch = await json("fetch.json");
const budgets = { today: await json("budget-today.json"), "10m": await json("budget-10m.json") };

// Compile: per (rows, tables at once), medians over the repetitions.
let compile: any[] | null = null;
if (compileLines) {
  const groups = new Map<string, any[]>();
  for (const l of compileLines) {
    const k = `${l.rows}/${l.tables_at_once}`;
    groups.set(k, [...(groups.get(k) ?? []), l]);
  }
  compile = [...groups.values()].map((ls) => {
    const tables = ls.flatMap((l) => l.tables);
    return {
      rows: ls[0].rows,
      tablesAtOnce: ls[0].tables_at_once,
      reps: ls.length,
      rayonThreads: ls[0].rayon_threads,
      wallMs: median(ls.map((l) => l.wall_ms)),
      wallMsAll: ls.map((l) => l.wall_ms),
      perTableMedianMs: {
        build: median(tables.map((t) => t.build_ms)),
        hints: median(tables.map((t) => t.hints_ms)),
        packing: median(tables.map((t) => t.packing_ms)),
      },
    };
  });
}

const s = sysStart ?? {};
const smi = s.nvidiaSmi?.[0] ?? null;
const machine = {
  hostname: s.hostname,
  gpu: smi?.name ?? s.webgpu?.summary ?? null,
  driver: smi?.driver_version ?? null,
  vramMiB: smi ? Number(smi["memory.total"]) : null,
  vramUsedAtStartMiB: smi ? Number(smi["memory.used"]) : null,
  vulkan: s.vulkan,
  webgpuAdapter: s.webgpu,
  cpu: s.cpuModel,
  logicalCpus: s.logicalCpus,
  physicalCpus: s.physicalCpus,
  ramGiB: s.memTotalKiB ? Math.round((s.memTotalKiB / 2 ** 20) * 10) / 10 : null,
  os: s.distro ?? s.os,
  kernel: s.osRelease,
  deno: s.deno?.deno,
  rustc: s.rustc,
  loadavgStart: s.loadavg,
  loadavgEnd: sysEnd?.loadavg ?? null,
};
const results = { schema: "ipir-benchmark-nvidia/1", meta, machine, compile, fetch, budget: budgets, sysinfoStart: sysStart, sysinfoEnd: sysEnd };
await Deno.writeTextFile(outJson, JSON.stringify(results, null, 1));

// Markdown.
const f = (x: number | null | undefined, d = 1) => (x === null || x === undefined || Number.isNaN(x) ? "—" : x.toFixed(d));
const lines: string[] = [];
lines.push(`# IPIR+SP GPU benchmark: ${machine.hostname ?? "host"}, ${meta?.date ?? ""}${meta?.quick ? " (quick)" : ""}`, "");
lines.push(
  `GPU ${machine.gpu ?? "?"}, driver ${machine.driver ?? "?"}, ${machine.vramMiB ?? "?"} MiB (${machine.vramUsedAtStartMiB ?? "?"} used at start); ` +
    `WebGPU adapter "${machine.webgpuAdapter?.summary ?? "?"}"${machine.webgpuAdapter?.timestampQuery ? " with timestamp queries" : ", no timestamp queries"}. ` +
    `CPU ${machine.cpu ?? "?"}, ${machine.physicalCpus ?? "?"} cores / ${machine.logicalCpus ?? "?"} threads, ${machine.ramGiB ?? "?"} GiB RAM. ` +
    `${machine.os ?? ""} ${machine.kernel ?? ""}, Deno ${machine.deno ?? "?"}, ${machine.rustc ?? "rustc ?"}. ` +
    `Load average at start ${machine.loadavgStart?.map((x: number) => x.toFixed(2)).join(" ") ?? "?"}, at end ${machine.loadavgEnd?.map((x: number) => x.toFixed(2)).join(" ") ?? "?"}. ` +
    `Run took ${meta?.seconds ? Math.round(meta.seconds / 60) + " min" : "?"}.`,
  "",
);

lines.push("## 1. `fetch` on the GPU", "");
if (fetch) {
  lines.push("Median of " + fetch.runs + " calls after a warm-up. Wall: from the call to the answer in host memory. Exec: the GPU's own time (timestamp queries).", "");
  lines.push("| rows × cols | resident | upload | call of 1: wall (exec) | call of 32: wall (exec) |", "|---|---|---|---|---|");
  const row = (r: any, tag = "") => {
    if (!r.ok) {
      lines.push(`| ${r.rows} × ${r.cols}${tag} | ${f(r.residentBytes / 2 ** 20, 0)} MiB | — | ${r.skipped ? "does not fit" : "failed"}: ${String(r.reason ?? "").slice(0, 80)} | |`);
      return;
    }
    const t1 = r.timing["1"], t32 = r.timing["32"];
    lines.push(
      `| ${r.rows} × ${r.cols}${tag} | ${f(r.residentBytes / 2 ** 20, 0)} MiB | ${f(r.uploadMs / 1000, 2)} s | ${f(t1.wallMs.median)} ms (${f(t1.execMedianMs.total, 2)} ms) | ${f(t32.wallMs.median)} ms (${f(t32.execMedianMs.total, 2)} ms) |`,
    );
  };
  for (const r of fetch.results) row(r);
  for (const r of fetch.largest?.tried ?? []) row(r, " (largest)");
  lines.push("");
} else lines.push("Not run or failed (see the raw files).", "");

lines.push("## 2. `compile` one column block on the CPU", "");
if (compile) {
  lines.push("Whole build of a one-block table (2,048 columns): hint columns + packing matrix. Median of the repetitions.", "");
  lines.push("| rows | at once | wall | per table: build | hints | packing |", "|---|---|---|---|---|---|");
  for (const c of compile) {
    lines.push(`| ${c.rows} | ${c.tablesAtOnce} | ${f(c.wallMs)} ms | ${f(c.perTableMedianMs.build)} ms | ${f(c.perTableMedianMs.hints)} ms | ${f(c.perTableMedianMs.packing)} ms |`);
  }
  lines.push("");
} else lines.push("Not run or failed.", "");

lines.push("## 3. The 3 s block budget, simulated", "");
lines.push(
  "Per chain block: compile one changed column block of each of six tables on the CPU, then upload the six to the GPU, while the GPU answers calls of 32 at a steady rate. " +
    "The compile recomputes each block's whole hint columns: an upper bound, since hint updates for new records are not built. " +
    "Latency: from the moment a call is due to its answer; \"during\" = calls overlapping a block's work.",
  "",
);
for (const [name, b] of Object.entries(budgets)) {
  if (!b) continue;
  lines.push(`### Shapes: ${name === "today" ? "today's starting shapes" : "10 million outputs"}`, "");
  lines.push(b.tables.map((t: any) => `${t.name} ${t.rows} × ${t.cols}`).join(", ") + `; ${f(b.residentBytes / 2 ** 30, 2)} GiB resident.`, "");
  if (!b.ok) {
    lines.push(`Failed at ${b.stage}: ${b.error}`, "");
    continue;
  }
  lines.push(`Pool: ${b.worker?.pool_threads} threads (${b.worker?.pool_priority}); ${b.blocks} blocks every ${b.intervalMs / 1000} s.`, "");
  lines.push(
    "| calls/s | mode | block work p50 / p99 | compile p50 / p99 | upload p50 | over 3 s | answer p50 / p99 during work | answer p50 / p99 no block work |",
    "|---|---|---|---|---|---|---|---|",
  );
  const base = new Map<number, any>(b.runs.filter((r: any) => r.mode === "none").map((r: any) => [r.rate, r]));
  for (const r of b.runs.filter((r: any) => r.mode !== "none")) {
    const bl = base.get(r.rate)?.answerLatencyMs;
    lines.push(
      `| ${r.rate} | ${r.mode} | ${f(r.blockWorkMs.p50 / 1000, 2)} / ${f(r.blockWorkMs.p99 / 1000, 2)} s | ${f(r.compileMs.p50 / 1000, 2)} / ${f(r.compileMs.p99 / 1000, 2)} s | ${f(r.uploadMs.p50)} ms | ${r.overBudget}/${r.blocks} | ` +
        `${f(r.answerLatencyMs.duringBlockWork.p50)} / ${f(r.answerLatencyMs.duringBlockWork.p99)} ms | ${f(bl?.p50)} / ${f(bl?.p99)} ms |`,
    );
  }
  lines.push("");
}
lines.push(`Full data: \`${outJson.split("/").pop()}\`.`, "");
await Deno.writeTextFile(outMd, lines.join("\n"));
console.error(`wrote ${outJson} and ${outMd}`);
