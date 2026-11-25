/**
 * Gathers a run's raw files into one results file and a short Markdown summary.
 *
 *   deno run <permissions> bench/report.ts <raw dir> <out.json> <out.md>
 *
 * The raw dir holds what `run-all.sh` wrote: `meta.json`, `sysinfo-start.json`, `sysinfo-end.json`, `fetch.json`,
 * and for each build of the vendored crates (`stock`, without patch 0002, and `patch0002`) `compile-<build>.jsonl`,
 * `budget-today-<build>.json` and `budget-10m-<build>.json` (any may be missing if its step failed or was skipped).
 * On a GPU backend also the blocks built on the GPU: `gpu-build-check.json` (its output against the CPU's),
 * `gpu-compile.json` (its time) and `budget-<set>-gpu.json`; in the results, `gpuCompile` and `budget.gpu`.
 *
 * The results file (schema `ipir-benchmark-nvidia/2`) records the backend (`backend.name`: `cpu`, `metal`, `nvidia`
 * or `nvidia-no-bar`; where the answers ran; resizable BAR; the memory-budget layer) and keeps the compile and the
 * budget per build: `compile.stock`, `compile.patch0002`, `budget.stock.today`, ...
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
const fetch = await json("fetch.json");
const BUILDS = ["stock", "patch0002"] as const;
const budget: Record<string, Record<string, any>> = {};
for (const b of [...BUILDS, "gpu"]) budget[b] = { today: await json(`budget-today-${b}.json`), "10m": await json(`budget-10m-${b}.json`) };
// The GPU build of the blocks: null when the run had none.
const gpuCompile = meta?.gpuCompile ? { check: await json("gpu-build-check.json"), timing: await json("gpu-compile.json") } : null;

// Compile: per (rows, tables at once), medians over the repetitions.
function compileOf(compileLines: any[] | null): any[] | null {
  if (!compileLines) return null;
  const groups = new Map<string, any[]>();
  for (const l of compileLines) {
    const k = `${l.rows}/${l.tables_at_once}`;
    groups.set(k, [...(groups.get(k) ?? []), l]);
  }
  return [...groups.values()].map((ls) => {
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
const compile: Record<string, any[] | null> = {};
for (const b of BUILDS) compile[b] = compileOf(await jsonl(`compile-${b}.jsonl`));

const s = sysStart ?? {};
const smi = s.nvidiaSmi?.[0] ?? null;
const { raw: _raw, ...nvidiaMemory } = s.nvidiaMemory ?? { raw: null };
const backend = {
  name: meta?.backend ?? null,
  answer: meta?.answer ?? null,
  gpuBackend: s.gpuBackend ?? null,
  resizableBar: s.nvidiaMemory?.resizableBar ?? null,
  nvidiaMemory: s.nvidiaMemory ? nvidiaMemory : null,
  memoryBudgetLayer: s.memoryBudgetLayer ?? false,
  fetchPatch0002: meta?.answer === "cpu" ? true : null,
};
const machine = {
  hostname: s.hostname,
  gpu: smi?.name ?? s.macGpu?.model ?? s.webgpu?.summary ?? null,
  gpuCores: s.macGpu?.cores ?? null,
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
  powerSource: s.powerSource ?? null,
  loadavgStart: s.loadavg,
  loadavgEnd: sysEnd?.loadavg ?? null,
};
const results = { schema: "ipir-benchmark-nvidia/2", meta, backend, machine, compile, gpuCompile, fetch, budget, sysinfoStart: sysStart, sysinfoEnd: sysEnd };
await Deno.writeTextFile(outJson, JSON.stringify(results, null, 1));

// Markdown.
const f = (x: number | null | undefined, d = 1) => (x === null || x === undefined || Number.isNaN(x) ? "—" : x.toFixed(d));
const lines: string[] = [];
lines.push(`# IPIR+SP benchmark: ${machine.hostname ?? "host"}, ${meta?.date ?? ""}, backend \`${backend.name ?? "?"}\`${meta?.quick ? " (quick)" : ""}`, "");
lines.push(
  `Answers on the ${backend.answer === "cpu" ? "CPU (the vendored crates' server path)" : "GPU (`GpuIpirTable`, WebGPU)"}. ` +
    `Resizable BAR: ${backend.resizableBar === null ? "not applicable or unknown" : backend.resizableBar ? "on" : "off"}` +
    `${backend.nvidiaMemory ? ` (BAR1 ${backend.nvidiaMemory.bar1TotalMiB} MiB, framebuffer ${backend.nvidiaMemory.fbTotalMiB} MiB)` : ""}. ` +
    `Memory-budget layer: ${backend.memoryBudgetLayer ? "loaded" : "not loaded"}.`,
  "",
);
lines.push(
  (machine.driver === null && machine.vramMiB === null
    ? `GPU ${machine.gpu ?? "?"}${machine.gpuCores ? `, ${machine.gpuCores} cores` : ""}; `
    : `GPU ${machine.gpu ?? "?"}, driver ${machine.driver ?? "?"}, ${machine.vramMiB ?? "?"} MiB (${machine.vramUsedAtStartMiB ?? "?"} used at start); `) +
    `WebGPU adapter "${machine.webgpuAdapter?.summary ?? "?"}"${machine.webgpuAdapter?.timestampQuery ? " with timestamp queries" : ", no timestamp queries"}. ` +
    `CPU ${machine.cpu ?? "?"}, ${machine.physicalCpus ?? "?"} cores / ${machine.logicalCpus ?? "?"} threads, ${machine.ramGiB ?? "?"} GiB RAM. ` +
    `${machine.os ?? ""} ${machine.kernel ?? ""}, Deno ${machine.deno ?? "?"}, ${machine.rustc ?? "rustc ?"}. ` +
    `Load average at start ${machine.loadavgStart?.map((x: number) => x.toFixed(2)).join(" ") ?? "?"}, at end ${machine.loadavgEnd?.map((x: number) => x.toFixed(2)).join(" ") ?? "?"}. ` +
    (machine.powerSource ? `Power: ${machine.powerSource}. ` : "") +
    (meta?.preflight ? `Before the run: ${JSON.stringify(meta.preflight)}. ` : "") +
    `Run took ${meta?.seconds ? Math.round(meta.seconds / 60) + " min" : "?"}.`,
  "",
);

lines.push(`## 1. \`fetch\` on the ${backend.answer === "cpu" ? "CPU" : "GPU"}`, "");
if (fetch) {
  if (backend.answer === "cpu") {
    lines.push(
      "Median of " + fetch.runs + " calls after a warm-up. Wall: from the call to the last answer; the requests of a call are answered one after another, " +
        "each in parallel over the columns on all cores. Build: the table's CPU build. GPU check: the CPU's answers to 32 requests compared byte for byte with `GpuIpirTable`'s.",
      "",
    );
    lines.push("| rows × cols | resident | build | call of 1: wall | call of 32: wall | GPU check |", "|---|---|---|---|---|---|");
  } else {
    lines.push("Median of " + fetch.runs + " calls after a warm-up. Wall: from the call to the answer in host memory. Exec: the GPU's own time (timestamp queries).", "");
    lines.push("| rows × cols | resident | upload | call of 1: wall (exec) | call of 32: wall (exec) |", "|---|---|---|---|---|");
  }
  const row = (r: any, tag = "") => {
    if (!r.ok) {
      lines.push(`| ${r.rows} × ${r.cols}${tag} | ${f(r.residentBytes / 2 ** 20, 0)} MiB | — | ${r.skipped ? "does not fit" : "failed"}: ${String(r.reason ?? "").slice(0, 80)} | |`);
      return;
    }
    const t1 = r.timing["1"], t32 = r.timing["32"];
    if (backend.answer === "cpu") {
      const check = r.gpuCheck === null || r.gpuCheck === undefined ? "not run" : r.gpuCheck.equal === true ? "equal" : r.gpuCheck.equal === false ? "**different**" : "failed";
      lines.push(`| ${r.rows} × ${r.cols}${tag} | ${f(r.residentBytes / 2 ** 20, 0)} MiB | ${f(r.buildMs / 1000, 2)} s | ${f(t1.wallMs.median)} ms | ${f(t32.wallMs.median)} ms | ${check} |`);
      return;
    }
    lines.push(
      `| ${r.rows} × ${r.cols}${tag} | ${f(r.residentBytes / 2 ** 20, 0)} MiB | ${f(r.uploadMs / 1000, 2)} s | ${f(t1.wallMs.median)} ms (${f(t1.execMedianMs.total, 2)} ms) | ${f(t32.wallMs.median)} ms (${f(t32.execMedianMs.total, 2)} ms) |`,
    );
  };
  for (const r of fetch.results) row(r);
  for (const r of fetch.largest?.tried ?? []) row(r, " (largest)");
  lines.push("");
} else lines.push("Not run or failed (see the raw files).", "");

const buildName = (b: string) => (b === "stock" ? "stock crates" : b === "gpu" ? "blocks built on the GPU" : "with patch 0002");
lines.push("## 2. `compile` one column block on the CPU", "");
if (BUILDS.some((b) => compile[b])) {
  lines.push("Whole build of a one-block table (2,048 columns): hint columns + packing matrix. Median of the repetitions.", "");
  lines.push("| build | rows | at once | wall | per table: build | hints | packing |", "|---|---|---|---|---|---|---|");
  for (const b of BUILDS) {
    for (const c of compile[b] ?? []) {
      lines.push(`| ${buildName(b)} | ${c.rows} | ${c.tablesAtOnce} | ${f(c.wallMs)} ms | ${f(c.perTableMedianMs.build)} ms | ${f(c.perTableMedianMs.hints)} ms | ${f(c.perTableMedianMs.packing)} ms |`);
    }
  }
  lines.push("");
} else lines.push("Not run or failed.", "");

if (gpuCompile) {
  lines.push("## 2b. `compile` one column block on the GPU", "");
  const checks: any[] = gpuCompile.check?.checks ?? [];
  const equal = checks.filter((c) => c.equal === true).length;
  lines.push(
    checks.length
      ? `The same work on the GPU (WebGPU kernels, \`gpu/block-build.ts\`). Output against the CPU's (\`pack_digest\`'s SHA-256 over every output word): ` +
        `**${equal} of ${checks.length} blocks equal**${equal === checks.length ? "" : " — THE GPU BUILD DIFFERS, its times mean nothing"}.`
      : "The same work on the GPU (WebGPU kernels, `gpu/block-build.ts`). The check of its output against the CPU's did not run.",
    "",
  );
  const t = gpuCompile.timing;
  if (t?.ok) {
    lines.push("Wall: from writing the block's cells to the build's end. GPU: the GPU's own time (timestamp queries), all stages and each alone. Median of the repetitions.", "");
    lines.push("| rows | at once | wall | GPU | hints | aggregation | chains | compile | GPU memory besides the cells |", "|---|---|---|---|---|---|---|---|---|");
    for (const s of t.shapes ?? []) {
      const st = s.stagesExecMedianMs ?? {};
      lines.push(`| ${s.rows} | 1 | ${f(s.wallMs?.median)} ms | ${f(s.execMs?.median)} ms | ${f(st.hints, 2)} ms | ${f(st.aggregation, 2)} ms | ${f(st.chains, 2)} ms | ${f(st.compile, 2)} ms | ${f(s.scratchBytes / 2 ** 20, 0)} MiB |`);
      for (const m of s.atOnce ?? []) lines.push(`| ${s.rows} | ${m.tables} | ${f(m.wallMs?.median)} ms | ${f(m.execMs?.median)} ms | | | | | ${f(m.scratchBytes / 2 ** 20, 0)} MiB |`);
    }
    lines.push("");
    if (t.sets?.length) {
      lines.push("One chain block: one column block of each of the six tables, built at once (all chains in one dispatch).", "");
      lines.push("| shapes | rows of the six tables | wall | GPU | GPU memory besides the cells |", "|---|---|---|---|---|");
      for (const s of t.sets) lines.push(`| ${s.set === "today" ? "today's starting shapes" : "10 million outputs"} | ${s.rows.join(", ")} | ${f(s.wallMs?.median)} ms | ${f(s.execMs?.median)} ms | ${f(s.scratchBytes / 2 ** 20, 0)} MiB |`);
      lines.push("");
    }
  } else lines.push(`Timing not run or failed${t?.error ? `: ${String(t.error).slice(0, 300)}` : ""}.`, "");
}

lines.push("## 3. The 3 s block budget, simulated", "");
lines.push(
  (backend.answer === "cpu"
    ? "Per chain block: compile one changed column block of each of six tables on the CPU while the CPU also answers calls of 32 at a steady rate (no upload: block work is the compile). "
    : "Per chain block: compile one changed column block of each of six tables on the CPU, then upload the six to the GPU, while the GPU answers calls of 32 at a steady rate. ") +
    "The compile recomputes each block's whole hint columns: an upper bound, since hint updates for new records are not built. " +
    (gpuCompile ? "\"Blocks built on the GPU\": the six blocks are built on the GPU that is answering instead (compile = their cells written and built in one submission; upload = their layout for the answer from the build's GPU buffers). " : "") +
    "Latency: from the moment a call is due to its answer; \"during\" = calls overlapping a block's work.",
  "",
);
for (const [build, name, b] of [...BUILDS, "gpu"].flatMap((x) => Object.entries(budget[x]!).map(([n, v]) => [x, n, v] as const))) {
  if (!b) continue;
  lines.push(`### Shapes: ${name === "today" ? "today's starting shapes" : "10 million outputs"}, ${buildName(build)}`, "");
  lines.push(b.tables.map((t: any) => `${t.name} ${t.rows} × ${t.cols}`).join(", ") + `; ${f(b.residentBytes / 2 ** 30, 2)} GiB resident.`, "");
  if (!b.ok) {
    lines.push(b.doesNotFit ? `Does not fit in the GPU's memory (at ${b.stage}): ${b.error}` : `Failed at ${b.stage}: ${b.error}`, "");
    continue;
  }
  lines.push(
    b.worker
      ? `Pool: ${b.worker.pool_threads} threads (${b.worker.pool_priority}); ${b.blocks} blocks every ${b.intervalMs / 1000} s.`
      : `${b.blocks} blocks every ${b.intervalMs / 1000} s; the build holds ${f((b.gpuBuild?.scratchBytes + b.gpuBuild?.cellsBytes) / 2 ** 20, 0)} MiB of GPU memory.`,
    "",
  );
  lines.push(
    "| calls/s | mode | block work p50 / p99 | compile p50 / p99 | upload p50 | over 3 s | answer p50 / p99 during work | answer p50 / p99 no block work | calls dropped |",
    "|---|---|---|---|---|---|---|---|---|",
  );
  const base = new Map<number, any>(b.runs.filter((r: any) => r.mode === "none").map((r: any) => [r.rate, r]));
  for (const r of b.runs.filter((r: any) => r.mode !== "none")) {
    const bl = base.get(r.rate)?.answerLatencyMs;
    lines.push(
      `| ${r.rate} | ${r.mode} | ${f(r.blockWorkMs.p50 / 1000, 2)} / ${f(r.blockWorkMs.p99 / 1000, 2)} s | ${f(r.compileMs.p50 / 1000, 2)} / ${f(r.compileMs.p99 / 1000, 2)} s | ${r.uploadMs ? f(r.uploadMs.p50) + " ms" : "—"} | ${r.overBudget}/${r.blocks} | ` +
        `${f(r.answerLatencyMs.duringBlockWork.p50)} / ${f(r.answerLatencyMs.duringBlockWork.p99)} ms | ${f(bl?.p50)} / ${f(bl?.p99)} ms | ${r.cancelled ?? 0} of ${r.issued ?? "?"} |`,
    );
  }
  lines.push("");
}
lines.push(`Full data: \`${outJson.split("/").pop()}\`.`, "");
await Deno.writeTextFile(outMd, lines.join("\n"));
console.error(`wrote ${outJson} and ${outMd}`);
