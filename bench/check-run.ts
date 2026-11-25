/**
 * Says whether a run is complete: every step that `run-all.sh` ran left its section in the results file, and nothing
 * in it failed. `run-all.sh` goes on when a step fails, so that one bad shape does not lose an 80-minute run; this is
 * the check afterwards. `bench.sh` runs it after the quick check (and stops there on a problem) and after a full run.
 *
 *   deno run --allow-read bench/check-run.ts results/<run>.json
 *
 * Prints one line per section and exits 1 when there is a problem. A shape that does not fit in the GPU's memory is
 * not a problem in a full run (it is recorded and left out); in a `--quick` run it is, because those shapes are small.
 */
// deno-lint-ignore no-explicit-any
type Any = any;

const path = Deno.args[0];
if (!path) throw new Error("usage: check-run.ts results/<run>.json");
const r: Any = JSON.parse(await Deno.readTextFile(path));
if (r.schema !== "ipir-benchmark-nvidia/2") throw new Error(`${path}: schema ${r.schema}, expected ipir-benchmark-nvidia/2`);

const quick = r.meta?.quick === true;
const builds: string[] = String(r.meta?.builds ?? "stock,patch0002").split(",");
const problems: string[] = [];
const say = (ok: boolean, text: string) => {
  console.log(`${ok ? "ok     " : "PROBLEM"} ${text}`);
  if (!ok) problems.push(text);
};

console.log(`${path}: backend ${r.backend?.name ?? "?"}, answers on the ${r.backend?.answer ?? "?"}${quick ? ", quick" : ""}`);

// compile, per build.
for (const b of builds) {
  const c: Any[] | null = r.compile?.[b] ?? null;
  say(Array.isArray(c) && c.length > 0, `compile, build ${b}: ${c?.length ?? 0} measurement(s)`);
}

// fetch.
const shapeOf = (x: Any) => `${x.rows}x${x.cols}`;
const results: Any[] | null = r.fetch?.results ?? null;
if (!results) say(false, "fetch: no results");
else {
  const ran = results.filter((x) => x.ok);
  const failed = results.filter((x) => !x.ok && !x.skipped);
  const noFit = results.filter((x) => !x.ok && x.skipped);
  say(ran.length > 0, `fetch: ${ran.length} of ${results.length} shape(s) ran`);
  for (const x of failed) say(false, `fetch ${shapeOf(x)} failed: ${String(x.reason ?? "no reason recorded").slice(0, 300)}`);
  for (const x of noFit) {
    const text = `fetch ${shapeOf(x)} does not fit: ${String(x.reason ?? "").slice(0, 300)}`;
    if (quick) say(false, text);
    else console.log(`note    ${text}`);
  }
  if (!quick) {
    const fit = r.fetch.largest?.fit;
    console.log(`note    largest shape that fits: ${fit ? shapeOf(fit) : "none of the candidates"}`);
  }
  // The CPU backend's answers, compared with the GPU's.
  const checked = ran.filter((x) => x.gpuCheck);
  const different = checked.filter((x) => x.gpuCheck.equal === false);
  const notCompared = checked.filter((x) => x.gpuCheck.equal !== true && x.gpuCheck.equal !== false);
  if (r.meta?.gpuCheck) {
    say(different.length === 0, `CPU answers against the GPU's: ${checked.length - different.length - notCompared.length} equal, ${different.length} different`);
    for (const x of notCompared) say(false, `CPU answers against the GPU's, ${shapeOf(x)}: the comparison did not run (${String(x.gpuCheck.error ?? "").slice(0, 300)})`);
  }
}

// budget, per build and shape set.
const sets = quick ? ["today"] : ["today", "10m"];
for (const b of builds) {
  for (const set of sets) {
    const s: Any = r.budget?.[b]?.[set] ?? null;
    if (!s) {
      say(false, `budget ${set}, build ${b}: missing`);
      continue;
    }
    const runs: Any[] = (s.runs ?? []).filter((x: Any) => x.mode !== "none");
    say(s.ok === true && runs.length > 0, `budget ${set}, build ${b}: ${runs.length} run(s)${s.ok === true ? "" : `, not ok${s.error ? ` (${String(s.error).slice(0, 300)})` : ""}`}`);
    const over = runs.filter((x) => x.overBudget > 0).length;
    if (over) console.log(`note    budget ${set}, build ${b}: ${over} of ${runs.length} run(s) had a block over 3 s (a result, not a failure)`);
  }
}

if (problems.length) {
  console.log(`\n${problems.length} problem(s) in ${path}`);
  Deno.exit(1);
}
console.log(`\ncomplete: ${path}`);
