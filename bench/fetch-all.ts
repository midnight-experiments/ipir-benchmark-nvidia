/**
 * Task 1 driver: runs `fetch-shape.ts` once per shape, each in a fresh Deno process with a time limit, and writes all
 * results (and every skip or failure, with its reason) to one JSON file.
 *
 *   deno run <permissions> bench/fetch-all.ts --out FILE [--shapes 2048x2048,...] [--largest 32768x32768,...]
 *     [--runs 9] [--vram-mib N] [--timeout-s 900]
 *
 * `--largest`: candidates tried in order after the shapes; the first that runs is kept as "the largest that fits"
 * and the rest are not tried. `--vram-mib`: the GPU's memory (from nvidia-smi); a shape whose resident bytes alone
 * exceed it is skipped without trying. Any other shape is tried, and an out-of-memory error is recorded as a skip.
 */
import { MIB, options, parseShapes, residentBytes } from "./lib.ts";

const opts = options(Deno.args);
const out = opts.get("out");
if (!out) throw new Error("--out FILE is required");
const shapes = parseShapes(
  opts.get("shapes") ??
    "2048x2048,2048x4096,2048x8192,4096x8192,4096x12288,6144x16384,6144x28672,8192x20480,8192x22528,10240x22528,12288x26624,14336x34816,16384x18432,16384x49152,20480x40960",
);
const largest = parseShapes(opts.get("largest") ?? "32768x32768,32768x30720,32768x28672,32768x26624,32768x24576,32768x22528,32768x20480,32768x16384");
const runs = Number(opts.get("runs") ?? 9);
const vramMib = opts.has("vram-mib") ? Number(opts.get("vram-mib")) : null;
const timeoutS = Number(opts.get("timeout-s") ?? 900);
const here = new URL(".", import.meta.url).pathname;

async function one(rows: number, cols: number): Promise<Record<string, unknown>> {
  const need = residentBytes(rows, cols);
  if (vramMib !== null && need > vramMib * MIB) {
    const r = { ok: false, skipped: true, rows, cols, residentBytes: need, reason: `resident bytes ${(need / MIB).toFixed(0)} MiB exceed the GPU's ${vramMib} MiB` };
    console.error(`fetch ${rows}x${cols}: skipped, ${r.reason}`);
    return r;
  }
  const t = performance.now();
  const child = new Deno.Command(Deno.execPath(), {
    args: ["run", "--allow-read", "--allow-env", `${here}fetch-shape.ts`, String(rows), String(cols), String(runs)],
    stdout: "piped",
    stderr: "piped",
  }).spawn();
  const timer = setTimeout(() => {
    try {
      child.kill("SIGKILL");
    } catch { /* gone */ }
  }, timeoutS * 1000);
  const res = await child.output();
  clearTimeout(timer);
  const seconds = (performance.now() - t) / 1000;
  const stdout = new TextDecoder().decode(res.stdout).trim().split("\n");
  const stderr = new TextDecoder().decode(res.stderr).trim();
  let parsed: Record<string, unknown> | null = null;
  try {
    parsed = JSON.parse(stdout[stdout.length - 1] ?? "");
  } catch { /* no JSON */ }
  if (res.success && parsed?.ok) {
    const t32 = (parsed.timing as Record<string, { wallMs: { median: number }; execMedianMs: { total: number | null } }>)["32"]!;
    console.error(`fetch ${rows}x${cols}: call of 32 wall ${t32.wallMs.median.toFixed(1)} ms, exec ${t32.execMedianMs.total?.toFixed(2) ?? "n/a"} ms (${seconds.toFixed(0)} s)`);
    return { ...parsed, processSeconds: seconds };
  }
  const error = (parsed?.error as string | undefined) ?? (res.signal ? `killed by ${res.signal} (time limit ${timeoutS} s?)` : `exit code ${res.code}`);
  const oom = /out of memory|OutOfMemory|allocat|device lost/i.test(error + stderr);
  console.error(`fetch ${rows}x${cols}: ${oom ? "does not fit" : "FAILED"}: ${error}`);
  return { ok: false, skipped: oom, rows, cols, residentBytes: need, stage: parsed?.stage ?? null, reason: error, stderrTail: stderr.slice(-2000), processSeconds: seconds };
}

const results: Record<string, unknown>[] = [];
for (const [r, c] of shapes) results.push(await one(r, c));
let largestFit: Record<string, unknown> | null = null;
const largestTried: Record<string, unknown>[] = [];
for (const [r, c] of largest) {
  const res = await one(r, c);
  largestTried.push(res);
  if (res.ok) {
    largestFit = res;
    break;
  }
}
await Deno.writeTextFile(out, JSON.stringify({ runs, vramMib, results, largest: { fit: largestFit, tried: largestTried } }, null, 1));
