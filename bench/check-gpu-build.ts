/**
 * Checks the GPU block build (`gpu/block-build.ts`) against the CPU build, for one column block.
 *
 *   deno run --allow-read --allow-write --allow-run --allow-env bench/check-gpu-build.ts [--rows 2048,8192]
 *     [--seeds 1,2,3] [--stages] [--steps N] [--bin ipir/target/release] [--out FILE]
 *
 * Default: for each row count and seed, builds the block `pack_digest` builds (`cells(rows, seed)`) on the GPU and
 * compares `pack_digest`'s SHA-256 over every output word (hint masks, both public masks, the compiled matrix) with
 * the CPU's. `--stages`: instead compares each stage's output with the CPU's (`gpu_build_prep --expect`), and names
 * the first words that differ; with `--steps N`, only the first N steps of each chain are checked (the kernel always
 * runs them all, so this is for a chain kernel changed to stop early).
 *
 * Prints one JSON line per block: `{"rows","seed","equal",...}`. Exits 1 when any block differs. `--out FILE` also
 * writes the lines as a JSON array.
 */
import { blockBuildDigest, GpuBlockBuild, parseBlockBuildPrep } from "../gpu/block-build.ts";
import { openBenchGpu, options } from "./lib.ts";

const opts = options(Deno.args);
const rowsList = (opts.get("rows") ?? "2048,8192,16384,32768").split(",").map(Number);
const seeds = (opts.get("seeds") ?? "1,2,3").split(",").map(Number);
const stages = opts.get("stages") === "true";
const bin = opts.get("bin") ?? "ipir/target/release";

async function run(cmd: string, args: string[]): Promise<string> {
  const r = await new Deno.Command(cmd, { args, stdout: "piped", stderr: "piped" }).output();
  if (!r.success) throw new Error(`${cmd} ${args.join(" ")} failed:\n${new TextDecoder().decode(r.stderr).slice(-2000)}`);
  return new TextDecoder().decode(r.stdout);
}

const gpu = await openBenchGpu();
if (!gpu.adapter.blockBuild) {
  console.error(`this adapter lacks the limits the GPU block build needs (${gpu.description})`);
  Deno.exit(2);
}
const device = gpu.device;
const errors: string[] = [];
device.addEventListener("uncapturederror", (e) => {
  const message = (e as GPUUncapturedErrorEvent).error.message;
  errors.push(message);
  console.error(`GPU error: ${message.slice(0, 2000)}`);
});
console.error(`GPU: ${gpu.description}`);

/** Indices of the first differing 64-bit words (as pairs of u32), and their count. */
function diff64(a: Uint8Array, b: Uint8Array): { count: number; first: number[] } {
  const x = new Uint32Array(a.buffer, a.byteOffset, a.byteLength / 4), y = new Uint32Array(b.buffer, b.byteOffset, b.byteLength / 4);
  if (x.length !== y.length) return { count: -1, first: [] };
  let count = 0;
  const first: number[] = [];
  for (let i = 0; i < x.length; i += 2) {
    if (x[i] !== y[i] || x[i + 1] !== y[i + 1]) {
      if (first.length < 4) first.push(i / 2);
      count++;
    }
  }
  return { count, first };
}

const tmp = await Deno.makeTempDir({ prefix: "ipir-gpu-build-" });
const lines: Record<string, unknown>[] = [];
let failed = false;
try {
  for (const rows of rowsList) {
    await run(`${bin}/gpu_build_prep`, [String(rows), `${tmp}/prep.bin`]);
    const prep = parseBlockBuildPrep(await Deno.readFile(`${tmp}/prep.bin`));
    const t0 = performance.now();
    const build = new GpuBlockBuild(device, prep);
    await device.queue.onSubmittedWorkDone();
    const setupMs = performance.now() - t0;
    const d = prep.header.d;
    for (const seed of seeds) {
      const line: Record<string, unknown> = { rows, seed };
      if (stages) {
        const steps = opts.has("steps") ? ["--steps", opts.get("steps")!] : [];
        await run(`${bin}/gpu_build_prep`, [String(rows), `${tmp}/prep.bin`, "--expect", String(seed), `${tmp}/ref`, ...steps]);
        const ref = (name: string) => Deno.readFile(`${tmp}/ref/${name}`);
        const cells = await ref("cells.u16");
        build.uploadCells(new Uint16Array(cells.buffer, cells.byteOffset, cells.byteLength / 2));
        const stage = async (names: ("hints" | "aggregation" | "chains" | "compile")[]) => {
          const enc = device.createCommandEncoder();
          for (const n of names) build.encodeStage(enc, n);
          device.queue.submit([enc.finish()]);
          await device.queue.onSubmittedWorkDone();
        };
        await stage(["hints"]);
        const masks = diff64(await build.read(build.masks), await ref("masks.u64"));
        await stage(["aggregation"]);
        const before = diff64(await build.read(build.slots), await ref("slots-before.u64"));
        await stage(["chains"]);
        const after = diff64(await build.read(build.slots), await ref("slots-after.u64"));
        const gd = new Int32Array((await build.read(build.digits)).buffer), rd = await ref("digits.i32");
        const rdi = new Int32Array(rd.buffer, rd.byteOffset, rd.byteLength / 4);
        let digits = 0;
        // Only the rows the reference's steps wrote (all of them without --steps); unwritten rows are zero there.
        for (let i = 0; i < rdi.length; i++) if (rdi[i] !== 0 && gd[i] !== rdi[i]) digits++;
        Object.assign(line, { masksDiffer: masks.count, masksFirst: masks.first.map((i) => [Math.floor(i / d), i % d]), slotsBeforeDiffer: before.count, slotsBeforeFirst: before.first.map((i) => [Math.floor(i / d), i % d]), slotsAfterDiffer: after.count, slotsAfterFirst: after.first.map((i) => [Math.floor(i / d), i % d]), digitsDiffer: digits });
        let equal = masks.count === 0 && before.count === 0 && after.count === 0 && digits === 0;
        if (!opts.has("steps")) {
          await stage(["compile"]);
          const m = new Int32Array((await build.read(build.matrix)).buffer);
          const want = await ref("matrix.u64");
          const w = new Uint32Array(want.buffer, want.byteOffset, want.byteLength / 4);
          let matrix = 0;
          const high = 2 ** (prep.header.qBits - 32) - 1;
          for (let i = 0; i < m.length; i++) if ((m[i]! >>> 0) !== w[2 * i] || (m[i]! < 0 ? high : 0) !== w[2 * i + 1]) matrix++;
          line.matrixDiffer = matrix;
          equal &&= matrix === 0;
        }
        line.equal = equal;
      } else {
        await run(`${bin}/gpu_build_prep`, [String(rows), `${tmp}/prep.bin`, "--cells", String(seed), `${tmp}/cells.u16`]);
        const cells = await Deno.readFile(`${tmp}/cells.u16`);
        build.uploadCells(new Uint16Array(cells.buffer, cells.byteOffset, cells.byteLength / 2));
        const t = performance.now();
        const enc = device.createCommandEncoder();
        build.encode(enc);
        device.queue.submit([enc.finish()]);
        await device.queue.onSubmittedWorkDone();
        const wallMs = performance.now() - t;
        const got = await blockBuildDigest(build);
        const want = (await run(`${bin}/pack_digest`, ["--parts", String(rows), String(seed)])).trim().split(/\s+/);
        const expected = { all: want[2], hints: want[3]?.replace("hints=", ""), packing: want[4]?.replace("packing=", "") };
        Object.assign(line, { equal: got.all === expected.all, hintsEqual: got.hints === expected.hints, packingEqual: got.packing === expected.packing, digest: got.all, expected: expected.all, wallMs: Math.round(wallMs * 10) / 10 });
      }
      if (errors.length) Object.assign(line, { equal: false, gpuErrors: errors.splice(0).map((e) => e.slice(0, 300)) });
      Object.assign(line, { setupMs: Math.round(setupMs) });
      if (!line.equal) failed = true;
      lines.push(line);
      console.log(JSON.stringify(line));
    }
    build.destroy();
  }
} finally {
  await Deno.remove(tmp, { recursive: true }).catch(() => {});
}
if (opts.has("out")) await Deno.writeTextFile(opts.get("out")!, JSON.stringify({ gpu: gpu.description, checks: lines }, null, 1));
device.destroy();
Deno.exit(failed ? 1 : 0);
