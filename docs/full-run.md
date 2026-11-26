# The full run

`./bench.sh` without `--report` runs the full benchmark, about 100 minutes on a GPU. Its results are in
[other-results.md](other-results.md). The report's shorter run (`./bench.sh --report`) is in the README.

## What it measures

| Step | Script | What |
|---|---|---|
| 1. `fetch` on the GPU | `bench/fetch-all.ts` → `bench/fetch-shape.ts` | For each table shape (rows × columns): 2048×2048, 2048×4096, 2048×8192, 4096×8192, 4096×12288, 6144×16384, 6144×28672, 8192×20480, 8192×22528, 10240×22528, 12288×26624, 14336×34816, 16384×18432, 16384×49152, 20480×40960, then the largest of 32768×32768, 32768×30720, … that fits. The time to answer a call of 32 requests (the padded call every wallet sends) and a call of 1. Reports wall time (from the call to the answer in host memory) and the GPU's own execution time per stage (timestamp queries). Median of 9 calls after a warm-up. Each shape runs in its own process, so a shape that does not fit in the GPU's memory is recorded as "does not fit" and the run goes on. |
| 1. `fetch` on the CPU (backend `cpu`) | `bench/fetch-all.ts --backend cpu` → `ipir/bench/src/bin/cpu_fetch.rs` | The same shapes and calls, answered by `NativeServer::respond` on the CPU: the requests of a call one after another, each in parallel over the columns on all cores, as the add-on answers a call. Wall time only (there is no GPU time). Also the table's CPU build. Once per shape, the CPU's answers to 32 requests are compared byte for byte with `GpuIpirTable`'s on the host's GPU, for the same cells and compiled matrices (`bench/check-cpu-answer.ts`). |
| 2. `compile` one column block | `ipir/bench/src/bin/compile_block.rs` | The CPU build of a table that is one column block (2,048 columns) at 2048, 8192, 16384 and 32768 rows, median of 3. Hint columns and packing matrix are timed separately. Then 4 at once in one process, sharing Rayon's pool, as the Node add-on would. |
| 2b. `compile` on the GPU (GPU backends) | `bench/check-gpu-build.ts`, `bench/gpu-compile.ts` → `gpu/block-build.ts` | The same build on the GPU, in WebGPU: hint columns, mask aggregation, the two collapse chains and the matrix compilation, from the block's cells to the same output words as the CPU. First the check: for 2048, 8192, 16384 and 32768 rows and seeds 1 to 3, and for one block of each table of the two six-table sets, `pack_digest`'s SHA-256 over every output word must equal the CPU's. Then the time, median of 5: one table (wall from writing the cells to the build's end, the GPU's own time, and each stage's), 4 and 6 tables of a row count at once, and one chain block of each six-table set (median of 10), measured one build right after another and again with the GPU left idle for 6 s before each build, as it is between chain blocks (median of 5). The tables of a set are built in one submission with all their chains in one dispatch, side by side. |
| 3. The block budget: a block's work in the 6 s between blocks | `bench/budget.ts` + `ipir/bench/src/bin/block_worker.rs` | Six tables stay resident on the GPU. The GPU answers calls of 32 against them, round robin, at a steady 2, 5 and 10 calls/s; calls are issued open loop, so a slow call does not delay the next. Every 6 s a chain block changes one column block of every table. The CPU compiles the six blocks, either one after another on all cores (`seq`) or all at once on a pool of half the cores at the lowest priority (`pool`). Then the six new blocks (cells and matrix) are uploaded to the GPU. Reported: the block's work time (p50, p99 over 30 blocks) against the 1 s goal and the 3 s ceiling and as a share of the 6 s between blocks, and the answer latency p50/p99 during block work and with no block work. On the `cpu` backend, the six tables stay in a `cpu_server` process (`ipir/bench/src/bin/cpu_server.rs`) that answers the calls on the CPU, all cores, while `block_worker` compiles; there is no upload, so a block's work is its compile. Calls the CPU cannot keep up with queue; those still waiting 30 s after a run's last block are dropped and counted. On a GPU backend the step runs a third time with the six blocks built on the GPU that is answering (mode `gpu`): a block's work is then writing the six blocks' cells, building them on the GPU in one submission, and laying each out for the answer from the build's GPU buffers, so no matrix crosses from the host. Run at today's starting shapes (2048×2048 read, 2048×2048 nullifier, 2048×4096 shielded groups, 6144×28672 Dust groups, 2048×2048 first coins, 2048×4096 generation groups) and at the 10-million-output shapes (14336×34816 read, 16384×18432 nullifier, 8192×20480 groups, and the same three Dust tables). Where the 10-million-output tables and the GPU build do not fit together in the GPU's memory, the step goes down in steps of half a million outputs (9.5, 9, 8.5, … million; the shapes are in `bench/gpu-build-sets.ts`) and measures the first size that fits: the largest tables that card keeps while it builds its blocks. |

Notes on what the numbers mean:

- The tables, matrices and requests are synthetic: xorshift cells, random packing matrices, and requests with the
  right magic and setup id. The time does not depend on the content, only on the shape.
- The compile in step 3 rebuilds each changed block's whole hint columns. The real design only updates the hints for
  the new records, and that code is not built yet, so these figures are an **upper bound** for the hints part.
- The setups use this design's patch 0001 (`new_without_columns`, seed and snapshot 32 zero bytes).
- Steps 2 and 3 run twice: with the stock crates and with patch 0002 (`ipir/patches/0002-faster-block-compile.patch`),
  which computes the same output words in less time. Patch 0002 is applied in `ipir/vendor/` (`ipir/target/release/`);
  `ipir/build-stock.sh` builds the same programs from a copy with the patch reversed (`ipir/target/stock/release/`).
  The results keep the two apart (`compile.stock`, `compile.patch0002`, and the same for `budget`). Step 1 on the CPU
  uses the patched build. Patch 0002 changes one helper of the answer path, the CRT reconstruction in `prepare_pack`;
  the GPU answer does not use the crates.
- The GPU build (step 2b) is a WebGPU port of the Metal build made for the design's question on building blocks on
  the GPU. It computes its exact dot products over three 30-bit primes, not the crates' 42-bit ones: the crates take
  the exact integer and only then reduce it modulo q, so the words are the same, and the check compares every one. It
  needs a device with 1,024 invocations per workgroup and 32 KB of workgroup memory (Apple GPUs and NVIDIA through
  Vulkan have them; WebGPU's defaults are 256 and 16 KB), and about 0.9 GiB of GPU memory for a six-table chain block
  besides the cells. With the 10-million-output tables resident (about 4 GiB), that does not fit a 6 GB card: the
  budget's GPU-build section then says "does not fit" and is followed by the largest tables that do fit (step 3), and
  the standalone time of step 2b is still measured. The
  production target for NVIDIA is CUDA; these kernels are the WebGPU form, and their times are an upper bound for it.
- Every Deno call pays a fixed read-back cost (about 12 ms on macOS). That is why wall time is much larger than
  execution time for small shapes.

## The steps by hand

`bench.sh` runs these; each also runs alone:

```sh
./setup.sh                  # checks the GPU stack, installs Deno 2.7.5 into .tools/, builds the Rust side (both builds)
./run-all.sh --quick        # a few minutes: small shapes, few blocks; checks everything works
./run-all.sh                # the full benchmark on this host's GPU, about 100 minutes
./run-all.sh --backend cpu  # the full benchmark with the answers on the CPU, about 1.5 hours
.tools/deno/deno run --allow-read bench/check-run.ts results/<run>.json   # is the run complete?
```

`setup.sh` needs network access once: it downloads Deno and runs `cargo fetch --locked`. After that, builds are
`--offline --locked`. To prepare an air-gapped host, run `cargo vendor` in `ipir/` on a connected machine and copy
the result over. `./setup.sh --cpu` is for the CPU backend alone: a missing or unusable GPU does not fail it.

`run-all.sh` writes `results/<host>-<date>-<backend>.json` (all data, including the machine description and the
backend) and `results/<host>-<date>-<backend>.md` (a short summary, also printed at the end). It also writes raw files
and a log in `results/raw/<host>-<date>-<backend>/`, which git ignores. Options: `--backend cpu`, `--builds
stock,patch0002`, `--quick`, `--skip-build`, `--skip-compile`, `--skip-fetch`, `--skip-budget`, `--no-gpu-check`,
`--budget-sets today`, `--skip-gpu-compile`, `--allow-non-nvidia`, `--name NAME`. `bench/budget.ts --set 8m` runs step 3 alone at a size between the two (`bench/gpu-build-sets.ts`), and `--tables ROWSxCOLS,...` at any six table shapes. `IPIR_BENCH_HOST=<name>` sets the host name used in the file names and results.
`run-all.sh` does not load the Vulkan layer: on a GPU without resizable BAR, load it first
(`tools/hide-memory-budget-layer/README.md`), or use `bench.sh`.

`run-all.sh` goes on when a step fails, so that one failed shape does not lose the rest of a long run.
`bench/check-run.ts` reads the results afterwards and lists what is missing or failed; it exits 1 on a problem. A
shape that does not fit in the GPU's memory is a problem only in a quick run.

How long it takes, in the full run: building about 4 min the first time (two builds). Step 2 is about 2 to 4 min for
both builds, and step 2b about 2 min. Step 1 is 5 to 15 min, mostly making the synthetic tables on the host. Step 3 is
about 80 min for the CPU builds (2 builds × 2 shape sets × 3 rates × 2 modes × 30 blocks × 6 s, plus baselines) and
about 20 min more with the blocks built on the GPU (2 shape sets × 3 rates × 30 blocks × 6 s, plus baselines).

## Backends, and the files for the overview page

`run-all.sh` decides the backend (`bench/sysinfo.ts`, recorded as `backend` in the results and `gpuBackend` in the
machine description):

- `metal`: macOS. Deno's WebGPU on Metal. A software adapter is refused, as on Linux.
- `nvidia` or `nvidia-no-bar`: Linux with an NVIDIA GPU. Resizable BAR is on when `nvidia-smi -q -d MEMORY` reports a
  BAR1 total at least as large as the framebuffer total; the results keep both totals and BAR1's use. The results also
  record whether the Vulkan layer of `tools/hide-memory-budget-layer/` was loaded (`VK_INSTANCE_LAYERS`). When
  `nvidia-smi` reports neither total, the run is named `nvidia-unknown-bar`, and the export needs
  `--backend nvidia` or `--backend nvidia-no-bar`.
- `cpu`: `--backend cpu`, on any host. Steps 1 and 3 answer on the CPU. If the host has a usable GPU, step 1 also
  checks the CPU's answers against it.

`bench.sh` makes the file the overview page reads. By hand:

```sh
.tools/deno/deno run --allow-read --allow-write --allow-run bench/export-site.ts results/<host>-<date>-<backend>.json
```

It writes `site/<host>-<date>-<backend>-site.json` (schema `ipir-benchmark-site/1`): the machine, the `fetch` time
of each shape that ran (wall time, and the GPU's own time, null on the CPU), the `compile` of one column block, and
the block budget's cells (each with the share of the 6 s its work takes, and the blocks over the 1 s goal and over the 3 s ceiling), the last two for `stock` and `patch0002`. A run on a GPU backend also has `gpuCompile` (how many
checked blocks equal the CPU's output, the GPU build's time per row count with its stages, several tables at once, and
one chain block of each six-table set) and `budget.gpu` (the budget cells with the blocks built on the GPU); both are
additions to `ipir-benchmark-site/1`, absent from the files exported before them. Shapes that did not fit and sections
that did not run are left out, not filled in. `source.commit` is the commit that holds the inputs, so by default the export wants them
committed; `--commit none` exports before the commit and leaves it null, which is what `bench.sh` does. To fill it
in, run the export again after the results are committed. The page picks the files up from `site/`. The files there
now:

| Backend | Site file | From |
|---|---|---|
| `nvidia-no-bar` | `site/pop-os-2026-11-25-nvidia-no-bar-site.json` | `results/pop-os-2026-11-25-nvidia-no-bar.json`: an RTX 2060 without resizable BAR, with the Vulkan layer |
| `metal` | `site/m4-max-2026-11-25-metal-site.json` | `results/m4-max-2026-11-25-metal.json`: an Apple M4 Max |
| `cpu` | not yet | `./bench.sh --cpu-only` on any host |
| `nvidia` | `site/rtx4090-2026-11-25-nvidia-site.json` | `results/rtx4090-2026-11-25-nvidia.json`: an RTX 4090 with resizable BAR; no 10-million-output budget (`results/rtx4090-notes.md`) |

A results file made before the backends (`results/pop-os-2026-11-24-*`, `results/pop-os-2026-11-25-hidebudget-*`)
holds one build and names no backend; the export takes it with its build after a colon and `--backend`, for example
`results/pop-os-2026-11-25-hidebudget-opt.json:patch0002 --backend nvidia-no-bar`.
