# ipir-benchmark-nvidia

This repository measures, on one Linux machine with an NVIDIA GPU (written for a GeForce RTX 2060 with 6 GB), how fast
a server answers IPIR+SP private information retrieval calls on the GPU, and whether it can also keep its tables up to
date as the chain grows. The answer code is the private shielded sync proof of concept's WebGPU implementation of
IPIR+SP's server answer (`GpuIpirTable`, byte for byte the same answers as the native add-on), run under Deno, which
uses Vulkan on Linux. The table preparation code is the Zcash IPIR+SP crates, in Rust, on the CPU.

It exists for the design of UmbraDB's private sync modules (spec 00065). There, every table follows the chain block by
block (one block every 6 s), and the server must finish a block's table work in under 3 s while it keeps answering
wallets. So far this has only been measured on an Apple M4 Max. The numbers here say whether a common, older GPU and
an ordinary desktop CPU are enough: how long a call of 32 requests takes for each table shape, how long the CPU takes
to compile one changed column block, and whether a block's work stays under 3 s while the GPU is answering calls.

## What it measures

| Step | Script | What |
|---|---|---|
| 1. `fetch` on the GPU | `bench/fetch-all.ts` → `bench/fetch-shape.ts` | For each table shape (rows × columns): 2048×2048, 2048×4096, 2048×8192, 4096×8192, 4096×12288, 6144×16384, 6144×28672, 8192×20480, 8192×22528, 10240×22528, 12288×26624, 14336×34816, 16384×18432, 16384×49152, 20480×40960, then the largest of 32768×32768, 32768×30720, … that fits. The time to answer a call of 32 requests (the padded call every wallet sends) and a call of 1. Reports wall time (from the call to the answer in host memory) and the GPU's own execution time per stage (timestamp queries). Median of 9 calls after a warm-up. Each shape runs in its own process, so a shape that does not fit in the GPU's memory is recorded as "does not fit" and the run goes on. |
| 2. `compile` one column block | `ipir/bench/src/bin/compile_block.rs` | The CPU build of a table that is one column block (2,048 columns) at 2048, 8192, 16384 and 32768 rows, median of 3. Hint columns and packing matrix are timed separately. Then 4 at once in one process, sharing Rayon's pool, as the Node add-on would. |
| 3. The 3 s block budget | `bench/budget.ts` + `ipir/bench/src/bin/block_worker.rs` | Six tables stay resident on the GPU. The GPU answers calls of 32 against them, round robin, at a steady 2, 5 and 10 calls/s; calls are issued open loop, so a slow call does not delay the next. Every 6 s a chain block changes one column block of every table. The CPU compiles the six blocks, either one after another on all cores (`seq`) or all at once on a pool of half the cores at the lowest priority (`pool`). Then the six new blocks (cells and matrix) are uploaded to the GPU. Reported: the block's work time (p50, p99 over 30 blocks) against 3 s, and the answer latency p50/p99 during block work and with no block work. Run at today's starting shapes (2048×2048 read, 2048×2048 nullifier, 2048×4096 shielded groups, 6144×28672 Dust groups, 2048×2048 first coins, 2048×4096 generation groups) and at the 10-million-output shapes (14336×34816 read, 16384×18432 nullifier, 8192×20480 groups, and the same three Dust tables). |

Notes on what the numbers mean:

- The tables, matrices and requests are synthetic: xorshift cells, random packing matrices, and requests with the
  right magic and setup id. The time does not depend on the content, only on the shape.
- The compile in step 3 rebuilds each changed block's whole hint columns. The real design only updates the hints for
  the new records, and that code is not built yet, so these figures are an **upper bound** for the hints part.
- The setups use this design's patch 0001 (`new_without_columns`, seed and snapshot 32 zero bytes).
- The CPU build of a column block uses patch 0002 (`ipir/patches/0002-faster-block-compile.patch`), which computes
  the same output words as the unpatched crates in less time.
- Every Deno call pays a fixed read-back cost (about 12 ms on macOS). That is why wall time is much larger than
  execution time for small shapes.

## Host requirements

- Linux, x86_64: Ubuntu 22.04 or 24.04, or similar. Root is needed only to install the driver and the packages below.
- NVIDIA driver **550 or newer** (535 should work; the RTX 2060 is Turing). The driver must include its **Vulkan ICD**:
  on Ubuntu that is `libnvidia-gl-<branch>`, which the desktop driver packages pull in. The headless/server packages
  (`nvidia-headless-*`) do not, so install `libnvidia-gl-<branch>` too.
  Check: `ls /usr/share/vulkan/icd.d/nvidia_icd.json`.
- Vulkan loader and tools: `sudo apt install libvulkan1 vulkan-tools`. `vulkaninfo --summary` must list the RTX 2060
  as `PHYSICAL_DEVICE_TYPE_DISCRETE_GPU`.
- Rust through rustup (https://rustup.rs). `rust-toolchain.toml` pins **1.92.0**, which rustup installs on first use.
  Also a C toolchain: `sudo apt install build-essential`.
- Deno **2.7.5**. `./setup.sh` downloads it into `.tools/deno/`, so a system Deno is not needed. WebGPU is enabled by
  default in this version; no `--unstable-webgpu` flag is needed. On Linux, Deno 2.7.5's WebGPU is wgpu 28 on
  Vulkan, and `run-all.sh` sets `DENO_WEBGPU_BACKEND=vulkan`.
- `curl`, `unzip`, `git`. About 16 GB RAM (the largest shape makes 2.5 GiB of cells and matrices on the host), 3 GB
  of disk for the Rust build.
- A machine that is otherwise idle. Close other GPU programs: `nvidia-smi` should show no other compute processes,
  and little memory in use by a desktop.

## Setup and run

```sh
git clone git@github.com:midnight-experiments/ipir-benchmark-nvidia.git
cd ipir-benchmark-nvidia
./setup.sh            # checks the driver and Vulkan, installs Deno 2.7.5 into .tools/, fetches and builds the Rust side
./run-all.sh --quick  # a few minutes: small shapes, few blocks; checks everything works
./run-all.sh          # the full benchmark, about 45 to 60 minutes
```

`setup.sh` needs network access once: it downloads Deno and runs `cargo fetch --locked`. After that, builds are
`--offline --locked`. To prepare an air-gapped host, run `cargo vendor` in `ipir/` on a connected machine and copy
the result over.

`run-all.sh` writes `results/<host>-<date>.json` (all data, including the machine description) and
`results/<host>-<date>.md` (a short summary, also printed at the end). It also writes raw files and a log in
`results/raw/<host>-<date>/`, which git ignores. Options: `--quick`, `--skip-build`, `--skip-compile`,
`--skip-fetch`, `--skip-budget`, `--allow-non-nvidia`, `--name NAME`.

How long it takes, in the full run: building about 2 min the first time. Step 2 is about 1 to 2 min. Step 1 is 5 to
15 min, mostly making the synthetic tables on the host. Step 3 is about 40 min (2 shape sets × 3 rates × 2 modes × 30
blocks × 6 s, plus baselines).

The results record: GPU name, driver, VRAM and VRAM in use (`nvidia-smi`); the Vulkan devices (`vulkaninfo
--summary`); the WebGPU adapter Deno picked, and whether it has timestamp queries; CPU model, cores and threads; RAM;
OS and kernel; Deno and Rust versions; the load average at the start and the end.

## Sending the results back

Either:

- commit `results/<host>-<date>.json` and `.md` to a branch and push it:
  `git checkout -b results/<host> && git add results/*.json results/*.md && git commit -m "Results from <host>" && git push -u origin HEAD`;
- or paste the `.md` summary into the conversation. Attach the `.json` if you can; it holds the per-call and
  per-block data.

## Troubleshooting

- **"no WebGPU adapter found"**: Deno found no Vulkan device. Run `vulkaninfo --summary`. If it fails or lists no
  NVIDIA GPU, the Vulkan ICD is missing: install `libnvidia-gl-<driver branch>`, or reinstall the driver with its
  Vulkan part. On a headless machine, Vulkan compute works without a display, so no X server is needed. Make sure
  `/dev/nvidia*` exist (`nvidia-smi` works) and that no `VK_ICD_FILENAMES` / `VK_DRIVER_FILES` points elsewhere.
- **"the WebGPU adapter is a software rasterizer"** (llvmpipe, lavapipe, SwiftShader): `run-all.sh` stops, because
  CPU-emulated Vulkan numbers mean nothing. The NVIDIA Vulkan driver is not loaded: same fix as above. Check that
  `vulkaninfo --summary` shows `driverName = NVIDIA`.
- **"WebGPU picked another adapter"**: on a machine with integrated graphics too, Deno picked the wrong GPU. Try
  `DENO_WEBGPU_BACKEND=vulkan ./run-all.sh` (the default on Linux). Disable the integrated GPU's Vulkan ICD for the
  run (`VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json`), or pass `--allow-non-nvidia` if that is really
  intended.
- **"not enough memory left" for every shape, even small ones**: on an NVIDIA GPU without resizable BAR, wgpu's
  memory-budget check counts the BAR heap, whose usage the driver sums over all processes, the desktop included. Load
  the Vulkan layer in `tools/hide-memory-budget-layer/`. Its README shows how to check whether your host needs it.
- **Shapes that do not fit in memory**: expected for the largest shapes on 6 GB. A shape whose resident bytes
  (2 bytes a cell plus 32 MiB a column block) exceed the VRAM is skipped without trying. Any other shape is tried, and
  an out-of-memory error, a lost device or a 15-minute timeout is recorded as a skip with its reason. The "largest"
  candidates are tried from 32768×32768 downwards until one runs. If the 10-million-output block-budget set (about
  4 GiB resident) does not fit, its section says so. Free VRAM by closing the desktop session or other GPU programs.
- **Timestamp queries missing** (exec times "—"): the adapter lacks `timestamp-query`. Wall times are still valid.
- **Build errors**: `rustup show` must report 1.92.0 in this folder, and `cc` must exist (`build-essential`). If
  `cargo fetch` fails behind a proxy, set `HTTPS_PROXY`.
- **Very slow or noisy numbers**: check `nvidia-smi -q -d PERFORMANCE` for thermal or power throttling. Set the CPU
  governor to `performance` (`sudo cpupower frequency-set -g performance`). Make sure nothing else runs. The results
  record the load average and governor.

## What was checked, and where

Developed and run on macOS (Apple M4 Max, Metal, Deno 2.7.5, Rust 1.92.0). There, the Rust programs build and run,
patch 0001's test passes, every Deno script type-checks and runs, `./run-all.sh --quick` completes, the 10-million
block-budget set runs, and the 32768×32768 shape and the skip logic work.

It has also run on Linux, on one desktop with an RTX 2060 (6 GB, driver 580.159.03, no resizable BAR) and a Ryzen 7
2700X. `results/pop-os-notes.md` has the details. There:

- `setup.sh`'s Deno download and `cargo fetch` work.
- Deno's WebGPU on Vulkan picks the NVIDIA GPU, not llvmpipe, and has timestamp queries. The adapter reports a
  `maxBufferSize` of 2^52, and the 2^30-byte chunks work.
- wgpu's memory-budget check refuses every buffer unless the Vulkan layer in `tools/hide-memory-budget-layer/` is
  loaded. With the layer, every shape up to 32768×32768 and the 10-million set fit in 6 GB, so how out-of-memory shows
  up on a 6 GB card is still not verified.
- The low-priority pool runs each thread at nice 19. On macOS the pool uses the utility QoS class instead.
- `bench/sysinfo.ts` records the CPU model, the RAM and the CPU governor as unknown, because Deno refuses reads of
  `/proc` and `/sys` without `--allow-all`.

## Layout and sources

| Path | What | From |
|---|---|---|
| `gpu/` | `GpuIpirTable` and its WGSL kernels, unchanged | UmbraDB, `examples/private-shielded-sync/gpu/` at `c620f0e7` |
| `ipir/sizes.ts` | IPIR+SP byte lengths, unchanged | UmbraDB, `examples/private-shielded-sync/ipir/` at `c620f0e7` |
| `ipir/vendor/` | Five IPIR+SP crates (`inspiring`, `reinspiring`, `ipir-native-client`, `ipir-sp`, `simplepir-kernel`) | UmbraDB at `4f362386`, copied from `valargroup/ipir-sp` at `791b528` |
| `ipir/patches/` | Patches applied to `ipir/vendor/`: this design's 0001, and 0002 (a faster CPU block compile with the same output words) | 0001: UmbraDB research, spec 00065; 0002: new |
| `ipir/bench/` | The compile benchmark programs | new, adapted from spec 00065's compile checks |
| `bench/` | The Deno benchmark scripts | new, adapted from spec 00065's GPU answer-time harness |
| `tools/hide-memory-budget-layer/` | A Vulkan layer that hides `VK_EXT_memory_budget` from wgpu, for NVIDIA GPUs without resizable BAR | new |

UmbraDB is https://github.com/midnightntwrk/umbradb (Apache-2.0); its notice is in `NOTICE`. The IPIR+SP crates are
MIT OR Apache-2.0 (`ipir/vendor/ORIGIN.md`, `ipir/vendor/LICENSE-MIT`, `ipir/vendor/LICENSE-APACHE`). This repository
is Apache-2.0 (`LICENSE`).
