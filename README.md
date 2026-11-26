# ipir-benchmark-nvidia

This repository measures how fast a server does the four operations of an IPIR+SP private information retrieval
table on a GPU: build the table, update it when a chain block arrives, answer one request, and answer a wallet's call
of 32 requests.

It exists for the design of UmbraDB's private sync modules (spec 00065). There, every table follows the chain block
by block. A block arrives every 6 s, and its table work takes time away from answering wallets: the goal is a block's
work under 1 s, and 3 s is the ceiling.

1. [Run instructions](#run-instructions)
2. [What we are testing today](#what-we-are-testing-today)
3. [Quick results summary](#quick-results-summary)
4. [Repo layout and sources](#repo-layout-and-sources)
5. [Troubleshooting](#troubleshooting)
6. [Licence](#licence)

## Run instructions

Run it on the machine to measure: Linux with an NVIDIA GPU and its driver, or a Mac with Apple silicon.
[docs/hosts.md](docs/hosts.md) lists what the machine needs.

### The standard run: the report

About 30 minutes. It measures the combinations of the next section and writes
`results/<short name>-<date>-report.json`.

```sh
git clone https://github.com/midnight-experiments/ipir-benchmark-nvidia.git
cd ipir-benchmark-nvidia
./bench.sh --install-deps --report --cuda --host <short name> --label "<GPU> + <CPU>"   # Linux, NVIDIA
./bench.sh --install-deps --report --host <short name> --label "<Mac model>"            # macOS
```

- `--install-deps` installs what is missing: the apt packages `build-essential curl unzip git patch ca-certificates
  libvulkan1 vulkan-tools` (as root or with `sudo`) and Rust with rustup into `$HOME`. Without it, `bench.sh` prints
  the command and stops. It never installs a GPU driver.
- `--cuda` adds the CUDA columns. It needs the CUDA driver library and NVRTC ([docs/hosts.md](docs/hosts.md)).
- On macOS the run also builds and measures the native Metal kernels in `metal/`.
- On an NVIDIA GPU without resizable BAR, `bench.sh` builds and loads the Vulkan layer of
  `tools/hide-memory-budget-layer/` by itself.
- `--quick` with `--report` is a short check of a few minutes; its numbers are not results.

Then write the tables from the machines' result files, here the three files on `main`:

```sh
.tools/deno/deno run --allow-read --allow-write bench/summary.ts --out REPORT.md --readme README.md --cpu M4 \
  "results/m4-max-2026-11-26-report.json=M4" \
  "results/pop-os-2026-11-26-report.json=RTX 20" \
  "results/rtx4090-2026-11-26-report.json=RTX 40"
```

This writes [`REPORT.md`](REPORT.md) and the table under [Quick results summary](#quick-results-summary). No figure
in either is typed by hand.

### The full run

About 100 minutes on a GPU. It measures more table shapes, the CPU compile with and without patch 0002, and a block's
work at three call rates ([docs/full-run.md](docs/full-run.md)). It takes over an hour, so start it inside `tmux` or
`screen`, or with `nohup ./bench.sh --install-deps > bench.log 2>&1 &`.

| Command | What it does |
|---|---|
| `./bench.sh --quick` | Setup and the quick check only, a few minutes. Its numbers are not results |
| `./bench.sh` | Setup, the quick check, the full run on this host's GPU, the overview page's file, the bundle |
| `./bench.sh --cpu` | The same, then the full run again with the answers on the CPU (about 1.5 hours more) |
| `./bench.sh --cpu-only` | The CPU backend alone, also on a host with no GPU |
| `./bench.sh --cuda` | Also upstream's CUDA kernel against the WebGPU path at five shapes, and upstream's own first-pass benchmark |

Other options:

- `--skip-quick` goes straight to the full run.
- `--budget-sets today` keeps the 10-million-output tables (about 4 GiB of GPU memory) out of the block budget.
- `--host NAME` sets the host's name in file names and results (rented machines have random names).
- `--label TEXT` sets the machine's name in the results and on the overview page, for example `--label "RTX 4090 + EPYC 7B13"`.

What a full run leaves, for a host named `box` on an NVIDIA GPU with resizable BAR:

| File | What |
|---|---|
| `results/box-<date>-nvidia.json` | All data, including the machine description and the backend |
| `results/box-<date>-nvidia.md` | A short summary, also printed at the end |
| `site/box-<date>-nvidia-site.json` | The file the overview page reads |
| `results/box-<date>-nvidia-quick.json`, `.md` | The quick check |
| `results/box-<date>-bundle.tar.gz` | All of the above and the run's log, in one file (git ignores it) |

### Sending results back

Either push them from the machine:

```sh
git checkout -b results/<short name> && git add results site && git commit -s -m "Results from <short name>" && git push -u origin HEAD
```

or copy the result files off it (`scp`; a full run packs them into the bundle) and put them at the top of a clone.

## What we are testing today

A combination is an operation, at a size, on a machine, with a code path.

**Operations**

| Operation | What is measured | When a server does it |
|---|---|---|
| Build a whole matrix from scratch | Every column block's hint columns and packing matrix computed from the cells | At server start, when a matrix changes its number of rows, when a matrix is added |
| Update, 1× | One column block of the matrix computed again | When data is added to that block |
| Update, 6× | One chain block: a column block of each of the six matrices, while the GPU answers 2 calls a second | Every 6 s |
| Fetch 1 | One request answered, from its bytes to the answer's bytes | |
| Fetch 32 | The call every wallet sends: 32 requests, padded so that the server cannot tell how many it needed | On every wallet call |

**Sizes.** A value is one shielded output's read record, 98 bytes. A matrix is the table of all values; a column
block is 2,048 of its columns.

| Values | Matrix (rows × columns) | Column blocks | On the GPU, with the packing matrices |
|---|---|---|---|
| 1M | 4,096 × 12,288 | 6 | 288 MiB |
| 10M | 14,336 × 34,816 | 17 | 1.5 GiB |

**Machines**

| Name | GPU | CPU | Conditions |
|---|---|---|---|
| M4 | Apple M4 Max, 40-core GPU | Apple M4 Max, 16 cores | A laptop in use |
| RTX 20 | GeForce RTX 2060, 6 GB, no resizable BAR | Ryzen 7 2700X | A desktop in use; needs the Vulkan layer |
| RTX 40 | GeForce RTX 4090, 24 GB, resizable BAR on | Ryzen 7 7700X | A shared host: another user's process holds about 21 of the card's 24 GiB |

**Code paths, and what each has code for**

| Path | Machines | Build from scratch | Update | Fetch |
|---|---|---|---|---|
| CPU: the IPIR+SP crates with patch 0002 | M4, as the reference | yes | yes | yes |
| WebGPU: Deno 2.7.5, on Metal on the Mac and on Vulkan on NVIDIA | M4, RTX 20, RTX 40 | yes | yes | yes, both steps on the GPU |
| Native Metal: `metal/` | M4 | computing only | computing only | no code |
| CUDA | RTX 20, RTX 40 | computing only | computing only | upstream's kernel: the first pass on the GPU, one request at a time, and the packing on the host's CPU |

- **Computing only**: the blocks are computed and compared with the CPU's, but the path has no fetch of its own, so
  nothing is installed into a table that answers and no calls are answered meanwhile.
- The CUDA build and update are this repository's port of the Metal kernels. Upstream's CUDA code has no build.
- Every GPU result is compared with the CPU's: answers byte for byte, built blocks by a SHA-256 digest over every
  output word.

That is 7 columns (one CPU, one Metal, three WebGPU, two CUDA) by 5 operations by 2 sizes: 70 cells.

The full run tests other combinations: 16 table shapes from 2048 × 2048 to 32768 × 32768, the six tables at the
chain's current size, the CPU compile with the stock crates and with patch 0002, a block's work at 2, 5 and 10 calls
a second, and a CPU backend ([docs/full-run.md](docs/full-run.md)).

## Quick results summary

<!-- report-summary:start -->
| Operation | Values | CPU M4 | Metal M4 | WebGPU M4 | WebGPU RTX 20 | WebGPU RTX 40 | CUDA RTX 20 | CUDA RTX 40 |
|---|---|---|---|---|---|---|---|---|
| Build from scratch | 1M | 622 ms | 78.1 ms\* | 305 ms | 307 ms | 153 ms | 116 ms\* | 55.7 ms\* |
| Build from scratch | 10M | 2.44 s | 250 ms\* | 966 ms | 1.34 s | not run (card in use) | 443 ms\* | not run (card in use) |
| Update, 1×: one column block | 1M | 101 ms | 28.4 ms | 61.5 ms | 59.3 ms | 47.4 ms | 41.8 ms | 22.5 ms |
| Update, 1×: one column block | 10M | 121 ms | 30.9 ms | 66.9 ms | 71.1 ms | 44.9 ms | 52.4 ms | not run (card in use) |
| Update, 6×: one chain block | 1M | 544 ms (9.1 %) | 53.1 ms (0.9 %)\* | 266 ms (4.4 %) | 252 ms (4.2 %) | not run (card in use) | 84.7 ms (1.4 %)\* | 34.5 ms (0.6 %)\* |
| Update, 6×: one chain block | 10M | 627 ms (10.4 %) | 60.7 ms (1.0 %)\* | 264 ms (4.4 %) | does not fit; 268 ms (4.5 %) at 8M | not run (card in use) | 106 ms (1.8 %)\* | not run (card in use) |
| Fetch 1 | 1M | 2.1 ms | no code | 15.2 ms | 13.4 ms | 12.3 ms | 7.8 ms | 6.5 ms |
| Fetch 1 | 10M | 11.0 ms | no code | 18.0 ms | 18.2 ms | 14.7 ms | 25.2 ms | 23.1 ms |
| Fetch 32 | 1M | 66.1 ms | no code | 29.6 ms | 30.1 ms | 15.6 ms | 257 ms | 217 ms |
| Fetch 32 | 10M | 363 ms | no code | 72.7 ms | 82.7 ms | 30.7 ms | 832 ms | 769 ms |

\* Computing only, with native kernels: nothing is installed into a table that answers, and no calls are answered meanwhile.

59 of the 70 cells hold a figure; the others: 6 `not run (card in use)`, 1 `does not fit`, 4 `no code`.

Medians of `./bench.sh --report`: M4 (Apple M4 Max) on 2026-11-26, RTX 20 (RTX 2060 + Ryzen 7 2700X) on 2026-11-26, RTX 40 (RTX 4090 + Ryzen 7 7700X) on 2026-11-26. The 6× rows give the share of the 6 s between two chain blocks in brackets.
<!-- report-summary:end -->

- **Update of a chain block (6×)**, the figure the 6 s depend on: every GPU path is under the 1 s goal. With WebGPU,
  while the GPU answers calls, it takes 4 to 5 % of the 6 s on the two GPUs where it ran. The native kernels compute
  the six blocks in under 2 % of the 6 s, computing only. The CPU takes 9 to 10 %.
- **Fetch 32**: WebGPU is the fastest path on every GPU. The CUDA path is 8 to 25 times slower on the same card,
  because its packing runs on the CPU.
- **Fetch 1**: the CPU is the fastest at 1M. WebGPU's time holds about 12 ms that do not depend on the call:
  submitting the work and reading the answer back in this harness.
- **One column block (1×)**: native Metal and native CUDA compute it in half to three quarters of the time WebGPU
  takes on the same GPU.
- **Not measured**: the RTX 40 cells marked `not run (card in use)` wait for the other user's process to leave the
  card. A 6 GB card cannot hold the six 10M matrices and build their blocks at once: the largest size at which it
  does both is 8M.
- **No code yet**: a native Metal fetch, and a CUDA fetch with the packing on the GPU. A real 6× update on those two
  paths needs them.

**Full standard results:** [`REPORT.md`](REPORT.md): each table with its definitions, the GPU memory every operation
needs, what fits on each GPU, and each run's conditions.

**Other results:**

- [docs/other-results.md](docs/other-results.md): the full runs. 16 table shapes, the six tables at the chain's
  current size, the CPU compile with and without patch 0002, the memory of a 6 GB card size by size, and upstream's
  CUDA kernel against WebGPU at five shapes.
- [`results/`](results/): every run's data (`.json`), its summary (`.md`), and the notes on each NVIDIA host
  ([`pop-os-notes.md`](results/pop-os-notes.md), [`rtx4090-notes.md`](results/rtx4090-notes.md)).
- [`site/`](site/): the full runs in the form the overview page reads (https://umbradb-private-sync.pages.dev/).
- [docs/verification.md](docs/verification.md): which parts ran on which machine, and what has not run.

## Repo layout and sources

| Path | What | From |
|---|---|---|
| `bench.sh` | The whole procedure in one command: tools, setup, the Vulkan layer when needed, the runs, the bundle | new |
| `setup.sh`, `run-all.sh` | The setup (GPU stack check, Deno into `.tools/`, the Rust builds) and the full run's steps | new |
| `REPORT.md` | The standard results | written by `bench/summary.ts` |
| `docs/` | Host requirements, the full run, the other results, what was checked | new |
| `bench/` | The Deno benchmark scripts. The report: `report-run.ts`, `summary.ts`, `table-build.ts`, `cuda-compare.ts`. The full run: `fetch-all.ts`, `gpu-compile.ts`, `budget.ts`, `check-run.ts`, `export-site.ts` | new, adapted from spec 00065's GPU answer-time harness |
| `gpu/ipir.ts`, `ipir-wgsl.ts`, `ipir-wire.ts`, `device.ts` | `GpuIpirTable`, the WebGPU answer, and its WGSL kernels. `ipir.ts` has `createEmpty` and `writeBlock` added, for a table built on the GPU; the others are unchanged | UmbraDB, `examples/private-shielded-sync/gpu/` at `c620f0e7` |
| `gpu/block-build.ts` | The build of a column block in WebGPU, and `pack_digest`'s digest of its output | new: a WGSL port of the Metal block build |
| `metal/` | The build of a column block with native Metal kernels (its own Rust crate, macOS only), and `metal_report`, the report's figures | the Metal block build of spec 00065's question 21; `metal_report` is new |
| `ipir/vendor/` | Five IPIR+SP crates (`inspiring`, `reinspiring`, `ipir-native-client`, `ipir-sp`, `simplepir-kernel`), with the four patches applied | UmbraDB at `4f362386`, copied from `valargroup/ipir-sp` at `791b528` |
| `ipir/patches/` | The patches to `ipir/vendor/`: 0001 (a table gains column blocks and keeps the ones it has), 0002 (a faster CPU block compile with the same output words), 0003 (read-only accessors for a GPU build), 0004 (a hook for timing upstream's CUDA kernel, and the CUDA release as a build choice) | 0001: UmbraDB research, spec 00065; the others new |
| `ipir/bench/src/bin/` | The CPU programs: `compile_block`, `block_worker`, `pack_digest`, `cpu_fetch`, `cpu_server`, `gpu_build_prep`; and with CUDA `cuda_fetch`, `cuda_build_digest`, `cuda_build_report` | new, adapted from spec 00065's compile checks |
| `ipir/bench/src/cuda_build.rs`, `cuda_build.cu` | The build of a column block in CUDA: the kernels, compiled at start with NVRTC, and their host side | new: a CUDA port of the kernels in `metal/` |
| `ipir/bench/src/gpu_prep.rs`, `rns32.rs` | The GPU builds' constant tables for a table shape, and the CPU's stage outputs to check them against | `rns32.rs` from the Metal block build; the rest new |
| `ipir/build-stock.sh` | Builds `ipir/bench/` without patches 0002 and 0003 | new |
| `ipir/sizes.ts` | IPIR+SP byte lengths, unchanged | UmbraDB, `examples/private-shielded-sync/ipir/` at `c620f0e7` |
| `tools/hide-memory-budget-layer/` | A Vulkan layer that hides `VK_EXT_memory_budget` from wgpu, for NVIDIA GPUs without resizable BAR | new |
| `tools/cuda-feature.sh` | Chooses the CUDA 12 or CUDA 13 bindings from the NVRTC library the host will load | new |
| `results/`, `site/` | The runs' data and notes, and the full runs' files for the overview page | written by the runs |

UmbraDB is https://github.com/midnightntwrk/umbradb. The IPIR+SP crates are the Zcash wallet services' repository,
https://github.com/valargroup/ipir-sp; `ipir/vendor/ORIGIN.md` has the commit and each crate's version.

## Troubleshooting

[docs/hosts.md](docs/hosts.md) has the host requirements, containers, and what CUDA needs.

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
  memory-budget check counts the BAR heap, whose usage the driver sums over all processes, the desktop included.
  `bench.sh` builds and loads the Vulkan layer in `tools/hide-memory-budget-layer/` when it sees this; `run-all.sh`
  alone does not. The layer's README shows how to check whether your host needs it and how to load it by hand.
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
- **`undefined symbol: nvrtcGetNVVM`** (with `--cuda`): the CUDA 12 bindings were built on a host whose NVRTC is
  CUDA 13's. `tools/cuda-feature.sh` chooses the bindings from the installed NVRTC; `IPIR_CUDA_FEATURE=cuda-13` (or
  `cuda-12`) chooses outright.
- **`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`** (with `--cuda`): the installed NVRTC is newer than the driver, which
  refuses what it compiles. `nvidia-smi` prints the CUDA release the driver supports. Put the NVRTC library files of
  that release, or an older one, in `.tools/nvrtc-<release>/` (for example `.tools/nvrtc-13.0/`): `bench.sh` puts
  that folder first in `LD_LIBRARY_PATH`.
- **`not run (card in use)` in the report**: other processes held more than a quarter of the card's memory when the
  run started. What runs out of memory then says nothing about the card, so it is not reported as `does not fit`,
  and the search for the largest size that fits is skipped. Run again when the card is free.
- **`does not fit` in the report's 6× update**: the six matrices and the block build do not fit in the GPU's memory
  together. The run then measures the largest smaller size that does (`bench/gpu-build-sets.ts`).

## Licence

This repository is licensed under Apache-2.0 ([`LICENSE`](LICENSE)).

- The code copied from UmbraDB is Apache-2.0; its notice is in [`NOTICE`](NOTICE).
- The IPIR+SP crates in `ipir/vendor/` are MIT OR Apache-2.0 (`ipir/vendor/ORIGIN.md`, `ipir/vendor/LICENSE-MIT`,
  `ipir/vendor/LICENSE-APACHE`).
