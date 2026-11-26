# ipir-benchmark-nvidia

This repository measures how fast a server answers IPIR+SP private information retrieval calls, and whether it can
also keep its tables up to date as the chain grows. It was written for one Linux machine with an NVIDIA GPU (a GeForce
RTX 2060 with 6 GB), and now measures four backends:

| Backend | Where the answers run | Host |
|---|---|---|
| `nvidia` | the GPU, WebGPU on Vulkan | Linux, NVIDIA GPU with resizable BAR |
| `nvidia-no-bar` | the GPU, WebGPU on Vulkan | Linux, NVIDIA GPU without resizable BAR (needs the Vulkan layer in `tools/hide-memory-budget-layer/`) |
| `metal` | the GPU, WebGPU on Metal | macOS, Apple silicon |
| `cpu` | the CPU, the IPIR+SP crates' server path | any of the above |

The GPU answer code is the private shielded sync proof of concept's WebGPU implementation of IPIR+SP's server answer
(`GpuIpirTable`, byte for byte the same answers as the native add-on), run under Deno, which uses Vulkan on Linux and
Metal on macOS. The CPU answer code is the add-on's: `NativeServer::respond` of the vendored IPIR+SP crates. The table
preparation code is the Zcash IPIR+SP crates, in Rust, on the CPU, for every backend.

It exists for the design of UmbraDB's private sync modules (spec 00065). There, every table follows the chain block by
block (one block every 6 s), and a block's table work takes time away from answering wallets: the goal is under 1 s
of each 6 s, and 3 s is the ceiling. The production target is NVIDIA through CUDA, with WebGPU for Apple's Metal and other GPUs, and a CPU
fallback. The numbers here say how long a call of 32 requests takes for each table shape on each backend, how long
it takes to compile one changed column block, on the CPU and on the GPU, and how much of the 6 s between two blocks a
block's work takes while the server is answering calls. What it leaves is the time the server has to answer. `bench/export-site.ts` turns a run into the file the overview page reads
(https://umbradb-private-sync.pages.dev/, the server timings per backend).

## Run it

On the machine to measure (Linux with an NVIDIA GPU and its driver, or a Mac with Apple silicon):

```sh
git clone https://github.com/midnight-experiments/ipir-benchmark-nvidia.git
cd ipir-benchmark-nvidia
./bench.sh --install-deps
```

`bench.sh` is the whole procedure in one command. It checks the tools and the GPU stack, builds, runs a quick check
of a few minutes, and stops there if anything failed. Then it runs the full benchmark on this host's GPU (about 100
minutes), writes the results and the overview page's file, and packs them into one file to copy off the machine. On
an NVIDIA GPU without resizable BAR it builds and loads the Vulkan layer by itself; with resizable BAR it does not
load it. A full run takes over an hour, so start it inside `tmux` or `screen`, or with
`nohup ./bench.sh --install-deps > bench.log 2>&1 &`.

| Command | What it does |
|---|---|
| `./bench.sh --quick` | Setup and the quick check only, a few minutes. Its numbers are not results |
| `./bench.sh` | Setup, the quick check, the full run on this host's GPU, the site file, the bundle |
| `./bench.sh --cpu` | The same, then the full run again with the answers on the CPU (about 1.5 hours more) |
| `./bench.sh --cpu-only` | The CPU backend alone, also on a host with no GPU |
| `./bench.sh --cuda` | Also upstream's CUDA kernel: a whole answer against the WebGPU path, in the full run's results, and upstream's own first-pass benchmark (NVIDIA; needs the CUDA driver library and NVRTC) |

Other options: `--install-deps` installs what is missing (the apt packages `build-essential curl unzip git patch
ca-certificates libvulkan1 vulkan-tools`, as root or with `sudo`, and Rust with rustup into `$HOME`); without it,
`bench.sh` prints the command and stops. It never installs a GPU driver. `--skip-quick` goes straight to the full
run. `--budget-sets today` keeps the 10-million-output tables (about 4 GiB of GPU memory) out of the block budget. `--host NAME` sets the host's name in file names and results (rented machines have random names). `--label TEXT`
sets the machine's name on the overview page, for example `--label "RTX 4090 + EPYC 7B13"`.

What a run leaves, for a host named `box` on an NVIDIA GPU with resizable BAR:

| File | What |
|---|---|
| `results/box-<date>-nvidia.json` | All data, including the machine description and the backend |
| `results/box-<date>-nvidia.md` | A short summary, also printed at the end |
| `site/box-<date>-nvidia-site.json` | The file the overview page reads |
| `results/box-<date>-nvidia-quick.json`, `.md` | The quick check |
| `results/box-<date>-bundle.tar.gz` | All of the above and the run's log, in one file (git ignores it) |

To send the results back, either push them from the machine
(`git checkout -b results/box && git add results site && git commit -s -m "Results from box" && git push -u origin HEAD`),
or copy the bundle off it (`scp`) and unpack it at the top of a clone. The `.md` summary alone is enough to read the
numbers; the `.json` holds the per-call and per-block data.

## Results so far

Three machines. Each figure comes from the results file named in the table; `site/` has the same figures in the form
the overview page reads.

| Machine | Backend | GPU | CPU | Results |
|---|---|---|---|---|
| RTX 2060 desktop | `nvidia-no-bar` | GeForce RTX 2060, 6 GB, no resizable BAR (Vulkan layer loaded) | Ryzen 7 2700X, 8 cores / 16 threads | `results/pop-os-2026-11-25-nvidia-no-bar.md`: a full run, on a desktop in use (load average about 3; `results/pop-os-notes.md`) |
| M4 Max | `metal` | Apple M4 Max, 40-core GPU | Apple M4 Max, 16 cores | `results/m4-max-2026-11-25-metal.md`; the GPU build of the blocks (steps 2b and 3) was measured on 2026-11-26 into the same results, with the machine in use (load average 7 at the start) |
| RTX 4090 host | `nvidia` | GeForce RTX 4090, 24 GB, resizable BAR on | Ryzen 7 7700X, 8 cores / 16 threads | `results/rtx4090-2026-11-25-nvidia.md`: a full run on a quiet machine, but with 3 GiB of the card free, so without the 10-million-output tables (`results/rtx4090-notes.md`) |

**Answering a call of 32 requests** (step 1): wall time, and in brackets the GPU's own time, in ms.

| rows × cols | RTX 2060 | M4 Max | RTX 4090 |
|---|---|---|---|
| 2048 × 2048 | 15.5 (1.6) | 16.0 (1.5) | 13.1 (0.3) |
| 6144 × 28672 | 56.6 (34.4) | 45.8 (27.5) | 20.5 (5.7) |
| 14336 × 34816 | 88.5 (64.0) | 76.1 (53.9) | 29.2 (13.4) |
| 32768 × 32768 | 144.6 (109.7) | 123.2 (94.5) | did not fit: 3 GiB of the card were free |

**Compiling one column block** (steps 2 and 2b): wall time in ms.

| | RTX 2060 host | M4 Max | RTX 4090 host |
|---|---|---|---|
| On the CPU, stock crates, 2,048 rows | 1,174 | 337 | 586 |
| On the CPU, patch 0002, 2,048 rows | 397 | 94 | 221 |
| On the CPU, patch 0002, 32,768 rows | 806 | 189 | 478 |
| On the GPU, 2,048 rows | 57 | 60 | 41 |
| On the GPU, 32,768 rows | 96 | 84 | 50 |

The GPU figures are the wall time from the block's cells in host memory to the end of the build. On the RTX 4090, of
the 50 ms at 32,768 rows, 29 ms are the GPU computing (the two chains 25.3 ms at every table height, the hints 0.4
to 2.9 ms, the aggregation 0.5 ms, the matrix compilation 0.3 ms), about 10 ms are writing the 128 MiB of cells to
the GPU, and about 12 ms are this harness's fixed cost of submitting work and noticing its end (the same gap as in a
call's answer). On the RTX 2060 the GPU computes for 44 to 60 ms: the chains 35.7 ms, the hints 2.1 to 17.8 ms, the
aggregation 3.2 ms, the matrix compilation 2.7 ms. On the M4 Max only wall times are given: Metal returned
the GPU's timestamps for few of the builds, and its per-stage times are not consistent.
The GPU build's output equals the CPU's in 24 of 24 checked blocks on each of the three GPUs.

**One chain block: one column block of each of six tables** (steps 2b and 3), at today's shapes and at the
10-million-output shapes. A block arrives every 6 s, and its work is time the server does not have for answering at
full speed. The goal is a block's work under 1 s; 3 s is the ceiling, the most that is tolerable.

The compile of the six column blocks alone, in seconds:

| | RTX 2060 host | M4 Max | RTX 4090 host |
|---|---|---|---|
| On the CPU, stock crates (one after another, p50) | 6.80 / 8.08 | 2.05 / 2.25 | 3.45 / not run |
| On the CPU, patch 0002 (one after another, p50) | 2.33 / 2.79 | 0.63 / 0.68 | 1.35 / not run |
| On the GPU, one build right after another (wall; GPU time in brackets) | 0.11 (0.087) / 0.15 (0.104) | 0.13 / 0.17 (wall only) | 0.05 (0.033) / 0.07 (0.036) |
| On the GPU, 6 s after the last build (wall; GPU time in brackets) | 0.39 (0.34) / 0.42 (0.32) | 0.17 / 0.20 (wall only) | 0.25 (0.21) / 0.32 (0.22) |

A block's whole work (the compile of the six column blocks, then their upload and layout for the answer) while the
GPU answers 2 calls a second: seconds at p50, and in brackets its share of the 6 s between two blocks.

| Block work | RTX 2060 host | M4 Max | RTX 4090 host |
|---|---|---|---|
| CPU compile, stock crates, one after another | 7.02 (117 %) / 8.32 (139 %) | 2.18 (36 %) / 2.39 (40 %) | 3.67 (61 %) / not run |
| CPU compile, stock crates, low-priority pool | 3.80 (63 %) / 5.01 (84 %) | 1.13 (19 %) / 1.64 (27 %) | 2.12 (35 %) / not run |
| CPU compile, patch 0002, one after another | 2.51 (42 %) / 3.15 (52 %) | 0.76 (13 %) / 0.83 (14 %) | 1.61 (27 %) / not run |
| CPU compile, patch 0002, low-priority pool | 1.94 (32 %) / 2.18 (36 %) | 0.54 (9 %) / 0.68 (11 %) | 1.43 (24 %) / not run |
| Blocks built on the GPU that is answering | 0.28 (5 %) / does not fit | 0.26 (4 %) / 0.27 (5 %) | 0.36 (6 %) / not run |
| The same at 8 million outputs, the largest tables the RTX 2060 keeps while it builds | 0.26 (4 %) | | |

- **Under the 1 s goal:** all three GPUs with the blocks built on the GPU (5 % of the 6 s on the RTX 2060, 4 to 5 %
  on the M4 Max, 6 % on the RTX 4090; no block over 1 s), and the M4 Max's CPU with patch 0002 (9 to 14 %; 0 to 4 of
  30 blocks over 1 s). On the M4 Max the GPU build takes the work from 0.54 to 0.26 s at today's shapes and from 0.68
  to 0.27 s at the 10-million-output shapes; there the share rises with the call rate, to 6.0 % at 10 calls/s. **Not
  under it:** every CPU compile on the RTX 2060 host and on the RTX 4090 host (24 % of the 6 s at best), and the M4
  Max with the stock crates.
- **Almost all of it is the compile of the column blocks.** The upload and layout are 2 to 4 % of the 6 s on every
  machine (0.12 to 0.26 s); with the blocks built on the GPU they are 1.7 % on the RTX 4090 (0.10 s), 1.5 % on the
  M4 Max (0.09 s) and 2.4 % on the RTX 2060 (0.14 s), since the matrix does not cross from the host.
- With the stock crates the RTX 2060 host's work does not fit in the 6 s at all when the blocks are compiled one
  after another (117 and 139 %). With patch 0002 it is over the 3 s ceiling in 28 of 30 blocks at the
  10-million-output shapes when compiled one after another, and in none on the pool.
- **A GPU that has been idle builds more slowly.** On the RTX 4090 the same six-table build takes 33 ms of GPU time
  right after another one and 210 ms when the GPU was idle for the 6 s between two chain blocks: the card lowers its
  clocks when idle. A small dispatch every 20 ms during the idle time brings it to 114 ms, not to 33 ms. In the block
  budget, where blocks are 6 s apart, the build takes 0.23 to 0.26 s for that reason; without it the block's work
  would be about 0.15 s, 2.5 % of the 6 s. The driver's power setting (maximum performance) is the likely remedy; it
  was not tried, as it needs rights this run did not have. The M4 Max slows less (0.13 to 0.17 s of wall time at today's shapes). The RTX 2060 shows the same as the RTX 4090 when it is measured alone
  (87 ms of GPU time right after another build, 341 ms after 6 s of idle), but not in its block budget, where the
  build took 0.13 s: there the card was also answering calls that take it tens of milliseconds each, and driving a
  desktop.
- While the GPU builds a block, the calls that arrive wait for it: at 2 calls/s the calls that overlap a block's
  work are answered in 252 ms (p50) on the RTX 4090, against 18 ms otherwise, and in 133 ms on the RTX 2060, against
  22 ms; none is dropped.
- The RTX 4090 run has no 10-million-output budget: another process held 20.7 of the card's 24.5 GiB. On the RTX
  2060 the blocks cannot be built on the GPU with the 10-million-output tables resident: 6 GB are not enough. The
  run then looks for the largest tables that fit with the build: 8 million shielded outputs, where a block's work
  is 0.26 s at 2 calls a second and 0.31 s at 10 (4.4 to 5.1 % of the 6 s), with no block of 90 over 1 s (see
  Memory).

**Memory.** On the GPU, a table keeps 2 bytes a cell and 32 MiB a column block (its compiled matrix). The six
tables of the block budget:

| GPU memory | Today's shapes (21 column blocks) | 10-million-output shapes (53 column blocks) |
|---|---|---|
| Resident tables: cells | 392 MiB | 2,208 MiB |
| Resident tables: compiled matrices | 672 MiB | 1,696 MiB |
| Resident tables, total | 1,064 MiB | 3,904 MiB |
| Building one chain block on the GPU: working memory | 898 MiB | 898 MiB |
| Building one chain block on the GPU: the six blocks' cells | 64 MiB | 192 MiB |

- The GPU build's working memory is 128 MiB a table built at once (its hint masks, slots, digits and matrix, 32 MiB
  each) and 128 MiB shared: 256 MiB for one table, 641 MiB for four, 898 MiB for six. The block budget runs it with
  one hint-mask buffer for all six tables, 774 MiB.
- Answering a call needs scratch memory for the time of the call: 13 MiB for a call of 32 at 2048 × 2048, 86 MiB at
  14336 × 34816, 167 MiB at 32768 × 32768 (under 11 MiB for a call of 1).
- **The largest size at which the 6 GB RTX 2060 builds its blocks on the GPU is 8 million shielded outputs.** When
  the 10-million-output tables do not fit with the build, the run goes down in steps of half a million outputs to
  the first size that runs 30 blocks at 2, 5 and 10 calls a second (`bench/gpu-build-sets.ts`; the three Dust tables
  stay the same). On that card, with `nvidia-smi` read every second and the desktop's 470 to 540 MiB counted in:

  | Shielded outputs | Resident tables | With the blocks built on the GPU |
  |---|---|---|
  | 10 million | 3,904 MiB | out of memory in the first blocks at 2 calls/s |
  | 9.5 million | 4,040 MiB | out of memory in the first blocks at 2 calls/s |
  | 9 million | 4,104 MiB | 30 blocks at 2 calls/s, with 5,688 MiB of the card in use; out of memory at 5 calls/s |
  | 8.5 million | 3,960 MiB | 30 blocks at 2 calls/s, with 5,542 MiB in use; out of memory at 5 calls/s |
  | 8 million | 3,656 MiB | fits at all three rates, 5,494 MiB in use at most; block work 0.26, 0.30 and 0.31 s (4.4, 5.1 and 5.1 % of the 6 s) |
  | 7 million (a trial of 5 blocks) | 3,416 MiB | fits, 5,074 MiB in use at most; block work 0.28 s (4.7 %) |

  No sample is above 5,690 MiB of the card's 6,144: beyond that a buffer is refused. Beside the tables the build
  holds about 0.9 GiB, and every call being answered holds scratch memory, so more calls a second need more (the
  reason 8.5 and 9 million run at 2 calls a second and not at 5). The shape rule picks each table's shape on its
  own, so memory does not fall at every step: the 10-million-output tables are smaller than the 9-million ones, but
  their column blocks are larger. Building the six tables three at a time would need about 0.3 GiB less; that is not
  built.

On the host, the CPU compile's peak memory (the whole process, with the block's cells; `/usr/bin/time -l
compile_block` on the M4 Max):

| CPU compile, peak memory | Patch 0002 | Stock crates |
|---|---|---|
| One block, 2,048 rows | 252 MiB | 326 MiB |
| One block, 32,768 rows | 384 MiB | 448 MiB |
| Four blocks at once, 2,048 rows | 662 MiB | 1,042 MiB |
| Four blocks at once, 32,768 rows | 1,172 MiB | 1,305 MiB |

Not measured: the memory of a whole server process, and of the CPU backend's resident tables.

**Upstream's CUDA kernel against the WebGPU path** (`--cuda`, step 4 of the report). Upstream's CUDA kernel
(`simplepir-kernel`, `CudaKernel`) does one part of an answer on the GPU: the first pass, the table times one
request. It takes one request per call, behind a lock, has no packing, and takes a table whole. So an answer with it
is the first pass on the GPU, 32 times for a call of 32, and the packing of every column block on the host's CPU. The
WebGPU path does both steps on the GPU, the 32 requests in one submission. Both were timed on the same table and the
same 32 requests, and every answer of both equals the CPU's, byte for byte. Times in ms, medians of 9 calls of 32.

RTX 4090 host (packing on the Ryzen 7 7700X; 3 GiB of the card were free):

| rows × cols | CUDA first pass, device | Packing on the CPU | CUDA path, whole answer | WebGPU, device | WebGPU, whole answer | WebGPU is faster by |
|---|---|---|---|---|---|---|
| 2048 × 2048 (today's read table) | 0.4 | 4.7 | 10.5 | 0.3 | 12.7 | the CUDA path is 2 ms faster |
| 12288 × 16384 (Dust commitment table) | 13.9 | 271 | 303 | 4.9 | 19.0 | 16 times |
| 6144 × 28672 (the same table before the cap of 8 column blocks) | 12.1 | 497 | 527 | 5.7 | 20.2 | 26 times |
| 14336 × 34816 (read table at 10 million outputs) | 34.0 | 611 | 669 | 11.3 | 27.4 | 24 times |
| 32768 × 32768 | 72.6 | 589 | 699 | did not fit in the 3 GiB free | | |

RTX 2060 desktop (packing on the Ryzen 7 2700X; no resizable BAR):

| rows × cols | CUDA first pass, device | Packing on the CPU | CUDA path, whole answer | WebGPU, device | WebGPU, whole answer | WebGPU is faster by |
|---|---|---|---|---|---|---|
| 2048 × 2048 | 2.9 | 39.8 | 52.8 | 2.2 | 15.8 | 3.3 times |
| 12288 × 16384 | 74.0 | 283 | 384 | 30.1 | 46.7 | 8.2 times |
| 6144 × 28672 | 70.1 | 489 | 585 | 32.9 | 54.0 | 10.8 times |
| 14336 × 34816 | 198 | 594 | 828 | 63.6 | 83.0 | 10.0 times |
| 32768 × 32768 | 368 | 563 | 986 | 110 | 147 | 6.7 times |

"Whole answer" is wall time from the call to the answer's bytes; for the CUDA path it also holds the reading of the
32 requests, the copies to and from the device and the writing of the answers, so it is more than the two columns
before it.

- **On the RTX 4090 the WebGPU path answers a call of 32 16 to 26 times faster than the CUDA path** at every table
  of more than one column block; at the one-block table the two are level (10.5 against 12.7 ms). **On the RTX 2060
  it is 3 to 11 times faster at every shape.**
- **The packing on the CPU is most of the CUDA path**: 34 to 37 ms a column block for the 32 requests on both hosts,
  0.3 to 0.6 s for the tables of more than one block. The CPU alone answers those calls in 0.7 to 3.3 s.
- **The first pass alone is slower too.** The device's time for the 32 first passes of the CUDA kernel is 1.2 to
  3.3 times the device's time for both steps on the WebGPU path: the kernel reads the table once per request, and
  the WebGPU kernels read it once for the 32.
- A call of 1 request goes the other way at small tables: 0.3 to 22 ms on the CUDA path against 12 to 14 ms on WebGPU
  on the RTX 4090, and 1.5 to 28 ms against 13 to 22 ms on the RTX 2060 (the CUDA path is the faster one up to the
  Dust commitment table, WebGPU from the 10-million read table on). The WebGPU wall time has about 12 ms that do
  not depend on the call (submitting the work and reading the answer back in this harness); its device time for the
  call is 0.2 to 9 ms. A wallet always sends 32.
- **A changed column block**: the CUDA kernel has no call for one block, so the table goes up again: 36 to 48 ms for
  336 to 384 MiB of cells, 99 to 119 ms for 952 MiB, 212 to 269 ms for 2 GiB. WebGPU writes the one block, its cells and
  its packing matrix, in 20 to 36 ms whatever the table's width (55 ms at 32,768 rows on the RTX 2060).
- Upstream's own `cuda_bench` (one request's first pass with a 56-bit prime modulus, 100 requests) gives the same
  first-pass time per request as the comparison: 1.06 ms at 14336 × 34816 and 2.26 ms at 32768 × 32768 on the RTX
  4090, 5.2 and 10.3 ms on the RTX 2060 (`results/rtx4090-2026-11-26-cuda-bench.txt`,
  `results/pop-os-2026-11-26-cuda-bench.txt`).

What it took, and what could not be run:

- Upstream wires the kernel into another server type (`YServer`), not into `NativeServer::respond`, which the design
  uses. The comparison joins the two with patch 0004 (`ipir/patches/README.md`): `respond` with the first pass
  handed to the CUDA kernel.
- Upstream fixes the bindings of CUDA 12.0, which do not load with the NVRTC of CUDA 13 that both hosts have. Patch
  0004 makes the release a build choice (`cuda-12`, `cuda-13`).
- The RTX 2060 desktop has the NVRTC of CUDA 13.4 and a driver for CUDA 13.0, which refuses what that NVRTC
  compiles. Its run used the two NVRTC 13.0 library files of the RTX 4090 host, copied into `.tools/nvrtc-13.0/`
  and put first in `LD_LIBRARY_PATH`.
- 32768 × 32768 on WebGPU on the RTX 4090: its 2.6 GiB did not fit beside another user's 20.7 GiB.
- The RTX 4090 figures are from a run while that shared host was quiet. A repeat while other users were compiling
  (load average 8 to 16) gave the same device times and a slower packing (832 ms against 611 ms at 14336 × 34816);
  `cuda_bench` ran during that load, so its CPU column there is not a quiet figure.

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

## Host requirements

On macOS (Apple silicon), only Rust and `curl`/`unzip` are needed: `setup.sh` installs Deno for darwin-aarch64, and
Deno's WebGPU uses Metal. There is no `nvidia-smi` or `vulkaninfo` to check there. The rest of this section is for
Linux with an NVIDIA GPU.

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

## The `nvidia` file: a rented Linux machine with resizable BAR

Rent a VM with one recent NVIDIA GPU passed through whole (an L4, L40S, A10, A100, H100 or an RTX 4090/5090 class
card; not a vGPU or MIG slice), 8 or more vCPUs, 32 GB of RAM and Ubuntu 22.04 or 24.04. Data-centre GPUs and most
recent clouds expose the full BAR (a T4 does not: its BAR1 is 256 MiB). A whole VM is simplest: the driver and its
Vulkan part are yours to install.

**A container instead of a VM** (RunPod, vast.ai and similar rent containers): the host's driver is mounted into
the container, and only the parts its `NVIDIA_DRIVER_CAPABILITIES` names. The default, `compute,utility`, leaves out
the Vulkan driver (`libGLX_nvidia.so.0`), and WebGPU then finds no NVIDIA device. Start the container with
`NVIDIA_DRIVER_CAPABILITIES=all` (an environment variable in the provider's template; with Docker,
`docker run --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all ...`). Do not install a driver inside it. If
`libGLX_nvidia.so.0` is there but `vulkaninfo` still sees no NVIDIA device, the ICD file is missing; `./setup.sh`
prints the one line that creates it.

Then:

1. On a VM, the driver with its Vulkan part, which `bench.sh` does not install:
   `sudo apt install nvidia-driver-570-server libnvidia-gl-570-server` (or the `-open` and desktop branches;
   whatever the image has, add its `libnvidia-gl-<branch>`), then reboot. `nvidia-smi` must list the GPU.
2. Clone and run the quick check first:
   ```sh
   git clone https://github.com/midnight-experiments/ipir-benchmark-nvidia.git && cd ipir-benchmark-nvidia
   ./bench.sh --install-deps --quick --host <short name>
   ```
   What to look for in its output:
   - `GPU backend: nvidia`, and `bar1 total` at least `fb total` (for example 32768 MiB BAR1 for a 24 GB card). If
     BAR1 is 256 MiB, resizable BAR is off, and the run would be `nvidia-no-bar`.
   - `without the layer, 1024 MiB: ok`: wgpu allows the buffers, and the layer is not loaded.
   - The WebGPU adapter is the NVIDIA GPU, not llvmpipe.
   - It ends with `The quick check passed`.
3. The full run, on an otherwise idle machine, inside `tmux`:
   ```sh
   ./bench.sh --skip-quick --host <short name> --label "<GPU> + <CPU>"          # the nvidia file, about 100 minutes
   ./bench.sh --skip-quick --host <short name> --label "<GPU> + <CPU>" --cpu --cuda   # or with that host's CPU and cuda_bench
   ```
   Then send back `results/<short name>-<date>-bundle.tar.gz`, or push (see "Run it").
4. `--cuda` measures upstream's CUDA kernel (`simplepir-kernel`, `CudaKernel`) against the WebGPU path. The kernel
   does the first pass of an answer on the GPU, one request at a time, and nothing else, so the comparison is a
   whole answer two ways (`bench/cuda-compare.ts`, in the full run as step 4 of the report):
   - CUDA: the first pass on the GPU and the packing of every column block on the host's CPU (`cuda_fetch`);
   - WebGPU: both steps on the GPU;
   - for the same table and the same 32 requests, with every answer compared with the CPU's, byte for byte.

   It also times the upload of the table again, which is what a changed column block costs on the CUDA kernel,
   against WebGPU's write of one block. `bench.sh --cuda` then runs upstream's own `cuda_bench` (one request's
   first pass with another modulus; `results/<short name>-<date>-cuda-bench.txt`).

   CUDA needs the driver library (`libcuda.so`, part of the NVIDIA driver) and NVRTC (`libnvrtc.so`, part of the
   CUDA toolkit), loaded at run time:
   - The bindings are built for CUDA 13 when `libnvrtc.so.13` is installed and for CUDA 12 otherwise
     (`IPIR_CUDA_FEATURE=cuda-12` or `cuda-13` chooses outright). Upstream fixes CUDA 12.0, whose bindings do not
     load with the NVRTC of CUDA 13 (`undefined symbol: nvrtcGetNVVM`); patch 0004 makes it a choice.
   - NVRTC must not be newer than the driver: `nvidia-smi` prints the CUDA release the driver supports, and an NVRTC
     of a later release compiles kernels the driver refuses (`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`).

   By hand, on a results folder that exists:
   ```sh
   (cd ipir && cargo build --release --locked -p ipir-bench --features cuda-13 --bin cuda_fetch)
   deno run --allow-read --allow-write --allow-run --allow-env --allow-sys bench/cuda-compare.ts --out results/raw/<run>/cuda-compare.json
   deno run --allow-read --allow-write --allow-run --allow-env --allow-sys bench/report.ts results/raw/<run> results/<run>.json results/<run>.md
   ```

The results record: GPU name, driver, VRAM and VRAM in use (`nvidia-smi`); the Vulkan devices (`vulkaninfo
--summary`); the WebGPU adapter Deno picked, and whether it has timestamp queries; CPU model, cores and threads; RAM;
OS and kernel; Deno and Rust versions; the load average at the start and the end.

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

## What was checked, and where

The backends, patch 0002's two builds and the site export were added and run on macOS (Apple M4 Max, 16 cores,
48 GB, macOS 15.7.3, Deno 2.7.5, Rust 1.92.0). The full GPU run is `results/m4-max-2026-11-25-metal.json` (78 min).
The CPU backend ran only as `--quick`: there, the CPU's answers equal the GPU's byte for byte at 2048 × 2048,
2048 × 4096 and 4096 × 8192. The full CPU run has not been made. Resizable BAR detection was checked against the
`nvidia-smi -q -d MEMORY` output of the RTX 2060 (BAR1 256 MiB, framebuffer 6144 MiB: `nvidia-no-bar`).

The GPU build of the blocks (step 2b) was written against an RTX 2060 through Vulkan (the desktop of
`results/pop-os-notes.md`, with the Vulkan layer): 12 of 12 digests equal the CPU's at 2,048 to 32,768 rows and seeds
1 to 3, and 12 of 12 for the two six-table sets; each stage's output was also compared with the CPU's on its own
(hint masks, slots before and after the chains, digits, matrix). `./bench.sh --quick` ran there from start to end,
which is the first run of the backends' code on Linux. It was then run on an RTX 4090 with resizable BAR
(`results/rtx4090-notes.md`): the backend is detected as `nvidia`, the Vulkan layer is not needed, 24 of 24 blocks
equal the CPU's, and `./bench.sh --skip-quick --budget-sets today` ran from start to end. On macOS (Metal, M4 Max) it ran
on 2026-11-26: 24 of 24 blocks equal the CPU's, and the block budget ran at both shape sets.

A full run with the GPU build was then made on each of the two NVIDIA hosts (the "Results so far" section).

Not run yet: the block budget at the 10-million-output shapes with the blocks built on an NVIDIA GPU (it needs more
than the 6 GB of the RTX 2060, where it was measured at 8 million outputs instead, and the RTX 4090's memory was
held by another user; on the M4 Max it ran), and the container check in `setup.sh`. Checked on macOS without a benchmark run:
`bench/check-run.ts` and the export against the committed Metal run and against copies of it renamed to `nvidia`,
`nvidia-no-bar` and `nvidia-unknown-bar`; resizable BAR detection against `nvidia-smi -q -d MEMORY` reports with
BAR1 above the framebuffer, at 256 MiB, and absent; and `bench.sh`'s order of steps, with stand-ins for `setup.sh`,
`run-all.sh` and the GPU tools (the layer loaded and not, a failed quick check, a stopped run, the CPU options).
The first `./bench.sh --quick` on a Linux host is the check of the rest.

Before the backends:

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
| `ipir/bench/` | The compile benchmark programs, and the CPU answer programs (`cpu_fetch`, `cpu_server`) | new, adapted from spec 00065's compile checks |
| `ipir/build-stock.sh` | Builds `ipir/bench/` without patch 0002 | new |
| `site/` | The files for the overview page, one per backend | new |
| `gpu/block-build.ts` | The GPU build of a column block in WebGPU (step 2b), and `pack_digest`'s digest of its output | new: a WGSL port of the Metal block build of spec 00065's question 21 |
| `ipir/bench/src/gpu_prep.rs`, `rns32.rs`, `bin/gpu_build_prep.rs` | The GPU build's constant tables for a table shape (three 30-bit primes, prepared masks, chain exponents), and the CPU's stage outputs to check it against | `rns32.rs` and the replica from the Metal block build; the rest new |
| `bench.sh` | The whole procedure in one command: tools, setup, the Vulkan layer when needed, the quick check, the full run, the site file, the bundle | new |
| `bench/` | The Deno benchmark scripts | new, adapted from spec 00065's GPU answer-time harness |
| `tools/hide-memory-budget-layer/` | A Vulkan layer that hides `VK_EXT_memory_budget` from wgpu, for NVIDIA GPUs without resizable BAR | new |

UmbraDB is https://github.com/midnightntwrk/umbradb (Apache-2.0); its notice is in `NOTICE`. The IPIR+SP crates are
MIT OR Apache-2.0 (`ipir/vendor/ORIGIN.md`, `ipir/vendor/LICENSE-MIT`, `ipir/vendor/LICENSE-APACHE`). This repository
is Apache-2.0 (`LICENSE`).
