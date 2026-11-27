# pop-os runs, 2026-11-24 and 2026-11-25

## Host

| Part | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 2060, 6 GB, driver 580.159.03. Resizable BAR is off, so the BAR heap is 246 MiB. |
| CPU | AMD Ryzen 7 2700X, 8 cores, 16 threads, governor `schedutil` |
| RAM | 31 GiB |
| OS | Pop!_OS 24.04 with the COSMIC desktop, kernel 7.0.11 |
| Tools | Deno 2.7.5, rustc 1.92.0 |

The summaries show the CPU and the RAM as `?`. `bench/sysinfo.ts` reads `/proc` and `/sys`, and Deno 2 refuses those
reads without `--allow-all` ("NotCapable"). The load average in each summary includes the benchmark's own threads, so
the end value (10.0 in the full run) mostly measures the benchmark. `vulkaninfo` is not installed, so the Vulkan device
list is empty. `gpuProcesses` holds program names, not full command lines.

## Runs

| Run | How it ran | Result |
|---|---|---|
| `pop-os-2026-11-24-quick` | as shipped, desktop using 88 MiB of BAR | every GPU step fails |
| `pop-os-2026-11-24-quick-stock-bar52` | as shipped, `--skip-compile`, Chrome and Zed closed (52 MiB of BAR) | every GPU step fails |
| `pop-os-2026-11-24-quick-hidebudget` | with the Vulkan layer below | every step runs |
| `pop-os-2026-11-24-hidebudget` | full run with the Vulkan layer below | every step runs |
| `pop-os-2026-11-25-hidebudget-quiet` | the same full run, with the host's other CPU-heavy work paused | every step runs. These are the stock numbers for this host. |
| `pop-os-2026-11-25-hidebudget-opt` | the quiet run again, right after it, with patch 0002 | every step runs, and 354 of 360 blocks finish within 3 s |

Background services use 2 to 3 of the 16 threads, measured between runs. In the 2026-11-24 full run, another CPU-heavy
test started at 21:26Z. It overlapped the last `today` budget run and all of the `10m` set. For the 2026-11-25 run,
that work was paused, and 8 to 9 % of the CPU was busy just before the run started.

## Why the GPU steps fail as shipped

Deno 2.7.5 runs wgpu 28 with memory-budget thresholds of 97 % and 99 %. Above 97 %, wgpu refuses a new buffer. Above
99 %, wgpu loses the device. wgpu checks every device-local Vulkan heap, and on this card that includes the 246 MiB BAR
heap (`DEVICE_LOCAL | HOST_VISIBLE`). The NVIDIA driver reports the usage of that heap summed over all processes, so
the desktop's share counts against the benchmark. Opening a WebGPU device adds a 64 MiB staging block to the same heap.

With D as the desktop's BAR use (`nvidia-smi -q -d MEMORY`, "BAR1 Memory Usage"), the largest buffer wgpu allows is
about 112.5 − 1.97 × D MiB:

- At D = 88 MiB, every buffer fails, 4 KiB included, and the device is lost.
- At D = 52 MiB, buffers up to 8 MiB pass. Buffers of 9 MiB and larger fail with "not enough memory left".
- At D = 0 (no desktop), the limit is about 112 MiB. The benchmark needs a 128 MiB buffer at 2048 × 8192, and buffers
  up to 1 GiB for the larger tables.

Closing programs or running headless does not get past this limit on a GPU without resizable BAR. The fetch table
reports these failures as "does not fit", although each table uses a small part of the 6 GB of VRAM.

## The Vulkan layer

The `*-hidebudget` runs load the Vulkan layer in `tools/hide-memory-budget-layer/`. It removes `VK_EXT_memory_budget`
from the list that `vkEnumerateDeviceExtensionProperties` returns. Without that extension, wgpu skips both budget
checks. The shaders, the buffers, and the allocator stay the same. The folder's README shows how to build the layer and
how to run the benchmark with it. The layer needs no root.

With the layer, buffers from 4 KiB to 1 GiB allocate, and every fetch shape up to 32768 × 32768 (2.5 GiB resident)
runs.

## What the numbers show

Without patch 0002, every simulated block takes longer than 3 s. In the quiet run, block work p50 is 6.9 to 7.8 s in `seq` mode and 3.4 to
4.6 s in `pool` mode, and all 360 blocks are over 3 s. The CPU compile sets that time. The packing matrix takes about
1.0 to 1.06 s per column block at every row count, and each block changes six tables. The GPU part is small. Each
block uploads in about 0.2 s, and a call of 32 on the largest table takes at most 110 ms of GPU time.

The 2026-11-24 full run's block work p50 is 6 to 16 % higher in every budget cell. The other test overlapped only the
last `today` cells and the `10m` set, and the first `today` cells differ by as much, so that test does not explain all
of it. The quiet run is the reference for this host. The GPU fetch times of the two runs agree, for example 2.16 and
2.15 ms of GPU time for a call of 32 at 2048 × 2048.

Patch 0002 makes the CPU compile about three times faster with the same output words. With it, 354 of the 360 blocks
finish within 3 s. `ipir/patches/0002-faster-block-compile.md` explains the patch and compares the two 2026-11-25
runs.

## The full run of 2026-11-25 with the blocks built on the GPU

`pop-os-2026-11-25-nvidia-no-bar` is one run of every step at `9e820f3`: `./bench.sh --skip-quick --host pop-os
--label "RTX 2060 + Ryzen 7 2700X"`, 99 minutes. `bench.sh` found that wgpu refused a 1 GiB buffer and loaded the
Vulkan layer by itself. It is the run behind `site/pop-os-2026-11-25-nvidia-no-bar-site.json` and the figures of
`docs/other-results.md`.

- The desktop was in use: the load average was 6.3 at the start (the build just before the run) and 2.9 at the end,
  and a chain node runs on the machine. Its CPU figures are 1 to 10 % above those of the quiet runs above
  (`hidebudget-quiet`, `hidebudget-opt`): for example a block's work with patch 0002 compiled one after another,
  2.51 s here against 2.36 s there.
- The GPU build's output equals the CPU's in 24 of 24 blocks.
- With the 10-million-output tables resident, the block budget with the blocks built on the GPU does not fit in
  the card's 6 GB (`GPU out of memory` when a built block is laid out for the answer); the run records it as such.
- Step 2b's times were measured again after the run with `bench/gpu-compile.ts` at `dcad312`, which adds the builds
  with the GPU idle before them, and the report was regenerated at that commit from the run's raw files: one chain
  block's build takes 87 ms of GPU time right after another one and 341 ms after 6 s of idle (104 and 320 ms at the
  10-million-output shapes).

## The largest tables with the blocks built on the GPU (2026-11-26)

The full run's block budget with the blocks built on the GPU does not fit at the 10-million-output shapes. On
2026-11-26 the search `run-all.sh` now makes in that case was run by hand with the full run's arguments (30 blocks at
2, 5 and 10 calls a second), into the full run's raw files, and the report and the site file were regenerated from
them: `budget.gpu.largest` in the results, `budgetLargest` in the site file. The desktop was using 470 MiB.

- 9.5 million outputs: out of memory in the first blocks. 9 and 8.5 million: 30 blocks at 2 calls a second, then out
  of memory at 5. 8 million: all three rates, block work 0.26, 0.30 and 0.31 s at p50 (4.4, 5.1 and 5.1 % of the
  6 s), no block of 90 over 1 s.
- `nvidia-smi`, read every second: 5,688 MiB in use with the 9-million tables, 5,542 MiB with the 8.5-million ones,
  5,494 MiB at most with the 8-million ones. No sample is above 5,690 MiB of 6,144.
- `run-all.sh`'s own path was run first with short runs (`--quick --skip-compile --skip-fetch --builds patch0002
  --budget-sets 10m`): it ended at 8 million too. There 9 million failed earlier, when the build's buffers were
  made, so a size at the limit does not always fail at the same point.
- Memory in use depends on the call rate. Whether that is only the scratch of calls being answered, or also buffers
  of replaced blocks that are freed late, was not looked into.

## Upstream's CUDA kernel against the WebGPU path (2026-11-26)

`bench/cuda-compare.ts` at commit `8866374`, bindings `cuda-13`, the Vulkan layer loaded for the WebGPU side. Its
figures are in the results file as `cuda`, in the run's summary as step 4 and in `docs/other-results.md`; upstream's `cuda_bench` is in
`results/pop-os-2026-11-26-cuda-bench.txt`.

- CUDA did not start with what the machine has installed: the toolkit's NVRTC is of CUDA 13.4
  (`/usr/local/cuda-13.4`) and the driver (580.159.03) supports CUDA 13.0, so loading the compiled kernels fails with
  `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`. The run used `libnvrtc.so.13.0.88` and `libnvrtc-builtins.so.13.0.88` of the
  RTX 4090 host (package `cuda-nvrtc-13-0`), copied to `.tools/nvrtc-13.0/` of the clone, with that folder first in
  `LD_LIBRARY_PATH`. Nothing installed on the machine was changed.
- The desktop was using 526 MiB of the card and the load average was about 1 at the start.
- Every answer of both paths equals the CPU's, byte for byte, at all five shapes.

## The CUDA fetch and the answers per second (2026-11-27)

`./bench.sh --report --cuda --steps fetch,update6cuda --merge results/pop-os-2026-11-26-report.json` at `5c0ed70`,
written to `results/pop-os-2026-11-27-report.json`: the fetch step measured again and the CUDA update while answering
added; the other steps are those of 2026-11-26. The desktop was in use (load average 7 at the start, 362 MiB of the
card held by the desktop).

- Fetch 32 with this repository's CUDA kernels: 7.9 ms at 1 million values and 30.3 ms at 10 million, of which the
  device 6.4 and 27.4 ms. WebGPU on the card: 25.8 and 81.9 ms. Upstream's CUDA kernel: 258 and 840 ms.
- The 6× update at 1 million values, built on the device and installed while the six matrices answer: 113 ms.
- At 10 million values the six matrices, the calls' buffers and the block build do not fit in the card's 6 GiB. The
  largest set of the ladder that fits is 5 million (142 ms, 5.1 GiB on the device); with the WebGPU build it is
  8 million. The CUDA block build keeps about 1.9 GiB of working buffers for six tables, twice the WebGPU build's.
- The layout sweep (`cuda_fetch_native --sweep`): the fastest first pass has 4 columns and 2 requests per thread,
  3 to 5 % faster than the default (4 and 4), which is the fastest on the RTX 4090. The default is kept for both.

