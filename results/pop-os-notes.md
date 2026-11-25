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
