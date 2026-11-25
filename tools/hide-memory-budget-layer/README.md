# Vulkan layer that hides `VK_EXT_memory_budget`

On an NVIDIA GPU without resizable BAR, Deno's WebGPU refuses the buffers the benchmark needs with "not enough memory
left", and every GPU step fails. This layer works around that. It removes `VK_EXT_memory_budget` from the device
extensions that the driver reports, so wgpu does not enable the extension and skips its memory-budget checks. The
shaders, the buffers and the allocator stay the same. The layer needs no root, and it applies only to the processes
that you start with its two environment variables.

## Why wgpu refuses the buffers

Deno 2.7.5 runs wgpu 28 with memory-budget thresholds of 97 % and 99 %. Above 97 % of a heap's budget, wgpu refuses a
new buffer. Above 99 %, wgpu loses the device. wgpu checks every device-local Vulkan heap. Without resizable BAR, that
includes the BAR heap (`DEVICE_LOCAL | HOST_VISIBLE`, 246 MiB on an RTX 2060). The NVIDIA driver reports the usage of
that heap summed over all processes, so the desktop's share counts against the benchmark. Opening a WebGPU device adds
a 64 MiB staging block to the same heap.

On the RTX 2060 in `results/pop-os-notes.md`, the largest buffer wgpu allows is about 112.5 − 1.97 × D MiB, where D is
the BAR memory that other programs use (`nvidia-smi -q -d MEMORY`, "BAR1 Memory Usage"). The benchmark needs a
128 MiB buffer at 2048 × 8192 and buffers up to 1 GiB for the larger tables, so it fails even with no desktop running.

## Build it

```sh
tools/hide-memory-budget-layer/build.sh
```

The script needs `cc`. It writes `libhide_budget.so` next to the manifest
`VK_LAYER_IPIRBENCH_hide_memory_budget.json`.

## Check whether your host needs it

Run `check.ts` without the layer, then with it:

```sh
export DENO_WEBGPU_BACKEND=vulkan
.tools/deno/deno run -A tools/hide-memory-budget-layer/check.ts 128
VK_LAYER_PATH=$PWD/tools/hide-memory-budget-layer VK_INSTANCE_LAYERS=VK_LAYER_IPIRBENCH_hide_memory_budget \
	.tools/deno/deno run -A tools/hide-memory-budget-layer/check.ts 128
```

If the first command prints `128 MiB: ok`, your host does not need the layer. On the RTX 2060 desktop, with 58 MiB of
BAR in use by other programs, the two commands print:

```
128 MiB: refused (not enough memory left)
128 MiB: ok
```

## Run the benchmark with it

```sh
export VK_LAYER_PATH=$PWD/tools/hide-memory-budget-layer
export VK_INSTANCE_LAYERS=VK_LAYER_IPIRBENCH_hide_memory_budget
./run-all.sh --name <host>-<date>-hidebudget
```

The result files do not record the layer. Put it in the run's name, as the `*-hidebudget` runs in `results/` do.

## Limits

- Linux only. It uses the Vulkan loader's layer interface. On macOS the benchmark runs on Metal.
- It tracks up to 16 devices for each process. Each benchmark script opens one.
- With the layer, wgpu no longer checks the budget before it allocates. On the RTX 2060, every shape that the benchmark
  tried fit in the 6 GB of VRAM, so a real out-of-memory with the layer has not been seen.
- Tested with NVIDIA driver 580.159.03 and the Vulkan loader 1.3.280 of Pop!_OS 24.04.
