# Report: a table's four operations at 1 million and 10 million values

Written by `bench/summary.ts` from the result files named at the end; every figure is a median measured by `./bench.sh --report` on that machine. A cell without a figure says why: `not run`, `does not fit` (the GPU's memory ran out), `failed`, `no code` (no native Metal code exists here), `no GPU code` (upstream's CUDA kernel does only the first pass of a fetch).

## 1. Definitions

- **Value**: one shielded output's read record, 98 bytes (49 cells of 2 bytes).
- **Matrix**: the table of all values, rows × columns of cells. One fetch reads one row.
- **Column block**: 2,048 columns of a matrix, the unit that is computed again when data is added.
- **CPU M4**: the vendored crates with patch 0002. **WebGPU**: Deno's WebGPU, on Metal on the Mac and on Vulkan on the NVIDIA cards. **CUDA**: upstream's kernel, the first pass of a fetch on the GPU and the packing on that host's CPU.

| Values | Matrix (rows × columns) | Column blocks | Cells | On the GPU, with the packing matrices | Request | Answer |
|---|---|---|---|---|---|---|
| 1M | 4,096 × 12,288 | 6 | 96 MiB | 288 MiB | 52.8 kB | 33.9 kB |
| 10M | 14,336 × 34,816 | 17 | 952 MiB | 1.5 GiB | 115.5 kB | 95.8 kB |

## 2. Build a whole matrix from scratch

At server start, when a matrix changes its number of rows, and when a matrix is added. Every column block's hint columns and packing matrix are computed from the cells.

| Values | CPU M4 | Metal M4 | WebGPU M4 | WebGPU RTX 20 | WebGPU RTX 40 | CUDA RTX 20 | CUDA RTX 40 |
|---|---|---|---|---|---|---|---|
| 1M | 622 ms | 78.1 ms\* | 305 ms | 307 ms | 154 ms | 116 ms\* | 57.5 ms\* |
| 10M | 2.44 s | 250 ms\* | 966 ms | 1.34 s | 495 ms | 443 ms\* | 317 ms\* |

The WebGPU figure is from the cells in host memory to a table that answers; the table built on the GPU answers 32 requests exactly as the CPU-built one.

\* Computing only, with native kernels: the blocks' hint columns and packing matrices are computed and compared with the CPU's, but there is no native fetch on this path, so nothing is installed into a table that answers and no calls are answered meanwhile.

## 3. Update: data added to one column block

The cost is the same whether one value or many were added, as long as they fall in one column block.

**1×: one column block of the matrix computed again**

| Values | CPU M4 | Metal M4 | WebGPU M4 | WebGPU RTX 20 | WebGPU RTX 40 | CUDA RTX 20 | CUDA RTX 40 |
|---|---|---|---|---|---|---|---|
| 1M | 101 ms | 28.4 ms | 61.5 ms | 59.3 ms | 47.9 ms | 41.8 ms | 23.1 ms |
| 10M | 121 ms | 30.9 ms | 66.9 ms | 71.1 ms | 45.1 ms | 52.4 ms | 34.2 ms |

**6×: one chain block, a column block in each of the six matrices**, while the GPU answers 2 calls a second: the time, and its share of the 6 s between blocks.

| Values | CPU M4 | Metal M4 | WebGPU M4 | WebGPU RTX 20 | WebGPU RTX 40 | CUDA RTX 20 | CUDA RTX 40 |
|---|---|---|---|---|---|---|---|
| 1M | 544 ms (9.1 %) | 53.1 ms (0.9 %)\* | 266 ms (4.4 %) | 252 ms (4.2 %) | 353 ms (5.9 %) | 84.7 ms (1.4 %)\* | 35.8 ms (0.6 %)\* |
| 10M | 627 ms (10.4 %) | 60.7 ms (1.0 %)\* | 264 ms (4.4 %) | does not fit; 268 ms (4.5 %) at 8M | 384 ms (6.4 %) | 106 ms (1.8 %)\* | 66.5 ms (1.1 %)\* |

The CPU column computes the six blocks on a low-priority pool of half the cores and writes them to the GPU that answers; the WebGPU columns compute them on the GPU that answers and install them. Upstream's CUDA kernel has no code for an update: a CUDA figure here is this benchmark's own port of the block build.

\* Computing only, with native kernels: the blocks' hint columns and packing matrices are computed and compared with the CPU's, but there is no native fetch on this path, so nothing is installed into a table that answers and no calls are answered meanwhile.

## 4. Fetch 1 value

One request: the wall time from the request's bytes to the answer's bytes.

| Values | CPU M4 | Metal M4 | WebGPU M4 | WebGPU RTX 20 | WebGPU RTX 40 | CUDA RTX 20 | CUDA RTX 40 |
|---|---|---|---|---|---|---|---|
| 1M | 2.1 ms | no code | 15.2 ms | 13.4 ms | 12.5 ms | 7.8 ms | 7.4 ms |
| 10M | 11.0 ms | no code | 18.0 ms | 18.2 ms | 14.5 ms | 25.2 ms | 27.1 ms |

## 5. Fetch 32 values (one wallet call)

The call every wallet sends: 32 requests, padded so that the server cannot tell how many it needed.

| Values | CPU M4 | Metal M4 | WebGPU M4 | WebGPU RTX 20 | WebGPU RTX 40 | CUDA RTX 20 | CUDA RTX 40 |
|---|---|---|---|---|---|---|---|
| 1M | 66.1 ms | no code | 29.6 ms | 30.1 ms | 14.9 ms | 257 ms | 235 ms |
| 10M | 363 ms | no code | 72.7 ms | 82.7 ms | 31.2 ms | 832 ms | 777 ms |

The CUDA cells are the first pass on the GPU, one request at a time, and the packing on that NVIDIA host's CPU. Every WebGPU and CUDA answer equals the CPU's, byte for byte.

## 6. Memory: what fits on each GPU

What the benchmark's buffers hold on the GPU, at most, during each operation (the same on every machine).

| Values | To answer (six matrices) | To answer and update (and one chain block being computed) | To build the matrix from scratch (alone on the card) | To re-dimension the matrix while it serves (old and new together) |
|---|---|---|---|---|
| 1M | 1.5 GiB | 2.4 GiB | 897 MiB | 1.2 GiB |
| 10M | 3.8 GiB | 4.8 GiB | 2.2 GiB | 3.8 GiB |

| Values | Operation | M4 | RTX 20 | RTX 40 |
|---|---|---|---|---|
| 1M | Answer | fits | fits | fits |
| 1M | Answer and update | fits | fits | fits |
| 1M | Build from scratch | fits | fits | fits |
| 1M | Re-dimension while serving | fits | fits | fits |
| 10M | Answer | fits | fits | fits |
| 10M | Answer and update | fits | does not fit; largest is 8M | fits |
| 10M | Build from scratch | fits | fits | fits |
| 10M | Re-dimension while serving | fits | fits | fits |

## 7. Detail and conditions

**M4** (Apple M4 Max), `results/m4-max-2026-11-26-report.json`, 2026-11-26:

- Load average (1 min) at the start: 7.8, at the end: 3.4.
- The machine was in use during the run: load average 8 at the start and 3 at the end.
- 1M: the GPU-built matrix answers as the CPU-built one (32 of 32); during the re-dimension the old matrix answered 1 call(s) of 32, 1 equal to the CPU's, in 142 ms (p50); fetch 32 while an update runs 196 ms, against 41.2 ms otherwise; no block of 30 over 1 s: yes; the six blocks computed alone on the GPU 146 ms, and 181 ms with the GPU idle for 6 s before; the native Metal build's output equals the CPU's in 3 of 3 blocks; its six blocks alone 53.1 ms, and 98.8 ms after 6 s of idle; WebGPU fetch 32: 10.2 ms on the device.
- 10M: the GPU-built matrix answers as the CPU-built one (32 of 32); during the re-dimension the old matrix answered 4 call(s) of 32, 4 equal to the CPU's, in 235 ms (p50); fetch 32 while an update runs 268 ms, against 67.7 ms otherwise; no block of 30 over 1 s: yes; the six blocks computed alone on the GPU 155 ms, and 199 ms with the GPU idle for 6 s before; the native Metal build's output equals the CPU's in 3 of 3 blocks; its six blocks alone 60.7 ms, and 114 ms after 6 s of idle; WebGPU fetch 32: 55.1 ms on the device.

**RTX 20** (RTX 2060 + Ryzen 7 2700X), `results/pop-os-2026-11-26-report.json`, 2026-11-26:

- GPU memory in use by others at the start: 530 MiB of 6.0 GiB. Load average (1 min) at the start: 2.8, at the end: 10.6.
- 1M: the GPU-built matrix answers as the CPU-built one (32 of 32); during the re-dimension the old matrix answered 1 call(s) of 32, 1 equal to the CPU's, in 133 ms (p50); fetch 32 while an update runs 327 ms, against 27.8 ms otherwise; no block of 30 over 1 s: yes; the six blocks computed alone on the GPU 115 ms, and 481 ms with the GPU idle for 6 s before; the native CUDA build's output equals the CPU's in 3 of 3 blocks; its six blocks alone 84.7 ms, and 101 ms after 6 s of idle; CUDA fetch 32: first pass 18.1 ms on the device, packing 222 ms on the CPU; WebGPU fetch 32: 14.2 ms on the device.
- 10M: the GPU-built matrix answers as the CPU-built one (32 of 32); during the re-dimension the old matrix answered 4 call(s) of 32, 4 equal to the CPU's, in 177 ms (p50); fetch 32 while an update runs 361 ms, against 55.4 ms otherwise; no block of 30 over 1 s: yes; the six blocks computed alone on the GPU 148 ms, and 423 ms with the GPU idle for 6 s before; the native CUDA build's output equals the CPU's in 3 of 3 blocks; its six blocks alone 106 ms, and 128 ms after 6 s of idle; CUDA fetch 32: first pass 199 ms on the device, packing 595 ms on the CPU; WebGPU fetch 32: 64.1 ms on the device.

**RTX 40** (RTX 4090 + Ryzen 7 7700X), `results/rtx4090-2026-11-27-report.json`, 2026-11-27:

- GPU memory in use by others at the start: 1 MiB of 24.0 GiB. Load average (1 min) at the start: 2.7, at the end: 35.1.
- This host is shared: other users' container builds were using its CPU when the run ended (load average 2.7 at the start, 35 at the end; 0.4 while the 6× update at 1M was measured), so the CUDA fetch cells, whose packing runs on this host's CPU, are slower than on a quiet host (669 ms for fetch 32 at 10M in a quiet run, `results/rtx4090-2026-11-25-nvidia.md`). The card itself was free: the other user's model was stopped for this run.
- 1M: the GPU-built matrix answers as the CPU-built one (32 of 32); during the re-dimension the old matrix answered 1 call(s) of 32, 1 equal to the CPU's, in 43.1 ms (p50); fetch 32 while an update runs 264 ms, against 23.0 ms otherwise; no block of 30 over 1 s: yes; the six blocks computed alone on the GPU 50.2 ms, and 257 ms with the GPU idle for 6 s before; the native CUDA build's output equals the CPU's in 3 of 3 blocks; its six blocks alone 35.8 ms, and 38.4 ms after 6 s of idle; CUDA fetch 32: first pass 3.7 ms on the device, packing 220 ms on the CPU; WebGPU fetch 32: 2.1 ms on the device.
- 10M: the GPU-built matrix answers as the CPU-built one (32 of 32); during the re-dimension the old matrix answered 2 call(s) of 32, 2 equal to the CPU's, in 27.4 ms (p50); fetch 32 while an update runs 368 ms, against 35.4 ms otherwise; no block of 30 over 1 s: yes; the six blocks computed alone on the GPU 67.0 ms, and 314 ms with the GPU idle for 6 s before; the native CUDA build's output equals the CPU's in 3 of 3 blocks; its six blocks alone 66.5 ms, and 69.8 ms after 6 s of idle; CUDA fetch 32: first pass 34.4 ms on the device, packing 711 ms on the CPU; WebGPU fetch 32: 11.2 ms on the device.

The WebGPU wall time of a fetch has about 12 ms that do not depend on the call (submitting the work and reading the answer back in this harness). A GPU left idle for the 6 s between blocks computes an update more slowly than one that just worked.
