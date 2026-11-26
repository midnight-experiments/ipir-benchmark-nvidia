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
| 1M | 608 ms | no code | 301 ms | 311 ms | 152 ms | no GPU code | no GPU code |
| 10M | 2.14 s | no code | 973 ms | 1.38 s | not run (card in use) | no GPU code | no GPU code |

The GPU figure is from the cells in host memory to a table that answers; the table built on the GPU answers 32 requests exactly as the CPU-built one.

## 3. Update: data added to one column block

The cost is the same whether one value or many were added, as long as they fall in one column block.

**1×: one column block of the matrix computed again**

| Values | CPU M4 | Metal M4 | WebGPU M4 | WebGPU RTX 20 | WebGPU RTX 40 | CUDA RTX 20 | CUDA RTX 40 |
|---|---|---|---|---|---|---|---|
| 1M | 99.2 ms | no code | 61.6 ms | 59.5 ms | 41.1 ms | no GPU code | no GPU code |
| 10M | 121 ms | no code | 70.7 ms | 71.4 ms | 49.1 ms | no GPU code | no GPU code |

**6×: one chain block, a column block in each of the six matrices**, while the GPU answers 2 calls a second: the time, and its share of the 6 s between blocks.

| Values | CPU M4 | Metal M4 | WebGPU M4 | WebGPU RTX 20 | WebGPU RTX 40 | CUDA RTX 20 | CUDA RTX 40 |
|---|---|---|---|---|---|---|---|
| 1M | 543 ms (9.0 %) | no code | 271 ms (4.5 %) | 253 ms (4.2 %) | not run (card in use) | no GPU code | no GPU code |
| 10M | 619 ms (10.3 %) | no code | 268 ms (4.5 %) | does not fit; 248 ms (4.1 %) at 8M | not run (card in use) | no GPU code | no GPU code |

The CPU column computes the six blocks on a low-priority pool of half the cores and writes them to the GPU that answers. With upstream's CUDA kernel the blocks would be computed on the CPU too, and each changed table uploaded whole.

## 4. Fetch 1 value

One request: the wall time from the request's bytes to the answer's bytes.

| Values | CPU M4 | Metal M4 | WebGPU M4 | WebGPU RTX 20 | WebGPU RTX 40 | CUDA RTX 20 | CUDA RTX 40 |
|---|---|---|---|---|---|---|---|
| 1M | 2.1 ms | no code | 15.0 ms | 13.8 ms | 13.2 ms | 8.2 ms | 8.0 ms |
| 10M | 11.5 ms | no code | 18.0 ms | 18.0 ms | 14.2 ms | 23.8 ms | 22.5 ms |

## 5. Fetch 32 values (one wallet call)

The call every wallet sends: 32 requests, padded so that the server cannot tell how many it needed.

| Values | CPU M4 | Metal M4 | WebGPU M4 | WebGPU RTX 20 | WebGPU RTX 40 | CUDA RTX 20 | CUDA RTX 40 |
|---|---|---|---|---|---|---|---|
| 1M | 65.9 ms | no code | 28.1 ms | 29.2 ms | 18.3 ms | 251 ms | 329 ms |
| 10M | 363 ms | no code | 80.3 ms | 84.7 ms | 27.9 ms | 829 ms | 752 ms |

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
| 1M | Answer and update | fits | fits | not run (card in use) |
| 1M | Build from scratch | fits | fits | fits |
| 1M | Re-dimension while serving | fits | fits | fits |
| 10M | Answer | fits | fits | not run (card in use) |
| 10M | Answer and update | fits | does not fit; largest is 8M | not run (card in use) |
| 10M | Build from scratch | fits | fits | not run (card in use) |
| 10M | Re-dimension while serving | fits | fits | not run (card in use) |

## 7. Detail and conditions

**M4** (Apple M4 Max), `results/m4-max-2026-11-26-report.json`, 2026-11-26:

- Load average (1 min) at the start: 9.6.
- The machine was in use during the run: load average about 10 at the start and about 5 at the end.
- 1M: the GPU-built matrix answers as the CPU-built one (32 of 32); during the re-dimension the old matrix answered 1 call(s) of 32, 1 equal to the CPU's, in 134 ms (p50); fetch 32 while an update runs 199 ms, against 38.5 ms otherwise; no block of 30 over 1 s: yes; the six blocks computed alone on the GPU 131 ms, and 159 ms with the GPU idle for 6 s before; WebGPU fetch 32: 10.2 ms on the device.
- 10M: the GPU-built matrix answers as the CPU-built one (32 of 32); during the re-dimension the old matrix answered 3 call(s) of 32, 3 equal to the CPU's, in 194 ms (p50); fetch 32 while an update runs 267 ms, against 56.3 ms otherwise; no block of 30 over 1 s: yes; the six blocks computed alone on the GPU 155 ms, and 196 ms with the GPU idle for 6 s before; WebGPU fetch 32: 54.0 ms on the device.

**RTX 20** (RTX 2060 + Ryzen 7 2700X), `results/pop-os-2026-11-26-report.json`, 2026-11-26:

- GPU memory in use by others at the start: 526 MiB of 6.0 GiB. Load average (1 min) at the start: 5.2.
- 1M: the GPU-built matrix answers as the CPU-built one (32 of 32); during the re-dimension the old matrix answered 1 call(s) of 32, 1 equal to the CPU's, in 136 ms (p50); fetch 32 while an update runs 146 ms, against 47.9 ms otherwise; no block of 30 over 1 s: yes; the six blocks computed alone on the GPU 116 ms, and 409 ms with the GPU idle for 6 s before; CUDA fetch 32: first pass 17.9 ms on the device, packing 217 ms on the CPU; WebGPU fetch 32: 14.3 ms on the device.
- 10M: the GPU-built matrix answers as the CPU-built one (32 of 32); during the re-dimension the old matrix answered 4 call(s) of 32, 4 equal to the CPU's, in 180 ms (p50); fetch 32 while an update runs 339 ms, against 56.0 ms otherwise; no block of 30 over 1 s: yes; the six blocks computed alone on the GPU 149 ms, and 419 ms with the GPU idle for 6 s before; CUDA fetch 32: first pass 196 ms on the device, packing 595 ms on the CPU; WebGPU fetch 32: 63.4 ms on the device.

**RTX 40** (RTX 4090 + Ryzen 7 7700X), `results/rtx4090-2026-11-26-report.json`, 2026-11-26:

- GPU memory in use by others at the start: 21.0 GiB of 24.0 GiB (a card in use: what did not fit beside that is `not run`). Load average (1 min) at the start: 2.6.
- This host is shared: other users were compiling during the run (load average 15 to 20 at its end), so the CUDA cells, whose packing runs on this host's CPU, are slower than on a quiet host (669 ms for fetch 32 at 10M in the quiet run of the same day, `results/rtx4090-2026-11-25-nvidia.md`).
- 1M: the GPU-built matrix answers as the CPU-built one (32 of 32); during the re-dimension the old matrix answered 1 call(s) of 32, 1 equal to the CPU's, in 43.7 ms (p50); the six blocks computed alone on the GPU 50.2 ms, and 259 ms with the GPU idle for 6 s before; CUDA fetch 32: first pass 4.1 ms on the device, packing 311 ms on the CPU; WebGPU fetch 32: 2.1 ms on the device.
- 10M: the six blocks computed alone on the GPU 82.9 ms, and 318 ms with the GPU idle for 6 s before; CUDA fetch 32: first pass 34.3 ms on the device, packing 687 ms on the CPU; WebGPU fetch 32: 11.3 ms on the device.

The WebGPU wall time of a fetch has about 12 ms that do not depend on the call (submitting the work and reading the answer back in this harness). A GPU left idle for the 6 s between blocks computes an update more slowly than one that just worked.
