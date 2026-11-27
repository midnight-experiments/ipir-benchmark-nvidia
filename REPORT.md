# Report: a table's four operations at 1 million and 10 million values

Written by `bench/summary.ts` from the result files named at the end; every figure is a median measured by `./bench.sh --report` on that machine. A cell without a figure says why: `not run`, `does not fit` (the GPU's memory ran out), `failed`, `no code` (no native Metal fetch exists here), `no GPU code`.

## 1. Definitions

- **Value**: one shielded output's read record, 98 bytes (49 cells of 2 bytes).
- **Matrix**: the table of all values, rows × columns of cells. One fetch reads one row.
- **Column block**: 2,048 columns of a matrix, the unit that is computed again when data is added.
- **CPU M4**: the vendored crates with patch 0002. **Metal**: native Metal kernels for the build of a column block; there is no native Metal fetch. **WebGPU**: Deno's WebGPU, on Metal on the Mac and on Vulkan on the NVIDIA cards.
- **CUDA**: this benchmark's own CUDA kernels. A fetch answers all the requests of a call in one pass over the matrix and does the packing on the GPU; a column block is built on the GPU and installed into the matrix that answers.
- **CUDA, upstream's kernel**: the CUDA kernel of the IPIR+SP crates as it is. It does the first pass of a fetch for one request per call; the packing runs on that host's CPU. It has no code for a build or an update.

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

\* Computing only, with native kernels: the blocks' hint columns and packing matrices are computed and compared with the CPU's, but nothing is installed into a matrix that answers and no calls are answered meanwhile. Metal has no native fetch to install into; the CUDA build from scratch is not joined to the CUDA fetch.

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
| 1M | 544 ms (9.1 %) | 53.1 ms (0.9 %)\* | 266 ms (4.4 %) | 252 ms (4.2 %) | 353 ms (5.9 %) | 113 ms (1.9 %) | 40.6 ms (0.7 %) |
| 10M | 627 ms (10.4 %) | 60.7 ms (1.0 %)\* | 264 ms (4.4 %) | does not fit; 268 ms (4.5 %) at 8M | 384 ms (6.4 %) | does not fit; 142 ms (2.4 %) at 5M | 58.9 ms (1.0 %) |

The CPU column computes the six blocks on a low-priority pool of half the cores and writes them to the GPU that answers. The WebGPU and CUDA columns compute them on the GPU that answers and install them into its matrices.

\* Computing only, with native kernels: the blocks' hint columns and packing matrices are computed and compared with the CPU's, but nothing is installed into a matrix that answers and no calls are answered meanwhile. Metal has no native fetch to install into; the CUDA build from scratch is not joined to the CUDA fetch.

## 4. Fetch 1 value

One request: the wall time from the request's bytes to the answer's bytes.

| Values | CPU M4 | Metal M4 | WebGPU M4 | WebGPU RTX 20 | WebGPU RTX 40 | CUDA RTX 20 | CUDA RTX 40 | CUDA, upstream's kernel RTX 20 | CUDA, upstream's kernel RTX 40 |
|---|---|---|---|---|---|---|---|---|---|
| 1M | 2.1 ms | no code | 15.3 ms | 13.2 ms | 12.2 ms | 1.1 ms | 0.4 ms | 8.0 ms | 6.5 ms |
| 10M | 12.8 ms | no code | 16.9 ms | 18.5 ms | 14.5 ms | 5.4 ms | 1.8 ms | 25.5 ms | 20.8 ms |

## 5. Fetch 32 values (one wallet call)

The call every wallet sends: 32 requests, padded so that the server cannot tell how many it needed.

| Values | CPU M4 | Metal M4 | WebGPU M4 | WebGPU RTX 20 | WebGPU RTX 40 | CUDA RTX 20 | CUDA RTX 40 | CUDA, upstream's kernel RTX 20 | CUDA, upstream's kernel RTX 40 |
|---|---|---|---|---|---|---|---|---|---|
| 1M | 72.1 ms | no code | 25.4 ms | 25.8 ms | 15.6 ms | 7.9 ms | 1.6 ms | 258 ms | 208 ms |
| 10M | 427 ms | no code | 73.7 ms | 81.9 ms | 25.9 ms | 30.3 ms | 6.8 ms | 840 ms | 666 ms |

The CUDA cells answer the 32 requests in one pass over the matrix, with the packing on the GPU. Upstream's kernel does the first pass one request at a time and leaves the packing to that NVIDIA host's CPU, so its call of 32 costs about 32 calls of 1. Every WebGPU and CUDA answer equals the CPU's, byte for byte.

## 6. Answers per second

The calls a matrix answers per second when calls arrive without a pause: each caller sends its next call as soon as its last one is answered, for 3 seconds, with 1, 2, 4, 8 and 16 callers. A cell is the highest rate found; section 8 gives the number of callers that reached it. For calls of 32 it also gives the values per second (calls × 32). Upstream's CUDA kernel takes one request at a time, so it has one caller.

**Calls of 1 per second**

| Values | CPU M4 | Metal M4 | WebGPU M4 | WebGPU RTX 20 | WebGPU RTX 40 | CUDA RTX 20 | CUDA RTX 40 | CUDA, upstream's kernel RTX 20 | CUDA, upstream's kernel RTX 40 |
|---|---|---|---|---|---|---|---|---|---|
| 1M | 517 | no code | 442 | 416 | 846 | 953 | 2671 | 124 | 153 |
| 10M | 83.8 | no code | 193 | 158 | 407 | 190 | 561 | 37.7 | 47.6 |

**Calls of 32 per second**

| Values | CPU M4 | Metal M4 | WebGPU M4 | WebGPU RTX 20 | WebGPU RTX 40 | CUDA RTX 20 | CUDA RTX 40 | CUDA, upstream's kernel RTX 20 | CUDA, upstream's kernel RTX 40 |
|---|---|---|---|---|---|---|---|---|---|
| 1M | 15.4 (493 values) | no code | 73.8 (2,360 values) | 63.5 (2,034 values) | 235 (7,533 values) | 147 (4,699 values) | 695 (22,227 values) | 3.9 (123 values) | 4.8 (153 values) |
| 10M | 2.4 (76 values) | no code | 13.5 (431 values) | 13.3 (425 values) | 67.9 (2,172 values) | 27.0 (864 values) | 160 (5,111 values) | 1.2 (38 values) | 1.5 (48 values) |

## 7. Memory: what fits on each GPU

What the WebGPU path's buffers hold on the GPU, at most, during each operation (the same on every machine). The CUDA path's memory is in section 8.

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

## 8. Detail and conditions

**M4** (Apple M4 Max), `results/m4-max-2026-11-27-report.json`, 2026-11-26:

- Load average (1 min) at the start: 7.8, at the end: 3.4.
- The machine was in use during the run: load average 8 at the start and 3 at the end.
- Measured again on 2026-11-27: fetch (1M, 10M); load average 5.9 at its start.
- 1M: the GPU-built matrix answers as the CPU-built one (32 of 32); during the re-dimension the old matrix answered 1 call(s) of 32, 1 equal to the CPU's, in 142 ms (p50); fetch 32 while an update runs 196 ms, against 41.2 ms otherwise; no block of 30 over 1 s: yes; the six blocks computed alone on the GPU 146 ms, and 181 ms with the GPU idle for 6 s before; the native Metal build's output equals the CPU's in 3 of 3 blocks; its six blocks alone 53.1 ms, and 98.8 ms after 6 s of idle; callers at the highest rate of answers per second, for calls of 1 and of 32: the CPU 4 and 8, WebGPU 8 and 8; WebGPU fetch 32: 10.2 ms on the device.
- 10M: the GPU-built matrix answers as the CPU-built one (32 of 32); during the re-dimension the old matrix answered 4 call(s) of 32, 4 equal to the CPU's, in 235 ms (p50); fetch 32 while an update runs 268 ms, against 67.7 ms otherwise; no block of 30 over 1 s: yes; the six blocks computed alone on the GPU 155 ms, and 199 ms with the GPU idle for 6 s before; the native Metal build's output equals the CPU's in 3 of 3 blocks; its six blocks alone 60.7 ms, and 114 ms after 6 s of idle; callers at the highest rate of answers per second, for calls of 1 and of 32: the CPU 16 and 1, WebGPU 16 and 1; WebGPU fetch 32: 55.8 ms on the device.

**RTX 20** (RTX 2060 + Ryzen 7 2700X), `results/pop-os-2026-11-27-report.json`, 2026-11-26:

- GPU memory in use by others at the start: 530 MiB of 6.0 GiB. Load average (1 min) at the start: 2.8, at the end: 10.6.
- Measured again on 2026-11-27: fetch, update6cuda (1M, 10M); load average 7.4 at its start, 362 MiB of the GPU's memory in use by others.
- 1M: the GPU-built matrix answers as the CPU-built one (32 of 32); during the re-dimension the old matrix answered 1 call(s) of 32, 1 equal to the CPU's, in 133 ms (p50); fetch 32 while an update runs 327 ms, against 27.8 ms otherwise; no block of 30 over 1 s: yes; the six blocks computed alone on the GPU 115 ms, and 481 ms with the GPU idle for 6 s before; the native CUDA build's output equals the CPU's in 3 of 3 blocks; its six blocks alone 84.7 ms, and 101 ms after 6 s of idle; CUDA fetch 32: 6.4 ms on the device (first pass 2.9 ms, packing 3.2 ms), and on the host 0.7 ms to read and copy the requests in and 0.4 ms to write the answers, the SHA-256 of the requests (0.2 ms) running while the device works; CUDA: the matrix holds 288 MiB on the device and a call's buffers 106 MiB; one changed column block is written from the host in 7.0 ms, and installed from the block build's buffers in 1.7 ms; CUDA 6× update: 103 ms to build the six blocks and 9.4 ms to install them; fetch 32 while it runs 11.6 ms, against 5.3 ms otherwise; no block of 30 over 1 s: yes; the six matrices, the calls' buffers and the build hold 3.7 GiB on the device; callers at the highest rate of answers per second, for calls of 1 and of 32: the CPU 16 and 8, WebGPU 16 and 2, CUDA 2 and 2; upstream's CUDA kernel, fetch 32: first pass 17.9 ms on the device, packing 223 ms on the CPU; WebGPU fetch 32: 10.6 ms on the device.
- 10M: the GPU-built matrix answers as the CPU-built one (32 of 32); during the re-dimension the old matrix answered 4 call(s) of 32, 4 equal to the CPU's, in 177 ms (p50); fetch 32 while an update runs 361 ms, against 55.4 ms otherwise; no block of 30 over 1 s: yes; the six blocks computed alone on the GPU 148 ms, and 423 ms with the GPU idle for 6 s before; the native CUDA build's output equals the CPU's in 3 of 3 blocks; its six blocks alone 106 ms, and 128 ms after 6 s of idle; CUDA fetch 32: 27.4 ms on the device (first pass 20.3 ms, packing 6.8 ms), and on the host 1.5 ms to read and copy the requests in and 0.8 ms to write the answers, the SHA-256 of the requests (0.4 ms) running while the device works; CUDA: the matrix holds 1.5 GiB on the device and a call's buffers 130 MiB; one changed column block is written from the host in 16.1 ms, and installed from the block build's buffers in 7.0 ms; CUDA 6× update (at 5M): 130 ms to build the six blocks and 12.2 ms to install them; fetch 32 while it runs 32.2 ms, against 13.9 ms otherwise; no block of 30 over 1 s: yes; the six matrices, the calls' buffers and the build hold 5.1 GiB on the device; callers at the highest rate of answers per second, for calls of 1 and of 32: the CPU 8 and 16, WebGPU 4 and 4, CUDA 2 and 8; upstream's CUDA kernel, fetch 32: first pass 196 ms on the device, packing 607 ms on the CPU; WebGPU fetch 32: 63.9 ms on the device.

**RTX 40** (RTX 4090 + Ryzen 7 7700X), `results/rtx4090-2026-11-27-report.json`, 2026-11-27:

- GPU memory in use by others at the start: 1 MiB of 24.0 GiB. Load average (1 min) at the start: 2.7, at the end: 35.1.
- This host is shared: other users' container builds were using its CPU when the first run of the day ended (load average 2.7 at the start, 35 at the end; 0.4 while the WebGPU 6× update at 1M was measured). The card itself was free: the other user's model was stopped for the day's runs. The fetch step was measured again later that day with the CPU quiet.
- Measured again on 2026-11-27: fetch, update6cuda (1M, 10M); load average 4.2 at its start, 1 MiB of the GPU's memory in use by others.
- 1M: the GPU-built matrix answers as the CPU-built one (32 of 32); during the re-dimension the old matrix answered 1 call(s) of 32, 1 equal to the CPU's, in 43.1 ms (p50); fetch 32 while an update runs 264 ms, against 23.0 ms otherwise; no block of 30 over 1 s: yes; the six blocks computed alone on the GPU 50.2 ms, and 257 ms with the GPU idle for 6 s before; the native CUDA build's output equals the CPU's in 3 of 3 blocks; its six blocks alone 35.8 ms, and 38.4 ms after 6 s of idle; CUDA fetch 32: 1.0 ms on the device (first pass 0.4 ms, packing 0.5 ms), and on the host 0.2 ms to read and copy the requests in and 0.1 ms to write the answers, the SHA-256 of the requests (0.1 ms) running while the device works; CUDA: the matrix holds 288 MiB on the device and a call's buffers 106 MiB; one changed column block is written from the host in 6.5 ms, and installed from the block build's buffers in 0.5 ms; CUDA 6× update: 38.1 ms to build the six blocks and 2.5 ms to install them; fetch 32 while it runs 9.5 ms, against 1.6 ms otherwise; no block of 30 over 1 s: yes; the six matrices, the calls' buffers and the build hold 3.7 GiB on the device; callers at the highest rate of answers per second, for calls of 1 and of 32: the CPU 8 and 4, WebGPU 16 and 8, CUDA 2 and 4; upstream's CUDA kernel, fetch 32: first pass 3.6 ms on the device, packing 194 ms on the CPU; WebGPU fetch 32: 2.1 ms on the device.
- 10M: the GPU-built matrix answers as the CPU-built one (32 of 32); during the re-dimension the old matrix answered 2 call(s) of 32, 2 equal to the CPU's, in 27.4 ms (p50); fetch 32 while an update runs 368 ms, against 35.4 ms otherwise; no block of 30 over 1 s: yes; the six blocks computed alone on the GPU 67.0 ms, and 314 ms with the GPU idle for 6 s before; the native CUDA build's output equals the CPU's in 3 of 3 blocks; its six blocks alone 66.5 ms, and 69.8 ms after 6 s of idle; CUDA fetch 32: 5.2 ms on the device (first pass 3.8 ms, packing 1.2 ms), and on the host 0.7 ms to read and copy the requests in and 0.3 ms to write the answers, the SHA-256 of the requests (0.1 ms) running while the device works; CUDA: the matrix holds 1.5 GiB on the device and a call's buffers 130 MiB; one changed column block is written from the host in 12.0 ms, and installed from the block build's buffers in 1.0 ms; CUDA 6× update: 54.9 ms to build the six blocks and 4.0 ms to install them; fetch 32 while it runs 26.4 ms, against 4.4 ms otherwise; no block of 30 over 1 s: yes; the six matrices, the calls' buffers and the build hold 6.2 GiB on the device; callers at the highest rate of answers per second, for calls of 1 and of 32: the CPU 16 and 16, WebGPU 16 and 8, CUDA 4 and 2; upstream's CUDA kernel, fetch 32: first pass 34.0 ms on the device, packing 608 ms on the CPU; WebGPU fetch 32: 11.2 ms on the device.

The WebGPU wall time of a fetch has about 12 ms that do not depend on the call (submitting the work and reading the answer back in this harness). A GPU left idle for the 6 s between blocks computes an update more slowly than one that just worked.
