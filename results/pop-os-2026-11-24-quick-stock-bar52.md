# IPIR+SP GPU benchmark: pop-os, 2026-11-24 (quick)

GPU NVIDIA GeForce RTX 2060, driver 580.159.03, 6144 MiB (362 used at start); WebGPU adapter "4318 7944 NVIDIA GeForce RTX 2060" with timestamp queries. CPU ?, 8 cores / 16 threads, ? GiB RAM. Pop!_OS 24.04 LTS 7.0.11-76070011-generic, Deno 2.7.5, rustc 1.92.0 (ded5c06cf 2025-12-08). Load average at start 2.34 2.65 3.59, at end 2.31 2.64 3.58. Run took 0 min.

## 1. `fetch` on the GPU

Median of 3 calls after a warm-up. Wall: from the call to the answer in host memory. Exec: the GPU's own time (timestamp queries).

| rows × cols | resident | upload | call of 1: wall (exec) | call of 32: wall (exec) |
|---|---|---|---|---|
| 2048 × 2048 | 40 MiB | — | does not fit: uploading the table: GPU out of memory: not enough memory left | |
| 2048 × 4096 | 80 MiB | — | does not fit: uploading the table: GPU out of memory: not enough memory left | |
| 4096 × 8192 | 192 MiB | — | does not fit: uploading the table: GPU out of memory: not enough memory left | |

## 2. `compile` one column block on the CPU

Not run or failed.

## 3. The 3 s block budget, simulated

Per chain block: compile one changed column block of each of six tables on the CPU, then upload the six to the GPU, while the GPU answers calls of 32 at a steady rate. The compile recomputes each block's whole hint columns: an upper bound, since hint updates for new records are not built. Latency: from the moment a call is due to its answer; "during" = calls overlapping a block's work.

### Shapes: today's starting shapes

read 2048 × 2048, nullifier 2048 × 2048, shielded groups 2048 × 4096, Dust groups 6144 × 28672, first coins 2048 × 2048, generation groups 2048 × 4096; 1.04 GiB resident.

Failed at uploading table read 2048x2048: uploading table read 2048x2048: GPU out of memory: not enough memory left

Full data: `pop-os-2026-11-24-quick-stock-bar52.json`.
