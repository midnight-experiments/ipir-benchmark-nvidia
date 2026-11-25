# IPIR+SP GPU benchmark: pop-os, 2026-11-24 (quick)

GPU NVIDIA GeForce RTX 2060, driver 580.159.03, 6144 MiB (688 used at start); WebGPU adapter "4318 7944 NVIDIA GeForce RTX 2060" with timestamp queries. CPU ?, 8 cores / 16 threads, ? GiB RAM. Pop!_OS 24.04 LTS 7.0.11-76070011-generic, Deno 2.7.5, rustc 1.92.0 (ded5c06cf 2025-12-08). Load average at start 3.57 3.06 2.82, at end 5.56 4.03 3.21. Run took 2 min.

## 1. `fetch` on the GPU

Median of 3 calls after a warm-up. Wall: from the call to the answer in host memory. Exec: the GPU's own time (timestamp queries).

| rows × cols | resident | upload | call of 1: wall (exec) | call of 32: wall (exec) |
|---|---|---|---|---|
| 2048 × 2048 | 40 MiB | 0.03 s | 12.5 ms (0.40 ms) | 18.0 ms (2.17 ms) |
| 2048 × 4096 | 80 MiB | 0.07 s | 12.7 ms (0.50 ms) | 18.4 ms (4.12 ms) |
| 4096 × 8192 | 192 MiB | 0.08 s | 13.2 ms (0.91 ms) | 26.8 ms (9.73 ms) |

## 2. `compile` one column block on the CPU

Whole build of a one-block table (2,048 columns): hint columns + packing matrix. Median of the repetitions.

| rows | at once | wall | per table: build | hints | packing |
|---|---|---|---|---|---|
| 2048 | 1 | 1206.9 ms | 1198.1 ms | 97.3 ms | 1092.5 ms |
| 8192 | 1 | 1315.8 ms | 1304.1 ms | 226.1 ms | 1067.7 ms |
| 2048 | 4 | 4192.5 ms | 2882.3 ms | 339.1 ms | 2539.3 ms |
| 8192 | 4 | 3441.7 ms | 3046.3 ms | 997.5 ms | 2041.0 ms |

## 3. The 3 s block budget, simulated

Per chain block: compile one changed column block of each of six tables on the CPU, then upload the six to the GPU, while the GPU answers calls of 32 at a steady rate. The compile recomputes each block's whole hint columns: an upper bound, since hint updates for new records are not built. Latency: from the moment a call is due to its answer; "during" = calls overlapping a block's work.

### Shapes: today's starting shapes

read 2048 × 2048, nullifier 2048 × 2048, shielded groups 2048 × 4096, Dust groups 6144 × 28672, first coins 2048 × 2048, generation groups 2048 × 4096; 1.04 GiB resident.

Pool: 8 threads (linux: thread nice 19); 4 blocks every 6 s.

| calls/s | mode | block work p50 / p99 | compile p50 / p99 | upload p50 | over 3 s | answer p50 / p99 during work | answer p50 / p99 no block work |
|---|---|---|---|---|---|---|---|
| 2 | seq | 7.08 / 7.44 s | 6.83 / 7.05 s | 178.8 ms | 4/4 | 23.1 / 313.4 ms | 23.3 / 228.1 ms |
| 2 | pool | 3.59 / 4.37 s | 3.42 / 4.03 s | 230.5 ms | 4/4 | 25.1 / 330.1 ms | 23.3 / 228.1 ms |
| 5 | seq | 7.00 / 7.14 s | 6.82 / 6.93 s | 186.1 ms | 4/4 | 20.4 / 244.2 ms | 22.3 / 217.9 ms |
| 5 | pool | 3.42 / 4.16 s | 3.22 / 3.98 s | 180.2 ms | 4/4 | 20.0 / 256.9 ms | 22.3 / 217.9 ms |

Full data: `pop-os-2026-11-24-quick-hidebudget.json`.
