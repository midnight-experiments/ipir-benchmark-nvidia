# IPIR+SP GPU benchmark: pop-os, 2026-11-24

GPU NVIDIA GeForce RTX 2060, driver 580.159.03, 6144 MiB (342 used at start); WebGPU adapter "4318 7944 NVIDIA GeForce RTX 2060" with timestamp queries. CPU ?, 8 cores / 16 threads, ? GiB RAM. Pop!_OS 24.04 LTS 7.0.11-76070011-generic, Deno 2.7.5, rustc 1.92.0 (ded5c06cf 2025-12-08). Load average at start 2.31 2.89 3.39, at end 8.20 6.97 6.28. Run took 46 min.

## 1. `fetch` on the GPU

Median of 9 calls after a warm-up. Wall: from the call to the answer in host memory. Exec: the GPU's own time (timestamp queries).

| rows × cols | resident | upload | call of 1: wall (exec) | call of 32: wall (exec) |
|---|---|---|---|---|
| 2048 × 2048 | 40 MiB | 0.02 s | 12.6 ms (0.40 ms) | 16.2 ms (2.15 ms) |
| 2048 × 4096 | 80 MiB | 0.05 s | 12.9 ms (0.49 ms) | 17.7 ms (4.09 ms) |
| 2048 × 8192 | 160 MiB | 0.06 s | 13.0 ms (0.79 ms) | 23.0 ms (7.81 ms) |
| 4096 × 8192 | 192 MiB | 0.06 s | 13.1 ms (0.90 ms) | 23.9 ms (9.66 ms) |
| 4096 × 12288 | 288 MiB | 0.09 s | 13.6 ms (1.17 ms) | 28.7 ms (14.30 ms) |
| 6144 × 16384 | 448 MiB | 0.12 s | 14.5 ms (1.84 ms) | 37.6 ms (19.37 ms) |
| 6144 × 28672 | 784 MiB | 0.18 s | 15.3 ms (2.94 ms) | 53.2 ms (33.62 ms) |
| 8192 × 20480 | 640 MiB | 0.15 s | 14.8 ms (2.53 ms) | 45.9 ms (31.18 ms) |
| 8192 × 22528 | 704 MiB | 0.16 s | 15.1 ms (2.70 ms) | 49.1 ms (32.91 ms) |
| 10240 × 22528 | 792 MiB | 0.18 s | 15.5 ms (3.06 ms) | 54.2 ms (32.95 ms) |
| 12288 × 26624 | 1040 MiB | 0.37 s | 16.5 ms (3.86 ms) | 67.0 ms (44.47 ms) |
| 14336 × 34816 | 1496 MiB | 0.53 s | 17.4 ms (5.20 ms) | 87.1 ms (63.81 ms) |
| 16384 × 18432 | 864 MiB | 0.20 s | 16.0 ms (3.27 ms) | 55.9 ms (35.55 ms) |
| 16384 × 49152 | 2304 MiB | 0.70 s | 20.4 ms (7.91 ms) | 129.7 ms (100.83 ms) |
| 20480 × 40960 | 2240 MiB | 0.81 s | 20.4 ms (7.58 ms) | 136.5 ms (96.02 ms) |
| 32768 × 32768 (largest) | 2560 MiB | 0.90 s | 21.6 ms (8.83 ms) | 147.5 ms (109.30 ms) |

## 2. `compile` one column block on the CPU

Whole build of a one-block table (2,048 columns): hint columns + packing matrix. Median of the repetitions.

| rows | at once | wall | per table: build | hints | packing |
|---|---|---|---|---|---|
| 2048 | 1 | 1163.5 ms | 1155.1 ms | 88.7 ms | 1063.0 ms |
| 8192 | 1 | 1264.4 ms | 1253.4 ms | 217.9 ms | 1021.7 ms |
| 16384 | 1 | 1412.8 ms | 1405.4 ms | 393.4 ms | 1002.3 ms |
| 32768 | 1 | 2214.5 ms | 2191.6 ms | 1153.1 ms | 1009.9 ms |
| 2048 | 4 | 2542.4 ms | 1772.9 ms | 329.1 ms | 1444.9 ms |
| 8192 | 4 | 3082.0 ms | 2999.3 ms | 869.4 ms | 2155.6 ms |
| 16384 | 4 | 3679.0 ms | 2978.8 ms | 1561.8 ms | 1390.2 ms |
| 32768 | 4 | 7519.8 ms | 6294.7 ms | 4581.6 ms | 1694.9 ms |

## 3. The 3 s block budget, simulated

Per chain block: compile one changed column block of each of six tables on the CPU, then upload the six to the GPU, while the GPU answers calls of 32 at a steady rate. The compile recomputes each block's whole hint columns: an upper bound, since hint updates for new records are not built. Latency: from the moment a call is due to its answer; "during" = calls overlapping a block's work.

### Shapes: today's starting shapes

read 2048 × 2048, nullifier 2048 × 2048, shielded groups 2048 × 4096, Dust groups 6144 × 28672, first coins 2048 × 2048, generation groups 2048 × 4096; 1.04 GiB resident.

Pool: 8 threads (linux: thread nice 19); 30 blocks every 6 s.

| calls/s | mode | block work p50 / p99 | compile p50 / p99 | upload p50 | over 3 s | answer p50 / p99 during work | answer p50 / p99 no block work |
|---|---|---|---|---|---|---|---|
| 2 | seq | 6.93 / 7.34 s | 6.71 / 7.13 s | 212.2 ms | 30/30 | 23.8 / 331.7 ms | 18.4 / 291.1 ms |
| 2 | pool | 3.46 / 4.31 s | 3.27 / 4.01 s | 189.5 ms | 30/30 | 19.7 / 393.3 ms | 18.4 / 291.1 ms |
| 5 | seq | 6.89 / 7.12 s | 6.70 / 6.89 s | 182.3 ms | 30/30 | 21.2 / 260.5 ms | 18.2 / 236.1 ms |
| 5 | pool | 3.40 / 4.49 s | 3.22 / 4.31 s | 178.1 ms | 30/30 | 20.9 / 254.3 ms | 18.2 / 236.1 ms |
| 10 | seq | 6.90 / 7.08 s | 6.71 / 6.91 s | 174.8 ms | 30/30 | 20.5 / 248.3 ms | 18.1 / 223.9 ms |
| 10 | pool | 3.46 / 4.33 s | 3.22 / 4.16 s | 190.3 ms | 30/30 | 20.3 / 223.6 ms | 18.1 / 223.9 ms |

### Shapes: 10 million outputs

read 14336 × 34816, nullifier 16384 × 18432, shielded groups 8192 × 20480, Dust groups 6144 × 28672, first coins 2048 × 2048, generation groups 2048 × 4096; 3.81 GiB resident.

Pool: 8 threads (linux: thread nice 19); 30 blocks every 6 s.

| calls/s | mode | block work p50 / p99 | compile p50 / p99 | upload p50 | over 3 s | answer p50 / p99 during work | answer p50 / p99 no block work |
|---|---|---|---|---|---|---|---|
| 2 | seq | 7.79 / 7.98 s | 7.56 / 7.77 s | 207.4 ms | 30/30 | 67.4 / 380.0 ms | 61.1 / 375.5 ms |
| 2 | pool | 4.60 / 6.15 s | 4.40 / 5.68 s | 202.5 ms | 30/30 | 64.5 / 393.4 ms | 61.1 / 375.5 ms |
| 5 | seq | 7.80 / 7.93 s | 7.59 / 7.67 s | 214.5 ms | 30/30 | 65.6 / 171.2 ms | 61.0 / 117.5 ms |
| 5 | pool | 4.39 / 5.76 s | 4.19 / 5.56 s | 215.1 ms | 30/30 | 62.8 / 133.2 ms | 61.0 / 117.5 ms |
| 10 | seq | 7.82 / 7.98 s | 7.64 / 7.70 s | 176.5 ms | 30/30 | 58.9 / 248.0 ms | 56.4 / 96.0 ms |
| 10 | pool | 4.57 / 6.26 s | 4.36 / 6.00 s | 221.2 ms | 30/30 | 57.2 / 117.8 ms | 56.4 / 96.0 ms |

Full data: `pop-os-2026-11-25-hidebudget-quiet.json`.
