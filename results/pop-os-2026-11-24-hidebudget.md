# IPIR+SP GPU benchmark: pop-os, 2026-11-24

GPU NVIDIA GeForce RTX 2060, driver 580.159.03, 6144 MiB (689 used at start); WebGPU adapter "4318 7944 NVIDIA GeForce RTX 2060" with timestamp queries. CPU ?, 8 cores / 16 threads, ? GiB RAM. Pop!_OS 24.04 LTS 7.0.11-76070011-generic, Deno 2.7.5, rustc 1.92.0 (ded5c06cf 2025-12-08). Load average at start 5.16 4.01 3.22, at end 10.04 9.98 9.01. Run took 48 min.

## 1. `fetch` on the GPU

Median of 9 calls after a warm-up. Wall: from the call to the answer in host memory. Exec: the GPU's own time (timestamp queries).

| rows × cols | resident | upload | call of 1: wall (exec) | call of 32: wall (exec) |
|---|---|---|---|---|
| 2048 × 2048 | 40 MiB | 0.02 s | 12.7 ms (0.40 ms) | 15.8 ms (2.16 ms) |
| 2048 × 4096 | 80 MiB | 0.08 s | 12.8 ms (0.50 ms) | 17.9 ms (4.10 ms) |
| 2048 × 8192 | 160 MiB | 0.08 s | 12.9 ms (0.80 ms) | 22.5 ms (7.82 ms) |
| 4096 × 8192 | 192 MiB | 0.08 s | 13.3 ms (0.90 ms) | 24.4 ms (9.68 ms) |
| 4096 × 12288 | 288 MiB | 0.10 s | 13.5 ms (1.18 ms) | 28.9 ms (14.28 ms) |
| 6144 × 16384 | 448 MiB | 0.13 s | 14.2 ms (1.85 ms) | 36.5 ms (21.52 ms) |
| 6144 × 28672 | 784 MiB | 0.20 s | 15.4 ms (2.95 ms) | 51.6 ms (33.81 ms) |
| 8192 × 20480 | 640 MiB | 0.16 s | 14.8 ms (2.53 ms) | 42.5 ms (26.94 ms) |
| 8192 × 22528 | 704 MiB | 0.17 s | 15.2 ms (2.71 ms) | 48.5 ms (30.58 ms) |
| 10240 × 22528 | 792 MiB | 0.20 s | 15.7 ms (3.07 ms) | 54.0 ms (32.97 ms) |
| 12288 × 26624 | 1040 MiB | 0.47 s | 16.4 ms (3.88 ms) | 66.3 ms (43.67 ms) |
| 14336 × 34816 | 1496 MiB | 0.51 s | 17.7 ms (5.22 ms) | 88.7 ms (63.76 ms) |
| 16384 × 18432 | 864 MiB | 0.35 s | 15.7 ms (3.20 ms) | 54.1 ms (35.60 ms) |
| 16384 × 49152 | 2304 MiB | 0.86 s | 20.7 ms (7.94 ms) | 121.3 ms (100.53 ms) |
| 20480 × 40960 | 2240 MiB | 0.76 s | 20.2 ms (7.60 ms) | 137.6 ms (97.21 ms) |
| 32768 × 32768 (largest) | 2560 MiB | 1.07 s | 21.8 ms (8.85 ms) | 149.0 ms (110.02 ms) |

## 2. `compile` one column block on the CPU

Whole build of a one-block table (2,048 columns): hint columns + packing matrix. Median of the repetitions.

| rows | at once | wall | per table: build | hints | packing |
|---|---|---|---|---|---|
| 2048 | 1 | 1197.8 ms | 1189.5 ms | 88.4 ms | 1098.8 ms |
| 8192 | 1 | 1306.9 ms | 1290.4 ms | 220.0 ms | 1065.7 ms |
| 16384 | 1 | 1476.6 ms | 1459.7 ms | 403.2 ms | 1047.6 ms |
| 32768 | 1 | 2295.0 ms | 2270.0 ms | 1192.9 ms | 1049.1 ms |
| 2048 | 4 | 3384.9 ms | 2555.9 ms | 344.5 ms | 2203.7 ms |
| 8192 | 4 | 3234.5 ms | 2419.1 ms | 895.4 ms | 1521.4 ms |
| 16384 | 4 | 4596.2 ms | 3729.9 ms | 1622.1 ms | 2081.7 ms |
| 32768 | 4 | 7692.4 ms | 6758.4 ms | 4709.6 ms | 2025.8 ms |

## 3. The 3 s block budget, simulated

Per chain block: compile one changed column block of each of six tables on the CPU, then upload the six to the GPU, while the GPU answers calls of 32 at a steady rate. The compile recomputes each block's whole hint columns: an upper bound, since hint updates for new records are not built. Latency: from the moment a call is due to its answer; "during" = calls overlapping a block's work.

### Shapes: today's starting shapes

read 2048 × 2048, nullifier 2048 × 2048, shielded groups 2048 × 4096, Dust groups 6144 × 28672, first coins 2048 × 2048, generation groups 2048 × 4096; 1.04 GiB resident.

Pool: 8 threads (linux: thread nice 19); 30 blocks every 6 s.

| calls/s | mode | block work p50 / p99 | compile p50 / p99 | upload p50 | over 3 s | answer p50 / p99 during work | answer p50 / p99 no block work |
|---|---|---|---|---|---|---|---|
| 2 | seq | 7.50 / 7.93 s | 7.31 / 7.74 s | 187.8 ms | 30/30 | 23.3 / 346.5 ms | 20.3 / 288.3 ms |
| 2 | pool | 3.86 / 5.37 s | 3.61 / 5.18 s | 216.0 ms | 30/30 | 23.1 / 328.6 ms | 20.3 / 288.3 ms |
| 5 | seq | 7.52 / 7.79 s | 7.32 / 7.45 s | 181.2 ms | 30/30 | 21.5 / 250.6 ms | 18.2 / 245.1 ms |
| 5 | pool | 3.77 / 4.63 s | 3.59 / 4.43 s | 179.9 ms | 30/30 | 21.9 / 254.5 ms | 18.2 / 245.1 ms |
| 10 | seq | 7.62 / 7.89 s | 7.44 / 7.73 s | 177.0 ms | 30/30 | 20.5 / 237.3 ms | 18.2 / 188.7 ms |
| 10 | pool | 4.00 / 5.32 s | 3.74 / 5.08 s | 179.2 ms | 30/30 | 20.6 / 232.3 ms | 18.2 / 188.7 ms |

### Shapes: 10 million outputs

read 14336 × 34816, nullifier 16384 × 18432, shielded groups 8192 × 20480, Dust groups 6144 × 28672, first coins 2048 × 2048, generation groups 2048 × 4096; 3.81 GiB resident.

Pool: 8 threads (linux: thread nice 19); 30 blocks every 6 s.

| calls/s | mode | block work p50 / p99 | compile p50 / p99 | upload p50 | over 3 s | answer p50 / p99 during work | answer p50 / p99 no block work |
|---|---|---|---|---|---|---|---|
| 2 | seq | 8.25 / 8.60 s | 8.04 / 8.31 s | 201.9 ms | 30/30 | 64.7 / 399.1 ms | 59.5 / 369.7 ms |
| 2 | pool | 4.95 / 6.32 s | 4.76 / 6.13 s | 203.9 ms | 30/30 | 67.1 / 379.8 ms | 59.5 / 369.7 ms |
| 5 | seq | 8.31 / 8.62 s | 8.08 / 8.40 s | 221.0 ms | 30/30 | 64.5 / 177.0 ms | 63.5 / 121.9 ms |
| 5 | pool | 5.04 / 5.99 s | 4.81 / 5.71 s | 220.0 ms | 30/30 | 62.0 / 133.4 ms | 63.5 / 121.9 ms |
| 10 | seq | 8.38 / 8.59 s | 8.16 / 8.38 s | 221.9 ms | 30/30 | 58.7 / 247.5 ms | 56.1 / 97.6 ms |
| 10 | pool | 5.02 / 5.78 s | 4.80 / 5.56 s | 234.1 ms | 30/30 | 57.8 / 111.1 ms | 56.1 / 97.6 ms |

Full data: `pop-os-2026-11-24-hidebudget.json`.
