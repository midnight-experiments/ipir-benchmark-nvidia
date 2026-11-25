# IPIR+SP GPU benchmark: pop-os, 2026-11-25

GPU NVIDIA GeForce RTX 2060, driver 580.159.03, 6144 MiB (459 used at start); WebGPU adapter "4318 7944 NVIDIA GeForce RTX 2060" with timestamp queries. CPU ?, 8 cores / 16 threads, ? GiB RAM. Pop!_OS 24.04 LTS 7.0.11-76070011-generic, Deno 2.7.5, rustc 1.92.0 (ded5c06cf 2025-12-08). Load average at start 3.94 5.96 5.97, at end 2.99 4.21 4.49. Run took 40 min.

## 1. `fetch` on the GPU

Median of 9 calls after a warm-up. Wall: from the call to the answer in host memory. Exec: the GPU's own time (timestamp queries).

| rows × cols | resident | upload | call of 1: wall (exec) | call of 32: wall (exec) |
|---|---|---|---|---|
| 2048 × 2048 | 40 MiB | 0.02 s | 12.6 ms (0.40 ms) | 15.4 ms (2.17 ms) |
| 2048 × 4096 | 80 MiB | 0.08 s | 12.5 ms (0.50 ms) | 18.0 ms (4.11 ms) |
| 2048 × 8192 | 160 MiB | 0.08 s | 12.9 ms (0.80 ms) | 21.8 ms (7.84 ms) |
| 4096 × 8192 | 192 MiB | 0.08 s | 13.4 ms (0.91 ms) | 24.0 ms (9.71 ms) |
| 4096 × 12288 | 288 MiB | 0.11 s | 13.5 ms (1.17 ms) | 28.6 ms (14.12 ms) |
| 6144 × 16384 | 448 MiB | 0.13 s | 14.2 ms (1.85 ms) | 36.6 ms (22.30 ms) |
| 6144 × 28672 | 784 MiB | 0.20 s | 15.3 ms (2.95 ms) | 53.9 ms (33.30 ms) |
| 8192 × 20480 | 640 MiB | 0.16 s | 15.0 ms (2.52 ms) | 41.5 ms (25.78 ms) |
| 8192 × 22528 | 704 MiB | 0.17 s | 15.1 ms (2.69 ms) | 50.0 ms (29.52 ms) |
| 10240 × 22528 | 792 MiB | 0.20 s | 15.2 ms (3.06 ms) | 53.3 ms (35.29 ms) |
| 12288 × 26624 | 1040 MiB | 0.41 s | 16.7 ms (3.84 ms) | 65.7 ms (43.64 ms) |
| 14336 × 34816 | 1496 MiB | 0.56 s | 17.7 ms (5.20 ms) | 88.2 ms (63.66 ms) |
| 16384 × 18432 | 864 MiB | 0.33 s | 15.7 ms (3.23 ms) | 57.3 ms (35.80 ms) |
| 16384 × 49152 | 2304 MiB | 0.85 s | 20.3 ms (7.91 ms) | 122.5 ms (100.63 ms) |
| 20480 × 40960 | 2240 MiB | 0.73 s | 20.3 ms (7.58 ms) | 136.1 ms (96.13 ms) |
| 32768 × 32768 (largest) | 2560 MiB | 1.09 s | 22.0 ms (8.82 ms) | 148.9 ms (109.50 ms) |

## 2. `compile` one column block on the CPU

Whole build of a one-block table (2,048 columns): hint columns + packing matrix. Median of the repetitions.

| rows | at once | wall | per table: build | hints | packing |
|---|---|---|---|---|---|
| 2048 | 1 | 389.1 ms | 385.0 ms | 31.4 ms | 350.5 ms |
| 8192 | 1 | 431.4 ms | 421.3 ms | 73.8 ms | 327.6 ms |
| 16384 | 1 | 490.2 ms | 479.3 ms | 130.3 ms | 328.1 ms |
| 32768 | 1 | 771.4 ms | 748.8 ms | 359.5 ms | 342.0 ms |
| 2048 | 4 | 1249.3 ms | 935.1 ms | 123.0 ms | 794.8 ms |
| 8192 | 4 | 1415.4 ms | 1212.4 ms | 290.4 ms | 871.7 ms |
| 16384 | 4 | 1680.3 ms | 1438.1 ms | 511.6 ms | 878.9 ms |
| 32768 | 4 | 2484.6 ms | 2416.4 ms | 1398.1 ms | 963.3 ms |

## 3. The 3 s block budget, simulated

Per chain block: compile one changed column block of each of six tables on the CPU, then upload the six to the GPU, while the GPU answers calls of 32 at a steady rate. The compile recomputes each block's whole hint columns: an upper bound, since hint updates for new records are not built. Latency: from the moment a call is due to its answer; "during" = calls overlapping a block's work.

### Shapes: today's starting shapes

read 2048 × 2048, nullifier 2048 × 2048, shielded groups 2048 × 4096, Dust groups 6144 × 28672, first coins 2048 × 2048, generation groups 2048 × 4096; 1.04 GiB resident.

Pool: 8 threads (linux: thread nice 19); 30 blocks every 6 s.

| calls/s | mode | block work p50 / p99 | compile p50 / p99 | upload p50 | over 3 s | answer p50 / p99 during work | answer p50 / p99 no block work |
|---|---|---|---|---|---|---|---|
| 2 | seq | 2.36 / 2.48 s | 2.18 / 2.31 s | 174.5 ms | 0/30 | 22.0 / 422.0 ms | 18.1 / 320.1 ms |
| 2 | pool | 1.93 / 2.06 s | 1.73 / 1.89 s | 184.4 ms | 0/30 | 31.7 / 397.1 ms | 18.1 / 320.1 ms |
| 5 | seq | 2.33 / 2.39 s | 2.16 / 2.21 s | 165.5 ms | 0/30 | 25.0 / 255.5 ms | 20.8 / 251.6 ms |
| 5 | pool | 1.73 / 2.23 s | 1.55 / 2.04 s | 187.5 ms | 0/30 | 21.8 / 262.4 ms | 20.8 / 251.6 ms |
| 10 | seq | 2.36 / 2.51 s | 2.19 / 2.33 s | 166.6 ms | 0/30 | 22.5 / 232.1 ms | 18.4 / 244.4 ms |
| 10 | pool | 1.78 / 1.95 s | 1.60 / 1.79 s | 181.0 ms | 0/30 | 22.4 / 231.2 ms | 18.4 / 244.4 ms |

### Shapes: 10 million outputs

read 14336 × 34816, nullifier 16384 × 18432, shielded groups 8192 × 20480, Dust groups 6144 × 28672, first coins 2048 × 2048, generation groups 2048 × 4096; 3.81 GiB resident.

Pool: 8 threads (linux: thread nice 19); 30 blocks every 6 s.

| calls/s | mode | block work p50 / p99 | compile p50 / p99 | upload p50 | over 3 s | answer p50 / p99 during work | answer p50 / p99 no block work |
|---|---|---|---|---|---|---|---|
| 2 | seq | 2.94 / 3.23 s | 2.54 / 3.01 s | 387.1 ms | 6/30 | 74.5 / 379.5 ms | 68.8 / 376.9 ms |
| 2 | pool | 2.06 / 2.30 s | 1.86 / 2.07 s | 204.3 ms | 0/30 | 71.4 / 386.6 ms | 68.8 / 376.9 ms |
| 5 | seq | 2.77 / 2.81 s | 2.57 / 2.59 s | 202.8 ms | 0/30 | 70.3 / 153.4 ms | 64.7 / 138.4 ms |
| 5 | pool | 2.00 / 2.27 s | 1.82 / 2.09 s | 187.5 ms | 0/30 | 66.8 / 133.1 ms | 64.7 / 138.4 ms |
| 10 | seq | 2.77 / 2.84 s | 2.57 / 2.67 s | 201.1 ms | 0/30 | 62.0 / 160.2 ms | 57.6 / 97.8 ms |
| 10 | pool | 2.09 / 2.47 s | 1.87 / 2.21 s | 226.4 ms | 0/30 | 59.6 / 155.7 ms | 57.6 / 97.8 ms |

Full data: `pop-os-2026-11-25-hidebudget-opt.json`.
