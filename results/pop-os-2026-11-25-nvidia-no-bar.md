# IPIR+SP benchmark: pop-os, 2026-11-25, backend `nvidia-no-bar`

Answers on the GPU (`GpuIpirTable`, WebGPU). Resizable BAR: off (BAR1 256 MiB, framebuffer 6144 MiB). Memory-budget layer: loaded.

GPU NVIDIA GeForce RTX 2060, driver 580.159.03, 6144 MiB (538 used at start); WebGPU adapter "4318 7944 NVIDIA GeForce RTX 2060" with timestamp queries. CPU AMD Ryzen 7 2700X Eight-Core Processor, 8 cores / 16 threads, 31.3 GiB RAM. Pop!_OS 24.04 LTS 7.0.11-76070011-generic, Deno 2.7.5, rustc 1.92.0 (ded5c06cf 2025-12-08). Load average at start 6.30 2.84 2.20, at end 2.85 3.50 4.87. Run took 99 min.

## 1. `fetch` on the GPU

Median of 9 calls after a warm-up. Wall: from the call to the answer in host memory. Exec: the GPU's own time (timestamp queries).

| rows × cols | resident | upload | call of 1: wall (exec) | call of 32: wall (exec) |
|---|---|---|---|---|
| 2048 × 2048 | 40 MiB | 0.02 s | 12.4 ms (0.31 ms) | 15.5 ms (1.63 ms) |
| 2048 × 4096 | 80 MiB | 0.07 s | 12.5 ms (0.50 ms) | 18.8 ms (4.16 ms) |
| 2048 × 8192 | 160 MiB | 0.08 s | 13.1 ms (0.80 ms) | 22.0 ms (7.79 ms) |
| 4096 × 8192 | 192 MiB | 0.09 s | 13.4 ms (0.90 ms) | 25.3 ms (9.70 ms) |
| 4096 × 12288 | 288 MiB | 0.11 s | 13.4 ms (1.17 ms) | 29.4 ms (14.24 ms) |
| 6144 × 16384 | 448 MiB | 0.13 s | 14.5 ms (1.85 ms) | 37.1 ms (22.59 ms) |
| 6144 × 28672 | 784 MiB | 0.20 s | 15.3 ms (2.95 ms) | 56.6 ms (34.39 ms) |
| 8192 × 20480 | 640 MiB | 0.16 s | 14.9 ms (2.53 ms) | 46.6 ms (26.43 ms) |
| 8192 × 22528 | 704 MiB | 0.17 s | 15.1 ms (2.69 ms) | 48.1 ms (33.28 ms) |
| 10240 × 22528 | 792 MiB | 0.20 s | 15.7 ms (3.07 ms) | 52.3 ms (32.64 ms) |
| 12288 × 26624 | 1040 MiB | 0.41 s | 18.4 ms (5.49 ms) | 65.9 ms (44.55 ms) |
| 14336 × 34816 | 1496 MiB | 0.57 s | 18.0 ms (5.22 ms) | 88.5 ms (64.05 ms) |
| 16384 × 18432 | 864 MiB | 0.43 s | 17.7 ms (4.78 ms) | 54.4 ms (37.61 ms) |
| 16384 × 49152 | 2304 MiB | 1.02 s | 20.5 ms (7.93 ms) | 126.0 ms (100.93 ms) |
| 20480 × 40960 | 2240 MiB | 0.77 s | 20.1 ms (7.61 ms) | 132.6 ms (96.42 ms) |
| 32768 × 32768 (largest) | 2560 MiB | 1.02 s | 21.9 ms (8.86 ms) | 144.6 ms (109.66 ms) |

## 2. `compile` one column block on the CPU

Whole build of a one-block table (2,048 columns): hint columns + packing matrix. Median of the repetitions.

| build | rows | at once | wall | per table: build | hints | packing |
|---|---|---|---|---|---|---|
| stock crates | 2048 | 1 | 1174.1 ms | 1166.3 ms | 85.3 ms | 1078.9 ms |
| stock crates | 8192 | 1 | 1283.2 ms | 1271.6 ms | 223.0 ms | 1040.9 ms |
| stock crates | 16384 | 1 | 1454.9 ms | 1440.8 ms | 412.9 ms | 1004.0 ms |
| stock crates | 32768 | 1 | 2208.6 ms | 2187.9 ms | 1168.5 ms | 1009.4 ms |
| stock crates | 2048 | 4 | 2510.2 ms | 2140.0 ms | 340.8 ms | 1794.6 ms |
| stock crates | 8192 | 4 | 3256.2 ms | 2807.7 ms | 921.0 ms | 1871.7 ms |
| stock crates | 16384 | 4 | 4841.3 ms | 4299.3 ms | 2219.7 ms | 2100.0 ms |
| stock crates | 32768 | 4 | 7667.9 ms | 7280.8 ms | 5309.6 ms | 1879.2 ms |
| with patch 0002 | 2048 | 1 | 397.1 ms | 392.3 ms | 33.6 ms | 356.4 ms |
| with patch 0002 | 8192 | 1 | 441.3 ms | 433.2 ms | 79.9 ms | 344.0 ms |
| with patch 0002 | 16384 | 1 | 526.0 ms | 511.8 ms | 142.0 ms | 344.1 ms |
| with patch 0002 | 32768 | 1 | 806.1 ms | 782.9 ms | 406.0 ms | 338.4 ms |
| with patch 0002 | 2048 | 4 | 1385.2 ms | 1235.1 ms | 136.7 ms | 1097.3 ms |
| with patch 0002 | 8192 | 4 | 1517.2 ms | 1308.6 ms | 318.4 ms | 976.9 ms |
| with patch 0002 | 16384 | 4 | 1814.0 ms | 1441.1 ms | 541.4 ms | 886.4 ms |
| with patch 0002 | 32768 | 4 | 2812.7 ms | 2694.5 ms | 1619.9 ms | 1038.0 ms |

## 2b. `compile` one column block on the GPU

The same work on the GPU (WebGPU kernels, `gpu/block-build.ts`). Output against the CPU's (`pack_digest`'s SHA-256 over every output word): **24 of 24 blocks equal**.

Wall: from writing the block's cells to the build's end. GPU: the GPU's own time (timestamp queries), all stages and each alone. Median of the repetitions.

| rows | at once | wall | GPU | hints | aggregation | chains | compile | GPU memory besides the cells |
|---|---|---|---|---|---|---|---|---|
| 2048 | 1 | 56.9 ms | 43.7 ms | 2.05 ms | 3.24 ms | 35.59 ms | 2.72 ms | 256 MiB |
| 2048 | 4 | 86.4 ms | 68.4 ms | | | | | 641 MiB |
| 2048 | 6 | 106.0 ms | 85.1 ms | | | | | 898 MiB |
| 8192 | 1 | 64.5 ms | 47.0 ms | 5.18 ms | 3.23 ms | 35.84 ms | 2.72 ms | 256 MiB |
| 8192 | 4 | 115.5 ms | 80.9 ms | | | | | 642 MiB |
| 8192 | 6 | 149.5 ms | 103.8 ms | | | | | 898 MiB |
| 16384 | 1 | 74.1 ms | 51.0 ms | 9.25 ms | 3.25 ms | 35.83 ms | 2.72 ms | 257 MiB |
| 16384 | 4 | 263.5 ms | 98.5 ms | | | | | 642 MiB |
| 16384 | 6 | 404.9 ms | 129.1 ms | | | | | 900 MiB |
| 32768 | 1 | 95.5 ms | 59.9 ms | 17.78 ms | 3.24 ms | 35.81 ms | 2.72 ms | 257 MiB |
| 32768 | 4 | 533.4 ms | 131.7 ms | | | | | 644 MiB |
| 32768 | 6 | 785.8 ms | 179.8 ms | | | | | 902 MiB |

One chain block: one column block of each of the six tables, built at once (all chains in one dispatch). "Back to back": one build right after another. "After idle": the GPU left idle before each build, as between chain blocks; a GPU that lowers its clocks when idle is slower then.

| shapes | rows of the six tables | back to back: wall | GPU | after idle: wall | GPU | GPU memory besides the cells |
|---|---|---|---|---|---|---|
| today's starting shapes | 2048, 2048, 2048, 6144, 2048, 2048 | 111.5 ms | 87.3 ms | 391.8 ms (6 s idle) | 340.8 ms | 898 MiB |
| 10 million outputs | 14336, 16384, 8192, 6144, 2048, 2048 | 149.8 ms | 103.5 ms | 422.8 ms (6 s idle) | 320.3 ms | 898 MiB |

## 3. The block budget, simulated: a block's work in the 6 s between blocks

The goal is a block's work under 1 s; 3 s is the ceiling. "Of the 6 s": the share of the time between two blocks that the work takes (p50).

Per chain block: compile one changed column block of each of six tables on the CPU, then upload the six to the GPU, while the GPU answers calls of 32 at a steady rate. The compile recomputes each block's whole hint columns: an upper bound, since hint updates for new records are not built. "Blocks built on the GPU": the six blocks are built on the GPU that is answering instead (compile = their cells written and built in one submission; upload = their layout for the answer from the build's GPU buffers). Latency: from the moment a call is due to its answer; "during" = calls overlapping a block's work.

### Shapes: today's starting shapes, stock crates

read 2048 × 2048, nullifier 2048 × 2048, shielded groups 2048 × 4096, Dust groups 6144 × 28672, first coins 2048 × 2048, generation groups 2048 × 4096; 1.04 GiB resident.

Pool: 8 threads (linux: thread nice 19); 30 blocks every 6 s.

| calls/s | mode | block work p50 / p99 | of the 6 s | compile p50 / p99 | upload p50 | over 1 s | over 3 s | answer p50 / p99 during work | answer p50 / p99 no block work | calls dropped |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | seq | 7.02 / 7.70 s | 117.0 % | 6.80 / 7.47 s | 202.0 ms | 30/30 | 30/30 | 24.6 / 377.8 ms | 18.0 / 305.5 ms | 0 of 429 |
| 2 | pool | 3.80 / 4.93 s | 63.3 % | 3.57 / 4.75 s | 234.5 ms | 30/30 | 30/30 | 24.2 / 339.8 ms | 18.0 / 305.5 ms | 0 of 359 |
| 5 | seq | 7.04 / 7.70 s | 117.3 % | 6.85 / 7.51 s | 182.3 ms | 30/30 | 30/30 | 21.6 / 260.5 ms | 19.6 / 260.4 ms | 0 of 1071 |
| 5 | pool | 3.89 / 5.61 s | 64.8 % | 3.70 / 5.22 s | 186.0 ms | 30/30 | 30/30 | 22.2 / 263.3 ms | 19.6 / 260.4 ms | 0 of 898 |
| 10 | seq | 7.16 / 7.71 s | 119.3 % | 6.95 / 7.38 s | 183.4 ms | 30/30 | 30/30 | 20.9 / 218.1 ms | 18.5 / 194.4 ms | 0 of 2177 |
| 10 | pool | 3.83 / 4.87 s | 63.8 % | 3.65 / 4.69 s | 180.3 ms | 30/30 | 30/30 | 21.2 / 215.5 ms | 18.5 / 194.4 ms | 0 of 1798 |

### Shapes: 10 million outputs, stock crates

read 14336 × 34816, nullifier 16384 × 18432, shielded groups 8192 × 20480, Dust groups 6144 × 28672, first coins 2048 × 2048, generation groups 2048 × 4096; 3.81 GiB resident.

Pool: 8 threads (linux: thread nice 19); 30 blocks every 6 s.

| calls/s | mode | block work p50 / p99 | of the 6 s | compile p50 / p99 | upload p50 | over 1 s | over 3 s | answer p50 / p99 during work | answer p50 / p99 no block work | calls dropped |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | seq | 8.32 / 8.89 s | 138.7 % | 8.08 / 8.66 s | 210.9 ms | 30/30 | 30/30 | 66.8 / 377.1 ms | 67.2 / 377.1 ms | 0 of 507 |
| 2 | pool | 5.01 / 6.23 s | 83.6 % | 4.82 / 6.03 s | 207.2 ms | 30/30 | 30/30 | 66.9 / 332.0 ms | 67.2 / 377.1 ms | 0 of 363 |
| 5 | seq | 8.46 / 9.02 s | 141.0 % | 8.23 / 8.79 s | 222.0 ms | 30/30 | 30/30 | 64.1 / 185.5 ms | 64.2 / 133.9 ms | 0 of 1284 |
| 5 | pool | 5.01 / 6.58 s | 83.5 % | 4.69 / 6.35 s | 212.7 ms | 30/30 | 30/30 | 62.7 / 129.7 ms | 64.2 / 133.9 ms | 0 of 906 |
| 10 | seq | 8.29 / 8.98 s | 138.2 % | 8.06 / 8.77 s | 223.5 ms | 30/30 | 30/30 | 59.0 / 203.1 ms | 56.9 / 93.9 ms | 0 of 2521 |
| 10 | pool | 5.01 / 5.70 s | 83.6 % | 4.70 / 5.47 s | 222.8 ms | 30/30 | 30/30 | 57.5 / 110.6 ms | 56.9 / 93.9 ms | 0 of 1807 |

### Shapes: today's starting shapes, with patch 0002

read 2048 × 2048, nullifier 2048 × 2048, shielded groups 2048 × 4096, Dust groups 6144 × 28672, first coins 2048 × 2048, generation groups 2048 × 4096; 1.04 GiB resident.

Pool: 8 threads (linux: thread nice 19); 30 blocks every 6 s.

| calls/s | mode | block work p50 / p99 | of the 6 s | compile p50 / p99 | upload p50 | over 1 s | over 3 s | answer p50 / p99 during work | answer p50 / p99 no block work | calls dropped |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | seq | 2.51 / 2.58 s | 41.8 % | 2.33 / 2.41 s | 174.3 ms | 30/30 | 0/30 | 28.3 / 357.9 ms | 22.5 / 365.3 ms | 0 of 357 |
| 2 | pool | 1.94 / 2.11 s | 32.3 % | 1.74 / 1.94 s | 183.2 ms | 30/30 | 0/30 | 33.1 / 371.4 ms | 22.5 / 365.3 ms | 0 of 356 |
| 5 | seq | 2.50 / 2.64 s | 41.7 % | 2.32 / 2.46 s | 171.0 ms | 30/30 | 0/30 | 25.8 / 268.1 ms | 19.0 / 241.0 ms | 0 of 893 |
| 5 | pool | 1.88 / 2.22 s | 31.4 % | 1.61 / 2.04 s | 199.3 ms | 30/30 | 0/30 | 24.5 / 263.8 ms | 19.0 / 241.0 ms | 0 of 889 |
| 10 | seq | 2.49 / 2.56 s | 41.5 % | 2.30 / 2.40 s | 171.9 ms | 30/30 | 0/30 | 23.5 / 211.3 ms | 18.6 / 195.7 ms | 0 of 1786 |
| 10 | pool | 1.81 / 2.09 s | 30.1 % | 1.63 / 1.90 s | 174.6 ms | 30/30 | 0/30 | 22.8 / 170.3 ms | 18.6 / 195.7 ms | 0 of 1778 |

### Shapes: 10 million outputs, with patch 0002

read 14336 × 34816, nullifier 16384 × 18432, shielded groups 8192 × 20480, Dust groups 6144 × 28672, first coins 2048 × 2048, generation groups 2048 × 4096; 3.81 GiB resident.

Pool: 8 threads (linux: thread nice 19); 30 blocks every 6 s.

| calls/s | mode | block work p50 / p99 | of the 6 s | compile p50 / p99 | upload p50 | over 1 s | over 3 s | answer p50 / p99 during work | answer p50 / p99 no block work | calls dropped |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | seq | 3.15 / 3.37 s | 52.5 % | 2.79 / 3.14 s | 230.9 ms | 30/30 | 28/30 | 78.3 / 375.0 ms | 62.3 / 359.1 ms | 0 of 359 |
| 2 | pool | 2.18 / 2.94 s | 36.4 % | 1.98 / 2.57 s | 210.2 ms | 30/30 | 0/30 | 73.7 / 383.8 ms | 62.3 / 359.1 ms | 0 of 358 |
| 5 | seq | 2.97 / 3.11 s | 49.6 % | 2.76 / 2.90 s | 215.5 ms | 30/30 | 7/30 | 68.4 / 167.7 ms | 62.9 / 136.4 ms | 0 of 896 |
| 5 | pool | 2.06 / 2.50 s | 34.3 % | 1.87 / 2.21 s | 189.1 ms | 30/30 | 0/30 | 66.4 / 135.0 ms | 62.9 / 136.4 ms | 0 of 892 |
| 10 | seq | 3.01 / 3.17 s | 50.1 % | 2.77 / 2.88 s | 233.5 ms | 30/30 | 18/30 | 64.2 / 176.7 ms | 56.2 / 96.8 ms | 0 of 1793 |
| 10 | pool | 2.14 / 2.60 s | 35.6 % | 1.92 / 2.32 s | 220.4 ms | 30/30 | 0/30 | 60.7 / 151.9 ms | 56.2 / 96.8 ms | 0 of 1784 |

### Shapes: today's starting shapes, blocks built on the GPU

read 2048 × 2048, nullifier 2048 × 2048, shielded groups 2048 × 4096, Dust groups 6144 × 28672, first coins 2048 × 2048, generation groups 2048 × 4096; 1.04 GiB resident.

30 blocks every 6 s; the build holds 802 MiB of GPU memory.

| calls/s | mode | block work p50 / p99 | of the 6 s | compile p50 / p99 | upload p50 | over 1 s | over 3 s | answer p50 / p99 during work | answer p50 / p99 no block work | calls dropped |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | gpu | 0.28 / 0.49 s | 4.7 % | 0.13 / 0.35 s | 141.6 ms | 0/30 | 0/30 | 133.5 / 355.2 ms | 21.3 / 323.1 ms | 0 of 353 |
| 5 | gpu | 0.27 / 0.40 s | 4.4 % | 0.13 / 0.25 s | 135.6 ms | 0/30 | 0/30 | 69.5 / 258.9 ms | 19.4 / 244.9 ms | 0 of 882 |
| 10 | gpu | 0.40 / 0.55 s | 6.6 % | 0.22 / 0.43 s | 134.4 ms | 0/30 | 0/30 | 64.4 / 390.7 ms | 19.0 / 195.9 ms | 0 of 1766 |

### Shapes: 10 million outputs, blocks built on the GPU

read 14336 × 34816, nullifier 16384 × 18432, shielded groups 8192 × 20480, Dust groups 6144 × 28672, first coins 2048 × 2048, generation groups 2048 × 4096; 3.81 GiB resident.

Does not fit in the GPU's memory (at gpu at 2 calls/s): uploading a block: GPU out of memory: not enough memory left

### Shapes: 8 million outputs, the largest tables that fit in this GPU's memory with the build, blocks built on the GPU

read 12288 × 32768, nullifier 8192 × 30720, shielded groups 8192 × 16384, Dust groups 6144 × 28672, first coins 2048 × 2048, generation groups 2048 × 4096; 3.57 GiB resident.

30 blocks every 6 s; the build holds 890 MiB of GPU memory.

| calls/s | mode | block work p50 / p99 | of the 6 s | compile p50 / p99 | upload p50 | over 1 s | over 3 s | answer p50 / p99 during work | answer p50 / p99 no block work | calls dropped |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | gpu | 0.26 / 0.30 s | 4.4 % | 0.16 / 0.22 s | 80.6 ms | 0/30 | 0/30 | 341.8 / 493.6 ms | 51.7 / 335.3 ms | 0 of 354 |
| 5 | gpu | 0.30 / 0.36 s | 5.1 % | 0.21 / 0.25 s | 93.5 ms | 0/30 | 0/30 | 130.2 / 358.6 ms | 57.7 / 119.9 ms | 0 of 883 |
| 10 | gpu | 0.31 / 0.38 s | 5.1 % | 0.20 / 0.29 s | 106.3 ms | 0/30 | 0/30 | 86.1 / 362.6 ms | 46.8 / 88.8 ms | 0 of 1765 |

Full data: `pop-os-2026-11-25-nvidia-no-bar.json`.
