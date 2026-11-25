# IPIR+SP benchmark: rtx4090, 2026-11-25, backend `nvidia`

Answers on the GPU (`GpuIpirTable`, WebGPU). Resizable BAR: on (BAR1 32768 MiB, framebuffer 24564 MiB). Memory-budget layer: not loaded.

GPU NVIDIA GeForce RTX 4090, driver 580.126.18, 24564 MiB (21466 used at start); WebGPU adapter "4318 9860 NVIDIA GeForce RTX 4090" with timestamp queries. CPU AMD Ryzen 7 7700X 8-Core Processor, 8 cores / 16 threads, 62 GiB RAM. Pop!_OS 24.04 LTS 6.18.7-76061807-generic, Deno 2.7.5, rustc 1.92.0 (ded5c06cf 2025-12-08). Load average at start 21.38 18.37 9.84, at end 7.95 14.67 10.28. Run took 4 min.

## 1. `fetch` on the GPU

Median of 9 calls after a warm-up. Wall: from the call to the answer in host memory. Exec: the GPU's own time (timestamp queries).

| rows × cols | resident | upload | call of 1: wall (exec) | call of 32: wall (exec) |
|---|---|---|---|---|
| 2048 × 2048 | 40 MiB | 0.02 s | 13.1 ms (0.18 ms) | 14.9 ms (0.33 ms) |
| 2048 × 4096 | 80 MiB | 0.05 s | 13.1 ms (0.27 ms) | 15.3 ms (0.61 ms) |
| 2048 × 8192 | 160 MiB | 0.06 s | 12.6 ms (0.33 ms) | 14.3 ms (1.12 ms) |
| 4096 × 8192 | 192 MiB | 0.06 s | 12.4 ms (0.37 ms) | 14.7 ms (1.39 ms) |
| 4096 × 12288 | 288 MiB | 0.08 s | 12.5 ms (0.46 ms) | 16.0 ms (2.13 ms) |
| 6144 × 16384 | 448 MiB | 0.11 s | 12.6 ms (0.65 ms) | 18.0 ms (3.48 ms) |
| 6144 × 28672 | 784 MiB | 0.16 s | 12.9 ms (1.04 ms) | 20.7 ms (5.71 ms) |
| 8192 × 20480 | 640 MiB | 0.14 s | 12.7 ms (0.90 ms) | 18.9 ms (4.77 ms) |
| 8192 × 22528 | 704 MiB | 0.15 s | 12.8 ms (0.97 ms) | 20.6 ms (5.41 ms) |
| 10240 × 22528 | 792 MiB | 0.16 s | 13.0 ms (1.10 ms) | 20.2 ms (6.17 ms) |
| 12288 × 26624 | 1040 MiB | 0.21 s | 13.2 ms (1.31 ms) | 23.3 ms (7.95 ms) |
| 14336 × 34816 | 1496 MiB | 0.42 s | 38.7 ms (26.51 ms) | 28.3 ms (11.46 ms) |
| 16384 × 18432 | 864 MiB | 0.18 s | 13.1 ms (1.14 ms) | 22.4 ms (6.80 ms) |
| 16384 × 49152 | 2304 MiB | — | does not fit: uploading the table: GPU out of memory: not enough memory left | |
| 20480 × 40960 | 2240 MiB | — | failed: validation error occurred | |
| 32768 × 32768 (largest) | 2560 MiB | — | failed: validation error occurred | |
| 32768 × 30720 (largest) | 2400 MiB | — | does not fit: uploading the table: GPU out of memory: not enough memory left | |
| 32768 × 28672 (largest) | 2240 MiB | — | failed: validation error occurred | |
| 32768 × 26624 (largest) | 2080 MiB | — | does not fit: answering calls of 32: GPU out of memory: not enough memory left | |
| 32768 × 24576 (largest) | 1920 MiB | 0.62 s | 46.6 ms (33.50 ms) | 38.1 ms (14.93 ms) |

## 2. `compile` one column block on the CPU

Whole build of a one-block table (2,048 columns): hint columns + packing matrix. Median of the repetitions.

| build | rows | at once | wall | per table: build | hints | packing |
|---|---|---|---|---|---|---|
| stock crates | 2048 | 1 | 642.5 ms | 637.0 ms | 44.2 ms | 586.8 ms |
| stock crates | 8192 | 1 | 658.8 ms | 650.8 ms | 112.7 ms | 529.5 ms |
| stock crates | 16384 | 1 | 751.5 ms | 743.4 ms | 204.9 ms | 524.7 ms |
| stock crates | 32768 | 1 | 1192.7 ms | 1180.8 ms | 589.8 ms | 554.8 ms |
| stock crates | 2048 | 4 | 1529.7 ms | 1441.2 ms | 176.3 ms | 1257.4 ms |
| stock crates | 8192 | 4 | 1736.5 ms | 1487.0 ms | 453.6 ms | 1024.9 ms |
| stock crates | 16384 | 4 | 2436.3 ms | 2047.3 ms | 831.2 ms | 1180.0 ms |
| stock crates | 32768 | 4 | 5233.1 ms | 4676.6 ms | 3041.6 ms | 1622.9 ms |
| with patch 0002 | 2048 | 1 | 307.3 ms | 307.1 ms | 27.1 ms | 276.8 ms |
| with patch 0002 | 8192 | 1 | 349.4 ms | 346.4 ms | 63.0 ms | 275.7 ms |
| with patch 0002 | 16384 | 1 | 397.9 ms | 389.0 ms | 111.0 ms | 261.4 ms |
| with patch 0002 | 32768 | 1 | 641.8 ms | 633.4 ms | 310.2 ms | 279.8 ms |
| with patch 0002 | 2048 | 4 | 1139.3 ms | 938.4 ms | 108.3 ms | 799.9 ms |
| with patch 0002 | 8192 | 4 | 1278.4 ms | 1152.9 ms | 239.5 ms | 875.0 ms |
| with patch 0002 | 16384 | 4 | 1392.2 ms | 1342.2 ms | 407.2 ms | 890.5 ms |
| with patch 0002 | 32768 | 4 | 2193.0 ms | 1984.3 ms | 1126.3 ms | 783.4 ms |

## 2b. `compile` one column block on the GPU

The same work on the GPU (WebGPU kernels, `gpu/block-build.ts`). Output against the CPU's (`pack_digest`'s SHA-256 over every output word): **24 of 24 blocks equal**.

Wall: from writing the block's cells to the build's end. GPU: the GPU's own time (timestamp queries), all stages and each alone. Median of the repetitions.

| rows | at once | wall | GPU | hints | aggregation | chains | compile | GPU memory besides the cells |
|---|---|---|---|---|---|---|---|---|
| 2048 | 1 | 38.7 ms | 26.4 ms | 0.36 ms | 0.51 ms | 25.25 ms | 0.31 ms | 256 MiB |
| 2048 | 4 | 43.9 ms | 30.2 ms | | | | | 641 MiB |
| 2048 | 6 | 51.8 ms | 32.7 ms | | | | | 898 MiB |
| 8192 | 1 | 41.3 ms | 26.9 ms | 0.87 ms | 0.51 ms | 25.24 ms | 0.28 ms | 256 MiB |
| 8192 | 4 | 55.0 ms | 32.3 ms | | | | | 642 MiB |
| 8192 | 6 | 90.2 ms | 49.1 ms | | | | | 898 MiB |
| 16384 | 1 | 45.4 ms | 27.6 ms | 1.56 ms | 0.51 ms | 25.24 ms | 0.29 ms | 257 MiB |
| 16384 | 4 | 87.3 ms | 48.2 ms | | | | | 642 MiB |
| 16384 | 6 | 122.7 ms | 42.5 ms | | | | | 900 MiB |
| 32768 | 1 | 51.7 ms | 29.0 ms | 2.93 ms | 0.53 ms | 25.24 ms | 0.29 ms | 257 MiB |
| 32768 | 4 | 141.1 ms | 43.2 ms | | | | | 644 MiB |
| 32768 | 6 | failed: validation error occurred | | | | | | |

One chain block: one column block of each of the six tables, built at once (all chains in one dispatch).

| shapes | rows of the six tables | wall | GPU | GPU memory besides the cells |
|---|---|---|---|---|
| today's starting shapes | 2048, 2048, 2048, 6144, 2048, 2048 | 50.5 ms | 33.1 ms | 898 MiB |
| 10 million outputs | 14336, 16384, 8192, 6144, 2048, 2048 | 70.3 ms | 35.8 ms | 898 MiB |

## 3. The 3 s block budget, simulated

Per chain block: compile one changed column block of each of six tables on the CPU, then upload the six to the GPU, while the GPU answers calls of 32 at a steady rate. The compile recomputes each block's whole hint columns: an upper bound, since hint updates for new records are not built. "Blocks built on the GPU": the six blocks are built on the GPU that is answering instead (compile = their cells written and built in one submission; upload = their layout for the answer from the build's GPU buffers). Latency: from the moment a call is due to its answer; "during" = calls overlapping a block's work.

Full data: `rtx4090-2026-11-25-nvidia-nobudget.json`.
