# Other results: the full runs

The standard results are in [`REPORT.md`](../REPORT.md): a table's four operations at 1 million and 10 million
values. This page has the results of the full runs (`./bench.sh` without `--report`, described in
[full-run.md](full-run.md)), which measure more combinations:

- 16 table shapes, from 2048 × 2048 to 32768 × 32768, not the two of the report;
- the six tables at the chain's current size ("today's shapes") and at 10 million outputs;
- the CPU compile with the stock crates and with patch 0002;
- a block's work at 2, 5 and 10 calls a second;
- upstream's CUDA kernel against the WebGPU path at five shapes.

The step numbers are those of [full-run.md](full-run.md).

## The machines

Three machines. Each figure comes from the results file named in the table; `site/` has the same figures in the form
the overview page reads.

| Machine | Backend | GPU | CPU | Results |
|---|---|---|---|---|
| RTX 2060 desktop | `nvidia-no-bar` | GeForce RTX 2060, 6 GB, no resizable BAR (Vulkan layer loaded) | Ryzen 7 2700X, 8 cores / 16 threads | `results/pop-os-2026-11-25-nvidia-no-bar.md`: a full run, on a desktop in use (load average about 3; `results/pop-os-notes.md`) |
| M4 Max | `metal` | Apple M4 Max, 40-core GPU | Apple M4 Max, 16 cores | `results/m4-max-2026-11-25-metal.md`; the GPU build of the blocks (steps 2b and 3) was measured on 2026-11-26 into the same results, with the machine in use (load average 7 at the start) |
| RTX 4090 host | `nvidia` | GeForce RTX 4090, 24 GB, resizable BAR on | Ryzen 7 7700X, 8 cores / 16 threads | `results/rtx4090-2026-11-25-nvidia.md`: a full run on a quiet machine, but with 3 GiB of the card free, so without the 10-million-output tables (`results/rtx4090-notes.md`) |

## Answering a call of 32 requests

Step 1: wall time, and in brackets the GPU's own time, in ms.

| rows × cols | RTX 2060 | M4 Max | RTX 4090 |
|---|---|---|---|
| 2048 × 2048 | 15.5 (1.6) | 16.0 (1.5) | 13.1 (0.3) |
| 6144 × 28672 | 56.6 (34.4) | 45.8 (27.5) | 20.5 (5.7) |
| 14336 × 34816 | 88.5 (64.0) | 76.1 (53.9) | 29.2 (13.4) |
| 32768 × 32768 | 144.6 (109.7) | 123.2 (94.5) | did not fit: 3 GiB of the card were free |

## Compiling one column block

Steps 2 and 2b: wall time in ms.

| | RTX 2060 host | M4 Max | RTX 4090 host |
|---|---|---|---|
| On the CPU, stock crates, 2,048 rows | 1,174 | 337 | 586 |
| On the CPU, patch 0002, 2,048 rows | 397 | 94 | 221 |
| On the CPU, patch 0002, 32,768 rows | 806 | 189 | 478 |
| On the GPU, 2,048 rows | 57 | 60 | 41 |
| On the GPU, 32,768 rows | 96 | 84 | 50 |

The GPU figures are the wall time from the block's cells in host memory to the end of the build. On the RTX 4090, of
the 50 ms at 32,768 rows, 29 ms are the GPU computing (the two chains 25.3 ms at every table height, the hints 0.4
to 2.9 ms, the aggregation 0.5 ms, the matrix compilation 0.3 ms), about 10 ms are writing the 128 MiB of cells to
the GPU, and about 12 ms are this harness's fixed cost of submitting work and noticing its end (the same gap as in a
call's answer). On the RTX 2060 the GPU computes for 44 to 60 ms: the chains 35.7 ms, the hints 2.1 to 17.8 ms, the
aggregation 3.2 ms, the matrix compilation 2.7 ms. On the M4 Max only wall times are given: Metal returned
the GPU's timestamps for few of the builds, and its per-stage times are not consistent.
The GPU build's output equals the CPU's in 24 of 24 checked blocks on each of the three GPUs.

## One chain block: one column block of each of six tables

Steps 2b and 3, at today's shapes and at the
10-million-output shapes. A block arrives every 6 s, and its work is time the server does not have for answering at
full speed. The goal is a block's work under 1 s; 3 s is the ceiling, the most that is tolerable.

The compile of the six column blocks alone, in seconds:

| | RTX 2060 host | M4 Max | RTX 4090 host |
|---|---|---|---|
| On the CPU, stock crates (one after another, p50) | 6.80 / 8.08 | 2.05 / 2.25 | 3.45 / not run |
| On the CPU, patch 0002 (one after another, p50) | 2.33 / 2.79 | 0.63 / 0.68 | 1.35 / not run |
| On the GPU, one build right after another (wall; GPU time in brackets) | 0.11 (0.087) / 0.15 (0.104) | 0.13 / 0.17 (wall only) | 0.05 (0.033) / 0.07 (0.036) |
| On the GPU, 6 s after the last build (wall; GPU time in brackets) | 0.39 (0.34) / 0.42 (0.32) | 0.17 / 0.20 (wall only) | 0.25 (0.21) / 0.32 (0.22) |

A block's whole work (the compile of the six column blocks, then their upload and layout for the answer) while the
GPU answers 2 calls a second: seconds at p50, and in brackets its share of the 6 s between two blocks.

| Block work | RTX 2060 host | M4 Max | RTX 4090 host |
|---|---|---|---|
| CPU compile, stock crates, one after another | 7.02 (117 %) / 8.32 (139 %) | 2.18 (36 %) / 2.39 (40 %) | 3.67 (61 %) / not run |
| CPU compile, stock crates, low-priority pool | 3.80 (63 %) / 5.01 (84 %) | 1.13 (19 %) / 1.64 (27 %) | 2.12 (35 %) / not run |
| CPU compile, patch 0002, one after another | 2.51 (42 %) / 3.15 (52 %) | 0.76 (13 %) / 0.83 (14 %) | 1.61 (27 %) / not run |
| CPU compile, patch 0002, low-priority pool | 1.94 (32 %) / 2.18 (36 %) | 0.54 (9 %) / 0.68 (11 %) | 1.43 (24 %) / not run |
| Blocks built on the GPU that is answering | 0.28 (5 %) / does not fit | 0.26 (4 %) / 0.27 (5 %) | 0.36 (6 %) / not run |
| The same at 8 million outputs, the largest tables the RTX 2060 keeps while it builds | 0.26 (4 %) | | |

- **Under the 1 s goal:** all three GPUs with the blocks built on the GPU (5 % of the 6 s on the RTX 2060, 4 to 5 %
  on the M4 Max, 6 % on the RTX 4090; no block over 1 s), and the M4 Max's CPU with patch 0002 (9 to 14 %; 0 to 4 of
  30 blocks over 1 s). On the M4 Max the GPU build takes the work from 0.54 to 0.26 s at today's shapes and from 0.68
  to 0.27 s at the 10-million-output shapes; there the share rises with the call rate, to 6.0 % at 10 calls/s. **Not
  under it:** every CPU compile on the RTX 2060 host and on the RTX 4090 host (24 % of the 6 s at best), and the M4
  Max with the stock crates.
- **Almost all of it is the compile of the column blocks.** The upload and layout are 2 to 4 % of the 6 s on every
  machine (0.12 to 0.26 s); with the blocks built on the GPU they are 1.7 % on the RTX 4090 (0.10 s), 1.5 % on the
  M4 Max (0.09 s) and 2.4 % on the RTX 2060 (0.14 s), since the matrix does not cross from the host.
- With the stock crates the RTX 2060 host's work does not fit in the 6 s at all when the blocks are compiled one
  after another (117 and 139 %). With patch 0002 it is over the 3 s ceiling in 28 of 30 blocks at the
  10-million-output shapes when compiled one after another, and in none on the pool.
- **A GPU that has been idle builds more slowly.** On the RTX 4090 the same six-table build takes 33 ms of GPU time
  right after another one and 210 ms when the GPU was idle for the 6 s between two chain blocks: the card lowers its
  clocks when idle. A small dispatch every 20 ms during the idle time brings it to 114 ms, not to 33 ms. In the block
  budget, where blocks are 6 s apart, the build takes 0.23 to 0.26 s for that reason; without it the block's work
  would be about 0.15 s, 2.5 % of the 6 s. The driver's power setting (maximum performance) is the likely remedy; it
  was not tried, as it needs rights this run did not have. The M4 Max slows less (0.13 to 0.17 s of wall time at today's shapes). The RTX 2060 shows the same as the RTX 4090 when it is measured alone
  (87 ms of GPU time right after another build, 341 ms after 6 s of idle), but not in its block budget, where the
  build took 0.13 s: there the card was also answering calls that take it tens of milliseconds each, and driving a
  desktop.
- While the GPU builds a block, the calls that arrive wait for it: at 2 calls/s the calls that overlap a block's
  work are answered in 252 ms (p50) on the RTX 4090, against 18 ms otherwise, and in 133 ms on the RTX 2060, against
  22 ms; none is dropped.
- The RTX 4090 run has no 10-million-output budget: another process held 20.7 of the card's 24.5 GiB. On the RTX
  2060 the blocks cannot be built on the GPU with the 10-million-output tables resident: 6 GB are not enough. The
  run then looks for the largest tables that fit with the build: 8 million shielded outputs, where a block's work
  is 0.26 s at 2 calls a second and 0.31 s at 10 (4.4 to 5.1 % of the 6 s), with no block of 90 over 1 s (see
  Memory).

## Memory

On the GPU, a table keeps 2 bytes a cell and 32 MiB a column block (its compiled matrix). The six
tables of the block budget:

| GPU memory | Today's shapes (21 column blocks) | 10-million-output shapes (53 column blocks) |
|---|---|---|
| Resident tables: cells | 392 MiB | 2,208 MiB |
| Resident tables: compiled matrices | 672 MiB | 1,696 MiB |
| Resident tables, total | 1,064 MiB | 3,904 MiB |
| Building one chain block on the GPU: working memory | 898 MiB | 898 MiB |
| Building one chain block on the GPU: the six blocks' cells | 64 MiB | 192 MiB |

- The GPU build's working memory is 128 MiB a table built at once (its hint masks, slots, digits and matrix, 32 MiB
  each) and 128 MiB shared: 256 MiB for one table, 641 MiB for four, 898 MiB for six. The block budget runs it with
  one hint-mask buffer for all six tables, 774 MiB.
- Answering a call needs scratch memory for the time of the call: 13 MiB for a call of 32 at 2048 × 2048, 86 MiB at
  14336 × 34816, 167 MiB at 32768 × 32768 (under 11 MiB for a call of 1).
- **The largest size at which the 6 GB RTX 2060 builds its blocks on the GPU is 8 million shielded outputs.** When
  the 10-million-output tables do not fit with the build, the run goes down in steps of half a million outputs to
  the first size that runs 30 blocks at 2, 5 and 10 calls a second (`bench/gpu-build-sets.ts`; the three Dust tables
  stay the same). On that card, with `nvidia-smi` read every second and the desktop's 470 to 540 MiB counted in:

  | Shielded outputs | Resident tables | With the blocks built on the GPU |
  |---|---|---|
  | 10 million | 3,904 MiB | out of memory in the first blocks at 2 calls/s |
  | 9.5 million | 4,040 MiB | out of memory in the first blocks at 2 calls/s |
  | 9 million | 4,104 MiB | 30 blocks at 2 calls/s, with 5,688 MiB of the card in use; out of memory at 5 calls/s |
  | 8.5 million | 3,960 MiB | 30 blocks at 2 calls/s, with 5,542 MiB in use; out of memory at 5 calls/s |
  | 8 million | 3,656 MiB | fits at all three rates, 5,494 MiB in use at most; block work 0.26, 0.30 and 0.31 s (4.4, 5.1 and 5.1 % of the 6 s) |
  | 7 million (a trial of 5 blocks) | 3,416 MiB | fits, 5,074 MiB in use at most; block work 0.28 s (4.7 %) |

  No sample is above 5,690 MiB of the card's 6,144: beyond that a buffer is refused. Beside the tables the build
  holds about 0.9 GiB, and every call being answered holds scratch memory, so more calls a second need more (the
  reason 8.5 and 9 million run at 2 calls a second and not at 5). The shape rule picks each table's shape on its
  own, so memory does not fall at every step: the 10-million-output tables are smaller than the 9-million ones, but
  their column blocks are larger. Building the six tables three at a time would need about 0.3 GiB less; that is not
  built.

On the host, the CPU compile's peak memory (the whole process, with the block's cells; `/usr/bin/time -l
compile_block` on the M4 Max):

| CPU compile, peak memory | Patch 0002 | Stock crates |
|---|---|---|
| One block, 2,048 rows | 252 MiB | 326 MiB |
| One block, 32,768 rows | 384 MiB | 448 MiB |
| Four blocks at once, 2,048 rows | 662 MiB | 1,042 MiB |
| Four blocks at once, 32,768 rows | 1,172 MiB | 1,305 MiB |

Not measured: the memory of a whole server process, and of the CPU backend's resident tables.

## Upstream's CUDA kernel against the WebGPU path

With `--cuda`, step 4 of a run's summary. Upstream's CUDA kernel
(`simplepir-kernel`, `CudaKernel`) does one part of an answer on the GPU: the first pass, the table times one
request. It takes one request per call, behind a lock, has no packing, and takes a table whole. So an answer with it
is the first pass on the GPU, 32 times for a call of 32, and the packing of every column block on the host's CPU. The
WebGPU path does both steps on the GPU, the 32 requests in one submission. Both were timed on the same table and the
same 32 requests, and every answer of both equals the CPU's, byte for byte. Times in ms, medians of 9 calls of 32.

RTX 4090 host (packing on the Ryzen 7 7700X; 3 GiB of the card were free):

| rows × cols | CUDA first pass, device | Packing on the CPU | CUDA path, whole answer | WebGPU, device | WebGPU, whole answer | WebGPU is faster by |
|---|---|---|---|---|---|---|
| 2048 × 2048 (today's read table) | 0.4 | 4.7 | 10.5 | 0.3 | 12.7 | the CUDA path is 2 ms faster |
| 12288 × 16384 (Dust commitment table) | 13.9 | 271 | 303 | 4.9 | 19.0 | 16 times |
| 6144 × 28672 (the same table before the cap of 8 column blocks) | 12.1 | 497 | 527 | 5.7 | 20.2 | 26 times |
| 14336 × 34816 (read table at 10 million outputs) | 34.0 | 611 | 669 | 11.3 | 27.4 | 24 times |
| 32768 × 32768 | 72.6 | 589 | 699 | did not fit in the 3 GiB free | | |

RTX 2060 desktop (packing on the Ryzen 7 2700X; no resizable BAR):

| rows × cols | CUDA first pass, device | Packing on the CPU | CUDA path, whole answer | WebGPU, device | WebGPU, whole answer | WebGPU is faster by |
|---|---|---|---|---|---|---|
| 2048 × 2048 | 2.9 | 39.8 | 52.8 | 2.2 | 15.8 | 3.3 times |
| 12288 × 16384 | 74.0 | 283 | 384 | 30.1 | 46.7 | 8.2 times |
| 6144 × 28672 | 70.1 | 489 | 585 | 32.9 | 54.0 | 10.8 times |
| 14336 × 34816 | 198 | 594 | 828 | 63.6 | 83.0 | 10.0 times |
| 32768 × 32768 | 368 | 563 | 986 | 110 | 147 | 6.7 times |

"Whole answer" is wall time from the call to the answer's bytes; for the CUDA path it also holds the reading of the
32 requests, the copies to and from the device and the writing of the answers, so it is more than the two columns
before it.

- **On the RTX 4090 the WebGPU path answers a call of 32 16 to 26 times faster than the CUDA path** at every table
  of more than one column block; at the one-block table the two are level (10.5 against 12.7 ms). **On the RTX 2060
  it is 3 to 11 times faster at every shape.**
- **The packing on the CPU is most of the CUDA path**: 34 to 37 ms a column block for the 32 requests on both hosts,
  0.3 to 0.6 s for the tables of more than one block. The CPU alone answers those calls in 0.7 to 3.3 s.
- **The first pass alone is slower too.** The device's time for the 32 first passes of the CUDA kernel is 1.2 to
  3.3 times the device's time for both steps on the WebGPU path: the kernel reads the table once per request, and
  the WebGPU kernels read it once for the 32.
- Upstream's own `cuda_bench` (one request's first pass with a 56-bit prime modulus, 100 requests) gives the same
  first-pass time per request as the comparison: 1.06 ms at 14336 × 34816 and 2.26 ms at 32768 × 32768 on the RTX
  4090, 5.2 and 10.3 ms on the RTX 2060 (`results/rtx4090-2026-11-26-cuda-bench.txt`,
  `results/pop-os-2026-11-26-cuda-bench.txt`).

## The update when a chain block arrives

Every 6 s one column block of each of the six tables changes: its hint
columns and its packing matrix are computed again from its cells, and the new block is put where the answers read
it. Three ways, in seconds at 2 calls a second (p50), with the share of the 6 s in brackets, at today's shapes / at
the 10-million-output shapes:

| The update of one chain block (six column blocks) | RTX 2060 host | RTX 4090 host |
|---|---|---|
| WebGPU, blocks computed on the GPU: computed there, then laid out for the answer | 0.28 (5 %) / does not fit; 0.26 (4 %) at 8 million outputs | 0.36 (6 %) / not run |
| of which the computing | 0.13 / 0.16 at 8 million | 0.26 |
| WebGPU, blocks computed on the CPU (patch 0002, pool): then the six blocks written to the GPU | 1.94 (32 %) / 2.18 (36 %) | 1.43 (24 %) / not run |
| of which the computing on the CPU | 1.74 / 1.98 | 1.20 |
| of which writing the six blocks to the GPU | 0.18 / 0.21 | 0.23 |
| CUDA kernel: blocks computed on the CPU (it has no other way), then the six tables uploaded whole again | about 1.78 (30 %) / 2.21 (37 %), computed | about 1.25 (21 %), computed |
| of which uploading the six tables again | about 0.04 / 0.23 | about 0.05 |

- **Only the WebGPU path has the computing on the GPU**, and that is what brings the update from a quarter or a
  third of the 6 s to 4 to 6 %. The CUDA kernel has no code for it: with that kernel the blocks are computed on the
  CPU, as in the second row.
- The CUDA row is computed, not measured as a set: the CPU computing of the second row, plus the six tables' cells
  (392 MiB at today's shapes, 2,208 MiB at 10 million outputs) at the upload rate measured per table below. The
  kernel has no call for one block, and it drops the table it holds before the new one is up
  (`CudaKernel::try_prepare`), so the table does not answer during its upload.
- Measured per table, a changed block on the device: the CUDA kernel uploads the table again in 36 to 48 ms for 336
  to 384 MiB of cells, 99 to 119 ms for 952 MiB, 212 to 269 ms for 2 GiB (8 to 10 MiB a millisecond). WebGPU writes
  the one block, its cells and its packing matrix, in 20 to 36 ms whatever the table's width (55 ms at 32,768 rows
  on the RTX 2060).
- One column block alone, computed again: the "Compiling one column block" table above (397 and 221 ms on the two
  hosts' CPUs with patch 0002 at 2,048 rows, 57 and 41 ms on their GPUs).

## A single answer: a call of 1 request

The same tables and paths as the call of 32 above, in ms.

RTX 4090 host:

| rows × cols | CUDA first pass, device | Packing on the CPU | CUDA path, whole answer | WebGPU, device | WebGPU, whole answer | CPU alone |
|---|---|---|---|---|---|---|
| 2048 × 2048 | 0.01 | 0.14 | 0.32 | 0.18 | 12.4 | 0.47 |
| 12288 × 16384 | 0.43 | 8.5 | 9.4 | 0.96 | 13.2 | 24.9 |
| 6144 × 28672 | 0.38 | 15.5 | 16.4 | 1.03 | 13.7 | 29.9 |
| 14336 × 34816 | 1.06 | 19.0 | 20.8 | 1.92 | 14.0 | 58.8 |
| 32768 × 32768 | 2.27 | 18.5 | 22.0 | did not fit | | 103 |

RTX 2060 desktop:

| rows × cols | CUDA first pass, device | Packing on the CPU | CUDA path, whole answer | WebGPU, device | WebGPU, whole answer | CPU alone |
|---|---|---|---|---|---|---|
| 2048 × 2048 | 0.09 | 1.13 | 1.53 | 0.40 | 12.9 | 1.79 |
| 12288 × 16384 | 2.31 | 8.7 | 11.9 | 2.65 | 15.1 | 22.6 |
| 6144 × 28672 | 2.19 | 15.3 | 18.3 | 2.94 | 15.7 | 27.5 |
| 14336 × 34816 | 5.14 | 18.4 | 24.7 | 5.24 | 17.9 | 52.7 |
| 32768 × 32768 | 9.09 | 17.1 | 27.8 | 8.85 | 21.5 | 87.5 |

- **For one request the two paths are close, and the CUDA path is the faster one at small tables**: up to the Dust
  commitment table on both hosts; WebGPU is faster from 6144 × 28672 on.
- The WebGPU wall time has about 12 ms that do not depend on the call: submitting the work and reading the answer
  back in this harness (Deno's WebGPU). Its device time for the single answer is 0.2 to 2 ms on the RTX 4090 and
  0.4 to 9 ms on the RTX 2060, so a server that does not pay those 12 ms would answer faster than either column.
- The packing of one request on the CPU is about 1.1 ms a column block, and it is most of the CUDA path here too.
- On the M4 Max (WebGPU on Metal; no CUDA) a single answer takes 15.1 ms at 2048 × 2048, 16.5 ms at 6144 × 28672
  and 17.9 ms at 14336 × 34816, of which 1.0, 2.2 and 3.5 ms on the device.
- A wallet always sends 32 requests in a call, so the call of 32 is what a server answers.

## What the CUDA comparison took, and what could not be run

- Upstream wires the kernel into another server type (`YServer`), not into `NativeServer::respond`, which the design
  uses. The comparison joins the two with patch 0004 (`ipir/patches/README.md`): `respond` with the first pass
  handed to the CUDA kernel.
- Upstream fixes the bindings of CUDA 12.0, which do not load with the NVRTC of CUDA 13 that both hosts have. Patch
  0004 makes the release a build choice (`cuda-12`, `cuda-13`).
- The RTX 2060 desktop has the NVRTC of CUDA 13.4 and a driver for CUDA 13.0, which refuses what that NVRTC
  compiles. Its run used the two NVRTC 13.0 library files of the RTX 4090 host, copied into `.tools/nvrtc-13.0/`
  and put first in `LD_LIBRARY_PATH`.
- 32768 × 32768 on WebGPU on the RTX 4090: its 2.6 GiB did not fit beside another user's 20.7 GiB.
- The RTX 4090 figures are from a run while that shared host was quiet. A repeat while other users were compiling
  (load average 8 to 16) gave the same device times and a slower packing (832 ms against 611 ms at 14336 × 34816);
  `cuda_bench` ran during that load, so its CPU column there is not a quiet figure.
