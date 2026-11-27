# Notes on the RTX 4090 run of 2026-11-25

`rtx4090-2026-11-25-nvidia` is a full run on a shared machine with a card that has resizable BAR, without the
10-million-output tables in the block budget. Its site file is `site/rtx4090-2026-11-25-nvidia-site.json`.

| | |
|---|---|
| GPU | GeForce RTX 4090, 24,564 MiB, driver 580.126.18; resizable BAR on (BAR1 32,768 MiB) |
| CPU | AMD Ryzen 7 7700X, 8 cores / 16 threads, 62 GiB RAM |
| OS | Pop!_OS 24.04 LTS, kernel 6.18.7; reached by SSH, no root |
| Command | `./bench.sh --skip-quick --budget-sets today --host rtx4090 --label "RTX 4090 + Ryzen 7 7700X"` at `94faa8f`, 51 minutes |
| Load | Load average 0.11 at the start and 0.27 at the end: the machine was quiet |

What limits it:

- **GPU memory.** Another user's process held 20.7 of the 24.5 GiB, idle (0 % utilisation), so about 3 GiB were
  free. Shapes above 16384 × 18432 did not fit or failed, and the largest that ran was 32768 × 24576; six tables of
  32,768 rows at once failed in step 2b. The errors that read "validation error occurred" are allocations the driver
  could not make, which is why `bench/check-run.ts` lists two problems for this run. The block budget ran at today's
  shapes only (`--budget-sets today`): the 10-million-output tables need 4 GiB.
- **Step 2b's times were measured again after the run**, with the same command line and `bench/gpu-compile.ts` at
  `9b14acd`, which adds the builds with the GPU idle before them; the report and the site file were regenerated from
  the run's raw files with that one replaced. The first measurement's six-table figures were the same within 1 ms.

What it shows:

- The backend is detected as `nvidia`, and wgpu allows a 1 GiB buffer without the Vulkan layer, which is not loaded.
- The GPU build's output equals the CPU's: 24 of 24 blocks (12 single tables, 12 of the two six-table sets).
- One chain block's compile on the GPU, six tables at once, one build right after another: 33 ms of GPU time (49 ms
  wall) at today's shapes, 36 ms (67 ms wall) at the 10-million-output shapes. One table: 27 to 29 ms, of which the
  chains are 25.3 ms.
- **The same build after 6 s of idle takes 210 ms of GPU time** (251 ms wall; 218 and 320 ms at the
  10-million-output shapes): the card lowers its clocks when idle. After 0.5 s of idle there is no effect; after 2 s
  it is 174 ms. A small dispatch every 20 ms during the idle time gives 114 ms, every 100 ms 159 ms, every 250 ms or
  more nothing. The driver's maximum-performance power mode was not tried (no root on this machine).
- The block budget at today's shapes, block work p50: 3.67 s with the stock crates compiled one after another (over
  3 s in every block) and 2.12 s on the pool; 1.61 s and 1.43 s with patch 0002; 0.36 s with the blocks built on the
  GPU (0.26 s of build, slowed by the idle clocks, and 0.10 s of layout for the answer).

Two things this machine brought out:

- The CPU build stores the compiled matrix packed in 27 bits there (`storage=packed27`; `i32` on the Ryzen 7 2700X
  and the M4 Max), and `pack_digest` hashes that name with the words. So `pack_digest`'s digest of the same words
  differs between machines; the GPU build's check hashes its words under the name the CPU reports.
- `vulkaninfo` is not installed, and the Vulkan loader prints "TU: error ... failed to open device" for Mesa's
  freedreno driver each time a device is opened. Neither matters: the adapter WebGPU opens is the NVIDIA GPU.

## Upstream's CUDA kernel against the WebGPU path (2026-11-26)

`bench/cuda-compare.ts` with CUDA 13.0 (the host's toolkit, driver 580.126.18), bindings `cuda-13`. Its figures are in
the results file as `cuda`, in the run's summary as step 4 and in `docs/other-results.md`.

- The figures kept are from a run at 06:25 local time with the host quiet (load average about 1), from a working copy
  with the same files as commit `8866374`. The run was repeated from that commit a few minutes later while other
  users were compiling (load average 8 to 16): the device times were the same within 2 %, the packing on the CPU was
  slower (832 ms against 611 ms for a call of 32 at 14336 × 34816). That repeat is not in the results.
- `results/rtx4090-2026-11-26-cuda-bench.txt` (upstream's `cuda_bench`) ran during that load: its GPU columns are
  not affected, its `cpu_wall_ms_per_query` is.
- 3 GiB of the card were free (another user's process held 20.7 GiB): 32768 × 32768 ran on CUDA (2 GiB of cells) and
  did not fit on WebGPU, which also keeps the packing matrices on the card (2.6 GiB).
- Every answer of both paths equals the CPU's, byte for byte, at all five shapes (four on WebGPU).

## The report run with the card free (2026-11-27)

`./bench.sh --report --cuda --host rtx4090 --label "RTX 4090 + Ryzen 7 7700X"` at `ec442eb`, 20 minutes,
`results/rtx4090-2026-11-27-report.json`. The owner of the model that usually holds 20.7 GiB of the card stopped it
for this run: 1 MiB of the card was in use at the start. It replaces the report run of 2026-11-26, in which six
cells could not run beside that model.

- Every cell of the RTX 40 columns now holds a figure, and the six matrices at 10 million values fit with a chain
  block being computed (4,930 MiB at most).
- The 6× update with WebGPU, while the GPU answers 2 calls a second: 353 ms at 1 million values and 384 ms at
  10 million (the largest of 30 blocks: 397 and 401 ms). Computing the six blocks is 256 and 283 ms of that, and
  installing them 97 and 100 ms.
- The same six blocks computed alone, one right after another, take 50 ms and 67 ms. The card is idle for most of
  the 6 s between two chain blocks, and after that idle time the WebGPU build is about five times slower (257 and
  314 ms measured alone after 6 s of idle). The CUDA build on this card does not slow: 36 and 67 ms, and 38 and
  70 ms after idle. The cause is not known.
- Other users' container builds were using the CPU when the run ended (load average 2.7 at the start, 35 at the
  end; 0.4 while the 6× update at 1 million was measured), so the CUDA fetch cells, whose packing runs on the CPU,
  are slower than on a quiet host.
