# Notes on the RTX 4090 run of 2026-11-25

`rtx4090-2026-11-25-nvidia-nobudget` is a partial run on a shared machine, made to check the `nvidia` path and the
GPU build of the blocks on a card with resizable BAR. It is not a result for the overview page, and it is not in
`site/`.

| | |
|---|---|
| GPU | GeForce RTX 4090, 24,564 MiB, driver 580.126.18; resizable BAR on (BAR1 32,768 MiB) |
| CPU | AMD Ryzen 7 7700X, 8 cores / 16 threads, 62 GiB RAM |
| OS | Pop!_OS 24.04 LTS, kernel 6.18.7; reached by SSH, no root |
| Command | `IPIR_BENCH_HOST=rtx4090 ./run-all.sh --skip-build --skip-budget --name rtx4090-2026-11-25-nvidia-nobudget` |

What limits it:

- **GPU memory.** Another user's process held 20.7 of the 24.5 GiB, idle (0 % utilisation), so about 3 GiB were
  free. Shapes above 16384 × 18432 did not fit or failed, and the largest that ran was 32768 × 24576; six tables of
  32,768 rows at once failed in step 2b. The errors that read "validation error occurred" are allocations the driver
  could not make. The block budget (step 3) was skipped: its 10-million-output tables need 4 GiB.
- **CPU load.** Another user's build had the load average at 21 when the run started (8 when it ended), so the CPU
  compile times of step 2 are too high. A pass earlier that day at a load of 1.3 gave 566 ms (stock) and 217 ms
  (patch 0002) for one 2,048-row block; its file was replaced by this run's.
- The GPU's own times (timestamp queries) do not depend on the host's CPU load; the wall times do.

What it shows:

- The backend is detected as `nvidia`, and wgpu allows a 1 GiB buffer without the Vulkan layer, which is not loaded.
- The GPU build's output equals the CPU's: 24 of 24 blocks (12 single tables, 12 of the two six-table sets).
- One chain block's compile on the GPU, six tables at once: 33 ms of GPU time (50 ms wall) at today's shapes, and
  36 ms (70 ms wall) at the 10-million-output shapes. One table: 26 to 29 ms, of which the chains are 25.2 ms.
- `./bench.sh --quick` passed there from start to end.

Two things this machine brought out:

- The CPU build stores the compiled matrix packed in 27 bits there (`storage=packed27`; `i32` on the Ryzen 7 2700X
  and the M4 Max), and `pack_digest` hashes that name with the words. So `pack_digest`'s digest of the same words
  differs between machines; the GPU build's check hashes its words under the name the CPU reports.
- `vulkaninfo` is not installed, and the Vulkan loader prints "TU: error ... failed to open device" for Mesa's
  freedreno driver each time a device is opened. Neither matters: the adapter WebGPU opens is the NVIDIA GPU.
