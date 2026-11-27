# What was checked, and where

Which parts of the benchmark ran on which machine, and what has not run. The step numbers are those of
[full-run.md](full-run.md).

The backends, patch 0002's two builds and the site export were added and run on macOS (Apple M4 Max, 16 cores,
48 GB, macOS 15.7.3, Deno 2.7.5, Rust 1.92.0). The full GPU run is `results/m4-max-2026-11-25-metal.json` (78 min).
The CPU backend ran only as `--quick`: there, the CPU's answers equal the GPU's byte for byte at 2048 × 2048,
2048 × 4096 and 4096 × 8192. The full CPU run has not been made. Resizable BAR detection was checked against the
`nvidia-smi -q -d MEMORY` output of the RTX 2060 (BAR1 256 MiB, framebuffer 6144 MiB: `nvidia-no-bar`).

The GPU build of the blocks (step 2b) was written against an RTX 2060 through Vulkan (the desktop of
`results/pop-os-notes.md`, with the Vulkan layer): 12 of 12 digests equal the CPU's at 2,048 to 32,768 rows and seeds
1 to 3, and 12 of 12 for the two six-table sets; each stage's output was also compared with the CPU's on its own
(hint masks, slots before and after the chains, digits, matrix). `./bench.sh --quick` ran there from start to end,
which is the first run of the backends' code on Linux. It was then run on an RTX 4090 with resizable BAR
(`results/rtx4090-notes.md`): the backend is detected as `nvidia`, the Vulkan layer is not needed, 24 of 24 blocks
equal the CPU's, and `./bench.sh --skip-quick --budget-sets today` ran from start to end. On macOS (Metal, M4 Max) it ran
on 2026-11-26: 24 of 24 blocks equal the CPU's, and the block budget ran at both shape sets.

A full run with the GPU build was then made on each of the two NVIDIA hosts ([other-results.md](other-results.md)).

Not run yet: the full run's block budget at the 10-million-output shapes with the blocks built on an NVIDIA GPU (it
needs more than the 6 GB of the RTX 2060, where it was measured at 8 million outputs instead, and the RTX 4090's
memory was held by another user during its full run; on the M4 Max it ran), and the container check in `setup.sh`.
The report run of 2026-11-27 on the RTX 4090, with the card free, did measure it at 2 calls a second
([REPORT.md](../REPORT.md), section 3). Checked on macOS without a benchmark run:
`bench/check-run.ts` and the export against the committed Metal run and against copies of it renamed to `nvidia`,
`nvidia-no-bar` and `nvidia-unknown-bar`; resizable BAR detection against `nvidia-smi -q -d MEMORY` reports with
BAR1 above the framebuffer, at 256 MiB, and absent; and `bench.sh`'s order of steps, with stand-ins for `setup.sh`,
`run-all.sh` and the GPU tools (the layer loaded and not, a failed quick check, a stopped run, the CPU options).
The first `./bench.sh --quick` on a Linux host is the check of the rest.

Before the backends:

Developed and run on macOS (Apple M4 Max, Metal, Deno 2.7.5, Rust 1.92.0). There, the Rust programs build and run,
patch 0001's test passes, every Deno script type-checks and runs, `./run-all.sh --quick` completes, the 10-million
block-budget set runs, and the 32768×32768 shape and the skip logic work.

It has also run on Linux, on one desktop with an RTX 2060 (6 GB, driver 580.159.03, no resizable BAR) and a Ryzen 7
2700X. `results/pop-os-notes.md` has the details. There:

- `setup.sh`'s Deno download and `cargo fetch` work.
- Deno's WebGPU on Vulkan picks the NVIDIA GPU, not llvmpipe, and has timestamp queries. The adapter reports a
  `maxBufferSize` of 2^52, and the 2^30-byte chunks work.
- wgpu's memory-budget check refuses every buffer unless the Vulkan layer in `tools/hide-memory-budget-layer/` is
  loaded. With the layer, every shape up to 32768×32768 and the 10-million set fit in 6 GB, so how out-of-memory shows
  up on a 6 GB card is still not verified.
- The low-priority pool runs each thread at nice 19. On macOS the pool uses the utility QoS class instead.
- `bench/sysinfo.ts` records the CPU model, the RAM and the CPU governor as unknown, because Deno refuses reads of
  `/proc` and `/sys` without `--allow-all`.
