# Patch 0002: a faster CPU compile with the same output

Patch 0002 changes how the vendored IPIR+SP crates compute a table's hints and packing state. It does not change what
they compute. Every hint word and every word of the compiled packing state is the same as without the patch, so
queries, answers and keys are unaffected. On the benchmark host (Ryzen 7 2700X, 8 cores, 16 threads) the CPU build
of one 2,048-row column block drops from about 1.16 s to about 0.4 s. In the full run, 354 of 360 chain blocks finish
within the 3 s budget, against none of 360 without the patch.

This note explains where the time went, what each change does, why the output stays the same, and what the patch
costs. The patch is `0002-faster-block-compile.patch` next to this file. The code is already applied in `ipir/vendor/`.

## Where the stock compile spends its time

`compile_block` builds one table per column block (2,048 columns). The build has two parts:

- The hints: one `public_dot` per column, run in parallel over columns.
- The packing state, built by `NativePreprocessed::build_two_mask`:
  1. `aggregate_masks` sums the public masks into a d × d integer matrix (d = 2,048) with an integer ring FFT.
  2. Two collapse chains (`trace_half`), one per mask, run side by side. Each chain is 1,023 steps in a row, and each
     step needs the previous one. A step splits a polynomial into gadget digits (`decompose`), applies an
     automorphism to the digits (`tau_coeffs`), multiplies them by a prepared key (`LiftContext::public_dot`),
     applies the automorphism back and adds the result into the next slot.
  3. `compile_fast` turns the digits of each gadget level into the packing matrix.

With the stock code, one 2,048-row table took 1,062 ms of packing, and the chains took about 750 ms of that. The chains
keep two threads busy and leave the other 14 idle, so more cores do not help one table.

A chain step took about 735 µs, and `public_dot` was about 80 % of it. For each of its two auxiliary NTT primes it
allocated new spiral-rs buffers, copied the prepared operand, converted the input with `i128` arithmetic and ran two
forward NTTs, two multiplies and one inverse NTT through spiral (`to_ntt_alloc`, `multiply`, `from_ntt_alloc`). Then it
rebuilt each output word from the residues with `u128` remainders. Two costs dominated:

- spiral's scalar NTT does a `u128` Shoup multiply and a `subtle` constant-time select in every butterfly, about
  5.6 ns per butterfly on this CPU. The NTTs were 57 % of a step.
- spiral's `multiply` and the CRT step reduce with `u128 %`, which compiles to a library call of tens of cycles. They
  were 17 % of a step.

Building with `-C target-cpu=native` does not change this much (packing −4 to −5 %). spiral-rs has AVX2 and AVX-512
NTT code, but the lift context gives each `Params` a single prime, and spiral uses its scalar `_alt` NTT for one
prime.

## The five changes

Each change is its own commit on the branch, measured against the commit before it (packing for one 2,048-row table,
median of 6 alternating runs):

| Change | What it does | Packing |
|---|---|---|
| `public_dot` without division or allocation | Runs the same butterflies as spiral's single-prime NTT, on spiral's own root tables, with a 64-bit lazy Shoup multiply (a precomputed quotient per fixed multiplier, values kept in [0, 2p)) and a branch-free select instead of the `subtle` one. The NTT and the Shoup helpers are in `reinspiring/src/lazy_ntt.rs`. Its buffers are allocated once per call and prime. The pointwise multiply uses a Shoup quotient that `prepare_public_dot` stores for every word of the prepared operand. The CRT reconstruction for two and three primes uses Shoup products instead of `u128 %`. A butterfly costs about 1.8 ns. | 1,075 → 647 ms |
| `decompose` and `tau_coeffs` without division | `NativeParams` only allows q = 2^k with k ≤ 56 and digit and dropped widths of at most 27 bits, so the digit split works in `i64` with masks and shifts instead of `i128` division. `tau_coeffs` takes a direct path when d and q are powers of two and g is odd: then g permutes the slots, each slot is written once, and reductions are masks. | 640 → 507 ms |
| Both primes of a chain step in parallel | A chain step works on its auxiliary primes independently until the reconstruction. The chain now runs the primes as parallel Rayon tasks (`public_dot_parallel_primes`). The hints keep the sequential loop, because their loop over columns already fills the pool. | 521 → 491 ms |
| Matrix compilation without copies | The chain's digit polynomials move into compilation instead of being cloned and added into zeroed rows. The ring FFT writes into one reused scratch row instead of cloning a row for every butterfly. `NativeMatrix::from_blocks` checks its blocks with a parallel scan, in the same order, so the same error wins. | 474 → 381 ms |
| Mask aggregation without serial steps | `aggregate_masks` allocated its d × d matrix by cloning one row d times and widened it to `i128` in a serial loop. Rows are now allocated where they are filled, in parallel, and the widening is parallel. | 388 → 336 ms |

The hints need no change of their own: they use the same `public_dot`, and at 32,768 rows they drop from 1,137 ms to
351 ms.

## Why the output stays the same

All of this code is exact integer arithmetic with no rounding. The changes only reorder independent work or compute
the same remainder another way, and each one keeps every intermediate residue equal to the stock one:

- Shoup multiplication by a fixed w < p returns a value congruent to x · w in [0, 2p). The butterflies keep values in
  [0, 4p) and congruent to spiral's, and a final pass reduces them to the same canonical residues.
- The `i64` digit split equals the `i128` definition because `NativeParams` bounds every intermediate value, as above.
- The parallel steps write disjoint data and combine it in a fixed order.

Three kinds of check confirm it:

- New unit tests compare each new routine word for word with the code it replaces: the NTTs with spiral's, the
  reconstruction with the division formula, `public_dot` with the stock implementation at production shapes,
  `decompose` with the `i128` definition for every valid profile at every rounding and digit boundary, and
  `tau_coeffs` with the general loop, including inputs the fast path does not take. A test also checks that
  `ring_fft_eval_pow2` gives the unpatched output when a row is longer than the first.
- `ipir/bench/src/bin/pack_digest.rs` computes the hints and the packing state through the public calls that
  `NativeServer::build_with_concurrency` makes: `LiftContext::public_dot` for each column, then
  `NativePreprocessed::build_two_mask`. It hashes the hint masks, both public masks, and the compiled matrix with its
  storage width. The matrix comes from `NativePreprocessed::matrix()`, the one hidden accessor the patch adds. The
  reference digests come from the unpatched crates with only that accessor added. The hashes are equal at 2,048,
  8,192, 16,384 and 32,768 rows, for seeds 1, 2 and 3. 32,768 rows is the size where the hints take the three-prime
  path. A deliberate +1 on one matrix word changes the hash, and so does a +1 on one coefficient of every hint mask.
- The crates' own tests pass: `cargo test --release -p reinspiring -p ipir-sp`, 144 passed, plus the ignored
  degree-2,048 roundtrip when run explicitly.

The checks ran on a Ryzen 7 2700X, which lacks AVX-512 VBMI, so they did not reach the bit-packed `Packed27` and
`Packed28` matrix storage that `NativeMatrix::from_blocks` picks on CPUs with VBMI. The patched scan picks those widths
from the same centered range that decides between `i32` and `i64` words, and the test
`blocks_roundtrip_and_multiply_at_storage_boundaries` covers that choice on the 2700X.

## Results

`compile_block`, stock and patched builds run alternately, 6 runs each, median [range]:

| Case | Stock | Patched |
|---|---:|---:|
| One 2,048-row table: packing | 1,062 ms [1,045–1,086] | 359 ms [340–370] |
| One 2,048-row table: whole build | 1,164 ms [1,147–1,192] | 411 ms [391–424] |
| One 8,192-row table: whole build | 1,266 ms [1,224–1,308] | 428 ms [391–435] |
| Six 2,048-row tables at once | 3,893 ms [3,320–4,720] | 1,876 ms [1,655–2,002] |
| CPU time per table, one thread | ~2.7 s | ~0.98 s |

Two full `run-all.sh` runs on the same host, back to back, both with the Vulkan layer in
`tools/hide-memory-budget-layer/`: `results/pop-os-2026-11-25-hidebudget-quiet` (stock) and
`results/pop-os-2026-11-25-hidebudget-opt` (patched). `results/pop-os-notes.md` describes the host. Block work is the
CPU build of the six tables plus their upload to the GPU, 30 blocks per cell, p50 / p99:

| Set, mode, calls/s | Stock | Patched |
|---|---:|---:|
| `today`, `seq`, 5 | 6,894 / 7,118 ms | 2,329 / 2,394 ms |
| `today`, `pool`, 5 | 3,402 / 4,487 ms | 1,731 / 2,229 ms |
| `10m`, `seq`, 5 | 7,803 / 7,927 ms | 2,773 / 2,807 ms |
| `10m`, `pool`, 5 | 4,386 / 5,755 ms | 2,005 / 2,272 ms |
| `10m`, `seq`, 2 | 7,791 / 7,976 ms | 2,941 / 3,234 ms |
| Blocks over 3 s, all 12 cells | 360 of 360 | 6 of 360 |

All six blocks over 3 s are in one cell, `10m` `seq` at 2 calls/s. Its CPU build (p50 2,543 ms) is the same as at 5
and 10 calls/s, but its upload took 387 ms at p50 instead of the 166 to 234 ms of every other cell. A second run of
that cell alone gave 5 of 30 over. During it the GPU moved between power states P0 and P8 and between PCIe Gen 3 and
Gen 1 several times. The cause is not proven: finding it needs the GPU clocks pinned, which needs root. The upload is
GPU code this patch does not touch. `10m` `seq` therefore has about 0.2 s of margin, and the upload decides whether a
block fits.

The GPU fetch times are the same in both runs. Answer latency during block work stays within 12 ms of the stock run
at p50.

## What the patch costs

- About 450 lines of code and 320 lines of tests added to `reinspiring`, and 170 lines removed. The patch touches 7
  files and creates one of them, `reinspiring/src/lazy_ntt.rs`. `ipir-sp` is unchanged. The vendored crates then differ
  from `valargroup/ipir-sp` beyond patch 0001.
- Each `PreparedPublicDot` stores a Shoup quotient for every transform word, which doubles its size. It stays under
  1 MB for the benchmark's profile. The client's decoder keeps only the transforms of its masks and computes no
  quotients.
- `compile_fast` copies its input once up front.
- One `#[doc(hidden)]` read-only accessor, `NativePreprocessed::matrix()`, for the equality check, and `sha2` in the
  bench crate. The public API is otherwise unchanged.
- `std::hint::select_unpredictable` needs Rust 1.88. The crates declare `rust-version = "1.89"`.
- The new NTT and pointwise multiply run only inside `public_dot`, which handles public preprocessing data: the
  public setup, the masks it expands to, and the database the server already holds. The stock code documents
  `PreparedPublicDot` as "for public preprocessing data; validation is not constant-time". Dropping spiral's
  `subtle` select there exposes nothing.
- One changed helper also handles secret-dependent data. The two-prime CRT reconstruction (`reconstruct_two`) is
  shared with `sum_cached_two`. The client's `prepare_request` calls it with the client's secret, and the server's
  `prepare_pack` calls it with the packing keys the client sent. The patch replaces its `u128 %`, a library division
  whose time depends on its operands, with a Shoup multiply and a branch-free select. Its existing branches stay as
  they were. Neither version is constant-time, and this one has no data-dependent
  division. A reviewer who cares about timing should still look at this function.

## What limits it now

For one table, the two collapse chains are about half of packing, and about 80 % of a chain step is the NTT. The
NTT is bound by the 2700X's single 64-bit integer multiplier: each butterfly needs three 64-bit multiplies and takes
about 1.8 ns. In `seq` mode the six tables run one after another, so this sets the block time.

With six tables at once, the limit is how the shared Rayon pool schedules six builds that each have long sequential
chains, not CPU work. The six builds use about 10 of 16 threads on average and finish at about 2.5× the time eight
cores would need for their total work. A larger pool helps a little. `--pool-threads 16` gave a `today` p50 of
1,696 ms against 1,848 ms with 8 threads, in one run of 10 blocks.

Not tried in this patch:

- An NTT with different arithmetic, such as a floating-point quotient estimate or AVX2 lanes. It would move work off
  the integer multiplier, but it needs its own error-bound proof.
- `-C target-cpu=native` as the default. On the patched code it gives −5 % to −7.5 % packing and −13 % to −15 % hints,
  but the binary then needs a CPU of this class.

## What it means for the design

The stock result, every chain block over 3 s on this host, came from the implementation, not from the design. With the
same parameters and the same output, an 8-core 2018 desktop CPU finishes a `today` block's work in 2.3 s one table
after another, and in 1.7 s in the low-priority pool, while the GPU keeps answering calls. The `10m` shapes fit in `pool`
mode with at least 0.5 s to spare at p99. In `seq` mode their p50 is 2.77 to 2.94 s, close to the limit.

The real server builds with the IPIR+SP crates its Node add-on depends on, not with this vendored copy. For the design
to count on these numbers, the changes need to land in `valargroup/ipir-sp`, or the add-on needs to carry this patch
the way it carries patch 0001.

The benchmark still rebuilds a changed block's whole hint columns. The design updates only the hints for the new
records, so the hint part of these numbers stays an upper bound.
