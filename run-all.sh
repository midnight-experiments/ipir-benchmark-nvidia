#!/usr/bin/env bash
# Runs the whole benchmark for one backend and writes results/<host>-<date>-<backend>.json and .md.
# ./bench.sh is the one command around this script: setup, a quick check, this run, and the overview page's file.
#
#   ./run-all.sh                 # this host's GPU: metal on macOS, nvidia or nvidia-no-bar on Linux (about 1.5 hours)
#   ./run-all.sh --backend cpu   # the answers on the CPU instead of the GPU
#   ./run-all.sh --quick         # small shapes and few blocks (a few minutes), to check that it works
#
# Steps 2 (compile) and 3 (budget) run twice on the CPU: with the stock vendored crates and with patch 0002. On a GPU
# backend they also run with the blocks built on the GPU (gpu/block-build.ts), after a check that the GPU's output
# equals the CPU's.
# Other options: --builds stock,patch0002 (which of the two), --skip-build, --skip-compile, --skip-fetch,
# --skip-budget, --budget-sets today (which shape sets step 3 runs: today, 10m; the 10M tables need about 4 GiB of
# GPU memory), --skip-gpu-compile (no GPU build of the blocks), --no-gpu-check (CPU backend: skip comparing its answers with the GPU's), --allow-non-nvidia (run on
# a GPU that is not NVIDIA's), --name NAME (results file name instead of <host>-<date>-<backend>).
# Environment: DENO (the deno binary), DENO_WEBGPU_BACKEND (default "vulkan" on Linux), CARGO_BUILD_JOBS,
# IPIR_BENCH_HOST (the host name in file names and results, instead of `hostname -s`), IPIR_BENCH_PREFLIGHT (a JSON
# object recorded in the results as `meta.preflight`, for example the load and the wait before the run).
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$PWD
DENO_VERSION_PINNED=2.7.5

# Keep a Mac awake during the run (the benchmark was tried on macOS; Linux needs nothing here).
if [[ "$(uname)" == Darwin && -z "${IPIR_BENCH_CAFFEINATED:-}" ]] && command -v caffeinate >/dev/null; then
  IPIR_BENCH_CAFFEINATED=1 exec caffeinate -s -i -m "$0" "$@"
fi

QUICK=0 SKIP_BUILD=0 SKIP_COMPILE=0 SKIP_FETCH=0 SKIP_BUDGET=0 SKIP_GPU_COMPILE=0 ALLOW_NON_NVIDIA=0 NAME="" BACKEND_OPT=gpu
BUILDS=stock,patch0002 GPU_CHECK=1 BUDGET_SETS_OPT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) QUICK=1 ;;
    --backend) BACKEND_OPT="$2"; shift ;;
    --builds) BUILDS="$2"; shift ;;
    --no-gpu-check) GPU_CHECK=0 ;;
    --skip-build) SKIP_BUILD=1 ;;
    --skip-compile) SKIP_COMPILE=1 ;;
    --skip-fetch) SKIP_FETCH=1 ;;
    --skip-budget) SKIP_BUDGET=1 ;;
    --budget-sets) BUDGET_SETS_OPT="$2"; shift ;;
    --skip-gpu-compile) SKIP_GPU_COMPILE=1 ;;
    --allow-non-nvidia) ALLOW_NON_NVIDIA=1 ;;
    --name) NAME="$2"; shift ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
[[ "$BACKEND_OPT" == gpu || "$BACKEND_OPT" == cpu ]] || { echo "--backend is gpu (this host's GPU) or cpu" >&2; exit 2; }
IFS=, read -r -a BUILD_LIST <<<"$BUILDS"
for b in "${BUILD_LIST[@]}"; do
  [[ "$b" == stock || "$b" == patch0002 ]] || { echo "--builds takes stock and patch0002" >&2; exit 2; }
done
# Where each build's programs are: patch 0002 is applied in ipir/vendor/; ipir/build-stock.sh builds without it.
bindir() { if [[ "$1" == stock ]]; then echo ipir/target/stock/release; else echo ipir/target/release; fi; }

if [[ -x "$ROOT/.tools/deno/deno" ]]; then DENO="${DENO:-$ROOT/.tools/deno/deno}"; else DENO="${DENO:-deno}"; fi
command -v "$DENO" >/dev/null || { echo "deno not found: run ./setup.sh first" >&2; exit 1; }
command -v cargo >/dev/null || { echo "cargo not found: install Rust with rustup (README, Host requirements)" >&2; exit 1; }
DENO_VERSION=$("$DENO" --version | head -1 | awk '{print $2}')
if [[ "$DENO_VERSION" != "$DENO_VERSION_PINNED" ]]; then
  echo "warning: deno $DENO_VERSION, the benchmark was written against $DENO_VERSION_PINNED (./setup.sh installs it)" >&2
fi
if [[ "$(uname)" == Linux ]]; then export DENO_WEBGPU_BACKEND="${DENO_WEBGPU_BACKEND:-vulkan}"; fi

HOST=${IPIR_BENCH_HOST:-$(hostname -s 2>/dev/null || hostname)}
DATE=$(date +%Y-%m-%d)
PERMS=(--allow-read --allow-write --allow-run --allow-env --allow-sys)

# The backend: the GPU this host has (bench/sysinfo.ts decides metal, nvidia or nvidia-no-bar), or cpu.
SYSTMP=$(mktemp)
set +e
"$DENO" run "${PERMS[@]}" bench/sysinfo.ts > "$SYSTMP"
SYS=$?
set -e
GPU_BACKEND=$("$DENO" eval 'const s=JSON.parse(Deno.readTextFileSync(Deno.args[0]));console.log(s.gpuBackend??"webgpu")' "$SYSTMP")
if [[ "$BACKEND_OPT" == cpu ]]; then BACKEND=cpu; ANSWER=cpu; else BACKEND=$GPU_BACKEND; ANSWER=gpu; fi
if [[ -z "$NAME" ]]; then
  NAME="$HOST-$DATE-$BACKEND"; [[ $QUICK == 1 ]] && NAME="$NAME-quick"
  [[ -e "results/$NAME.json" ]] && NAME="$NAME-$(date +%H%M)"
fi
RAW="results/raw/$NAME"
mkdir -p "$RAW"
echo "$NAME" > results/raw/last-run   # bench.sh reads the name of the run from here
mv "$SYSTMP" "$RAW/sysinfo-start.json"
exec > >(tee -a "$RAW/log.txt") 2>&1
START=$(date +%s)
echo "== ipir-benchmark-nvidia: $NAME (backend=$BACKEND, builds=$BUILDS, quick=$QUICK) $(date)"
echo "load average at start: $(uptime | sed 's/.*load averages*: //')"

if [[ $QUICK == 1 ]]; then
  FETCH_ARGS=(--shapes 2048x2048,2048x4096,4096x8192 --largest "" --runs 3)
  COMPILE_ARGS=(2048,8192 1 1,4)
  BUDGET_SETS=(today)
  BUDGET_ARGS=(--rates 2,5 --blocks 4 --baseline-s 5)
  GPU_CHECK_ARGS=(--rows 2048,8192 --seeds 1 --sets today)
  GPU_COMPILE_ARGS=(--rows 2048,8192 --reps 3 --at-once 4 --sets today --set-reps 3 --idle-ms 2000 --idle-reps 2)
else
  FETCH_ARGS=(--runs 9)
  COMPILE_ARGS=(2048,8192,16384,32768 3 1,4)
  BUDGET_SETS=(today 10m)
  BUDGET_ARGS=(--rates 2,5,10 --blocks 30 --baseline-s 20)
  GPU_CHECK_ARGS=(--rows 2048,8192,16384,32768 --seeds 1,2,3 --sets today,10m)
  GPU_COMPILE_ARGS=(--rows 2048,8192,16384,32768 --reps 5 --at-once 4,6 --sets today,10m --set-reps 10)
fi

if [[ -n "$BUDGET_SETS_OPT" ]]; then
  IFS=, read -r -a BUDGET_SETS <<<"$BUDGET_SETS_OPT"
  for s in "${BUDGET_SETS[@]}"; do [[ "$s" == today || "$s" == 10m ]] || { echo "--budget-sets takes today and 10m" >&2; exit 2; }; done
fi

# 1. Build the CPU side: with patch 0002 (ipir/target/release/) and without (ipir/target/stock/release/).
if [[ $SKIP_BUILD == 0 ]]; then
  echo "== building ipir/ (cargo, release)"
  (cd ipir && { cargo build --release --locked --offline -p ipir-bench 2>/dev/null || cargo build --release --locked -p ipir-bench; })
  if [[ ",$BUILDS," == *,stock,* ]]; then echo "== building ipir/ without patch 0002"; ipir/build-stock.sh; fi
fi
for b in "${BUILD_LIST[@]}"; do
  for p in compile_block block_worker cpu_server; do
    [[ -x "$(bindir "$b")/$p" ]] || { echo "$(bindir "$b")/$p is missing: build failed?" >&2; exit 1; }
  done
done
for p in cpu_fetch gpu_build_prep pack_digest; do
  [[ -x ipir/target/release/$p ]] || { echo "ipir/target/release/$p is missing: build failed?" >&2; exit 1; }
done

# 2. The machine, and the adapter check. The CPU backend needs no GPU; it uses one, if there is a usable one, only
# to check its answers.
echo "== machine"
if [[ $SYS == 3 && $ANSWER == cpu ]]; then
  echo "the WebGPU adapter is a software rasterizer: the CPU backend runs without the GPU check"; GPU_CHECK=0
elif [[ $SYS == 3 ]]; then
  echo "ABORT: the WebGPU adapter is a software rasterizer (llvmpipe/lavapipe/SwiftShader); its numbers mean nothing." >&2
  echo "The NVIDIA Vulkan driver is not being used: see README, Troubleshooting." >&2
  exit 3
elif [[ $SYS != 0 ]]; then
  echo "ABORT: bench/sysinfo.ts failed ($SYS)" >&2; exit 1
fi
ADAPTER=$("$DENO" eval 'const s=JSON.parse(Deno.readTextFileSync(Deno.args[0]));const w=s.webgpu??{};console.log(w.error?("ERROR "+w.error):[w.summary,w.vendor,w.device].join(" | "))' "$RAW/sysinfo-start.json")
echo "WebGPU adapter: $ADAPTER"
"$DENO" eval 'const s=JSON.parse(Deno.readTextFileSync(Deno.args[0]));const m=s.nvidiaMemory;console.log(`backend: ${Deno.args[1]}; GPU backend of this host: ${s.gpuBackend}; resizable BAR: ${m?`${m.resizableBar} (BAR1 ${m.bar1TotalMiB} MiB, framebuffer ${m.fbTotalMiB} MiB)`:"n/a"}; memory-budget layer: ${s.memoryBudgetLayer}`)' "$RAW/sysinfo-start.json" "$BACKEND"
if [[ "$ADAPTER" == ERROR* && $ANSWER == cpu ]]; then
  echo "no usable WebGPU adapter: the CPU backend runs without the GPU check"; GPU_CHECK=0
elif [[ "$ADAPTER" == ERROR* ]]; then echo "ABORT: no usable WebGPU adapter ($ADAPTER). See README, Troubleshooting." >&2; exit 1; fi
if [[ $ANSWER == gpu ]] && command -v nvidia-smi >/dev/null && [[ $ALLOW_NON_NVIDIA == 0 ]] && ! grep -qiE 'nvidia|geforce|rtx|10de|4318' <<<"$ADAPTER"; then
  echo "ABORT: nvidia-smi sees an NVIDIA GPU but WebGPU picked another adapter ($ADAPTER)." >&2
  echo "Try DENO_WEBGPU_BACKEND=vulkan, check vulkaninfo --summary, or pass --allow-non-nvidia." >&2
  exit 1
fi
# The GPU build of the blocks: on a GPU backend whose adapter has the limits its kernels need.
GPU_COMPILE=0
if [[ $ANSWER == gpu && $SKIP_GPU_COMPILE == 0 ]]; then
  if [[ "$("$DENO" eval 'const s=JSON.parse(Deno.readTextFileSync(Deno.args[0]));console.log(s.webgpu?.blockBuild===true)' "$RAW/sysinfo-start.json")" == true ]]; then
    GPU_COMPILE=1
  else
    echo "the adapter lacks the limits the GPU block build needs (1,024 invocations per workgroup, 32 KB of workgroup memory): blocks are built on the CPU only"
  fi
fi
VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || true)
RAM_MIB=$("$DENO" eval 'const s=JSON.parse(Deno.readTextFileSync(Deno.args[0]));console.log(s.memTotalKiB?Math.floor(s.memTotalKiB/1024):"")' "$RAW/sysinfo-start.json")

cat > "$RAW/meta.json" <<JSON
{"name":"$NAME","date":"$DATE","quick":$([[ $QUICK == 1 ]] && echo true || echo false),"backend":"$BACKEND","answer":"$ANSWER","builds":"$BUILDS","gpuCheck":$([[ $GPU_CHECK == 1 && $ANSWER == cpu ]] && echo true || echo false),"gpuCompile":$([[ $GPU_COMPILE == 1 ]] && echo true || echo false),"budgetSets":"$([[ $SKIP_BUDGET == 0 ]] && (IFS=,; echo -n "${BUDGET_SETS[*]}"))","gpuCompileSteps":"$([[ $SKIP_COMPILE == 0 ]] && echo -n compile)$([[ $SKIP_BUDGET == 0 ]] && echo -n ,budget)","repoCommit":"$(git rev-parse --short HEAD 2>/dev/null || echo unknown)","repoDirty":$([[ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ]] && echo true || echo false),"denoWebgpuBackend":"${DENO_WEBGPU_BACKEND:-}","preflight":${IPIR_BENCH_PREFLIGHT:-null},"started":"$(date -u +%FT%TZ)"}
JSON

# 3. compile (task 2), for each build.
if [[ $SKIP_COMPILE == 0 ]]; then
  for b in "${BUILD_LIST[@]}"; do
    echo "== compile one column block on the CPU, build $b (rows, reps, at once: ${COMPILE_ARGS[*]})"
    "$(bindir "$b")/compile_block" "${COMPILE_ARGS[@]}" | tee "$RAW/compile-$b.jsonl" || echo "compile step ($b) failed" >&2
  done
fi

# 3b. compile on the GPU: first that its output equals the CPU's, then its time.
if [[ $SKIP_COMPILE == 0 && $GPU_COMPILE == 1 ]]; then
  echo "== compile on the GPU: its output against the CPU's (${GPU_CHECK_ARGS[*]})"
  "$DENO" run "${PERMS[@]}" bench/check-gpu-build.ts --out "$RAW/gpu-build-check.json" "${GPU_CHECK_ARGS[@]}" \
    || echo "the GPU build's check failed or found a difference" >&2
  echo "== compile on the GPU: time (${GPU_COMPILE_ARGS[*]})"
  "$DENO" run "${PERMS[@]}" bench/gpu-compile.ts --out "$RAW/gpu-compile.json" "${GPU_COMPILE_ARGS[@]}" || echo "gpu compile step failed" >&2
fi

# 4. fetch (task 1): on the GPU, or on the CPU (with patch 0002, the repository's default build).
if [[ $SKIP_FETCH == 0 ]]; then
  if [[ $ANSWER == cpu ]]; then
    echo "== fetch on the CPU"
    CPU_FETCH_ARGS=(--backend cpu --cpu-fetch ipir/target/release/cpu_fetch ${RAM_MIB:+--ram-mib "$RAM_MIB"})
    [[ $GPU_CHECK == 1 ]] && CPU_FETCH_ARGS+=(--check-gpu)
    "$DENO" run "${PERMS[@]}" bench/fetch-all.ts --out "$RAW/fetch.json" "${CPU_FETCH_ARGS[@]}" "${FETCH_ARGS[@]}" \
      || echo "fetch step failed" >&2
  else
    echo "== fetch on the GPU"
    "$DENO" run "${PERMS[@]}" bench/fetch-all.ts --out "$RAW/fetch.json" ${VRAM_MIB:+--vram-mib "$VRAM_MIB"} "${FETCH_ARGS[@]}" \
      || echo "fetch step failed" >&2
  fi
fi

# 5. The block budget (task 3), for each build.
if [[ $SKIP_BUDGET == 0 ]]; then
  for b in "${BUILD_LIST[@]}"; do
    ANSWER_ARGS=()
    [[ $ANSWER == cpu ]] && ANSWER_ARGS=(--answer cpu --cpu-server "$(bindir "$b")/cpu_server")
    for set in "${BUDGET_SETS[@]}"; do
      echo "== block budget, shapes: $set, build $b, answers on the $ANSWER"
      "$DENO" run "${PERMS[@]}" bench/budget.ts --set "$set" --worker "$(bindir "$b")/block_worker" --modes seq,pool \
        --out "$RAW/budget-$set-$b.json" ${ANSWER_ARGS[@]+"${ANSWER_ARGS[@]}"} "${BUDGET_ARGS[@]}" || echo "budget step ($set, $b) failed" >&2
    done
  done
  # The same with the blocks built on the GPU that is answering.
  if [[ $GPU_COMPILE == 1 ]]; then
    for set in "${BUDGET_SETS[@]}"; do
      echo "== block budget, shapes: $set, blocks built on the GPU"
      "$DENO" run "${PERMS[@]}" bench/budget.ts --set "$set" --modes gpu --gpu-prep ipir/target/release/gpu_build_prep \
        --out "$RAW/budget-$set-gpu.json" "${BUDGET_ARGS[@]}" || echo "budget step ($set, GPU build) failed or does not fit" >&2
    done
  fi
fi

"$DENO" run "${PERMS[@]}" bench/sysinfo.ts --no-adapter > "$RAW/sysinfo-end.json" || true
END=$(date +%s)
echo "load average at end: $(uptime | sed 's/.*load averages*: //')"
"$DENO" eval 'const p=Deno.args[0];const m=JSON.parse(Deno.readTextFileSync(p));m.seconds=Number(Deno.args[1]);Deno.writeTextFileSync(p,JSON.stringify(m))' "$RAW/meta.json" "$((END-START))"
"$DENO" run "${PERMS[@]}" bench/report.ts "$RAW" "results/$NAME.json" "results/$NAME.md"
echo
cat "results/$NAME.md"
echo
echo "== done in $(( (END-START)/60 )) min: results/$NAME.json and results/$NAME.md (raw files in $RAW)"
if [[ $QUICK == 0 ]]; then
  echo "== next: the overview page's file, site/$NAME-site.json (./bench.sh does this itself; see the README, 'Backends'):"
  echo "  $DENO run --allow-read --allow-write --allow-run bench/export-site.ts --commit none --label \"<GPU> + <CPU>\" results/$NAME.json"
  [[ $BACKEND == nvidia-unknown-bar ]] && echo "  (resizable BAR was not detected either way: add --backend nvidia or --backend nvidia-no-bar after checking nvidia-smi -q -d MEMORY)"
fi
true
