#!/usr/bin/env bash
# Runs the whole benchmark and writes results/<host>-<date>.json and results/<host>-<date>.md.
#
#   ./run-all.sh            # everything (about an hour)
#   ./run-all.sh --quick    # small shapes and few blocks (a few minutes), to check that it works
#
# Other options: --skip-build, --skip-compile, --skip-fetch, --skip-budget, --allow-non-nvidia (run on a GPU that is
# not NVIDIA's), --name NAME (results file name instead of <host>-<date>).
# Environment: DENO (the deno binary), DENO_WEBGPU_BACKEND (default "vulkan" on Linux).
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$PWD
DENO_VERSION_PINNED=2.7.5

# Keep a Mac awake during the run (the benchmark was tried on macOS; Linux needs nothing here).
if [[ "$(uname)" == Darwin && -z "${IPIR_BENCH_CAFFEINATED:-}" ]] && command -v caffeinate >/dev/null; then
  IPIR_BENCH_CAFFEINATED=1 exec caffeinate -i "$0" "$@"
fi

QUICK=0 SKIP_BUILD=0 SKIP_COMPILE=0 SKIP_FETCH=0 SKIP_BUDGET=0 ALLOW_NON_NVIDIA=0 NAME=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) QUICK=1 ;;
    --skip-build) SKIP_BUILD=1 ;;
    --skip-compile) SKIP_COMPILE=1 ;;
    --skip-fetch) SKIP_FETCH=1 ;;
    --skip-budget) SKIP_BUDGET=1 ;;
    --allow-non-nvidia) ALLOW_NON_NVIDIA=1 ;;
    --name) NAME="$2"; shift ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done

if [[ -x "$ROOT/.tools/deno/deno" ]]; then DENO="${DENO:-$ROOT/.tools/deno/deno}"; else DENO="${DENO:-deno}"; fi
command -v "$DENO" >/dev/null || { echo "deno not found: run ./setup.sh first" >&2; exit 1; }
command -v cargo >/dev/null || { echo "cargo not found: install Rust with rustup (README, Host requirements)" >&2; exit 1; }
DENO_VERSION=$("$DENO" --version | head -1 | awk '{print $2}')
if [[ "$DENO_VERSION" != "$DENO_VERSION_PINNED" ]]; then
  echo "warning: deno $DENO_VERSION, the benchmark was written against $DENO_VERSION_PINNED (./setup.sh installs it)" >&2
fi
if [[ "$(uname)" == Linux ]]; then export DENO_WEBGPU_BACKEND="${DENO_WEBGPU_BACKEND:-vulkan}"; fi

HOST=$(hostname -s 2>/dev/null || hostname)
DATE=$(date +%Y-%m-%d)
if [[ -z "$NAME" ]]; then
  NAME="$HOST-$DATE"; [[ $QUICK == 1 ]] && NAME="$NAME-quick"
  [[ -e "results/$NAME.json" ]] && NAME="$NAME-$(date +%H%M)"
fi
RAW="results/raw/$NAME"
mkdir -p "$RAW"
exec > >(tee -a "$RAW/log.txt") 2>&1
START=$(date +%s)
echo "== ipir-benchmark-nvidia: $NAME (quick=$QUICK) $(date)"
PERMS=(--allow-read --allow-write --allow-run --allow-env --allow-sys)

if [[ $QUICK == 1 ]]; then
  FETCH_ARGS=(--shapes 2048x2048,2048x4096,4096x8192 --largest "" --runs 3)
  COMPILE_ARGS=(2048,8192 1 1,4)
  BUDGET_SETS=(today)
  BUDGET_ARGS=(--rates 2,5 --modes seq,pool --blocks 4 --baseline-s 5)
else
  FETCH_ARGS=(--runs 9)
  COMPILE_ARGS=(2048,8192,16384,32768 3 1,4)
  BUDGET_SETS=(today 10m)
  BUDGET_ARGS=(--rates 2,5,10 --modes seq,pool --blocks 30 --baseline-s 20)
fi

# 1. Build the CPU side.
if [[ $SKIP_BUILD == 0 ]]; then
  echo "== building ipir/ (cargo, release)"
  (cd ipir && { cargo build --release --locked --offline -p ipir-bench 2>/dev/null || cargo build --release --locked -p ipir-bench; })
fi
for b in compile_block block_worker; do
  [[ -x "ipir/target/release/$b" ]] || { echo "ipir/target/release/$b is missing: build failed?" >&2; exit 1; }
done

# 2. The machine, and the adapter check.
echo "== machine"
set +e
"$DENO" run "${PERMS[@]}" bench/sysinfo.ts > "$RAW/sysinfo-start.json"
SYS=$?
set -e
if [[ $SYS == 3 ]]; then
  echo "ABORT: the WebGPU adapter is a software rasterizer (llvmpipe/lavapipe/SwiftShader); its numbers mean nothing." >&2
  echo "The NVIDIA Vulkan driver is not being used: see README, Troubleshooting." >&2
  exit 3
elif [[ $SYS != 0 ]]; then
  echo "ABORT: bench/sysinfo.ts failed ($SYS)" >&2; exit 1
fi
ADAPTER=$("$DENO" eval 'const s=JSON.parse(Deno.readTextFileSync(Deno.args[0]));const w=s.webgpu??{};console.log(w.error?("ERROR "+w.error):[w.summary,w.vendor,w.device].join(" | "))' "$RAW/sysinfo-start.json")
echo "WebGPU adapter: $ADAPTER"
if [[ "$ADAPTER" == ERROR* ]]; then echo "ABORT: no usable WebGPU adapter ($ADAPTER). See README, Troubleshooting." >&2; exit 1; fi
if command -v nvidia-smi >/dev/null && [[ $ALLOW_NON_NVIDIA == 0 ]] && ! grep -qiE 'nvidia|geforce|rtx|10de|4318' <<<"$ADAPTER"; then
  echo "ABORT: nvidia-smi sees an NVIDIA GPU but WebGPU picked another adapter ($ADAPTER)." >&2
  echo "Try DENO_WEBGPU_BACKEND=vulkan, check vulkaninfo --summary, or pass --allow-non-nvidia." >&2
  exit 1
fi
VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || true)

cat > "$RAW/meta.json" <<JSON
{"name":"$NAME","date":"$DATE","quick":$([[ $QUICK == 1 ]] && echo true || echo false),"repoCommit":"$(git rev-parse --short HEAD 2>/dev/null || echo unknown)","denoWebgpuBackend":"${DENO_WEBGPU_BACKEND:-}","started":"$(date -u +%FT%TZ)"}
JSON

# 3. compile (task 2).
if [[ $SKIP_COMPILE == 0 ]]; then
  echo "== compile one column block on the CPU (rows, reps, at once: ${COMPILE_ARGS[*]})"
  ipir/target/release/compile_block "${COMPILE_ARGS[@]}" | tee "$RAW/compile.jsonl" || echo "compile step failed" >&2
fi

# 4. fetch (task 1).
if [[ $SKIP_FETCH == 0 ]]; then
  echo "== fetch on the GPU"
  "$DENO" run "${PERMS[@]}" bench/fetch-all.ts --out "$RAW/fetch.json" ${VRAM_MIB:+--vram-mib "$VRAM_MIB"} "${FETCH_ARGS[@]}" \
    || echo "fetch step failed" >&2
fi

# 5. The block budget (task 3).
if [[ $SKIP_BUDGET == 0 ]]; then
  for set in "${BUDGET_SETS[@]}"; do
    echo "== block budget, shapes: $set"
    "$DENO" run "${PERMS[@]}" bench/budget.ts --set "$set" --worker ipir/target/release/block_worker \
      --out "$RAW/budget-$set.json" "${BUDGET_ARGS[@]}" || echo "budget step ($set) failed" >&2
  done
fi

"$DENO" run "${PERMS[@]}" bench/sysinfo.ts --no-adapter > "$RAW/sysinfo-end.json" || true
END=$(date +%s)
"$DENO" eval 'const p=Deno.args[0];const m=JSON.parse(Deno.readTextFileSync(p));m.seconds=Number(Deno.args[1]);Deno.writeTextFileSync(p,JSON.stringify(m))' "$RAW/meta.json" "$((END-START))"
"$DENO" run "${PERMS[@]}" bench/report.ts "$RAW" "results/$NAME.json" "results/$NAME.md"
echo
cat "results/$NAME.md"
echo
echo "== done in $(( (END-START)/60 )) min: results/$NAME.json and results/$NAME.md (raw files in $RAW)"
