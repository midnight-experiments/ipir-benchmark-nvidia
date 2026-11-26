#!/usr/bin/env bash
# One command from a fresh clone to the results:
#
#   ./bench.sh                 # setup, a quick check, the full run on this host's GPU, the overview page's file
#   ./bench.sh --quick         # setup and the quick check only (a few minutes)
#   ./bench.sh --install-deps  # first install what is missing: apt packages (as root or with sudo), Rust with rustup
#
# More options:
#   --cpu          after the GPU run, the same benchmark with the answers on the CPU (about 1.5 hours more)
#   --cpu-only     no GPU run: the CPU backend alone (also for a host with no GPU)
#   --report       only the report's measurements (bench/report-run.ts, about 30 minutes): a table built from
#                  scratch, an update, fetch 1 and fetch 32 at 1M and 10M values, and the GPU memory each needs;
#                  with --cuda also upstream's CUDA kernel for the fetches, with --quick a short check of it
#   --cuda         also upstream's CUDA kernel: a whole answer against the WebGPU path (in the full run's results),
#                  and upstream's own first-pass benchmark (NVIDIA, with the CUDA driver library and NVRTC)
#   --skip-quick   go straight to the full run
#   --budget-sets today   only today's shapes in the block budget: the 10-million-output tables need about 4 GiB of
#                  GPU memory (default: today,10m)
#   --host NAME    the host's name in file names and results, instead of `hostname -s` (rented machines have random
#                  names); the same as IPIR_BENCH_HOST=NAME
#   --label TEXT   the machine's name on the overview page, for example "RTX 4090 + EPYC 7B13" (default: the GPU's and
#                  the CPU's names)
#
# What it does, in order:
#   1. Checks the tools (curl, unzip, a C compiler, patch, git, cargo). With --install-deps it
#      installs the missing ones; without, it prints the command and stops. It never installs a GPU driver.
#   2. ./setup.sh: checks the GPU stack, installs the pinned Deno into .tools/, builds the Rust side.
#   3. Linux: tries a 1 GiB WebGPU buffer. If wgpu refuses it (an NVIDIA GPU without resizable BAR), it builds the
#      Vulkan layer in tools/hide-memory-budget-layer/ and loads it for the runs. With resizable BAR the layer is not
#      needed and is not loaded. The results record which of the two happened.
#   4. ./run-all.sh --quick, then bench/check-run.ts on its results: stops here when something failed.
#   5. ./run-all.sh (about 100 minutes), the same check, and bench/export-site.ts: site/<run>-site.json.
#   6. Packs the results into results/<host>-<date>-bundle.tar.gz and prints how to send them back.
#
# A full run takes over an hour: start it inside tmux or screen, or with `nohup ./bench.sh > bench.log 2>&1 &`.
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$PWD

QUICK_ONLY=0 SKIP_QUICK=0 INSTALL=0 WITH_CPU=0 CPU_ONLY=0 CUDA=0 REPORT=0 LABEL="" FULL_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) QUICK_ONLY=1 ;;
    --skip-quick) SKIP_QUICK=1 ;;
    --install-deps) INSTALL=1 ;;
    --cpu) WITH_CPU=1 ;;
    --cpu-only) CPU_ONLY=1 ;;
    --cuda) CUDA=1; FULL_ARGS+=(--cuda) ;;
    --report) REPORT=1 ;;
    --budget-sets) FULL_ARGS+=(--budget-sets "$2"); shift ;;
    --host) export IPIR_BENCH_HOST="$2"; shift ;;
    --label) LABEL="$2"; shift ;;
    -h|--help) sed -n '2,31p' "$0"; exit 0 ;;
    *) echo "unknown option $1 (./bench.sh --help)" >&2; exit 2 ;;
  esac
  shift
done
[[ $QUICK_ONLY == 1 && $SKIP_QUICK == 1 ]] && { echo "--quick and --skip-quick exclude each other" >&2; exit 2; }
OS=$(uname)
HOST=${IPIR_BENCH_HOST:-$(hostname -s 2>/dev/null || hostname)}
DATE=$(date +%Y-%m-%d)
step() { echo; echo "==== bench.sh: $*"; }

# 1. Tools.
step "tools"
# A rustup installed earlier in this shell's lifetime is not on PATH yet.
if ! command -v cargo >/dev/null && [[ -f "$HOME/.cargo/env" ]]; then . "$HOME/.cargo/env"; fi
missing=""
for c in curl unzip cc patch git; do command -v "$c" >/dev/null || missing="$missing $c"; done
APT_PACKAGES="build-essential curl unzip git patch ca-certificates libvulkan1 vulkan-tools"
if [[ -n "$missing" ]]; then
  echo "missing:$missing"
  if [[ $INSTALL == 1 ]] && command -v apt-get >/dev/null; then
    SUDO=""; [[ "$(id -u)" != 0 ]] && SUDO=sudo
    $SUDO apt-get update
    $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends $APT_PACKAGES
  elif [[ $OS == Darwin ]]; then
    echo "install the Xcode command line tools (xcode-select --install), then run ./bench.sh again"; exit 1
  else
    echo "install them, then run ./bench.sh again:"
    echo "  sudo apt-get update && sudo apt-get install -y $APT_PACKAGES"
    echo "or let this script do it: ./bench.sh --install-deps"
    exit 1
  fi
fi
if ! command -v cargo >/dev/null; then
  if [[ $INSTALL == 1 ]]; then
    echo "installing Rust with rustup into \$HOME (rust-toolchain.toml selects the version)"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
    . "$HOME/.cargo/env"
  else
    echo "cargo not found. Install Rust with rustup, then run ./bench.sh again:"
    echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal && . \"\$HOME/.cargo/env\""
    echo "or let this script do it: ./bench.sh --install-deps"
    exit 1
  fi
fi
echo "tools: ok"

# 2. Setup: the GPU stack, Deno, the Rust build.
step "setup"
if [[ $CPU_ONLY == 1 ]]; then ./setup.sh --cpu; else ./setup.sh; fi
DENO="$ROOT/.tools/deno/deno"
export DENO

# 3. Linux with a GPU: does wgpu allow the benchmark's buffers, or does this host need the Vulkan layer?
LAYER=VK_LAYER_IPIRBENCH_hide_memory_budget
if [[ $OS == Linux && $CPU_ONLY == 0 ]]; then
  step "WebGPU buffers"
  export DENO_WEBGPU_BACKEND="${DENO_WEBGPU_BACKEND:-vulkan}"
  if command -v nvidia-smi >/dev/null; then
    others=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)
    if [[ -n "$others" ]]; then echo "warning: other processes are using the GPU; the timings will include their load:"; echo "$others"; fi
  fi
  probe() { "$DENO" run -A tools/hide-memory-budget-layer/check.ts 1024 2>&1 | tail -1 || true; }
  if [[ ":${VK_INSTANCE_LAYERS:-}:" == *":$LAYER:"* ]]; then
    echo "the memory-budget layer is already loaded (VK_INSTANCE_LAYERS)"
  else
    out=$(probe)
    echo "without the layer, $out"
    if [[ "$out" != *"MiB: ok" ]]; then
      echo "wgpu refuses the buffer (as on an NVIDIA GPU without resizable BAR): building and loading the layer"
      tools/hide-memory-budget-layer/build.sh
      export VK_LAYER_PATH="$ROOT/tools/hide-memory-budget-layer${VK_LAYER_PATH:+:$VK_LAYER_PATH}"
      export VK_INSTANCE_LAYERS="$LAYER${VK_INSTANCE_LAYERS:+:$VK_INSTANCE_LAYERS}"
      out=$(probe)
      echo "with the layer, $out"
      if [[ "$out" != *"MiB: ok" ]]; then
        echo "ABORT: wgpu refuses a 1 GiB buffer even with the layer. See README, Troubleshooting." >&2
        exit 1
      fi
    fi
  fi
fi

# An NVRTC kept beside the tools goes first: for a host whose installed NVRTC is newer than its driver (README, Troubleshooting).
for d in "$ROOT"/.tools/nvrtc-*; do
  [[ -d "$d" ]] && export LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
done

# The report's measurements only.
if [[ $REPORT == 1 ]]; then
  step "the report's measurements"
  OUT="results/$HOST-$DATE-report.json"
  args=(--out "$OUT")
  [[ -n "$LABEL" ]] && args+=(--label "$LABEL")
  [[ $QUICK_ONLY == 1 ]] && args+=(--quick)
  if [[ $CUDA == 1 ]]; then
    feature=$("$ROOT/tools/cuda-feature.sh")
    echo "building cuda_fetch ($feature)"
    (cd ipir && { cargo build --release --locked --offline -p ipir-bench --features "$feature" --bin cuda_fetch 2>/dev/null \
      || cargo build --release --locked -p ipir-bench --features "$feature" --bin cuda_fetch; }) || { echo "cuda_fetch did not build" >&2; exit 1; }
    # The CUDA port of the block build (bins cuda_build_report, cuda_build_digest), when this checkout has it.
    if grep -q 'name = "cuda_build_report"' ipir/bench/Cargo.toml; then
      (cd ipir && { cargo build --release --locked --offline -p ipir-bench --features "$feature" --bin cuda_build_report 2>/dev/null \
        || cargo build --release --locked -p ipir-bench --features "$feature" --bin cuda_build_report; }) || echo "cuda_build_report did not build: the CUDA build cells stay empty" >&2
    fi
    args+=(--cuda)
  fi
  if [[ $OS == Darwin && -d metal ]]; then
    echo "building the native Metal block build (metal/)"
    (cd metal && cargo build --release --locked) || echo "the Metal build did not compile: the Metal cells stay empty" >&2
  fi
  KEEP_AWAKE=()
  [[ $OS == Darwin ]] && command -v caffeinate >/dev/null && KEEP_AWAKE=(caffeinate -s -i -m)
  ${KEEP_AWAKE[@]+"${KEEP_AWAKE[@]}"} "$DENO" run --allow-read --allow-write --allow-run --allow-env --allow-sys bench/report-run.ts "${args[@]}" || exit 1
  echo
  echo "== the report's measurements: $OUT"
  echo "   the tables: $DENO run --allow-read --allow-write bench/summary.ts (README, 'Run instructions')"
  exit 0
fi

# 4 and 5. The runs. $1 is a note for the log, the rest are run-all.sh's options. Sets NAME to the run's name, or to
# nothing when run-all.sh stopped before it wrote results. Returns 1 when the run stopped or its check found problems.
# (`set -e` does not apply inside a function called on the left of `||`, so every step returns its own failure.)
NAME=""
FILES=()
PROBLEMS=0
run() {
  local what=$1 name; shift
  step "$what"
  NAME=""
  rm -f results/raw/last-run
  ./run-all.sh --skip-build "$@" || { echo "run-all.sh stopped with an error (above)" >&2; return 1; }
  name=$(cat results/raw/last-run 2>/dev/null) || return 1
  [[ -f "results/$name.json" ]] || { echo "results/$name.json was not written" >&2; return 1; }
  NAME=$name
  FILES+=("results/$NAME.json" "results/$NAME.md")
  if [[ -f "results/raw/$NAME/log.txt" ]]; then FILES+=("results/raw/$NAME/log.txt"); fi
  echo
  "$DENO" run --allow-read bench/check-run.ts "results/$NAME.json" || return 1
}
export_site() {  # $1: the run's name; $2: "label" to pass --label
  local args=(--commit none --out "site/$1-site.json")
  if [[ $2 == label && -n "$LABEL" ]]; then args+=(--label "$LABEL"); fi
  mkdir -p site
  if "$DENO" run --allow-read --allow-write --allow-run bench/export-site.ts "${args[@]}" "results/$1.json"; then
    FILES+=("site/$1-site.json")
  else
    echo "the overview page's file was not written for $1 (the message above says why). By hand, once that is settled:"
    echo "  $DENO run --allow-read --allow-write --allow-run bench/export-site.ts --commit none [--backend nvidia|nvidia-no-bar] results/$1.json"
    PROBLEMS=1
  fi
}

# setup.sh has just built both builds, so every run skips the build.
if [[ $SKIP_QUICK == 0 ]]; then
  if [[ $CPU_ONLY == 0 ]]; then
    run "quick check, this host's GPU" --quick \
      || { echo "ABORT: the quick check on the GPU found problems (above); the full run was not started." >&2; exit 1; }
  fi
  if [[ $CPU_ONLY == 1 || $WITH_CPU == 1 ]]; then
    run "quick check, CPU backend" --quick --backend cpu \
      || { echo "ABORT: the quick check on the CPU found problems (above); the full run was not started." >&2; exit 1; }
  fi
fi

if [[ $QUICK_ONLY == 0 ]]; then
  if [[ $CPU_ONLY == 0 ]]; then
    run "full run, this host's GPU (about 100 minutes)" ${FULL_ARGS[@]+"${FULL_ARGS[@]}"} || PROBLEMS=1
    if [[ -n "$NAME" ]]; then export_site "$NAME" label; fi
  fi
  if [[ $CPU_ONLY == 1 || $WITH_CPU == 1 ]]; then
    run "full run, CPU backend (about 1.5 hours)" --backend cpu ${FULL_ARGS[@]+"${FULL_ARGS[@]}"} || PROBLEMS=1
    if [[ -n "$NAME" ]]; then export_site "$NAME" nolabel; fi
  fi
fi

# Upstream's CUDA first-pass kernel: one query's first pass, checked against the CPU kernel. Not the same quantity
# as step 1's call of 32; kept next to the results for comparison.
if [[ $CUDA == 1 ]]; then
  step "upstream's cuda_bench"
  CUDA_OUT="results/$HOST-$DATE-cuda-bench.txt"
  FILES+=("$CUDA_OUT")
  if (
    cd ipir || exit 1
    # The bindings follow the installed NVRTC library, as in run-all.sh.
    feature=$("$ROOT/tools/cuda-feature.sh")
    for shape in "2048 2048" "12288 16384" "14336 34816" "32768 32768"; do
      echo "== cuda_bench $shape 100 ($feature)"
      # shellcheck disable=SC2086
      cargo run --release --locked -p simplepir-kernel --features "$feature" --example cuda_bench -- $shape 100 || exit 1
    done
  ) 2>&1 | tee "$CUDA_OUT"; then
    :
  else
    echo "cuda_bench failed (it needs the CUDA driver library and NVRTC: docs/hosts.md); its output is in $CUDA_OUT"
    PROBLEMS=1
  fi
fi

# 6. One file to copy off the machine.
step "results"
if [[ ${#FILES[@]} == 0 ]]; then echo "no results were written: see the errors above" >&2; exit 1; fi
BUNDLE="results/$HOST-$DATE-bundle.tar.gz"
tar czf "$BUNDLE" "${FILES[@]}"
printf '  %s\n' "${FILES[@]}"
echo "all of them in one file: $BUNDLE"
echo
if [[ $QUICK_ONLY == 1 ]]; then
  echo "The quick check passed. Its numbers are not results. The full run: ./bench.sh --skip-quick"
else
  echo "To send the results back, either push them:"
  echo "  git checkout -b results/$HOST && git add results site && git commit -s -m \"Results from $HOST\" && git push -u origin HEAD"
  echo "or copy $BUNDLE off this machine (scp), and unpack it at the top of a clone."
fi
if [[ $PROBLEMS == 1 ]]; then echo; echo "Some steps had problems: see the PROBLEM lines above."; exit 1; fi
