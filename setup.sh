#!/usr/bin/env bash
# One-time setup: checks the GPU stack, installs the pinned Deno into .tools/ (no root needed), fetches the Rust
# dependencies and builds the CPU side, with patch 0002 and without it. Run once after cloning; then ./run-all.sh.
#
# Linux: an NVIDIA GPU through Vulkan (nvidia-smi, vulkaninfo). macOS: Apple silicon through Metal (no NVIDIA tools).
#
# It does not install system packages or drivers (that needs root): it prints the commands to run when something
# is missing.
set -euo pipefail
cd "$(dirname "$0")"
DENO_VERSION=2.7.5
ok=1

echo "== GPU"
if [[ "$(uname)" == Darwin ]]; then
  system_profiler SPDisplaysDataType 2>/dev/null | grep -E 'Chipset Model|Total Number of Cores|Metal' || true
elif command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || ok=0
  # Resizable BAR is on when the BAR1 aperture covers the framebuffer.
  nvidia-smi -q -d MEMORY | awk '/FB Memory Usage/{s="fb"} /BAR1 Memory Usage/{s="bar1"} /Total/ && s {print s" total: "$3" "$4; s=""}'
else
  echo "nvidia-smi not found: install the NVIDIA driver (Ubuntu: sudo ubuntu-drivers install, or sudo apt install nvidia-driver-570)"
  echo "(only the CPU backend, ./run-all.sh --backend cpu, runs without a GPU)"
  ok=0
fi
if [[ "$(uname)" == Linux ]]; then
  if command -v vulkaninfo >/dev/null; then
    vulkaninfo --summary 2>/dev/null | grep -E 'deviceName|deviceType|driverName|driverInfo|apiVersion' || true
    if vulkaninfo --summary 2>/dev/null | grep -qi 'llvmpipe\|lavapipe'; then
      if ! vulkaninfo --summary 2>/dev/null | grep -qi 'nvidia'; then
        echo "Vulkan sees only a software device (llvmpipe): the NVIDIA Vulkan driver is missing (README, Troubleshooting)"
        ok=0
      fi
    fi
  else
    echo "vulkaninfo not found: sudo apt install vulkan-tools libvulkan1"
    ok=0
  fi
fi

echo "== Deno $DENO_VERSION"
if [[ -x .tools/deno/deno ]] && [[ "$(.tools/deno/deno --version | head -1 | awk '{print $2}')" == "$DENO_VERSION" ]]; then
  echo "already in .tools/deno"
else
  case "$(uname -s)-$(uname -m)" in
    Linux-x86_64) target=x86_64-unknown-linux-gnu ;;
    Linux-aarch64) target=aarch64-unknown-linux-gnu ;;
    Darwin-arm64) target=aarch64-apple-darwin ;;
    Darwin-x86_64) target=x86_64-apple-darwin ;;
    *) echo "no Deno build for $(uname -s)-$(uname -m)"; exit 1 ;;
  esac
  command -v unzip >/dev/null || { echo "unzip not found: sudo apt install unzip"; exit 1; }
  mkdir -p .tools/deno
  url="https://github.com/denoland/deno/releases/download/v$DENO_VERSION/deno-$target.zip"
  echo "downloading $url"
  curl -fsSL "$url" -o .tools/deno/deno.zip
  unzip -o -q .tools/deno/deno.zip -d .tools/deno && rm .tools/deno/deno.zip
  chmod +x .tools/deno/deno
fi
.tools/deno/deno --version | head -1

echo "== Rust"
if ! command -v cargo >/dev/null; then
  echo "cargo not found: install rustup (https://rustup.rs):"
  echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal && . \"\$HOME/.cargo/env\""
  echo "then run ./setup.sh again (rust-toolchain.toml selects Rust 1.92.0, which rustup installs on first use)."
  exit 1
fi
command -v cc >/dev/null || { echo "no C compiler/linker: sudo apt install build-essential"; exit 1; }
(cd ipir && cargo fetch --locked && cargo build --release --locked --offline -p ipir-bench)
ipir/build-stock.sh
ipir/target/release/compile_block 2048 1 1 >/dev/null && echo "compile_block runs (patch 0002)"
ipir/target/stock/release/compile_block 2048 1 1 >/dev/null && echo "compile_block runs (stock)"

echo "== WebGPU adapter as Deno sees it"
[[ "$(uname)" == Linux ]] && export DENO_WEBGPU_BACKEND="${DENO_WEBGPU_BACKEND:-vulkan}"
set +e
.tools/deno/deno run --allow-read --allow-run --allow-env --allow-sys bench/sysinfo.ts > /tmp/ipir-bench-sysinfo.$$.json
rc=$?
set -e
.tools/deno/deno eval 'const s=JSON.parse(Deno.readTextFileSync(Deno.args[0]));const w=s.webgpu??{};console.log(w.error?"ERROR: "+w.error:`${w.summary} (vendor ${w.vendor}, device ${w.device}, timestamp queries: ${w.timestampQuery}, software: ${w.software})`);console.log(`GPU backend: ${s.gpuBackend}; memory-budget layer: ${s.memoryBudgetLayer}`)' /tmp/ipir-bench-sysinfo.$$.json
rm -f /tmp/ipir-bench-sysinfo.$$.json
[[ $rc == 3 ]] && { echo "the adapter is a software rasterizer: fix the Vulkan driver before running (README, Troubleshooting)"; ok=0; }

if [[ $ok == 1 ]]; then echo "== ready: ./run-all.sh --quick to check, then ./run-all.sh"; else echo "== fix the items above, then run ./setup.sh again"; exit 1; fi
