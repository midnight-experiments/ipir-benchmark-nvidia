#!/usr/bin/env bash
# One-time setup: checks the GPU stack, installs the pinned Deno into .tools/ (no root needed), fetches the Rust
# dependencies and builds the CPU side, with patch 0002 and without it. Run once after cloning; then ./run-all.sh.
#
# Linux: an NVIDIA GPU through Vulkan (nvidia-smi, vulkaninfo). macOS: Apple silicon through Metal (no NVIDIA tools).
#
# It does not install system packages or drivers (that needs root): it prints the commands to run when something
# is missing. ./bench.sh runs this script first, and with --install-deps installs the packages too.
#
#   ./setup.sh         # for a run on this host's GPU
#   ./setup.sh --cpu   # for the CPU backend only: a missing or unusable GPU is reported, and does not fail the setup
set -euo pipefail
cd "$(dirname "$0")"
DENO_VERSION=2.7.5
CPU_ONLY=0
[[ "${1:-}" == --cpu ]] && CPU_ONLY=1
ok=1       # what every backend needs
gpu_ok=1   # what a run on the GPU needs

echo "== GPU"
if [[ "$(uname)" == Darwin ]]; then
  system_profiler SPDisplaysDataType 2>/dev/null | grep -E 'Chipset Model|Total Number of Cores|Metal' || true
elif command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || gpu_ok=0
  # Resizable BAR is on when the BAR1 aperture covers the framebuffer.
  nvidia-smi -q -d MEMORY | awk '/FB Memory Usage/{s="fb"} /BAR1 Memory Usage/{s="bar1"} /Total/ && s {print s" total: "$3" "$4; s=""}'
else
  echo "nvidia-smi not found: install the NVIDIA driver (Ubuntu: sudo ubuntu-drivers install, or sudo apt install nvidia-driver-570)"
  echo "(only the CPU backend runs without a GPU: ./bench.sh --cpu-only)"
  gpu_ok=0
fi
if [[ "$(uname)" == Linux ]]; then
  if command -v vulkaninfo >/dev/null; then
    vk=$(vulkaninfo --summary 2>/dev/null || true)
    grep -E 'deviceName|deviceType|driverName|driverInfo|apiVersion' <<<"$vk" || true
    # The NVIDIA Vulkan driver is libGLX_nvidia.so.0, found through an ICD file. Containers (most GPU rentals) often
    # have the library but not the file, or neither when the container lacks the `graphics` driver capability.
    if command -v nvidia-smi >/dev/null && ! grep -i 'nvidia' <<<"$vk" >/dev/null; then
      echo "Vulkan does not see the NVIDIA GPU$(grep -i 'llvmpipe\|lavapipe' <<<"$vk" >/dev/null && echo ' (only a software device, llvmpipe)')."
      if ldconfig -p 2>/dev/null | grep 'libGLX_nvidia.so.0' >/dev/null; then
        echo "The driver library is there, so its ICD file is missing: create it with"
        echo "  sudo mkdir -p /usr/share/vulkan/icd.d && echo '{\"file_format_version\":\"1.0.0\",\"ICD\":{\"library_path\":\"libGLX_nvidia.so.0\",\"api_version\":\"1.3\"}}' | sudo tee /usr/share/vulkan/icd.d/nvidia_icd.json"
      elif [[ -f /.dockerenv || -f /run/.containerenv ]]; then
        echo "This is a container without the NVIDIA Vulkan driver (libGLX_nvidia.so.0): start the container with"
        echo "  NVIDIA_DRIVER_CAPABILITIES=all (or compute,utility,graphics); see docs/hosts.md"
      else
        echo "The NVIDIA Vulkan driver (libGLX_nvidia.so.0) is missing: sudo apt install libnvidia-gl-<driver branch>"
      fi
      gpu_ok=0
    fi
  else
    # Without vulkaninfo there is no device list to show; the WebGPU adapter check below is the test.
    echo "vulkaninfo not found (sudo apt install vulkan-tools): skipping the Vulkan device list"
    ldconfig -p 2>/dev/null | grep 'libvulkan.so.1' >/dev/null || { echo "the Vulkan loader is missing too: sudo apt install libvulkan1"; gpu_ok=0; }
  fi
fi
command -v patch >/dev/null || { echo "patch not found (ipir/build-stock.sh needs it): sudo apt install patch"; ok=0; }

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
cp /tmp/ipir-bench-sysinfo.$$.json /tmp/ipir-bench-sysinfo.$$.keep 2>/dev/null || true
rm -f /tmp/ipir-bench-sysinfo.$$.json
[[ $rc == 3 ]] && { echo "the adapter is a software rasterizer: fix the Vulkan driver before running (README, Troubleshooting)"; gpu_ok=0; }
# On Linux with an NVIDIA GPU, the adapter must be that GPU (vendor 0x10de = 4318), whatever vulkaninfo could show.
if [[ "$(uname)" == Linux && $rc == 0 ]] && command -v nvidia-smi >/dev/null; then
  if ! grep -qiE '"vendor": *"?(4318|0x10de|nvidia)' /tmp/ipir-bench-sysinfo.$$.keep 2>/dev/null; then
    echo "WebGPU did not open the NVIDIA GPU: check the Vulkan driver (README, Troubleshooting)"; gpu_ok=0
  fi
fi
rm -f /tmp/ipir-bench-sysinfo.$$.keep

if [[ $ok == 0 ]]; then echo "== fix the items above, then run ./setup.sh again"; exit 1; fi
if [[ $gpu_ok == 1 ]]; then echo "== ready: ./bench.sh (or ./run-all.sh --quick to check, then ./run-all.sh)"
elif [[ $CPU_ONLY == 1 ]]; then echo "== ready for the CPU backend only: ./bench.sh --cpu-only (or ./run-all.sh --backend cpu)"
else echo "== fix the GPU items above, then run ./setup.sh again (or ./setup.sh --cpu for the CPU backend only)"; exit 1; fi
