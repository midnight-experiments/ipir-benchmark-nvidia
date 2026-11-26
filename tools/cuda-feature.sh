#!/usr/bin/env bash
# Prints the build feature for the CUDA bindings: cuda-13 or cuda-12, after the NVRTC library this host will load.
# IPIR_CUDA_FEATURE chooses outright; else an NVRTC kept in .tools/nvrtc-13* (bench.sh puts it first in
# LD_LIBRARY_PATH), else the installed one (ldconfig, which is in /sbin on some hosts' non-login shells).
ROOT=$(cd "$(dirname "$0")/.." && pwd)
if [[ -n "${IPIR_CUDA_FEATURE:-}" ]]; then echo "$IPIR_CUDA_FEATURE"; exit 0; fi
for d in "$ROOT"/.tools/nvrtc-13*; do [[ -d "$d" ]] && { echo cuda-13; exit 0; }; done
for d in "$ROOT"/.tools/nvrtc-12*; do [[ -d "$d" ]] && { echo cuda-12; exit 0; }; done
libs=$( (ldconfig -p 2>/dev/null || /sbin/ldconfig -p 2>/dev/null) | grep -o 'libnvrtc\.so\.[0-9]*' | sort -u)
if grep -q 'libnvrtc\.so\.13' <<<"$libs"; then echo cuda-13; else echo cuda-12; fi
