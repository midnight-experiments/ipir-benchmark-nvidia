#!/usr/bin/env bash
# Builds libhide_budget.so next to this script. Needs a C compiler; no Vulkan headers, no root.
set -euo pipefail
cd "$(dirname "$0")"
cc -O2 -shared -fPIC -fvisibility=hidden -o libhide_budget.so hide_budget.c
echo "built $PWD/libhide_budget.so"
