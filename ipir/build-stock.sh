#!/usr/bin/env bash
# Builds the benchmark programs against the vendored crates without patch 0002 ("stock"), next to the patched build
# in ipir/target/release/. The source is a copy of ipir/ (Cargo files, bench/, vendor/) in ipir/target/stock/src/,
# with patch 0002 reversed; the programs land in ipir/target/stock/release/. `pack_digest` and `cpu_fetch` read the
# matrix accessor that patch 0002 adds, so the stock build leaves them out.
#
#   ipir/build-stock.sh [cargo build options, e.g. -j 4]
set -euo pipefail
cd "$(dirname "$0")"
IPIR=$PWD
SRC=target/stock/src
STAMP=$( { cat patches/0002-faster-block-compile.patch Cargo.toml Cargo.lock; find bench vendor -type f -not -name .DS_Store | LC_ALL=C sort | xargs cat; } | shasum -a 256 | cut -c1-64)
if [[ "$(cat "$SRC/.stamp" 2>/dev/null)" != "$STAMP" ]]; then
  echo "== preparing the stock source in ipir/$SRC (patch 0002 reversed)"
  rm -rf "$SRC"
  mkdir -p "$SRC"
  cp -Rp Cargo.toml Cargo.lock bench vendor "$SRC/"
  (cd "$SRC/vendor/ipir-sp" && patch -R -p1 -s < "$IPIR/patches/0002-faster-block-compile.patch")
  echo "$STAMP" > "$SRC/.stamp"
fi
cargo build --release --locked --offline --manifest-path "$SRC/Cargo.toml" --target-dir target/stock -p ipir-bench \
  --bin compile_block --bin block_worker --bin cpu_server "$@" 2>/dev/null \
  || cargo build --release --locked --manifest-path "$SRC/Cargo.toml" --target-dir target/stock -p ipir-bench \
    --bin compile_block --bin block_worker --bin cpu_server "$@"
