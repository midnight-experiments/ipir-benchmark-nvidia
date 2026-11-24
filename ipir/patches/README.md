# Patches to the vendored IPIR+SP crates

Applied to `../vendor/ipir-sp/` from that folder with `patch -p1 < ../../patches/<file>`; the copy in this repository
already has them applied.

| Patch | Crate, function | What it changes |
|---|---|---|
| 0001 | `ipir-native-client`, `NativePublicSetup` | Adds `new_without_columns`: the setup identifier hashes the rows but not the column count, under its own domain tag (`/umbradb-without-columns-v1/`), so a table can gain column blocks and the masks and packing matrices of the blocks it has stay the same. `new` is unchanged. Its test: `cargo test --release -p ipir-native-client setup_without_columns`. |

The benchmark builds every table with `new_without_columns` and seed = snapshot = 32 zero bytes, as the design does.
