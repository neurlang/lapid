#!/usr/bin/env bash
# Build the lapid browser bundle into plans/web/pkg/:
#   lapid.js + lapid_bg.wasm (wasm-bindgen --target web), lapid_jspi.js (JSPI import module), snippets/
# Needs the wasm32-unknown-unknown target and the wasm-bindgen CLI matching Cargo.lock
# (installed with `cargo install wasm-bindgen-cli --version <x> --locked` when missing). No npm.
#
# Usage: tools/build_web.sh [--dev]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PROFILE=release
if [[ "${1:-}" == "--dev" ]]; then
  PROFILE=dev
fi
OUT="$ROOT/plans/web/pkg"
TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"

if ! rustup target list --installed 2>/dev/null | grep -q '^wasm32-unknown-unknown$'; then
  rustup target add wasm32-unknown-unknown
fi

cargo build --target wasm32-unknown-unknown -p lapid-web --profile "$PROFILE"
DIR="$PROFILE"
[[ "$PROFILE" == "dev" ]] && DIR=debug
WASM="$TARGET_DIR/wasm32-unknown-unknown/$DIR/lapid_web.wasm"

# wasm-bindgen CLI must match the wasm-bindgen crate version in Cargo.lock exactly
VER="$(awk '/^name = "wasm-bindgen"$/ { getline; gsub(/"/, "", $3); print $3; exit }' Cargo.lock)"
if [[ -z "$VER" ]]; then
  echo "build_web: wasm-bindgen not found in Cargo.lock" >&2
  exit 1
fi
if ! command -v wasm-bindgen >/dev/null || [[ "$(wasm-bindgen --version | awk '{print $2}')" != "$VER" ]]; then
  echo "build_web: installing wasm-bindgen-cli $VER"
  cargo install wasm-bindgen-cli --version "$VER" --locked
fi

rm -rf "$OUT"
wasm-bindgen --target web --out-dir "$OUT" --out-name lapid "$WASM"

# The raw `lapid_jspi` import module (WebAssembly.Suspending `lapid_yield`) is a bare
# specifier in the glue; point it at the file shipped next to it.
cp "$ROOT/crates/lapid-web/js/lapid_yield.js" "$OUT/lapid_jspi.js"
sed -i.bak -e "s#from 'lapid_jspi'#from './lapid_jspi.js'#g" -e 's#from "lapid_jspi"#from "./lapid_jspi.js"#g' "$OUT/lapid.js"
rm -f "$OUT/lapid.js.bak"
if ! grep -q "lapid_jspi.js" "$OUT/lapid.js"; then
  echo "build_web: could not find the lapid_jspi import in lapid.js (wasm-bindgen output changed?)" >&2
  exit 1
fi

ls -la "$OUT"
echo "build_web: done -> $OUT (serve with: python3 plans/web/serve.py)"
