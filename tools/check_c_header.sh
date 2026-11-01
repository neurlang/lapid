#!/usr/bin/env bash
# lapid-c's build.rs regenerates crates/lapid-c/include/lapid.h on every build.
# This rebuilds it and fails if the committed header differs.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo build -q -p lapid-c
git diff --exit-code crates/lapid-c/include/lapid.h && echo "ok: lapid.h is up to date"
