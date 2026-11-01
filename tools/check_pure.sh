#!/usr/bin/env bash
# Fails if any crate in the default build of the workspace (default features, normal +
# build dependencies, host target) depends on a C/C++ build helper (cc, cmake), i.e. would
# compile C/asm. See docs/PROJECT.md "Pure Rust".
#
# Allow-list (crate -> reason), checked by name:
#   blake3  built with the "pure" feature (workspace Cargo.toml): its build script lists
#           `cc` as a build-dependency but compiles nothing in pure mode.
set -euo pipefail
cd "$(dirname "$0")/.."

ALLOW=(blake3)
status=0
for helper in cc cmake; do
    # Direct dependents of the helper, one per line ("<depth><name> v<version>").
    out=$(cargo tree --workspace -e normal,build -i "$helper" --depth 1 --prefix depth 2>/dev/null || true)
    while read -r line; do
        [[ "$line" =~ ^1([A-Za-z0-9_-]+)\  ]] || continue
        dep="${BASH_REMATCH[1]}"
        allowed=0
        for a in "${ALLOW[@]}"; do [[ "$dep" == "$a" ]] && allowed=1; done
        if (( allowed )); then
            echo "ok: $dep uses $helper (allow-listed)"
        else
            echo "FAIL: $dep builds native code via $helper:" >&2
            cargo tree --workspace -e normal,build -i "$dep" >&2 || true
            status=1
        fi
    done <<< "$out"
done
(( status == 0 )) && echo "pure: no C/C++ builds in the default dependency tree"
exit $status
