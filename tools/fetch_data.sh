#!/usr/bin/env bash
# Fetch the small test assets lapid's verification and benchmarks use into data/ (gitignored).
#   tools/fetch_data.sh          # data/jfk.wav
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$ROOT/data"
mkdir -p "$DATA"

fetch() { # url dest
  local url="$1" dest="$2"
  if [[ -s "$dest" ]]; then
    echo "ok   $dest (cached)"
    return
  fi
  echo "get  $url"
  curl -fL --retry 3 --progress-bar -o "$dest.part" "$url"
  mv "$dest.part" "$dest"
  echo "ok   $dest ($(wc -c <"$dest") bytes)"
}

fetch "https://github.com/ggml-org/whisper.cpp/raw/master/samples/jfk.wav" "$DATA/jfk.wav"

# Sanity check: RIFF/WAVE header.
if [[ "$(head -c 4 "$DATA/jfk.wav")" != "RIFF" ]]; then
  echo "error: $DATA/jfk.wav is not a WAV file" >&2
  exit 1
fi
