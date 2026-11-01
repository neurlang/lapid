#!/usr/bin/env bash
# whisper.cpp baseline for lapid: build (CPU + Vulkan if glslc), fetch ggml-small (+ q8_0),
# run whisper-cli N times per configuration and print a JSON summary (medians, ms).
#
#   tools/bench/whisper_cpp.sh [--runs 5] [--threads 20] [--audio data/jfk.wav] [--no-build]
#
# Output: JSON on stdout, also written to data/whispercpp/bench.json.
# Per-run logs: data/whispercpp/logs/<config>-<run>.log
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="$ROOT/plans/whisper.cpp"
OUT="$ROOT/data/whispercpp"
RUNS=5
THREADS=20
AUDIO="$ROOT/data/jfk.wav"
BUILD=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runs) RUNS="$2"; shift 2 ;;
    --threads|-t) THREADS="$2"; shift 2 ;;
    --audio) AUDIO="$2"; shift 2 ;;
    --no-build) BUILD=0; shift ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

log() { echo "[whisper.cpp] $*" >&2; }

[[ -d "$SRC" ]] || { log "missing $SRC (git clone --depth 1 https://github.com/ggml-org/whisper.cpp plans/whisper.cpp)"; exit 1; }
[[ -s "$AUDIO" ]] || { log "missing $AUDIO (run tools/fetch_data.sh)"; exit 1; }
mkdir -p "$OUT/logs"

JOBS="$(nproc)"
HAVE_VULKAN=0
command -v glslc >/dev/null 2>&1 && HAVE_VULKAN=1

# ---------------------------------------------------------------- build
build() { # dir extra-cmake-args...
  local dir="$1"; shift
  log "configure $dir $*"
  cmake -S "$SRC" -B "$SRC/$dir" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON \
    -DWHISPER_BUILD_TESTS=OFF -DWHISPER_SDL2=OFF "$@" >&2
  log "build $dir"
  cmake --build "$SRC/$dir" --config Release -j "$JOBS" --target whisper-cli whisper-quantize >&2
}

if [[ "$BUILD" == 1 ]]; then
  build build-cpu -DGGML_VULKAN=OFF -DGGML_CUDA=OFF
  if [[ "$HAVE_VULKAN" == 1 ]]; then
    build build-vulkan -DGGML_VULKAN=ON -DGGML_CUDA=OFF
  else
    log "glslc not found: skipping Vulkan build"
  fi
fi

CLI_CPU="$SRC/build-cpu/bin/whisper-cli"
QUANTIZE="$SRC/build-cpu/bin/whisper-quantize"
CLI_VK="$SRC/build-vulkan/bin/whisper-cli"
[[ -x "$CLI_CPU" ]] || { log "missing $CLI_CPU (run without --no-build)"; exit 1; }

# ---------------------------------------------------------------- models
MODEL_F16="$OUT/ggml-small.bin"
MODEL_Q8="$OUT/ggml-small-q8_0.bin"
if [[ ! -s "$MODEL_F16" ]]; then
  log "download ggml-small.bin"
  curl -fL --retry 3 --progress-bar -o "$MODEL_F16.part" \
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin"
  mv "$MODEL_F16.part" "$MODEL_F16"
fi
if [[ ! -s "$MODEL_Q8" ]]; then
  [[ -x "$QUANTIZE" ]] || { log "missing $QUANTIZE"; exit 1; }
  log "quantize q8_0"
  "$QUANTIZE" "$MODEL_F16" "$MODEL_Q8" q8_0 >&2
fi

# ---------------------------------------------------------------- run
# Extract "<key> time = <ms>" values from a whisper-cli log as k=v lines.
parse_timings() { # logfile
  awk '
    /whisper_print_timings:/ && / time = / {
      line = $0
      sub(/.*whisper_print_timings:[ \t]*/, "", line)
      split(line, parts, " time = ")
      key = parts[1]; gsub(/[ \t]/, "", key)
      val = parts[2]; sub(/[ \t]*ms.*/, "", val); gsub(/[ \t]/, "", val)
      print key "=" val
    }' "$1"
}

median() { # numbers...
  printf '%s\n' "$@" | sort -g | awk '{a[NR]=$1} END { if (NR==0) {print "null"} else if (NR%2) {print a[(NR+1)/2]} else {print (a[NR/2]+a[NR/2+1])/2} }'
}

CONFIG_JSON=()

run_config() { # name cli model extra-args...
  local name="$1" cli="$2" model="$3"; shift 3
  [[ -x "$cli" ]] || { log "skip $name (no binary)"; return; }
  local load=() mel=() encode=() decode=() total=() text=""
  log "warmup $name"
  "$cli" -m "$model" -f "$AUDIO" -t "$THREADS" "$@" >"$OUT/logs/$name-warmup.log" 2>&1 || {
    log "$name failed, see $OUT/logs/$name-warmup.log"; return; }
  for ((i = 1; i <= RUNS; i++)); do
    local lf="$OUT/logs/$name-$i.log"
    "$cli" -m "$model" -f "$AUDIO" -t "$THREADS" "$@" >"$lf" 2>&1
    local kv; kv="$(parse_timings "$lf")"
    get() { echo "$kv" | awk -F= -v k="$1" '$1==k {print $2}'; }
    load+=("$(get load)")
    mel+=("$(get mel)")
    encode+=("$(get encode)")
    # whisper.cpp splits decoding into sample + decode + batchd + prompt.
    decode+=("$(awk -v a="$(get sample)" -v b="$(get decode)" -v c="$(get batchd)" -v d="$(get prompt)" \
      'BEGIN { printf "%.2f", a + b + c + d }')")
    total+=("$(get total)")
    log "$name run $i: total $(get total) ms"
  done
  text="$(grep -E '^\[[0-9:. >-]+\]' "$OUT/logs/$name-1.log" | sed -E 's/^\[[^]]*\][ ]*//' | tr '\n' ' ' \
    | sed -E 's/[ ]+$//; s/\\/\\\\/g; s/"/\\"/g')"
  CONFIG_JSON+=("$(printf '{"name":"%s","model":"%s","args":"%s","load_ms":%s,"mel_ms":%s,"encode_ms":%s,"decode_ms":%s,"total_ms":%s,"total_runs_ms":[%s],"text":"%s"}' \
    "$name" "$(basename "$model")" "$*" \
    "$(median "${load[@]}")" "$(median "${mel[@]}")" "$(median "${encode[@]}")" \
    "$(median "${decode[@]}")" "$(median "${total[@]}")" \
    "$(IFS=,; echo "${total[*]}")" "$text")")
}

run_config cpu-f16 "$CLI_CPU" "$MODEL_F16"
run_config cpu-q8_0 "$CLI_CPU" "$MODEL_Q8"
if [[ "$HAVE_VULKAN" == 1 ]]; then
  run_config vulkan-f16 "$CLI_VK" "$MODEL_F16"
  run_config vulkan-q8_0 "$CLI_VK" "$MODEL_Q8"
fi

AUDIO_S="$(awk -v b="$(wc -c <"$AUDIO")" 'BEGIN { printf "%.3f", (b - 44) / 32000 }')"  # 16 kHz s16 mono
COMMIT="$(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo unknown)"
JSON="$(printf '{"engine":"whisper.cpp","commit":"%s","audio":"%s","audio_seconds":%s,"threads":%s,"runs":%s,"configs":[%s]}' \
  "$COMMIT" "$(basename "$AUDIO")" "$AUDIO_S" "$THREADS" "$RUNS" "$(IFS=,; echo "${CONFIG_JSON[*]}")")"
echo "$JSON" > "$OUT/bench.json"
echo "$JSON"
