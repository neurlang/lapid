# /// script
# requires-python = ">=3.10"
# dependencies = ["sherpa-onnx", "numpy", "soundfile", "huggingface_hub"]
# ///
"""sherpa-onnx baseline for `lapid bench zipformer`: the same streaming Zipformer2 as ONNX, fed
in 100 ms pieces as fast as possible on the CPU. Reports RTF and first-partial latency (audio
fed when the first token appeared, and the compute of that call) like lapid's bench.

    # LibriSpeech model (icefall 2023-05-17 exported by k2-fsa, chunk 16 / left 128)
    uv run tools/reference/zipformer_sherpa_bench.py --repo csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26 \
        --prefix "-chunk-16-left-128" --audio data/jfk.wav [--int8] [--threads 4]
    # local export (e.g. the Hebrew model's sherpa-onnx/ folder: encoder.int8.onnx ...)
    uv run tools/reference/zipformer_sherpa_bench.py --dir DIR --int8 --audio data/reference/zipformer/fleurs_he_0.wav
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import sherpa_onnx
import soundfile as sf


def files(a):
    if a.dir:
        d = Path(a.dir)
        tag = ".int8" if a.int8 else ""
        return {k: str(d / f"{k}{a.prefix}{tag}.onnx") for k in ("encoder", "decoder", "joiner")} | {"tokens": str(d / "tokens.txt")}
    from huggingface_hub import hf_hub_download

    tag = ".int8" if a.int8 else ""
    out = {}
    for k in ("encoder", "decoder", "joiner"):
        name = f"{k}-epoch-99-avg-1{a.prefix}{tag}.onnx"
        out[k] = hf_hub_download(a.repo, name)
    out["tokens"] = hf_hub_download(a.repo, "tokens.txt")
    return out


def one_run(rec, audio, sr, piece):
    s = rec.create_stream()
    t = time.perf_counter()
    first = None
    fed = 0
    for i in range(0, len(audio), piece):
        tc = time.perf_counter()
        s.accept_waveform(sr, audio[i:i + piece])
        while rec.is_ready(s):
            rec.decode_stream(s)
        fed += len(audio[i:i + piece])
        if first is None and rec.get_result(s).strip():
            first = (fed / sr, (time.perf_counter() - tc) * 1e3)
    tail = np.zeros(int(0.66 * sr), dtype=np.float32)  # flush the look-ahead (sherpa-onnx examples)
    s.accept_waveform(sr, tail)
    s.input_finished()
    while rec.is_ready(s):
        rec.decode_stream(s)
    return (time.perf_counter() - t) * 1e3, first, rec.get_result(s).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26")
    ap.add_argument("--dir")
    ap.add_argument("--prefix", default="-chunk-16-left-128")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--audio", default="data/jfk.wav")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--provider", default="cpu")
    a = ap.parse_args()
    f = files(a)
    rec = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=f["tokens"], encoder=f["encoder"], decoder=f["decoder"], joiner=f["joiner"],
        num_threads=a.threads, provider=a.provider, decoding_method="greedy_search", model_type="zipformer2")
    audio, sr = sf.read(a.audio, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    piece = sr // 10
    one_run(rec, audio, sr, piece)
    runs = [one_run(rec, audio, sr, piece) for _ in range(a.runs)]
    total = float(np.median([r[0] for r in runs]))
    dur = len(audio) / sr
    print(json.dumps({
        "engine": "sherpa-onnx", "int8": a.int8, "threads": a.threads, "provider": a.provider, "audio": a.audio,
        "audio_seconds": dur, "total_ms_median": total, "realtime_factor": total / 1e3 / dur,
        "first_partial_audio_s": runs[-1][1][0] if runs[-1][1] else None,
        "first_partial_compute_ms": runs[-1][1][1] if runs[-1][1] else None,
        "text": runs[-1][2],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
