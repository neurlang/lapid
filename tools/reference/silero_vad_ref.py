# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.4",
#   "numpy",
#   "safetensors",
# ]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Reference data for lapid's Silero VAD v5 (official TorchScript model, fp32, CPU).

    uv run tools/reference/silero_vad_ref.py --out data/reference/vad

The model is Silero VAD **v5** from the `silero-vad==5.1.2` wheel on PyPI
(`silero_vad/data/silero_vad.jit`; MIT license). The torch hub cache
(`~/.cache/torch/hub/snakers4_silero-vad_master`) tracks master, which is v6 (6.2.x) and is
not used. The wheel is downloaded once into `~/.cache/lapid/reference/` (or pass `--wheel`).
Only `utils_vad.py` and the `.jit` are used; `torchaudio` (only needed by silero's
`read_audio`) is stubbed so no extra packages are installed.

Writes into `--out`:
  model/model.safetensors   16 kHz branch weights, torch names without the `_model.` prefix
  model/config.json         {"model_type": "silero_vad", "version": "v5", "sample_rate": 16000, ...}
  gaps.wav                  1089-134686-0001.wav + 1 s silence + 1221-135766-0002.wav (16 kHz)
  <sample>/result.json      audio path, sample count, `get_speech_timestamps` output for several
                            option sets, `VADIterator` events (512-sample chunks)
  <sample>/activations.safetensors
      probs [N]             per-window probabilities (`model.audio_forward`, state reset first)
      stft [K, 129, 4]      STFT magnitude of the first K windows (64-sample context prepended)
      encoder [K, 128]      encoder output
      lstm.h / lstm.c [K, 128]  LSTM cell state after each window
      prob [K]              decoder output

`--bench` also times the TorchScript model (and onnxruntime, if installed) on 60 s of audio.
"""

import argparse
import hashlib
import json
import os
import sys
import time
import types
import urllib.request
import wave
import zipfile
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

WHEEL_URL = (
    "https://files.pythonhosted.org/packages/98/f7/5ae11d13fbb733cd3bfd7ff1c3a3902e6f55437df4b72307c1f168146268/"
    "silero_vad-5.1.2-py3-none-any.whl"
)
SR = 16000
WINDOW = 512
CONTEXT = 64
TRACE_WINDOWS = 32
LIBRI = "/home/yakov/Documents/projects/audio/zipformer2-66m-streaming-he/models/en-streaming-zipformer-2023-05-17/test_wavs"

# get_speech_timestamps option sets (silero argument names); the first is the default.
OPTION_SETS = [
    {},
    {"threshold": 0.6, "neg_threshold": 0.3, "min_speech_duration_ms": 100, "min_silence_duration_ms": 300,
     "speech_pad_ms": 50, "max_speech_duration_s": 2.0},
    {"threshold": 0.4, "min_speech_duration_ms": 50, "min_silence_duration_ms": 50, "speech_pad_ms": 0,
     "max_speech_duration_s": 1.0},
]
# VADIterator option sets (it has no neg_threshold / min_speech / max_speech).
ITERATOR_SETS = [{}, {"threshold": 0.6, "min_silence_duration_ms": 300, "speech_pad_ms": 50}]


def fetch_wheel(path):
    if path:
        return Path(path)
    cache = Path.home() / ".cache" / "lapid" / "reference" / "silero_vad-5.1.2-py3-none-any.whl"
    if not cache.is_file():
        cache.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {WHEEL_URL}", file=sys.stderr)
        urllib.request.urlretrieve(WHEEL_URL, cache)
    return cache


def load_silero(wheel):
    root = wheel.parent / "silero_vad-5.1.2"
    if not (root / "silero_vad" / "data" / "silero_vad.jit").is_file():
        with zipfile.ZipFile(wheel) as z:
            z.extractall(root)
    sys.modules.setdefault("torchaudio", types.ModuleType("torchaudio"))
    sys.path.insert(0, str(root))
    from silero_vad import utils_vad  # noqa: E402

    jit = root / "silero_vad" / "data" / "silero_vad.jit"
    model = utils_vad.init_jit_model(str(jit))
    return model, utils_vad, jit


def read_wav(path):
    with wave.open(str(path)) as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2, path
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return data


def write_wav(path, pcm):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.astype("<i2").tobytes())


def export_model(model, jit, out):
    out.mkdir(parents=True, exist_ok=True)
    sd = {k[len("_model."):]: v.detach().clone().contiguous().float()
          for k, v in model.state_dict().items() if k.startswith("_model.")}
    save_file(sd, str(out / "model.safetensors"))
    cfg = {
        "model_type": "silero_vad",
        "version": "v5",
        "sample_rate": SR,
        "window_size_samples": WINDOW,
        "context_size_samples": CONTEXT,
        "filter_length": int(model._model.stft.filter_length),
        "hop_length": int(model._model.stft.hop_length),
        "encoder": [[129, 128, 1], [128, 64, 2], [64, 64, 2], [64, 128, 1]],
        "hidden_size": 128,
        "source": {"package": "silero-vad==5.1.2", "file": "silero_vad/data/silero_vad.jit",
                   "sha256": hashlib.sha256(jit.read_bytes()).hexdigest(), "license": "MIT"},
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")


@torch.no_grad()
def trace(model, x, k):
    """First k windows through the sub-modules, checking against the full model call."""
    m = model._model
    ctx = torch.zeros(1, CONTEXT)
    state = torch.zeros(2, 1, 128)
    model.reset_states()
    outs = {"stft": [], "encoder": [], "lstm.h": [], "lstm.c": [], "prob": []}
    for i in range(k):
        chunk = x[i * WINDOW:(i + 1) * WINDOW].unsqueeze(0)
        inp = torch.cat([ctx, chunk], 1)
        mag = m.stft(inp)
        enc = m.encoder(mag)
        out, state = m.decoder(enc, state)
        prob = out.squeeze(1).mean(1)
        full = model(chunk, SR)
        assert torch.equal(full.reshape(-1), prob.reshape(-1)), (i, full, prob)
        outs["stft"].append(mag[0])
        outs["encoder"].append(enc[0, :, 0])
        outs["lstm.h"].append(state[0, 0])
        outs["lstm.c"].append(state[1, 0])
        outs["prob"].append(prob.reshape(()))
        ctx = inp[:, -CONTEXT:]
    return {n: torch.stack(v).contiguous() for n, v in outs.items()}


def iterator_events(model, utils_vad, x, opts):
    it = utils_vad.VADIterator(model, sampling_rate=SR, **opts)
    events = []
    n = x.shape[0]
    padded = torch.nn.functional.pad(x, (0, (-n) % WINDOW))
    for i in range(0, padded.shape[0], WINDOW):
        ev = it(padded[i:i + WINDOW])
        if ev:
            events.append({"window": i // WINDOW, **{k: int(v) for k, v in ev.items()}})
    it.reset_states()
    return events


def sample(model, utils_vad, pcm, audio_rel, out):
    out.mkdir(parents=True, exist_ok=True)
    x = torch.from_numpy(pcm.astype(np.float32) / 32768.0)
    with torch.no_grad():
        probs = model.audio_forward(x.unsqueeze(0), SR)[0].float()
    acts = {"probs": probs.contiguous()}
    acts.update(trace(model, x, min(TRACE_WINDOWS, probs.shape[0])))
    segs = []
    for opts in OPTION_SETS:
        ts = utils_vad.get_speech_timestamps(x, model, sampling_rate=SR, **opts)
        segs.append({"options": opts, "segments": [[int(s["start"]), int(s["end"])] for s in ts]})
    # get_speech_timestamps runs the model window by window: same probabilities.
    probs2 = []
    model.reset_states()
    padded = torch.nn.functional.pad(x, (0, (-x.shape[0]) % WINDOW))
    with torch.no_grad():
        for i in range(0, padded.shape[0], WINDOW):
            probs2.append(model(padded[i:i + WINDOW], SR).item())
    assert np.array_equal(np.array(probs2, dtype=np.float32), probs.numpy()), "audio_forward != per-window calls"
    events = [{"options": o, "events": iterator_events(model, utils_vad, x, o)} for o in ITERATOR_SETS]
    save_file(acts, str(out / "activations.safetensors"))
    result = {
        "audio": audio_rel,
        "num_samples": int(x.shape[0]),
        "num_windows": int(probs.shape[0]),
        "trace_windows": int(acts["prob"].shape[0]),
        "speech_timestamps": segs,
        "iterator": events,
        "reference": "silero-vad 5.1.2 TorchScript (silero_vad.jit), fp32 CPU",
    }
    (out / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    print(f"{out}: {probs.shape[0]} windows, {len(segs[0]['segments'])} default segments", file=sys.stderr)


@torch.no_grad()
def bench(model, jit, x, runs=3):
    seconds = x.shape[0] / SR
    rows = {}
    best = float("inf")
    for _ in range(runs):
        t = time.perf_counter()
        model.audio_forward(x.unsqueeze(0), SR)
        best = min(best, time.perf_counter() - t)
    rows["torchscript_1thread"] = {"seconds": best, "rtf": best / seconds}
    try:
        import onnxruntime  # noqa: F401
        onnx = Path(jit).with_name("silero_vad.onnx")
        from silero_vad.utils_vad import OnnxWrapper

        ort = OnnxWrapper(str(onnx), force_onnx_cpu=True)
        best = float("inf")
        for _ in range(runs):
            t = time.perf_counter()
            ort.audio_forward(x.unsqueeze(0), SR)
            best = min(best, time.perf_counter() - t)
        rows["onnxruntime_1thread"] = {"seconds": best, "rtf": best / seconds}
    except ImportError:
        rows["onnxruntime_1thread"] = "onnxruntime not installed"
    print(json.dumps({"audio_seconds": seconds, **rows}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/reference/vad")
    ap.add_argument("--wheel", default=None, help="local silero_vad-5.1.2-py3-none-any.whl")
    ap.add_argument("--jfk", default="data/jfk.wav")
    ap.add_argument("--libri", default=LIBRI)
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    model, utils_vad, jit = load_silero(fetch_wheel(args.wheel))
    model.eval()
    export_model(model, jit, out / "model")

    jfk = read_wav(args.jfk)
    sample(model, utils_vad, jfk, os.path.relpath(args.jfk, out / "jfk"), out / "jfk")
    a = read_wav(Path(args.libri) / "1089-134686-0001.wav")
    b = read_wav(Path(args.libri) / "1221-135766-0002.wav")
    gaps = np.concatenate([a, np.zeros(SR, dtype=np.int16), b])
    write_wav(out / "gaps.wav", gaps)
    sample(model, utils_vad, gaps, "../gaps.wav", out / "gaps")

    if args.bench:
        x = torch.from_numpy(np.tile(jfk, 6)[: 60 * SR].astype(np.float32) / 32768.0)
        bench(model, jit, x)


if __name__ == "__main__":
    main()
