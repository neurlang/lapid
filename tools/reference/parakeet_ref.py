# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.4",
#   "transformers>=5.9",
#   "librosa",
#   "numpy",
#   "safetensors",
#   "soundfile",
#   "scipy",
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
"""Reference activations for lapid's Parakeet TDT verification (HF transformers
`ParakeetForTDT`, fp32, CPU, greedy TDT).

    uv run tools/reference/parakeet_ref.py --model nvidia/parakeet-tdt-0.6b-v3 \
        --audio data/jfk.wav --out data/reference/parakeet/en

`--model` may be a local directory with the HF files (config.json, model.safetensors,
tokenizer.json, processor_config.json).

Writes:
  <out>/activations.safetensors
      mel [T, 128]                  normalized log-mel, valid frames only (the extractor's
                                    extra last frame, masked to zero by HF, is dropped)
      encoder.subsampling [T', 1024] conv subsampling + linear output
      encoder.layers.{i} [T', 1024] FastConformer block outputs (after norm_out)
      encoder.projected [T', 640]   encoder_projector output (the joint's encoder input)
      decoder.step_logits [S, V+D]  joint logits of every greedy step (blank steps included)
  <out>/result.json
      tokens / token_frames / token_durations   emitted (non-blank) tokens with frame index and
                                                 duration (frames of 80 ms)
      steps: frames / tokens / durations        every greedy step (decoder replay path)
      text, words (NeMo-style word offsets), hf_generate_tokens (cross-check), audio, versions

Greedy decoding follows NeMo's GreedyTDTInfer: a blank with duration 0 advances one frame;
at most `max_symbols_per_step` symbols per frame. HF `generate` is run as a cross-check.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from safetensors.numpy import save_file
from scipy.signal import resample_poly
from transformers import AutoProcessor, ParakeetForTDT

SAMPLE_RATE = 16_000


def load_audio(path: Path) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        g = np.gcd(sr, SAMPLE_RATE)
        mono = resample_poly(mono, SAMPLE_RATE // g, sr // g).astype(np.float32)
    return np.ascontiguousarray(mono, dtype=np.float32)


def f32(t: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(t.detach().to(torch.float32).cpu().numpy())


@torch.no_grad()
def greedy_tdt(model: ParakeetForTDT, enc: torch.Tensor, n_valid: int):
    """NeMo greedy TDT over projected encoder frames enc [T, H]. Returns step records."""
    cfg = model.config
    blank, vocab = cfg.blank_token_id, cfg.vocab_size
    durations = list(cfg.durations)
    max_symbols = cfg.max_symbols_per_step
    dec = model.decoder

    def pred(token: int, state):
        emb = dec.embedding(torch.tensor([[token]]))
        out, state = dec.lstm(emb, state)
        return dec.decoder_projector(out)[0, 0], state

    g, state = pred(blank, None)
    t, symbols = 0, 0
    steps = {"frames": [], "tokens": [], "durations": []}
    logits_all = []
    emitted = {"tokens": [], "frames": [], "durations": []}
    while t < n_valid:
        logits = model.joint.head(model.joint.activation(enc[t] + g))
        logits_all.append(f32(logits))
        tok = int(torch.argmax(logits[:vocab]))
        dur = durations[int(torch.argmax(logits[vocab:]))]
        steps["frames"].append(t)
        steps["tokens"].append(tok)
        steps["durations"].append(dur)
        if tok != blank:
            emitted["tokens"].append(tok)
            emitted["frames"].append(t)
            emitted["durations"].append(dur)
            g, state = pred(tok, state)
            symbols += 1
        if tok == blank and dur == 0:
            dur = 1
        if dur == 0 and symbols >= max_symbols:
            dur = 1
        if dur > 0:
            symbols = 0
        t += dur
    return steps, emitted, np.stack(logits_all)


PUNCTUATION = {"?", "'", "¡", "¿", "-", ":", ",", "%", "/", ".", "!"}


def words_from_tokens(tokenizer, tokens, frames, durs, frame_s):
    """NeMo-style word offsets: token offsets (start frame, start + duration), punctuation
    tokens refined to the previous token's end (`_refine_timestamps_tdt`), then a word
    starts at every token beginning with '▁'."""
    words = []
    prev_end = 0.0
    for tok, fr, du in zip(tokens, frames, durs):
        piece = tokenizer.convert_ids_to_tokens(tok)
        start, end = fr * frame_s, (fr + du) * frame_s
        if piece in PUNCTUATION and words:
            start = end = prev_end
        prev_end = end
        if piece.startswith("▁") or not words:
            words.append({"ids": [tok], "start": start, "end": end})
        else:
            words[-1]["ids"].append(tok)
            words[-1]["end"] = end
    for w in words:
        w["word"] = tokenizer.decode(w.pop("ids"), skip_special_tokens=True, group_tokens=False).strip()
    return words


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nvidia/parakeet-tdt-0.6b-v3")
    ap.add_argument("--audio", default="data/jfk.wav")
    ap.add_argument("--out", default="data/reference/parakeet/en")
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(min(16, os.cpu_count() or 1))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    audio = load_audio(Path(args.audio))

    processor = AutoProcessor.from_pretrained(args.model)
    model = ParakeetForTDT.from_pretrained(args.model, torch_dtype=torch.float32, attn_implementation="eager").eval()
    cfg = model.config

    feats = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    mel_full, mask = feats["input_features"], feats["attention_mask"]
    n_mel = int(mask.sum())

    acts: dict[str, np.ndarray] = {"mel": f32(mel_full[0, :n_mel])}
    hooks = []
    enc = model.encoder
    hooks.append(enc.subsampling.register_forward_hook(lambda m, i, o: acts.__setitem__("encoder.subsampling", f32(o[0]))))
    for i, layer in enumerate(enc.layers):
        hooks.append(layer.register_forward_hook(lambda m, inp, o, i=i: acts.__setitem__(f"encoder.layers.{i}", f32((o[0] if isinstance(o, tuple) else o)[0]))))
    with torch.no_grad():
        audio_out = model.get_audio_features(input_features=mel_full, attention_mask=mask, output_attention_mask=True)
    for h in hooks:
        h.remove()
    n_valid = int(audio_out.attention_mask.sum()) if audio_out.attention_mask is not None else audio_out.pooler_output.shape[1]
    # Keep valid encoder frames only.
    for k in list(acts):
        if k.startswith("encoder."):
            acts[k] = np.ascontiguousarray(acts[k][:n_valid])
    projected = audio_out.pooler_output[0]
    acts["encoder.projected"] = f32(projected[:n_valid])

    steps, emitted, logits = greedy_tdt(model, projected, n_valid)
    acts["decoder.step_logits"] = logits

    tok = processor.tokenizer
    text = tok.decode(emitted["tokens"], skip_special_tokens=True, group_tokens=False).strip()

    with torch.no_grad():
        gen = model.generate(input_features=mel_full, attention_mask=mask)
    seq = gen.sequences[0].tolist()
    hf_tokens = [t for t in seq[1:] if t not in (cfg.blank_token_id, cfg.pad_token_id)]
    if hf_tokens != emitted["tokens"]:
        print(f"WARNING: HF generate tokens differ from NeMo-style greedy ({len(hf_tokens)} vs {len(emitted['tokens'])})")

    frame_s = processor.feature_extractor.hop_length / SAMPLE_RATE * 8
    result = {
        "model": args.model,
        "audio": os.path.relpath(Path(args.audio).resolve(), out.resolve()),
        "audio_samples": int(audio.shape[0]),
        "mel_frames": n_mel,
        "encoder_frames": n_valid,
        "frame_seconds": frame_s,
        "blank_id": cfg.blank_token_id,
        "vocab_size": cfg.vocab_size,
        "durations": list(cfg.durations),
        "tokens": emitted["tokens"],
        "token_frames": emitted["frames"],
        "token_durations": emitted["durations"],
        "steps": steps,
        "text": text,
        "words": words_from_tokens(tok, emitted["tokens"], emitted["frames"], emitted["durations"], frame_s),
        "hf_generate_tokens": hf_tokens,
        "hf_generate_text": tok.decode(hf_tokens, skip_special_tokens=True, group_tokens=False).strip(),
        "versions": {"torch": torch.__version__, "transformers": __import__("transformers").__version__},
    }
    save_file(acts, str(out / "activations.safetensors"))
    (out / "result.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(text)
    print({k: v.shape for k, v in acts.items() if not k.startswith("encoder.layers.")})


if __name__ == "__main__":
    main()
