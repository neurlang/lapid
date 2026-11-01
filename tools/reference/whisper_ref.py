# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.4",
#   "transformers>=4.45",
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
"""Reference activations for lapid's Whisper verification (HF transformers, fp32, CPU, greedy).

    uv run tools/reference/whisper_ref.py --model openai/whisper-small \
        --audio data/jfk.wav --out data/reference/whisper-small

Writes (see the model's verify.rs):
  <out>/activations.safetensors  mel [80,3000], encoder.embed [1500,d], encoder.layers.{i} [1500,d]
                                 (raw block outputs, before the final LayerNorm), encoder.out [1500,d]
                                 (after the final LayerNorm), decoder.logits.prompt [n_prompt, n_vocab]
  <out>/result.json              prompt_tokens, tokens (generated, prompt and EOT excluded,
                                 timestamps included), text_tokens, text, segments, special ids,
                                 audio (path relative to <out>), versions
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from safetensors.numpy import save_file
from scipy.signal import resample_poly
from transformers import WhisperForConditionalGeneration, WhisperProcessor

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


def strip_prompt(seq: list[int], prompt: list[int], sot: int, eot: int, ts_begin: int) -> list[int]:
    """Generated ids without the decoder prompt and without the trailing EOT."""
    if seq[: len(prompt)] == prompt:
        seq = seq[len(prompt) :]
    else:
        # Some transformers versions emit a slightly different forced prefix (or none):
        # drop leading non-timestamp special tokens up to the first generated token.
        i = 0
        while i < len(seq) and sot <= seq[i] < ts_begin:
            i += 1
        seq = seq[i:]
    while seq and seq[-1] == eot:
        seq = seq[:-1]
    return seq


def segments_from_tokens(tokens: list[int], ts_begin: int, eot: int, tokenizer) -> list[dict]:
    """Pairs <|t0|> text <|t1|> into segments (OpenAI whisper semantics, 20 ms per step)."""
    segments: list[dict] = []
    start: float | None = None
    body: list[int] = []
    for tok in tokens:
        if tok >= ts_begin:
            t = (tok - ts_begin) * 0.02
            if start is None:
                start = t
                body = []
            else:
                segments.append(
                    {
                        "start": round(start, 2),
                        "end": round(t, 2),
                        "text": tokenizer.decode(body, skip_special_tokens=True).strip(),
                        "tokens": body,
                    }
                )
                start = None
                body = []
        elif tok < eot:
            body.append(tok)
    if start is not None and body:
        segments.append(
            {
                "start": round(start, 2),
                "end": None,
                "text": tokenizer.decode(body, skip_special_tokens=True).strip(),
                "tokens": body,
            }
        )
    return segments


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="openai/whisper-small")
    ap.add_argument("--audio", default="data/jfk.wav", type=Path)
    ap.add_argument("--out", default="data/reference/whisper-small", type=Path)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    args.out.mkdir(parents=True, exist_ok=True)

    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()
    tok = processor.tokenizer

    audio = load_audio(args.audio)
    feats = processor.feature_extractor(
        audio, sampling_rate=SAMPLE_RATE, return_tensors="pt"
    ).input_features.to(torch.float32)  # [1, n_mels, 3000]

    sot = tok.convert_tokens_to_ids("<|startoftranscript|>")
    eot = tok.convert_tokens_to_ids("<|endoftext|>")
    en = tok.convert_tokens_to_ids("<|en|>")
    transcribe = tok.convert_tokens_to_ids("<|transcribe|>")
    no_ts = tok.convert_tokens_to_ids("<|notimestamps|>")
    ts_begin = tok.convert_tokens_to_ids("<|0.00|>")
    prompt = [sot, en, transcribe]

    # ---- encoder: hooks capture the raw output of every block (HF's hidden_states[-1]
    # is post-LayerNorm, so the last block's raw output is only reachable via a hook).
    encoder = model.model.encoder
    layer_out: dict[int, torch.Tensor] = {}

    def make_hook(i: int):
        def hook(_mod, _inp, out):
            layer_out[i] = out[0] if isinstance(out, (tuple, list)) else out

        return hook

    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(encoder.layers)]
    enc = encoder(feats, output_hidden_states=True, return_dict=True)
    for h in handles:
        h.remove()

    tensors: dict[str, np.ndarray] = {
        "mel": f32(feats[0]),
        "encoder.embed": f32(enc.hidden_states[0][0]),
    }
    for i in range(len(encoder.layers)):
        tensors[f"encoder.layers.{i}"] = f32(layer_out[i][0])
    tensors["encoder.out"] = f32(enc.last_hidden_state[0])

    # ---- decoder logits for the prompt (no cache), on the reference encoder output.
    dec_ids = torch.tensor([prompt], dtype=torch.long)
    out = model(encoder_outputs=(enc.last_hidden_state,), decoder_input_ids=dec_ids, use_cache=False)
    tensors["decoder.logits.prompt"] = f32(out.logits[0])

    save_file(tensors, str(args.out / "activations.safetensors"))

    # ---- full greedy transcription with timestamps.
    gen = model.generate(
        feats,
        return_timestamps=True,
        language="en",
        task="transcribe",
        num_beams=1,
        do_sample=False,
    )
    seq = gen.sequences[0].tolist() if hasattr(gen, "sequences") else gen[0].tolist()
    tokens = strip_prompt(seq, prompt, sot, eot, ts_begin)
    text_tokens = [t for t in tokens if t < eot]
    text = tok.decode(text_tokens, skip_special_tokens=True).strip()

    result = {
        "model": args.model,
        "dtype": "fp32",
        "audio": os.path.relpath(args.audio.resolve(), args.out.resolve()),
        "prompt_tokens": prompt,
        "tokens": tokens,
        "text_tokens": text_tokens,
        "text": text,
        "segments": segments_from_tokens(tokens, ts_begin, eot, tok),
        "special": {
            "sot": sot,
            "eot": eot,
            "transcribe": transcribe,
            "no_timestamps": no_ts,
            "timestamp_begin": ts_begin,
            "en": en,
        },
        "shapes": {k: list(v.shape) for k, v in tensors.items()},
        "versions": {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "numpy": np.__version__,
        },
    }
    (args.out / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    print(json.dumps({"out": str(args.out), "text": text, "n_tokens": len(tokens)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
