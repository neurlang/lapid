# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.4",
#   "numpy",
#   "safetensors",
#   "soundfile",
#   "lhotse>=1.20",
#   "packaging",
#   "sentencepiece",
#   "urllib3",
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
"""Reference data for lapid's streaming Zipformer2 transducer (icefall model code, fp32, CPU).

The model code is icefall's `egs/librispeech/ASR/zipformer` (zipformer.py, scaling.py,
subsampling.py, decoder.py, joiner.py, encoder_interface.py and the few `icefall.utils` helpers
they import). `--icefall` points at a Python package directory with those files using relative
imports, e.g. the copy in zipformer2-66m-streaming-he (`src/zipformer2_medium_streaming_he/zipformer`,
icefall c1eda54, imports made relative). k2 is replaced by a stub: inference only needs SwooshL /
SwooshR, implemented here with icefall's own formula.

Streaming follows icefall `zipformer/streaming_decode.py` exactly: lhotse fbank of the whole
utterance (80 mel, 25 / 10 ms, dither 0, snip_edges False), 13 + 30 tail frames of LOG_EPS, then
chunks of `2 * chunk + 13` frames advancing by `2 * chunk`, `encoder_embed.streaming_forward` +
`encoder.streaming_forward` (key padding mask from processed_lens, as icefall / sherpa-onnx),
`joiner.encoder_proj`, and greedy search with at most one symbol per frame, starting from the
hypothesis `[-1] * (context - 1) + [blank]`.

    # export a checkpoint (icefall pretrained.pt with {"model"}, or a model.pt with {"model",
    # "config", "tokenizer"}) to safetensors + config.json + tokens.txt (+ bpe.model)
    uv run tools/reference/zipformer_ref.py export --checkpoint exp/pretrained.pt \
        --bpe data/lang_bpe_500/bpe.model --out data/hf/zipformer2-en [--chunk 16 --left 128]

    # reference activations / transcript of an exported model for one wav
    uv run tools/reference/zipformer_ref.py run --model data/hf/zipformer2-en \
        --audio data/jfk.wav --out data/reference/zipformer/en [--states-chunks 0,1,last]

    # tiny random model with the real head dims / stack pattern (+ its references)
    uv run tools/reference/zipformer_ref.py tiny --bpe data/lang_bpe_500/bpe.model \
        --audio data/jfk.wav --out data/reference/zipformer/tiny

`run` writes `activations.safetensors` (fbank, per chunk: embed, stack outputs, projected encoder
output; all states after the chunks named by --states-chunks, in lapid's time-major layouts) and
`result.json` (tokens, per-chunk tokens, text, config).
"""

import argparse
import importlib
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

LOG_EPS = math.log(1e-10)
PAD = 7 + 2 * 3  # Conv2dSubsampling drops 7 frames; its ConvNeXt needs 3 frames of right context after /2
TAIL = 30  # icefall streaming_decode tail_pad_len
DEFAULT_ICEFALL = "/home/yakov/Documents/projects/audio/zipformer2-66m-streaming-he/src/zipformer2_medium_streaming_he/zipformer"

CONFIG_KEYS = [
    "num_encoder_layers", "downsampling_factor", "feedforward_dim", "num_heads", "encoder_dim",
    "encoder_unmasked_dim", "cnn_module_kernel", "query_head_dim", "value_head_dim", "pos_head_dim",
]
# icefall egs/librispeech/ASR/zipformer/train.py defaults (the 66M "medium" model)
MEDIUM = {
    "num_encoder_layers": "2,2,3,4,3,2", "downsampling_factor": "1,2,4,8,4,2",
    "feedforward_dim": "512,768,1024,1536,1024,768", "num_heads": "4,4,4,8,4,4",
    "encoder_dim": "192,256,384,512,384,256", "encoder_unmasked_dim": "192,192,256,256,256,192",
    "cnn_module_kernel": "31,31,15,15,15,31", "query_head_dim": "32", "value_head_dim": "12",
    "pos_head_dim": "4", "pos_dim": 48, "decoder_dim": 512, "joiner_dim": 512, "context_size": 2,
    "blank_id": 0, "feature_dim": 80, "causal": True, "vocab_size": 500,
}


def k2_stub():
    """Inference-only k2: the Swoosh activations (icefall's formulas)."""
    m = types.ModuleType("k2")

    def swoosh_l(x):
        return torch.logaddexp(torch.zeros_like(x), x - 4.0) - 0.08 * x - 0.035

    def swoosh_r(x):
        return torch.logaddexp(torch.zeros_like(x), x - 1.0) - 0.08 * x - 0.313261687

    m.swoosh_l_forward = m.swoosh_l = swoosh_l
    m.swoosh_r_forward = m.swoosh_r = swoosh_r
    return m


def import_icefall(path):
    sys.modules.setdefault("k2", k2_stub())
    p = Path(path).resolve()
    sys.path.insert(0, str(p.parent.parent if (p.parent / "__init__.py").exists() else p.parent))
    pkg = f"{p.parent.name}.{p.name}" if (p.parent / "__init__.py").exists() else p.name
    mods = {n: importlib.import_module(f"{pkg}.{n}") for n in ["zipformer", "subsampling", "decoder", "joiner", "scaling"]}
    return mods


def ints(s):
    return tuple(int(x) for x in str(s).split(","))


class Asr(torch.nn.Module):
    """encoder_embed + encoder + decoder + joiner with icefall's AsrModel parameter names."""

    def __init__(self, ice, cfg, chunk, left):
        super().__init__()
        enc_dim = ints(cfg["encoder_dim"])
        self.encoder_embed = ice["subsampling"].Conv2dSubsampling(in_channels=cfg["feature_dim"], out_channels=enc_dim[0], dropout=0.0)
        self.encoder = ice["zipformer"].Zipformer2(
            output_downsampling_factor=2,
            downsampling_factor=ints(cfg["downsampling_factor"]),
            num_encoder_layers=ints(cfg["num_encoder_layers"]),
            encoder_dim=enc_dim,
            encoder_unmasked_dim=ints(cfg["encoder_unmasked_dim"]),
            query_head_dim=ints(cfg["query_head_dim"]),
            pos_head_dim=ints(cfg["pos_head_dim"]),
            value_head_dim=ints(cfg["value_head_dim"]),
            pos_dim=int(cfg["pos_dim"]),
            num_heads=ints(cfg["num_heads"]),
            feedforward_dim=ints(cfg["feedforward_dim"]),
            cnn_module_kernel=ints(cfg["cnn_module_kernel"]),
            dropout=0.0,
            causal=True,
            chunk_size=(chunk,),
            left_context_frames=(left,),
        )
        self.decoder = ice["decoder"].Decoder(vocab_size=int(cfg["vocab_size"]), decoder_dim=int(cfg["decoder_dim"]),
                                              blank_id=int(cfg["blank_id"]), context_size=int(cfg["context_size"]))
        self.joiner = ice["joiner"].Joiner(encoder_dim=max(enc_dim), decoder_dim=int(cfg["decoder_dim"]),
                                           joiner_dim=int(cfg["joiner_dim"]), vocab_size=int(cfg["vocab_size"]))


# ---------------------------------------------------------------- export

def read_checkpoint(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"] if "model" in ck else ck
    cfg = dict(MEDIUM)
    if isinstance(ck.get("config"), dict):
        for k, v in ck["config"].items():
            if k in cfg:
                cfg[k] = v
        cfg["train_chunk_size"] = ck["config"].get("chunk_size")
        cfg["train_left_context_frames"] = ck["config"].get("left_context_frames")
    cfg["vocab_size"] = int(sd["decoder.embedding.weight"].shape[0])
    cfg["decoder_dim"] = int(sd["decoder.embedding.weight"].shape[1])
    cfg["joiner_dim"] = int(sd["joiner.output_linear.weight"].shape[1])
    cfg["context_size"] = int(sd["decoder.conv.weight"].shape[2]) if "decoder.conv.weight" in sd else 1
    return sd, cfg, ck.get("tokenizer")


def tokens_txt(sp):
    return "".join(f"{sp.id_to_piece(i)} {i}\n" for i in range(sp.get_piece_size()))


def write_export(out, sd, cfg, chunk, left, bpe_bytes, source, license="unknown", name=None):
    import sentencepiece as spm

    out.mkdir(parents=True, exist_ok=True)
    keep = {k: (v if v.dtype == torch.float16 else v.float()).contiguous() for k, v in sd.items()
            if not k.startswith(("simple_am_proj", "simple_lm_proj"))}
    save_file(keep, str(out / "model.safetensors"))
    config = {"model_type": "zipformer2_transducer", **{k: cfg[k] for k in MEDIUM}, "chunk_size": chunk,
              "left_context_frames": left, "source": source, "sample_rate": 16000, "license": license,
              "name": name or out.name}
    for k in ("train_chunk_size", "train_left_context_frames"):
        if cfg.get(k) is not None:
            config[k] = cfg[k]
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if bpe_bytes is not None:
        (out / "bpe.model").write_bytes(bpe_bytes)
        sp = spm.SentencePieceProcessor(model_proto=bpe_bytes)
        (out / "tokens.txt").write_text(tokens_txt(sp), encoding="utf-8")
    print(f"exported {len(keep)} tensors to {out}")


def cmd_export(a):
    sd, cfg, tok = read_checkpoint(a.checkpoint)
    bpe = Path(a.bpe).read_bytes() if a.bpe else tok
    write_export(Path(a.out), sd, cfg, a.chunk, a.left, bpe, a.source or str(a.checkpoint), a.license, a.name)


# ---------------------------------------------------------------- run

def load_model(ice, model_dir, chunk=None, left=None):
    cfg = json.loads((model_dir / "config.json").read_text())
    chunk = chunk or int(cfg["chunk_size"])
    left = left or int(cfg["left_context_frames"])
    asr = Asr(ice, cfg, chunk, left)
    sd = load_file(str(model_dir / "model.safetensors"))
    missing, unexpected = asr.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    asr.eval()
    return asr, cfg, chunk, left


def fbank(samples):
    from lhotse.features.kaldi.layers import Wav2LogFilterBank

    ext = Wav2LogFilterBank(sampling_rate=16000, num_filters=80, dither=0.0, snip_edges=False)
    with torch.no_grad():
        return ext(torch.from_numpy(samples).float()[None])[0]


def read_audio(path):
    import soundfile as sf

    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    assert sr == 16000, f"{path}: {sr} Hz (need 16 kHz)"
    return x.mean(axis=1)


def time_major_states(states, enc):
    """icefall state list -> {name: [T, C]-style arrays} in lapid's layouts."""
    out, layer = {}, 0
    for i, m in enumerate(enc.encoders):
        for j in range(m.num_layers):
            k, nl, v1, v2, c1, c2 = states[layer * 6: layer * 6 + 6]
            p = f"state.{i}.{j}"
            out[f"{p}.key"] = k[:, 0]
            out[f"{p}.nonlin"] = nl[0, 0]
            out[f"{p}.val1"] = v1[:, 0]
            out[f"{p}.val2"] = v2[:, 0]
            out[f"{p}.conv1"] = c1[0].T
            out[f"{p}.conv2"] = c2[0].T
            layer += 1
    return out


@torch.no_grad()
def stream(asr, cfg, chunk, left, feats, states_chunks, capture_all=False):
    enc, embed, dec, joiner = asr.encoder, asr.encoder_embed, asr.decoder, asr.joiner
    ctx, blank = int(cfg["context_size"]), int(cfg["blank_id"])
    stack_out = {}
    for i, m in enumerate(enc.encoders):
        orig = m.streaming_forward

        def wrapped(*args, _i=i, _orig=orig, **kw):
            r = _orig(*args, **kw)
            stack_out[_i] = r[0][:, 0].clone()
            return r

        m.streaming_forward = wrapped

    padded = torch.cat([feats, torch.full((PAD + TAIL, feats.shape[1]), LOG_EPS)])
    states = enc.get_init_states(1, torch.device("cpu"))
    embed_state = embed.get_init_states(1, torch.device("cpu"))
    processed = torch.zeros(1, dtype=torch.int64)
    hyp = [-1] * (ctx - 1) + [blank]

    def dec_out():
        y = torch.tensor([hyp[-ctx:]])
        return joiner.decoder_proj(dec(y, need_pad=False))[:, 0]

    d = dec_out()
    acts, chunk_tokens, n, c = {}, [], 0, 0
    total_chunks = 0
    while n < padded.shape[0]:
        total_chunks += 1
        n += 2 * chunk
    n = 0
    want = set()
    for s in states_chunks:
        want.add(total_chunks - 1 if s == "last" else int(s))
    while n < padded.shape[0]:
        x = padded[n: n + 2 * chunk + PAD]
        if x.shape[0] < 2 * chunk + PAD:
            x = torch.cat([x, torch.full((2 * chunk + PAD - x.shape[0], x.shape[1]), LOG_EPS)])
        x_lens = torch.tensor([x.shape[0]])
        e, e_lens, embed_state = embed.streaming_forward(x[None], x_lens, embed_state)
        mask = (processed.unsqueeze(1) <= torch.arange(left).expand(1, left)).flip(1)
        mask = torch.cat([mask, torch.zeros(1, chunk, dtype=torch.bool)], dim=1)
        out, _, states = enc.streaming_forward(e.permute(1, 0, 2), e_lens, states, mask)
        processed = processed + e_lens
        proj = joiner.encoder_proj(out[:, 0])
        if capture_all or c < 4 or c in want:
            acts[f"chunk.{c}.embed"] = e[0]
            for i in range(len(enc.encoders)):
                acts[f"chunk.{c}.stack.{i}"] = stack_out[i]
        acts[f"chunk.{c}.encoder_out"] = proj
        if c in want:
            for k, v in time_major_states(states, enc).items():
                acts[f"chunk.{c}.{k}"] = v
            acts[f"chunk.{c}.state.embed"] = embed_state[0].permute(1, 2, 0)
        emitted = []
        for t in range(proj.shape[0]):
            logits = joiner.output_linear(torch.tanh(proj[t:t + 1] + d))
            y = int(logits.argmax(-1))
            if c == 0 and t == 0:
                acts["joiner.logits.0"] = logits[0]
            if y != blank:
                hyp.append(y)
                emitted.append(y)
                d = dec_out()
        chunk_tokens.append(emitted)
        n += 2 * chunk
        c += 1
    for i, m in enumerate(enc.encoders):
        del m.streaming_forward
    return hyp[ctx:], chunk_tokens, acts, c


def text_of(model_dir, tokens):
    bpe = model_dir / "bpe.model"
    if bpe.is_file():
        import sentencepiece as spm

        return spm.SentencePieceProcessor(model_file=str(bpe)).decode(tokens)
    pieces = {}
    for line in (model_dir / "tokens.txt").read_text(encoding="utf-8").splitlines():
        p, i = line.rsplit(" ", 1)
        pieces[int(i)] = p
    return "".join(pieces[t] for t in tokens).replace("▁", " ").strip()


def run_one(ice, model_dir, audio, out, states_chunks, chunk=None, left=None, audio_rel=None):
    asr, cfg, chunk, left = load_model(ice, model_dir, chunk, left)
    samples = read_audio(audio)
    feats = fbank(samples)
    tokens, chunk_tokens, acts, chunks = stream(asr, cfg, chunk, left, feats, states_chunks)
    acts["fbank"] = feats
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.float().contiguous().clone() for k, v in acts.items()}, str(out / "activations.safetensors"))
    text = text_of(model_dir, tokens)
    result = {
        "audio": audio_rel or str(Path(audio).resolve()),
        "model": str(model_dir),
        "samples": int(samples.shape[0]),
        "fbank_frames": int(feats.shape[0]),
        "chunk_size": chunk,
        "left_context_frames": left,
        "chunks": chunks,
        "tokens": tokens,
        "chunk_tokens": chunk_tokens,
        "text": text,
        "states_chunks": sorted({chunks - 1 if s == "last" else int(s) for s in states_chunks}),
    }
    (out / "result.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{audio}: {chunks} chunks, {len(tokens)} tokens: {text}")


def cmd_run(a):
    ice = import_icefall(a.icefall)
    run_one(ice, Path(a.model), a.audio, Path(a.out), a.states_chunks.split(","), a.chunk, a.left, a.audio_rel)


# ---------------------------------------------------------------- tiny

TINY = {
    "num_encoder_layers": "1,1,1,2,1,1", "downsampling_factor": "1,2,4,8,4,2",
    "feedforward_dim": "64,96,128,160,128,96", "num_heads": "2,2,2,4,2,2",
    "encoder_dim": "48,64,96,128,96,64", "encoder_unmasked_dim": "48,48,64,64,64,48",
    "cnn_module_kernel": "31,31,15,15,15,31", "query_head_dim": "32", "value_head_dim": "12",
    "pos_head_dim": "4", "pos_dim": 48, "decoder_dim": 128, "joiner_dim": 128, "context_size": 2,
    "blank_id": 0, "feature_dim": 80, "causal": True,
}


@torch.no_grad()
def cmd_tiny(a):
    import sentencepiece as spm

    ice = import_icefall(a.icefall)
    torch.manual_seed(0)
    bpe = Path(a.bpe).read_bytes()
    cfg = dict(TINY, vocab_size=spm.SentencePieceProcessor(model_proto=bpe).get_piece_size())
    asr = Asr(ice, cfg, a.chunk, a.left)
    # Perturb every parameter so no module is an identity at init (zero conv scales, 0.5 bypass).
    for name, p in asr.named_parameters():
        if name.endswith("log_scale"):
            p.add_(0.1 * torch.randn_like(p))
        elif p.dim() <= 1 or "chunkwise_conv_scale" in name:
            p.add_(0.1 * torch.randn_like(p))
        else:
            p.mul_(1.0 + 0.1 * torch.randn_like(p))
    # Make the random joiner depend on the frame (else one token or blank wins everywhere).
    asr.joiner.encoder_proj.weight.mul_(4.0)
    asr.joiner.output_linear.weight.mul_(4.0)
    asr.joiner.output_linear.bias[int(cfg["blank_id"])] += a.blank_shift
    # Store f16 (small fixture); the reference runs on exactly those values in f32.
    for p in asr.parameters():
        p.copy_(p.half().float())
    sd = {k: v.half() for k, v in asr.state_dict().items()}
    out = Path(a.out)
    write_export(out / "model", sd, cfg, a.chunk, a.left, bpe, "random tiny model (tools/reference/zipformer_ref.py tiny)",
                 "apache-2.0", "zipformer2-tiny-random")
    run_one(ice, out / "model", a.audio, out, ["0", "1", "last"], audio_rel=a.audio_rel)
    r = json.loads((out / "result.json").read_text())
    r["random"] = True
    (out / "result.json").write_text(json.dumps(r, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--icefall", default=DEFAULT_ICEFALL, help="icefall zipformer package directory")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--bpe", help="bpe.model (default: the checkpoint's 'tokenizer')")
    e.add_argument("--out", required=True)
    e.add_argument("--chunk", type=int, default=16, help="default encoder chunk (frames at 50 Hz)")
    e.add_argument("--left", type=int, default=128, help="default left context (frames at 50 Hz)")
    e.add_argument("--source", help="source repo label for metadata")
    e.add_argument("--license", default="unknown", help="SPDX id (apache-2.0 for icefall LibriSpeech, cc-by-4.0 for the Hebrew fine-tune)")
    e.add_argument("--name", help="model name (default: the output directory name)")
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--audio", required=True)
    r.add_argument("--audio-rel", help="audio path to record in result.json (relative to the repo root)")
    r.add_argument("--out", required=True)
    r.add_argument("--chunk", type=int)
    r.add_argument("--left", type=int)
    r.add_argument("--states-chunks", default="0,1,last")
    t = sub.add_parser("tiny")
    t.add_argument("--bpe", required=True)
    t.add_argument("--audio", required=True)
    t.add_argument("--audio-rel")
    t.add_argument("--out", required=True)
    t.add_argument("--chunk", type=int, default=16)
    t.add_argument("--left", type=int, default=64)
    t.add_argument("--blank-shift", type=float, default=4.0)
    a = ap.parse_args()
    {"export": cmd_export, "run": cmd_run, "tiny": cmd_tiny}[a.cmd](a)


if __name__ == "__main__":
    main()
