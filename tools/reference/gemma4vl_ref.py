# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.6",
#   "transformers>=5.5",
#   "numpy",
#   "safetensors",
#   "pillow",
#   "torchvision",
# ]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# torchvision = { index = "pytorch-cpu" }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Reference data for lapid's Gemma 4 vision verification (HF transformers, fp32, CPU, eager, greedy).

    uv run tools/reference/gemma4vl_ref.py --out data/reference/gemma4vl
    uv run tools/reference/gemma4vl_ref.py --tiny --out data/reference/gemma4vl/tiny

The real run uses google/gemma-4-E2B-it on `image.png` (llama.cpp's tools/mtmd/test-1.jpeg saved
losslessly as PNG). `--tiny` builds a random tiny Gemma4ForConditionalGeneration (2-layer vision
tower with clipped linears + standardize, 6-layer text decoder with PLE, kv sharing and
`use_bidirectional_attention="vision"`), saves it with the real tokenizer / processor to
`<out>/model` and a synthetic `image.png`. `--tiny --causal` drops the bidirectional image
attention (llama.cpp treats Gemma 4 models with per-layer embeddings as causal, so the GGUF +
mmproj pair of this variant is checked: `data/reference/gemma4vl/tiny_causal`).

Writes (the shared vision-LM reference layout of `lapid_hub::runtime::vlm_verify`, one case
named `image`):
  <out>/image.png
  <out>/activations.safetensors
      image.pixel_values [N, 3*p*p] (real patches only, rescaled to [0,1]), image.positions [N, 2]
      (x, y; f32), image.vision.embed [N, Dv] (patch embedder), image.vision.layers.{i} [N, Dv],
      image.vision.pooled [M, Dv] (pooler + standardize), image.vision.projected [M, D]
      (embed_vision), image.embed [S, D] (merged input embeddings), image.layers.{i}.attn /
      image.layers.{i} [S, D], image.logits.last [1, V], image.gen.logits [K, V]
  <out>/result.json
      max_soft_tokens, token ids, versions; cases.image: messages_json, prompt_text (template
      output), prompt_tokens (expanded ids), image, enable_thinking, n_soft, grid [ph, pw],
      generated_tokens, generated_text
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.numpy import save_file
from transformers import AutoModelForImageTextToText, AutoProcessor, GenerationConfig

PROMPT = "Describe this image in one sentence."
REAL_IMAGE = Path("plans/llama.cpp-latest/tools/mtmd/test-1.jpeg")


def f32(t: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(t.detach().to(torch.float32).cpu().numpy())


def synthetic_image(path: Path):
    """Deterministic 150x110 RGB test picture (gradients + blocks)."""
    w, h = 150, 110
    y, x = np.mgrid[0:h, 0:w]
    img = np.stack([(x * 255 // w), (y * 255 // h), ((x + y) * 3) % 256], axis=-1).astype(np.uint8)
    img[20:60, 30:80] = [220, 40, 40]
    img[70:100, 90:140] = [30, 200, 90]
    Image.fromarray(img).save(path)


def tiny_model(src: str, out: Path, bidirectional: bool = True) -> str:
    from huggingface_hub import snapshot_download
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration, Gemma4TextConfig, Gemma4VisionConfig

    types = ["sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "sliding_attention", "full_attention"]
    text = Gemma4TextConfig(
        vocab_size=262144, hidden_size=64, intermediate_size=128, num_hidden_layers=6, num_attention_heads=4,
        num_key_value_heads=1, head_dim=32, global_head_dim=64, layer_types=types, sliding_window=16,
        num_kv_shared_layers=2, use_double_wide_mlp=True, hidden_size_per_layer_input=16,
        vocab_size_per_layer_input=262144, final_logit_softcapping=30.0, rms_norm_eps=1e-6,
        bos_token_id=2, eos_token_id=1, pad_token_id=0, tie_word_embeddings=True,
        use_bidirectional_attention="vision" if bidirectional else None,
        rope_parameters={"full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1e6},
                         "sliding_attention": {"rope_type": "default", "rope_theta": 1e4}},
    )
    vision = Gemma4VisionConfig(
        hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        head_dim=64, pooling_kernel_size=3, patch_size=16, position_embedding_size=256, use_clipped_linears=True,
        standardize=True, rms_norm_eps=1e-6, rope_parameters={"rope_type": "default", "rope_theta": 100.0},
    )
    cfg = Gemma4Config(text_config=text.to_dict(), vision_config=vision.to_dict(), audio_config=None,
                       tie_word_embeddings=True, eos_token_id=[1, 106])
    torch.manual_seed(0)
    model = Gemma4ForConditionalGeneration(cfg).eval()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith("norm.weight"):
                p.copy_(1.0 + 0.1 * torch.randn_like(p))
            elif "position_embedding_table" in n:
                p.copy_(0.5 * torch.randn_like(p))
            else:
                p.copy_(0.05 * torch.randn_like(p))
        for l in model.model.language_model.layers:
            l.layer_scalar.fill_(0.9)
        vt = model.model.vision_tower
        for name, buf in vt.named_buffers():
            if name.endswith("input_min"):
                buf.fill_(-2.0)
            elif name.endswith("input_max"):
                buf.fill_(2.0)
            elif name.endswith("output_min"):
                buf.fill_(-1.5)
            elif name.endswith("output_max"):
                buf.fill_(1.5)
        vt.std_bias.copy_(0.1 * torch.randn_like(vt.std_bias))
        vt.std_scale.copy_(1.0 + 0.1 * torch.randn_like(vt.std_scale))
    d = out / "model"
    d.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(d, safe_serialization=True)
    snap = Path(snapshot_download(src, allow_patterns=["tokenizer*", "chat_template.jinja", "generation_config.json", "processor_config.json"]))
    for f in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"]:
        (d / f).unlink(missing_ok=True)
        shutil.copyfile(snap / f, d / f)
    pc = json.loads((snap / "processor_config.json").read_text())
    pc["image_processor"]["max_soft_tokens"] = 70
    pc["image_seq_length"] = 70
    pc["image_processor"]["image_seq_length"] = 70
    (d / "processor_config.json").write_text(json.dumps(pc, indent=2))
    return str(d)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/gemma-4-E2B-it")
    ap.add_argument("--out", default="data/reference/gemma4vl", type=Path)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--causal", action="store_true", help="tiny: causal image attention (what llama.cpp assumes for PLE models; GGUF pair check)")
    ap.add_argument("--max-new-tokens", default=40, type=int)
    ap.add_argument("--gen-logit-steps", default=8, type=int)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    args.out.mkdir(parents=True, exist_ok=True)
    img_path = args.out / "image.png"
    if args.tiny:
        synthetic_image(img_path)
        src = tiny_model(args.model, args.out, bidirectional=not args.causal)
    else:
        Image.open(REAL_IMAGE).convert("RGB").save(img_path)
        src = args.model
    image = Image.open(img_path).convert("RGB")

    proc = AutoProcessor.from_pretrained(src)
    model = AutoModelForImageTextToText.from_pretrained(src, dtype=torch.float32, attn_implementation="eager").eval()
    gen_cfg = GenerationConfig.from_pretrained(src)
    eos = gen_cfg.eos_token_id if isinstance(gen_cfg.eos_token_id, list) else [gen_cfg.eos_token_id]
    tok = proc.tokenizer
    turn_end = tok.convert_tokens_to_ids("<turn|>")
    if turn_end not in eos:
        eos.append(turn_end)

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    text = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    inputs = proc(text=[text], images=[[image]], return_tensors="pt")
    ids = inputs["input_ids"]
    pos = inputs["image_position_ids"][0]
    real = (pos != -1).all(dim=-1)
    pv = inputs["pixel_values"][0][real]
    n_soft = int((ids[0] == model.config.image_token_id).sum())
    ph = int(pos[real][:, 1].max()) + 1
    pw = int(pos[real][:, 0].max()) + 1

    tensors: dict[str, np.ndarray] = {"pixel_values": f32(pv), "positions": f32(pos[real])}
    captured: dict[str, torch.Tensor] = {}

    def hook(name):
        def f(_m, _inp, out):
            captured[name] = out[0] if isinstance(out, tuple) else out

        return f

    vt = model.model.vision_tower
    tm = model.model.language_model
    handles = [vt.patch_embedder.register_forward_hook(hook("vision.embed"))]
    for i, layer in enumerate(vt.encoder.layers):
        handles.append(layer.register_forward_hook(hook(f"vision.layers.{i}")))
    handles.append(model.model.embed_vision.register_forward_hook(hook("vision.projected")))
    for i, layer in enumerate(tm.layers):
        handles.append(layer.register_forward_hook(hook(f"mm.layers.{i}")))
        handles.append(layer.self_attn.register_forward_hook(hook(f"mm.layers.{i}.attn")))
    handles.append(tm.norm.register_forward_hook(hook("mm.norm")))

    # Pooled (pre-projection) features: run the tower directly once.
    vout = vt(pixel_values=inputs["pixel_values"], pixel_position_ids=inputs["image_position_ids"])
    tensors["vision.pooled"] = f32(vout.last_hidden_state)
    captured.clear()

    def embeds_hook(_m, _args, kwargs):
        captured["mm.embed"] = kwargs["inputs_embeds"]

    handles.append(tm.register_forward_pre_hook(embeds_hook, with_kwargs=True))
    out = model(**inputs, use_cache=False)
    logits = out.logits[0]
    n_real = pv.shape[0]
    for k, v in captured.items():
        t = v[0]
        if k.startswith("vision.") and k != "vision.projected":
            t = t[:n_real]
        if k == "vision.projected":
            t = v.reshape(-1, v.shape[-1])
        tensors[k] = f32(t)
    tensors["mm.logits.last"] = f32(logits[-1:])
    for h in handles:
        h.remove()

    gout = model.generate(
        **inputs, max_new_tokens=args.max_new_tokens, do_sample=False, temperature=None, top_k=None, top_p=None,
        repetition_penalty=1.0, eos_token_id=eos, output_logits=True, return_dict_in_generate=True,
    )
    gen = gout.sequences[0, ids.shape[1]:].tolist()
    tensors["mm.gen.logits"] = f32(torch.stack([l[0] for l in gout.logits])[: args.gen_logit_steps])
    print(f"{ids.shape[1]} prompt tokens ({n_soft} soft, grid {ph}x{pw}) -> {tok.decode(gen)!r}", file=sys.stderr)

    tensors.pop("mm.norm", None)
    tensors = {"image." + (k[3:] if k.startswith("mm.") else k): v for k, v in tensors.items()}
    save_file(tensors, str(args.out / "activations.safetensors"))
    import transformers

    cfg = model.config
    result = {
        "model": args.model if not args.tiny else "tiny-random-gemma4-vision",
        "dtype": "fp32",
        "device": "cpu",
        "max_soft_tokens": proc.image_processor.max_soft_tokens,
        "cases": {"image": {
            "messages_json": json.dumps(messages),
            "prompt": PROMPT,
            "prompt_text": text,
            "prompt_tokens": ids[0].tolist(),
            "image": "image.png",
            "enable_thinking": False,
            "n_soft": n_soft,
            "grid": [ph, pw],
            "generated_tokens": gen,
            "generated_text": tok.decode(gen, skip_special_tokens=False),
        }},
        "image_token_id": cfg.image_token_id,
        "boi_token_id": cfg.boi_token_id,
        "eoi_token_id": cfg.eoi_token_id,
        "eos": eos,
        "n_vision_layers": len(vt.encoder.layers),
        "n_layers": len(tm.layers),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "python": sys.version.split()[0]},
    }
    (args.out / "result.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
