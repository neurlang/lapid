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
"""Reference data for lapid's Gemma 4 MoE verification (HF transformers, fp32, CPU, eager, greedy).

    uv run tools/reference/gemma4moe_ref.py --tiny --out data/reference/gemma4moe/tiny

`--tiny` builds a random tiny Gemma 4 MoE text model from the real google/gemma-4-26B-A4B-it
config.json (6 layers with the real sliding / full pattern, real head dims 256 / 512, k_eq_v
full layers, 16 experts top-4, bidirectional vision attention, sliding window 16) with the real
tokenizer / template, saves it to `<out>/model` (convertible with `lapid convert`) and records
its reference. The 26B checkpoint itself does not fit an fp32 reference on this machine.

Same layout as `gemma4_ref.py` (short / long cases, template and tokenizer cases), plus:
  inject.embeds [n, D]   random rows that replace the embeddings of tokens [start, start + n)
  inject.norm [S, D]     final-norm output of that forward (bidirectional image-block mask on
                         sliding layers when the config says use_bidirectional_attention="vision")
  inject.logits.last [1, V]
  result.json inject: {prompt_tokens, start, n}
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gemma4_ref as g4  # noqa: E402

REPO = "google/gemma-4-26B-A4B-it"
INJECT_MESSAGES = [
    {"role": "user", "content": "Here is a picture of the harbour at dawn. Describe what you would expect to see in it, "
     "including the boats, the light and the water."}
]


def tiny_model(src: str, out: Path) -> str:
    from huggingface_hub import hf_hub_download, snapshot_download
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig

    real = json.loads(Path(hf_hub_download(src, "config.json")).read_text())["text_config"]
    keep = {k: v for k, v in real.items() if k not in ("dtype", "architectures", "transformers_version", "_name_or_path")}
    keep.update(
        hidden_size=64, intermediate_size=48, moe_intermediate_size=32, num_hidden_layers=6,
        layer_types=real["layer_types"][:6], num_attention_heads=2, num_key_value_heads=1,
        num_global_key_value_heads=1, num_experts=16, top_k_experts=4, sliding_window=16,
    )
    cfg = Gemma4TextConfig(**keep)
    torch.manual_seed(0)
    model = Gemma4ForCausalLM(cfg).eval()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith("norm.weight") or n.endswith("router.scale"):
                p.copy_(1.0 + 0.1 * torch.randn_like(p))
            elif n.endswith("per_expert_scale"):
                p.copy_(1.0 + 0.2 * torch.rand_like(p))
            else:
                p.copy_(0.05 * torch.randn_like(p))
        for l in model.model.layers:
            l.layer_scalar.fill_(0.9)
    d = out / "model"
    d.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(d, safe_serialization=True)
    snap = Path(snapshot_download(src, allow_patterns=["tokenizer*", "chat_template.jinja", "generation_config.json"]))
    for f in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"]:
        if (snap / f).exists():
            shutil.copy(snap / f, d / f)
    return str(d)


def additive(allowed: torch.Tensor) -> torch.Tensor:
    m = torch.zeros(allowed.shape, dtype=torch.float32)
    m[~allowed] = torch.finfo(torch.float32).min
    return m[None, None]


def inject_case(model, tok, tensors: dict, n_inject: int) -> dict:
    """Forward with random rows replacing token embeddings [start, start + n) (an image block)."""
    text = tok.apply_chat_template(INJECT_MESSAGES, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids
    s = ids.shape[1]
    start = 6
    tm = model.model
    cfg = tm.config
    pad = cfg.pad_token_id
    llm_ids = ids.clone()
    llm_ids[0, start : start + n_inject] = pad
    emb = tm.embed_tokens(llm_ids)
    torch.manual_seed(1)
    rows = torch.randn(n_inject, cfg.hidden_size) * 2.0
    emb[0, start : start + n_inject] = rows
    i = torch.arange(s)[:, None]
    j = torch.arange(s)[None, :]
    causal = j <= i
    block = torch.full((s,), -1)
    block[start : start + n_inject] = 0
    same = (block[:, None] == block[None, :]) & (block[:, None] >= 0)
    bidir = getattr(cfg, "use_bidirectional_attention", None) == "vision"
    sliding = ((causal | same) if bidir else causal) & (j > i - cfg.sliding_window)
    masks = {"full_attention": additive(causal), "sliding_attention": additive(sliding)}
    per_layer = None
    if cfg.hidden_size_per_layer_input:
        per_layer = tm.get_per_layer_inputs(llm_ids, None)
    out = tm(inputs_embeds=emb, attention_mask=masks, per_layer_inputs=per_layer, use_cache=False)
    h = out.last_hidden_state
    logits = model.lm_head(h[:, -1:])
    if cfg.final_logit_softcapping:
        c = cfg.final_logit_softcapping
        logits = torch.tanh(logits / c) * c
    tensors["inject.embeds"] = g4.f32(rows)
    tensors["inject.norm"] = g4.f32(h[0])
    tensors["inject.logits.last"] = g4.f32(logits[0])
    print(f"inject: {s} tokens, rows {start}..{start + n_inject}, bidirectional={bidir}", file=sys.stderr)
    return {"prompt_tokens": ids[0].tolist(), "start": start, "n": n_inject, "bidirectional": bidir}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=REPO)
    ap.add_argument("--out", default="data/reference/gemma4moe/tiny", type=Path)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--max-new-tokens", default=40, type=int)
    ap.add_argument("--gen-logit-steps", default=12, type=int)
    ap.add_argument("--long-tail", default=8, type=int)
    ap.add_argument("--inject", default=20, type=int, help="injected rows (0 = no inject case)")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    args.out.mkdir(parents=True, exist_ok=True)
    src = tiny_model(args.model, args.out) if args.tiny else args.model

    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32, attn_implementation="eager").eval()
    gen_cfg = GenerationConfig.from_pretrained(src)
    eos = gen_cfg.eos_token_id if isinstance(gen_cfg.eos_token_id, list) else [gen_cfg.eos_token_id]
    turn_end = tok.convert_tokens_to_ids("<turn|>")
    if turn_end not in eos:
        eos.append(turn_end)

    tm = model.model.language_model if hasattr(model.model, "language_model") else model.model
    tensors: dict[str, np.ndarray] = {}
    captured: dict[str, torch.Tensor] = {}

    def hook(name):
        def f(_m, _inp, out):
            captured[name] = out[0] if isinstance(out, tuple) else out

        return f

    handles = [tm.embed_tokens.register_forward_hook(hook("embed"))]
    for i, layer in enumerate(tm.layers):
        handles.append(layer.register_forward_hook(hook(f"layers.{i}")))
        handles.append(layer.self_attn.register_forward_hook(hook(f"layers.{i}.attn")))
    handles.append(tm.norm.register_forward_hook(hook("norm")))

    cases = {}
    for name, messages in (("short", g4.SHORT_MESSAGES), ("long", g4.LONG_MESSAGES)):
        text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids
        captured.clear()
        logits = model(input_ids=ids, use_cache=False).logits[0]
        if name == "short":
            for k, v in captured.items():
                tensors[f"short.{k}"] = g4.f32(v[0])
            tensors["short.logits"] = g4.f32(logits)
        else:
            for k, v in captured.items():
                if k.startswith("layers.") and not k.endswith(".attn"):
                    tensors[f"long.{k}.tail"] = g4.f32(v[0, -args.long_tail :])
            tensors["long.logits.last"] = g4.f32(logits[-1:])
        gen, gen_logits = g4.greedy(model, ids, args.max_new_tokens, eos)
        tensors[f"{name}.gen.logits"] = g4.f32(gen_logits[: args.gen_logit_steps])
        cases[name] = {
            "messages_json": json.dumps(messages, ensure_ascii=False),
            "enable_thinking": False,
            "prompt_text": text,
            "prompt_tokens": ids[0].tolist(),
            "generated_tokens": gen,
            "generated_text": tok.decode(gen, skip_special_tokens=False),
        }
        print(f"{name}: {ids.shape[1]} prompt tokens -> {tok.decode(gen)!r}", file=sys.stderr)
    for h in handles:
        h.remove()

    inject = inject_case(model, tok, tensors, args.inject) if args.inject > 0 else None

    template_cases = []
    for c in g4.TEMPLATE_CASES:
        try:
            text = g4.render(tok, c)
        except Exception as e:  # noqa: BLE001 - a case the template rejects is skipped
            print(f"template case skipped: {e}", file=sys.stderr)
            continue
        template_cases.append(
            {
                "messages_json": json.dumps(c["messages"], ensure_ascii=False),
                "tools_json": json.dumps(c["tools"], ensure_ascii=False) if c.get("tools") else None,
                "add_generation_prompt": c["add_generation_prompt"],
                "enable_thinking": c.get("enable_thinking"),
                "text": text,
            }
        )
    tokenizer_cases = [{"text": t, "ids": tok(t, add_special_tokens=False).input_ids} for t in g4.TOKENIZER_CASES]

    save_file(tensors, str(args.out / "activations.safetensors"))
    import transformers

    result = {
        "model": args.model if not args.tiny else "tiny-random-gemma4-moe",
        "dtype": "fp32",
        "device": "cpu",
        "cases": cases,
        "template_cases": template_cases,
        "tokenizer_cases": tokenizer_cases,
        "special": {"bos": tok.bos_token_id, "eos": eos, "pad": tok.pad_token_id, "turn_end": turn_end},
        "n_layers": len(tm.layers),
        "layer_types": list(tm.config.layer_types),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "python": sys.version.split()[0]},
    }
    if inject is not None:
        result["inject"] = inject
    (args.out / "result.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
