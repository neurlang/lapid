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
"""Reference data for lapid's Gemma 4 verification (HF transformers, fp32, CPU, eager, greedy).

    uv run tools/reference/gemma4_ref.py --model google/gemma-4-E2B-it --out data/reference/gemma4
    uv run tools/reference/gemma4_ref.py --tiny --out data/reference/gemma4/tiny

`--tiny` builds a random tiny Gemma 4 text model (6 layers, sliding window 16, 2 kv-shared
layers, double-wide MLP, PLE) with the real tokenizer / template, saves it to `<out>/model`
(convertible with `lapid convert`) and records its reference — a cheap end-to-end check.

Writes:
  <out>/activations.safetensors
      short.embed [S,D] (scaled embedding), short.layers.{i} [S,D] (decoder layer outputs),
      short.layers.{i}.attn [S,D] (attention output after o_proj), short.norm [S,D],
      short.logits [S,V] (soft-capped), short.gen.logits [K,V] (greedy steps: row 0 = last prompt
      row, row k = after feeding generated token k-1), long.layers.{i}.tail [T,D],
      long.logits.last [1,V], long.gen.logits [K,V]
  <out>/result.json
      cases.{short,long}: messages_json, enable_thinking, prompt_text, prompt_tokens, generated_tokens,
      generated_text; template_cases; tokenizer_cases; special ids; versions
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

LONG_TEXT = """The history of the printing press is a story of many small improvements rather than a single \
invention. Long before movable type reached Europe, craftsmen in East Asia carved whole pages into wooden \
blocks, inked them, and pressed paper against the surface to make copies of religious texts, calendars and \
official notices. Later, artisans experimented with individual characters made of baked clay, wood and \
eventually metal, which could be arranged, printed and then reused for another page. In the fifteenth \
century a goldsmith in Mainz combined several existing ideas: a screw press adapted from wine making, an \
oil-based ink that clung to metal, and a hand mould that could cast large numbers of identical letters \
quickly. The result was not just faster copying but a new economy of knowledge. Printers needed paper \
mills, type founders, editors, binders and merchants who could carry finished books along trade routes. \
Within fifty years presses operated in hundreds of towns, and millions of volumes circulated across the \
continent. Scholars could compare identical copies of a text, cite page numbers, and build on each other's \
work with a precision that had been impossible when every manuscript differed. Pamphlets spread political \
and religious arguments at remarkable speed, while almanacs, grammars and arithmetic primers taught \
practical skills to merchants and apprentices. Governments soon tried to license and censor printers, and \
authors began to argue about who owned the words they wrote. Many historians therefore treat the press as \
a turning point, although they also stress that its effects depended on literacy, trade, and the \
institutions that grew up around it."""

LONG_TEXT_2 = """Railways changed the rhythm of daily life in a similar way. Before the steam locomotive, \
travel over land moved at the pace of horses, and a journey of a few hundred kilometres could take many \
days. Early lines were built to carry coal from mines to rivers and ports, but passengers quickly \
discovered that trains were faster, cheaper and more predictable than coaches. Towns that gained a station \
grew, while those that were bypassed often declined. Standard time zones were introduced largely because \
timetables needed a single clock across long distances, replacing the local solar time that each town had \
kept. Fresh food, newspapers and mail could now reach distant regions within a day, and seaside resorts \
appeared where workers spent their first holidays. Engineers solved difficult problems along the way: \
bridges had to carry heavy, moving loads, tunnels were dug through mountains, and signals were invented \
to keep trains on the same track from colliding. Governments argued about gauges, routes and ownership, \
and investors experienced booms and painful crashes. By the end of the century the railway had become a \
symbol of progress, admired for its speed and criticised for its noise, smoke and the power of the \
companies that ran it."""

SHORT_MESSAGES = [{"role": "user", "content": "What is the capital of France? Answer in one short sentence."}]
LONG_MESSAGES = [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": LONG_TEXT + "\n\n" + LONG_TEXT_2 + "\n\nSummarize both passages in three bullet points."},
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}, "unit": {"type": "string", "enum": ["c", "f"]}},
                "required": ["city"],
            },
        },
    }
]

TEMPLATE_CASES = [
    {"messages": [{"role": "user", "content": "Hello!"}], "add_generation_prompt": True},
    {
        "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}],
        "add_generation_prompt": True,
        "enable_thinking": False,
    },
    {
        "messages": [{"role": "system", "content": "Think first."}, {"role": "user", "content": "Why is the sky blue?"}],
        "add_generation_prompt": True,
        "enable_thinking": True,
    },
    {
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 = 4."},
            {"role": "user", "content": "And 3+3?"},
        ],
        "add_generation_prompt": True,
    },
    {
        "messages": [
            {"role": "user", "content": "Name a color."},
            {"role": "assistant", "content": "Blue.", "reasoning_content": "Pick any color."},
        ],
        "add_generation_prompt": False,
    },
    {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Describe"}, {"type": "image"}]},
            {"role": "assistant", "content": "A cat."},
            {"role": "user", "content": "Thanks"},
        ],
        "add_generation_prompt": True,
        "enable_thinking": False,
    },
    {
        "messages": [
            {"role": "system", "content": "Use tools when helpful."},
            {"role": "user", "content": "Weather in Paris?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris", "unit": "c"}}}
                ],
            },
            {"role": "tool", "content": "{\"temp\": 18, \"sky\": \"clear\"}", "name": "get_weather"},
        ],
        "tools": TOOLS,
        "add_generation_prompt": True,
    },
    {"messages": [{"role": "user", "content": "Ünïcödé — 日本語 🙂 tabs\tand\nnewlines"}], "add_generation_prompt": True},
]

TOKENIZER_CASES = [
    "Hello, world!",
    " leading space and  double  spaces",
    "Numbers: 1234567 and 3.14159, year 2026.",
    "def f(x):\n    return x * 2  # comment\n\n\n",
    "Ünïcödé — 日本語 🙂 emoji and accents: café naïve",
    "It's we're they'll I'd I'M",
    "<|turn>user\nspecial tokens inline<turn|>\n<|channel>thought\n<channel|>",
    "​ zero width, \x07 bell,  nbsp",
]


def f32(t: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(t.detach().to(torch.float32).cpu().numpy())


def render(tok, case: dict) -> str:
    kw = {}
    if case.get("enable_thinking") is not None:
        kw["enable_thinking"] = case["enable_thinking"]
    return tok.apply_chat_template(
        case["messages"], tools=case.get("tools"), add_generation_prompt=case["add_generation_prompt"], tokenize=False, **kw
    )


def greedy(model, ids: torch.Tensor, max_new: int, eos: list[int]):
    out = model.generate(
        ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=max_new,
        do_sample=False,
        temperature=None,
        top_k=None,
        top_p=None,
        repetition_penalty=1.0,
        eos_token_id=eos,
        output_logits=True,
        return_dict_in_generate=True,
    )
    gen = out.sequences[0, ids.shape[1] :].tolist()
    logits = torch.stack([l[0] for l in out.logits])
    return gen, logits


def tiny_model(src: str, out: Path):
    """Random tiny Gemma 4 text model with the real tokenizer, saved to out/model."""
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig
    from huggingface_hub import snapshot_download

    types = ["sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "sliding_attention", "full_attention"]
    cfg = Gemma4TextConfig(
        vocab_size=262144, hidden_size=64, intermediate_size=128, num_hidden_layers=6, num_attention_heads=4,
        num_key_value_heads=1, head_dim=32, global_head_dim=64, layer_types=types, sliding_window=16,
        num_kv_shared_layers=2, use_double_wide_mlp=True, hidden_size_per_layer_input=16,
        vocab_size_per_layer_input=262144, final_logit_softcapping=30.0, rms_norm_eps=1e-6,
        bos_token_id=2, eos_token_id=1, pad_token_id=0, tie_word_embeddings=True,
        rope_parameters={"full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1e6},
                         "sliding_attention": {"rope_type": "default", "rope_theta": 1e4}},
    )
    torch.manual_seed(0)
    model = Gemma4ForCausalLM(cfg).eval()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith("norm.weight"):
                p.copy_(1.0 + 0.1 * torch.randn_like(p))
            elif "layer_scalar" in n:
                p.fill_(0.9)
            else:
                p.copy_(0.05 * torch.randn_like(p))
        for l in model.model.layers:
            l.layer_scalar.fill_(0.9)
    d = out / "model"
    d.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(d, safe_serialization=True)
    snap = Path(snapshot_download(src, allow_patterns=["tokenizer*", "chat_template.jinja", "generation_config.json"]))
    for f in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"]:
        shutil.copy(snap / f, d / f)
    return str(d)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/gemma-4-E2B-it")
    ap.add_argument("--out", default="data/reference/gemma4", type=Path)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--max-new-tokens", default=40, type=int)
    ap.add_argument("--gen-logit-steps", default=12, type=int)
    ap.add_argument("--long-tail", default=8, type=int)
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
    for name, messages in (("short", SHORT_MESSAGES), ("long", LONG_MESSAGES)):
        text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids
        captured.clear()
        logits = model(input_ids=ids, use_cache=False).logits[0]
        if name == "short":
            for k, v in captured.items():
                tensors[f"short.{k}"] = f32(v[0])
            tensors["short.logits"] = f32(logits)
        else:
            for k, v in captured.items():
                if k.startswith("layers.") and not k.endswith(".attn"):
                    tensors[f"long.{k}.tail"] = f32(v[0, -args.long_tail :])
            tensors["long.logits.last"] = f32(logits[-1:])
        gen, gen_logits = greedy(model, ids, args.max_new_tokens, eos)
        tensors[f"{name}.gen.logits"] = f32(gen_logits[: args.gen_logit_steps])
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

    template_cases = []
    for c in TEMPLATE_CASES:
        try:
            text = render(tok, c)
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
    tokenizer_cases = [{"text": t, "ids": tok(t, add_special_tokens=False).input_ids} for t in TOKENIZER_CASES]

    save_file(tensors, str(args.out / "activations.safetensors"))
    import transformers

    result = {
        "model": args.model if not args.tiny else "tiny-random-gemma4",
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
    (args.out / "result.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
