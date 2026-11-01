# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.6",
#   "transformers>=5.5",
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
"""Reference data for lapid's Qwen3.5 verification (HF transformers, fp32, CPU, eager, greedy).

    uv run tools/reference/qwen35_ref.py --model Qwen/Qwen3.5-0.8B --out data/reference/qwen35
    uv run tools/reference/qwen35_ref.py --tiny-moe --out data/reference/qwen35/tiny_moe

`--tiny-moe` builds a random-weight `qwen3_5_moe` text model (4 layers: 3 Gated DeltaNet + 1
attention, 8 experts top-2 + shared expert, 2 k heads / 4 v heads so the v-head reordering is
exercised, the real Qwen3.5 tokenizer), saves it as an HF checkpoint in
`<out>/qwen35-tiny-moe/` (bf16 weights; the reference runs on the bf16-rounded values in fp32)
and records the same activations as the real model.

Writes:
  <out>/activations.safetensors
      short.embed [S,D], short.layers.{i} [S,D] (decoder layer outputs), short.layers.{i}.op [S,D]
      (token mixer output before the residual add), short.norm [S,D], short.logits [S,V],
      short.gen.logits [K,V] (raw logits of greedy steps: row 0 = last prompt row, row k = after
      feeding generated token k-1), long.layers.{i}.tail [T,D], long.logits.last [1,V], long.gen.logits [K,V]
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
from transformers import AutoTokenizer

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

SHORT_MESSAGES = [{"role": "user", "content": "What is the capital of France? Answer in one short sentence."}]
LONG_MESSAGES = [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": LONG_TEXT + "\n\nSummarize the passage above in three bullet points."},
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}, "unit": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

TEMPLATE_CASES = [
    {"messages": [{"role": "user", "content": "Hello!"}], "add_generation_prompt": True},
    {"messages": [{"role": "user", "content": "Hello!"}], "add_generation_prompt": True, "enable_thinking": True},
    {
        "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}],
        "add_generation_prompt": True,
        "enable_thinking": False,
    },
    {
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "<think>\nSimple sum.\n</think>\n\n2+2 = 4."},
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
            {"role": "user", "content": [{"type": "text", "text": "Describe"}, {"type": "image"}, {"type": "text", "text": "this"}]},
            {"role": "assistant", "content": "A cat."},
            {"role": "user", "content": "Thanks"},
        ],
        "add_generation_prompt": True,
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
            {"role": "tool", "content": "{\"temp\": 18, \"sky\": \"clear\"}", "tool_call_id": "a"},
            {"role": "tool", "content": "second", "tool_call_id": "a"},
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
    "<|im_start|>user\nspecial tokens inline<|im_end|>\n<think></think>",
    "é combining mark (NFC)",
]


def f32(t: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(t.detach().to(torch.float32).cpu().numpy())


def render(tok, case: dict) -> str:
    kw = {}
    if "enable_thinking" in case:
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


def tiny_moe(out_dir: Path, tokenizer_repo: str):
    """Random qwen3_5_moe text model saved as an HF checkpoint (bf16) next to the real tokenizer."""
    from huggingface_hub import hf_hub_download
    from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

    cfg = Qwen3_5MoeTextConfig(
        vocab_size=248320,
        hidden_size=64,
        num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=48,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        eos_token_id=248046,
        rope_parameters={"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25},
    )
    torch.manual_seed(1234)
    model = Qwen3_5MoeForCausalLM(cfg)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith("norm.weight") and "linear_attn" not in n:
                p.normal_(0.0, 0.1)  # zero-centred (1 + w) norms
            elif n.endswith("linear_attn.norm.weight"):
                p.normal_(1.0, 0.1)
            elif n.endswith("A_log"):
                p.uniform_(-1.0, 1.0)
            elif n.endswith("dt_bias"):
                p.normal_(0.0, 0.5)
            else:
                p.normal_(0.0, 0.08)
    model = model.to(torch.bfloat16).to(torch.float32).eval()
    ckpt = out_dir / "qwen35-tiny-moe"
    model.to(torch.bfloat16).save_pretrained(ckpt)
    model = model.to(torch.float32)
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        shutil.copy(hf_hub_download(tokenizer_repo, f), ckpt / f)
    return model, model.model.layers, model.model


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--out", default="data/reference/qwen35", type=Path)
    ap.add_argument("--tiny-moe", action="store_true")
    ap.add_argument("--max-new-tokens", default=40, type=int)
    ap.add_argument("--gen-logit-steps", default=12, type=int)
    ap.add_argument("--long-tail", default=8, type=int)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    args.out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if args.tiny_moe:
        model, layers, text = tiny_moe(args.out, args.model)
        name = "qwen35-tiny-moe (random)"
    else:
        from transformers import Qwen3_5ForConditionalGeneration

        model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model, dtype=torch.float32, attn_implementation="eager").eval()
        text = model.model.language_model
        layers = text.layers
        name = args.model
    model.config._attn_implementation = "eager"
    tcfg = text.config
    eos = [248046, 248044]

    tensors: dict[str, np.ndarray] = {}
    captured: dict[str, torch.Tensor] = {}

    def hook(key):
        def f(_m, _inp, out):
            captured[key] = out[0] if isinstance(out, tuple) else out

        return f

    handles = [text.embed_tokens.register_forward_hook(hook("embed"))]
    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(hook(f"layers.{i}")))
        op = layer.self_attn if hasattr(layer, "self_attn") else layer.linear_attn
        handles.append(op.register_forward_hook(hook(f"layers.{i}.op")))
    handles.append(text.norm.register_forward_hook(hook("norm")))

    cases = {}
    for cname, messages, thinking in (("short", SHORT_MESSAGES, False), ("long", LONG_MESSAGES, False)):
        prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=thinking)
        ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids
        captured.clear()
        logits = model(input_ids=ids, use_cache=False).logits[0]
        if cname == "short":
            for k, v in captured.items():
                tensors[f"short.{k}"] = f32(v[0])
            tensors["short.logits"] = f32(logits)
        else:
            for k, v in captured.items():
                if k.startswith("layers.") and not k.endswith(".op"):
                    tensors[f"long.{k}.tail"] = f32(v[0, -args.long_tail :])
            tensors["long.logits.last"] = f32(logits[-1:])
        gen, gen_logits = greedy(model, ids, args.max_new_tokens, eos)
        tensors[f"{cname}.gen.logits"] = f32(gen_logits[: args.gen_logit_steps])
        cases[cname] = {
            "messages_json": json.dumps(messages, ensure_ascii=False),
            "enable_thinking": thinking,
            "prompt_text": prompt,
            "prompt_tokens": ids[0].tolist(),
            "generated_tokens": gen,
            "generated_text": tok.decode(gen, skip_special_tokens=False),
        }
        print(f"{cname}: {ids.shape[1]} prompt tokens -> {tok.decode(gen)!r}", file=sys.stderr)
    for h in handles:
        h.remove()

    template_cases = [
        {
            "messages_json": json.dumps(c["messages"], ensure_ascii=False),
            "tools_json": json.dumps(c["tools"], ensure_ascii=False) if c.get("tools") else None,
            "add_generation_prompt": c["add_generation_prompt"],
            "enable_thinking": c.get("enable_thinking"),
            "text": render(tok, c),
        }
        for c in TEMPLATE_CASES
    ]
    tokenizer_cases = [{"text": t, "ids": tok(t, add_special_tokens=False).input_ids} for t in TOKENIZER_CASES]

    save_file(tensors, str(args.out / "activations.safetensors"))
    import transformers

    result = {
        "model": name,
        "dtype": "fp32",
        "device": "cpu",
        "cases": cases,
        "template_cases": template_cases,
        "tokenizer_cases": tokenizer_cases,
        "special": {"eos": eos, "im_start": tok.convert_tokens_to_ids("<|im_start|>"), "im_end": 248046},
        "n_layers": len(layers),
        "layer_types": list(tcfg.layer_types),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "python": sys.version.split()[0]},
    }
    (args.out / "result.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
