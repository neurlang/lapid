# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.6",
#   "transformers>=5.5",
#   "numpy",
#   "safetensors",
#   "sentencepiece",
#   "protobuf",
#   "pyyaml",
#   "tqdm",
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
"""Reference data for lapid's Llama-family verification (`lapid verify llama`): Llama 3.x,
Qwen2 / 2.5, Qwen3 dense and Qwen3-MoE. HF transformers, fp32, CPU, eager, greedy.

    uv run tools/reference/llama_ref.py --family qwen3            # Qwen/Qwen3-0.6B
    uv run tools/reference/llama_ref.py --family qwen2            # Qwen/Qwen2.5-0.5B-Instruct
    uv run tools/reference/llama_ref.py --family llama            # meta-llama/Llama-3.2-1B-Instruct
    uv run tools/reference/llama_ref.py --family qwen3moe --tiny  # tiny from Qwen/Qwen3-30B-A3B's config
    uv run tools/reference/llama_ref.py --family qwen3 --tiny     # (any family)

Default output: `data/reference/llama/<family>` (real) or `data/reference/llama/<family>/tiny`.

`--tiny` builds a random-weight model from the real `config.json` (3 layers, hidden 256, 4 heads /
2 kv heads, the real head dim, RoPE (incl. llama3 scaling), biases, q/k norms, tying; MoE: 4 layers
(llama.cpp has no `mlp_only_layers`), 8 experts top-2) with the real tokenizer and chat template,
saves it as a bf16 HF checkpoint `<out>/llama-tiny-<family>/` (the reference runs on the
bf16-rounded values in fp32), and writes tiny GGUFs with llama.cpp's converter + `llama-quantize`
(`<out>/llama-tiny-<family>-{bf16,q8_0,q4_k_m}.gguf`) when `--llama-cpp` (default
`plans/llama.cpp-latest`) has them.

Writes:
  <out>/activations.safetensors
      short.embed [S,D], short.layers.{i} [S,D], short.layers.{i}.op [S,D] (self-attention output),
      short.norm [S,D], short.logits [S,V], short.gen.logits [K,V] (raw logits of greedy steps),
      long.layers.{i}.tail [T,D], long.logits.last [1,V], long.gen.logits [K,V]
  <out>/result.json
      model (repo id, or the tiny checkpoint dir name), default_dtype (tiny: fp32), date_string,
      cases.{short,long}, template_cases (cases the HF template rejects are left out),
      tokenizer_cases, stop ids, versions
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

REPOS = {
    "qwen3": "Qwen/Qwen3-0.6B",
    "qwen2": "Qwen/Qwen2.5-0.5B-Instruct",
    "llama": "meta-llama/Llama-3.2-1B-Instruct",
    "qwen3moe": "Qwen/Qwen3-30B-A3B",
}
DATE = "26 Jul 2024"
STOP_NAMES = ["<|eot_id|>", "<|eom_id|>", "<|end_of_text|>", "<|im_end|>", "<|endoftext|>"]

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
    {"messages": [{"role": "user", "content": "Hello!"}], "add_generation_prompt": True, "enable_thinking": False},
    {"messages": [{"role": "user", "content": "Hello!"}], "add_generation_prompt": False},
    {
        "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}],
        "add_generation_prompt": True,
        "enable_thinking": True,
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
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
        "tools": TOOLS,
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
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>",
    "é combining mark (NFC)",
]


def f32(t: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(t.detach().to(torch.float32).cpu().numpy())


def render(tok, case: dict, add_generation_prompt: bool | None = None) -> str:
    kw = {"date_string": DATE}
    if "enable_thinking" in case:
        kw["enable_thinking"] = case["enable_thinking"]
    agp = case["add_generation_prompt"] if add_generation_prompt is None else add_generation_prompt
    return tok.apply_chat_template(case["messages"], tools=case.get("tools"), add_generation_prompt=agp, tokenize=False, **kw)


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


def tiny(family: str, repo: str, out_dir: Path):
    """Random-weight model from the real config, saved as a bf16 HF checkpoint + tokenizer."""
    from huggingface_hub import hf_hub_download

    base = AutoConfig.from_pretrained(repo)
    d = base.to_dict()
    for k in ("layer_types", "architectures", "torch_dtype", "dtype", "transformers_version", "quantization_config"):
        d.pop(k, None)
    head_dim = getattr(base, "head_dim", None) or base.hidden_size // base.num_attention_heads
    moe = family == "qwen3moe"
    n_layers = 4 if moe else 3
    d.update(
        num_hidden_layers=n_layers,
        hidden_size=256,
        intermediate_size=512,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=head_dim,
        max_window_layers=n_layers,
        use_cache=True,
    )
    if moe:
        d.update(num_experts=8, num_local_experts=8, num_experts_per_tok=2, moe_intermediate_size=256, decoder_sparse_step=1, mlp_only_layers=[])
        d["tie_word_embeddings"] = False
    cfg = type(base).from_dict(d)
    torch.manual_seed(1234)
    model = AutoModelForCausalLM.from_config(cfg)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith("norm.weight"):
                p.normal_(1.0, 0.1)
            elif n.endswith(".bias"):
                p.normal_(0.0, 0.1)
            elif n.endswith("mlp.gate.weight"):
                p.normal_(0.0, 1.0)  # decisive routing: no top-k near-ties for quantized checks to flip
            else:
                p.normal_(0.0, 0.08)
    # bf16 round trip of the weights only: buffers (RoPE inv_freq) stay fp32 as in a fresh load.
    buffers = {n: b.clone() for n, b in model.named_buffers()}
    name = f"llama-tiny-{family}"
    ckpt = out_dir / name
    model = model.to(torch.bfloat16)
    model.save_pretrained(ckpt)
    model = model.to(torch.float32).eval()
    for n, b in model.named_buffers():
        b.copy_(buffers[n])
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json"):
        try:
            shutil.copy(hf_hub_download(repo, f), ckpt / f)
        except Exception:
            pass
    return model, name


def tiny_ggufs(family: str, out_dir: Path, llama_cpp: Path):
    """bf16 / Q8_0 / Q4_K_M GGUFs of the tiny checkpoint (llama.cpp converter + quantizer)."""
    conv = llama_cpp / "convert_hf_to_gguf.py"
    quant = next((p for p in (llama_cpp / "build-cpu/bin/llama-quantize", llama_cpp / "build/bin/llama-quantize") if p.is_file()), None)
    if not conv.is_file() or quant is None:
        print(f"skipping tiny GGUFs: no {conv} / llama-quantize", file=sys.stderr)
        return []
    name = f"llama-tiny-{family}"
    bf16 = out_dir / f"{name}-bf16.gguf"
    subprocess.run([sys.executable, str(conv), str(out_dir / name), "--outtype", "bf16", "--outfile", str(bf16)], check=True)
    outs = [bf16.name]
    for q in ("Q8_0", "Q4_K_M"):
        o = out_dir / f"{name}-{q.lower()}.gguf"
        subprocess.run([str(quant), str(bf16), str(o), q], check=True, capture_output=True)
        outs.append(o.name)
    return outs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=sorted(REPOS), default="qwen3")
    ap.add_argument("--model", help="HF repo (default: the family's smallest official checkpoint)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--llama-cpp", default="plans/llama.cpp-latest", type=Path)
    ap.add_argument("--max-new-tokens", default=40, type=int)
    ap.add_argument("--gen-logit-steps", default=12, type=int)
    ap.add_argument("--long-tail", default=8, type=int)
    args = ap.parse_args()
    repo = args.model or REPOS[args.family]
    if args.family == "qwen3moe" and not args.tiny:
        print("qwen3moe: only --tiny is supported (the smallest real model is 30B)", file=sys.stderr)
        return 2
    out = args.out or Path("data/reference/llama") / args.family / ("tiny" if args.tiny else "")
    out.mkdir(parents=True, exist_ok=True)

    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(repo)
    if args.tiny:
        model, name = tiny(args.family, repo, out)
    else:
        model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32, attn_implementation="eager").eval()
        name = repo
    model.config._attn_implementation = "eager"
    text = model.model
    layers = text.layers

    eos = []
    for e in [model.config.eos_token_id, getattr(model.generation_config, "eos_token_id", None)]:
        for x in e if isinstance(e, list) else [e]:
            if x is not None and x not in eos:
                eos.append(x)
    for t in STOP_NAMES:
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i != tok.unk_token_id and i not in eos and t in tok.get_vocab():
            eos.append(i)

    tensors: dict[str, np.ndarray] = {}
    captured: dict[str, torch.Tensor] = {}

    def hook(key):
        def f(_m, _inp, o):
            captured[key] = o[0] if isinstance(o, tuple) else o

        return f

    handles = [text.embed_tokens.register_forward_hook(hook("embed"))]
    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(hook(f"layers.{i}")))
        handles.append(layer.self_attn.register_forward_hook(hook(f"layers.{i}.op")))
    handles.append(text.norm.register_forward_hook(hook("norm")))

    cases = {}
    for cname, messages in (("short", SHORT_MESSAGES), ("long", LONG_MESSAGES)):
        case = {"messages": messages, "add_generation_prompt": True, "enable_thinking": False}
        prompt = render(tok, case)
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
            "enable_thinking": False,
            "prompt_text": prompt,
            "prompt_tokens": ids[0].tolist(),
            "generated_tokens": gen,
            "generated_text": tok.decode(gen, skip_special_tokens=False),
        }
        print(f"{cname}: {ids.shape[1]} prompt tokens -> {tok.decode(gen)!r}", file=sys.stderr)
    for h in handles:
        h.remove()

    template_cases = []
    for c in TEMPLATE_CASES:
        try:
            rendered = render(tok, c)
        except Exception as e:  # the family's template rejects this conversation shape
            print(f"template case skipped: {e}", file=sys.stderr)
            continue
        template_cases.append(
            {
                "messages_json": json.dumps(c["messages"], ensure_ascii=False),
                "tools_json": json.dumps(c["tools"], ensure_ascii=False) if c.get("tools") else None,
                "add_generation_prompt": c["add_generation_prompt"],
                "enable_thinking": c.get("enable_thinking"),
                "text": rendered,
            }
        )
    tokenizer_cases = [{"text": t, "ids": tok(t, add_special_tokens=False).input_ids} for t in TOKENIZER_CASES]

    ggufs = tiny_ggufs(args.family, out, args.llama_cpp) if args.tiny else []
    save_file(tensors, str(out / "activations.safetensors"))
    import transformers

    result = {
        "model": name,
        "family": args.family,
        "source_repo": repo,
        "dtype": "fp32",
        "device": "cpu",
        "date_string": DATE,
        "cases": cases,
        "template_cases": template_cases,
        "tokenizer_cases": tokenizer_cases,
        "chat_template": tok.chat_template,
        "stop": eos,
        "n_layers": len(layers),
        "ggufs": ggufs,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "python": sys.version.split()[0]},
    }
    if args.tiny:
        result["default_dtype"] = "fp32"
    (out / "result.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
