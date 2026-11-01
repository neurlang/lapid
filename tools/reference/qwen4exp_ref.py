# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.6",
#   "transformers>=5.22",
#   "numpy",
#   "safetensors",
#   "sentencepiece",
#   "protobuf",
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
"""Reference data for lapid's Qwen4-Exp verification (HF transformers, fp32, CPU, eager, greedy).

    uv run tools/reference/qwen4exp_ref.py --out data/reference/qwen4exp/tiny [--gguf]

The released checkpoint (Qwen/Qwen3.8-Flash-Next, 177B with a 51B-parameter n-gram table)
does not fit an fp32 reference here, so this builds a tiny random-weight `qwen4_exp` text model
from the real `config.json`: 4 layers (3 Gated DeltaNet + 1 QSA attention, PLE on layer 2 as
in the release), hidden 256 with the real head dims (attention 256, DeltaNet 128 with 16:48 ->
2:6 k:v heads, indexer 4 x 128), hc 4 streams (rank 32), 8 experts top-2, the real tokenizer
and hash scheme (n-gram vocab base shrunk to 1000), and a QSA budget of 16 tokens so the
long prompt exercises block selection. It is saved as an HF checkpoint in
`<out>/qwen4exp-tiny/` (bf16; the reference runs on the bf16-rounded values in fp32).

`--gguf` also writes `<out>/qwen4exp-tiny-f16.gguf` with llama.cpp's converter
(plans/llama.cpp-latest) and, when `llama-quantize` is built, `qwen4exp-tiny-q4_k_m.gguf`.

Writes:
  <out>/activations.safetensors
      short.embed [S,D], short.layers.{i}.ple [S,hc*D] (PLE injection), short.layers.{i}.op [S,D]
      (token mixer output), short.layers.{i} [S,hc*D] (decoder layer output: the hc streams),
      short.norm [S,D] (final hyper-connection mixer), short.logits [S,V], short.gen.logits [K,V]
      (raw logits of greedy steps, uncached forwards: see `greedy`), long.layers.{i}.tail [T,hc*D], long.logits.last [1,V], long.gen.logits [K,V]
  <out>/result.json
      cases.{short,long}: messages_json, enable_thinking, prompt_text, prompt_tokens, generated_tokens,
      generated_text; template_cases; tokenizer_cases; versions
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import AutoTokenizer

REPO = "Qwen/Qwen3.8-Flash-Next"
ROOT = Path(__file__).resolve().parents[2]

LONG_TEXT = """The history of the printing press is a story of many small improvements rather than a single \
invention. Long before movable type reached Europe, craftsmen in East Asia carved whole pages into wooden \
blocks, inked them, and pressed paper against the surface to make copies of religious texts, calendars and \
official notices. Later, artisans experimented with individual characters made of baked clay, wood and \
eventually metal, which could be arranged, printed and then reused for another page. In the fifteenth \
century a goldsmith in Mainz combined several existing ideas: a screw press adapted from wine making, an \
oil-based ink that clung to metal, and a hand mould that could cast large numbers of identical letters \
quickly. The result was not just faster copying but a new economy of knowledge."""

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
    {
        "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}],
        "add_generation_prompt": True,
        "enable_thinking": False,
    },
    {
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 = 4.", "reasoning_content": "Simple sum."},
            {"role": "user", "content": "And 3+3?"},
        ],
        "add_generation_prompt": True,
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
    "<|im_start|>user\nspecial tokens inline<|im_end|>\n<think></think>",
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
    """Greedy decoding by uncached full forwards (the model's definition). transformers'
    cached `generate` deviates from its own uncached forward once QSA selects blocks during
    decoding (dense QSA budgets agree to 3e-6), so the cached path is only reported."""
    seq = ids[0].tolist()
    gen, logits = [], []
    for _ in range(max_new):
        lg = model(input_ids=torch.tensor([seq]), use_cache=False).logits[0, -1]
        logits.append(lg)
        t = int(lg.argmax())
        gen.append(t)
        seq.append(t)
        if t in eos:
            break
    return gen, torch.stack(logits)


def greedy_cached(model, ids: torch.Tensor, max_new: int, eos: list[int]):
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


def tiny(out_dir: Path):
    """Random qwen4_exp text model from the real config, saved as an HF checkpoint (bf16)."""
    from huggingface_hub import hf_hub_download
    from transformers import Qwen4ExpForCausalLM, Qwen4ExpTextConfig

    real = json.loads(Path(hf_hub_download(REPO, "config.json")).read_text())["text_config"]
    t = dict(real)
    for k in ("model_type", "dtype", "mtp", "transformers_version"):
        t.pop(k, None)
    t.update(
        hidden_size=256,
        num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        linear_num_key_heads=2,
        linear_num_value_heads=6,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=256,
        shared_expert_intermediate_size=256,
        hc_lowrank=32,
        indexer_budget=16,
        ngram_vocab_size_base=1000,
        ple_embed_dim=256,
        split_ngram_parts=4,
        mtp_num_hidden_layers=0,
        tie_word_embeddings=False,
    )
    cfg = Qwen4ExpTextConfig(**t)
    torch.manual_seed(1234)
    model = Qwen4ExpForCausalLM(cfg)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith("linear_attn.norm.weight"):
                p.normal_(1.0, 0.1)
            elif "norm" in n:
                p.normal_(0.0, 0.1)  # zero-centred (1 + w) norms, incl. hc / PLE / indexer
            elif n.endswith("A_log"):
                p.uniform_(-1.0, 1.0)
            elif n.endswith("dt_bias"):
                p.normal_(0.0, 0.5)
            elif "ngram_embedding" in n:
                p.normal_(0.0, 0.5)
            else:
                p.normal_(0.0, 0.06)
    model = model.to(torch.bfloat16).to(torch.float32).eval()
    ckpt = out_dir / "qwen4exp-tiny"
    model.to(torch.bfloat16).save_pretrained(ckpt)
    # transformers saves the QSA layers as "indexed_attention"; the release (and llama.cpp's
    # converter, which only enables QSA on "full_attention" layers) uses "full_attention".
    cj = json.loads((ckpt / "config.json").read_text())
    cj["layer_types"] = ["full_attention" if t == "indexed_attention" else t for t in cj["layer_types"]]
    (ckpt / "config.json").write_text(json.dumps(cj, indent=2))
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
        shutil.copyfile(hf_hub_download(REPO, f), ckpt / f)
    # The reference runs the saved checkpoint as loaded (what converters read): the freshly
    # constructed module differs from it in the QSA attention layer (up to 8e-4).
    model = Qwen4ExpForCausalLM.from_pretrained(ckpt, dtype=torch.float32).eval()
    return model, ckpt


def write_gguf(ckpt: Path, out: Path):
    llama = ROOT / "plans" / "llama.cpp-latest"
    f16 = out / "qwen4exp-tiny-f16.gguf"
    env = dict(os.environ, PYTHONPATH=f"{llama / 'gguf-py'}:{llama}", NO_LOCAL_GGUF="")
    # llama.cpp's converter reads the config through transformers, which renames the QSA layers
    # "indexed_attention", and then enables QSA only on "full_attention" layers (compress ratios
    # all 0 = dense). Map the names back before it writes the parameters.
    patch = (
        "import sys, runpy\n"
        "from conversion.qwen4exp import Qwen4ExpTextModel as M\n"
        "orig = M.set_gguf_parameters\n"
        "def fixed(self):\n"
        "    lt = self.hparams.get('layer_types') or []\n"
        "    self.hparams['layer_types'] = ['full_attention' if t == 'indexed_attention' else t for t in lt]\n"
        "    orig(self)\n"
        "M.set_gguf_parameters = fixed\n"
        f"sys.argv = {[str(llama / 'convert_hf_to_gguf.py'), str(ckpt), '--outfile', str(f16), '--outtype', 'f16', '--no-mtp']!r}\n"
        f"runpy.run_path({str(llama / 'convert_hf_to_gguf.py')!r}, run_name='__main__')\n"
    )
    subprocess.run([sys.executable, "-c", patch], check=True, env=env)
    quant = llama / "build-cpu" / "bin" / "llama-quantize"
    if quant.is_file():
        subprocess.run([str(quant), str(f16), str(out / "qwen4exp-tiny-q4_k_m.gguf"), "Q4_K_M"], check=True)
        subprocess.run([str(quant), str(f16), str(out / "qwen4exp-tiny-q8_0.gguf"), "Q8_0"], check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/reference/qwen4exp/tiny", type=Path)
    ap.add_argument("--max-new-tokens", default=24, type=int)
    ap.add_argument("--gen-logit-steps", default=10, type=int)
    ap.add_argument("--long-tail", default=8, type=int)
    ap.add_argument("--gguf", action="store_true")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    args.out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(REPO)
    model, ckpt = tiny(args.out)
    model.config._attn_implementation = "eager"
    text = model.model
    layers = text.layers
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
        if layer.ple is not None:
            handles.append(layer.ple.register_forward_hook(hook(f"layers.{i}.ple")))
    handles.append(text.hyper_connection_mixer.register_forward_hook(hook("norm")))

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
                if k.startswith("layers.") and k.count(".") == 1:
                    tensors[f"long.{k}.tail"] = f32(v[0, -args.long_tail :])
            tensors["long.logits.last"] = f32(logits[-1:])
        gen, gen_logits = greedy(model, ids, args.max_new_tokens, eos)
        tensors[f"{cname}.gen.logits"] = f32(gen_logits[: args.gen_logit_steps])
        hf_cached, _ = greedy_cached(model, ids, args.max_new_tokens, eos)
        cases[cname] = {
            "messages_json": json.dumps(messages, ensure_ascii=False),
            "enable_thinking": thinking,
            "prompt_text": prompt,
            "prompt_tokens": ids[0].tolist(),
            "generated_tokens": gen,
            "generated_text": tok.decode(gen, skip_special_tokens=False),
            "hf_cached_generated_tokens": hf_cached,
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
        "model": "qwen4exp-tiny (random, real config shape-scaled)",
        "dtype": "fp32",
        "device": "cpu",
        "cases": cases,
        "template_cases": template_cases,
        "tokenizer_cases": tokenizer_cases,
        "n_layers": len(layers),
        "layer_types": list(model.config.layer_types),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "python": sys.version.split()[0]},
    }
    (args.out / "result.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    if args.gguf:
        write_gguf(ckpt, args.out)
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
