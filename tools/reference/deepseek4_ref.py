# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.6",
#   "transformers>=5.22",
#   "numpy",
#   "safetensors",
#   "sentencepiece",
#   "pyyaml",
#   "requests",
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
"""Reference data for lapid's DeepSeek V4 verification (HF transformers `DeepseekV4ForCausalLM`,
fp32, CPU, eager, greedy).

    uv run tools/reference/deepseek4_ref.py --tiny --out data/reference/deepseek4/tiny [--gguf]

The real checkpoints (V4-Flash: 284B params, ~160 GB FP8/FP4) do not fit this machine, so the
main proof is `--tiny`: a random-weight model built from the real V4-Flash config with every
architectural feature kept (head dim 512, rope dim 64 at the tail, 1 shared KV head + attention
sinks, q low-rank, grouped low-rank output projection, layer pattern sliding / CSA (ratio 4,
overlapping compressor + lightning indexer top-k) / HCA (ratio 128), YaRN on compressed layers,
hyper-connections (4 streams, Sinkhorn 20), hash-routed and sqrtsoftplus-routed MoE with a
shared expert and SwiGLU clamping, the real tokenizer and chat encoding), only narrower /
fewer layers / experts. Saved as an HF checkpoint (original DeepSeek tensor names, bf16) in
`<out>/model/`; the reference runs on the bf16-rounded weights in fp32.

`--gguf` also writes `<out>/deepseek4-tiny-bf16.gguf` with llama.cpp's own converter
(`plans/llama.cpp-latest/conversion`, routed experts stacked as plain tensors instead of the
MXFP4 repack the converter applies to the real FP4 checkpoint) and quantizes it with
`llama-quantize` to `<out>/deepseek4-tiny-Q4_K_M.gguf` and `-Q8_0.gguf`.

Writes:
  <out>/activations.safetensors
      short.embed [S,D], short.layers.{i} [S,4D] (hyper-connection streams after layer i, flattened),
      short.layers.{i}.op [S,D] (attention output), short.norm [S,D], short.logits [S,V],
      short.gen.logits [K,V] (raw logits of greedy steps), long.layers.{i}.tail [T,4D],
      long.logits.last [1,V], long.gen.logits [K,V]
  <out>/result.json
      cases.{short,long}: messages_json, thinking, prompt_text, prompt_tokens, generated_tokens,
      generated_text; template_cases (DeepSeek V4 `encoding_dsv4.encode_messages`); tokenizer_cases
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file

REPO = "deepseek-ai/DeepSeek-V4-Flash"
ROOT = Path(__file__).resolve().parents[2]
LLAMA = ROOT / "plans" / "llama.cpp-latest"

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
work with a precision that had been impossible when every manuscript differed."""

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
                "properties": {"city": {"type": "string", "description": "City name"}, "days": {"type": "integer"}},
                "required": ["city"],
            },
        },
    }
]

TEMPLATE_CASES = [
    {"messages": [{"role": "user", "content": "Hello!"}], "thinking": False},
    {"messages": [{"role": "user", "content": "Hello!"}], "thinking": True},
    {"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}], "thinking": False},
    {
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 = 4.", "reasoning_content": "Simple sum."},
            {"role": "user", "content": "And 3+3?"},
        ],
        "thinking": True,
    },
    {
        "messages": [
            {"role": "user", "content": "Name a color."},
            {"role": "assistant", "content": "Blue.", "reasoning_content": "Pick any color."},
        ],
        "thinking": True,
    },
    {
        "messages": [
            {"role": "user", "content": "Name a color."},
            {"role": "assistant", "content": "Blue."},
            {"role": "user", "content": "Another?"},
        ],
        "thinking": False,
    },
    {
        "messages": [
            {"role": "system", "content": "Use tools when helpful.", "tools": TOOLS},
            {"role": "user", "content": "Weather in Paris?"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "Need the tool.",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\", \"days\": 2}"}}
                ],
            },
            {"role": "tool", "content": "{\"temp\": 18, \"sky\": \"clear\"}", "tool_call_id": "c1"},
            {"role": "tool", "content": "second", "tool_call_id": "c1"},
        ],
        "thinking": True,
    },
    {
        "messages": [
            {"role": "system", "content": "Use tools.", "tools": TOOLS},
            {"role": "user", "content": "Weather in Rome?"},
        ],
        "thinking": False,
    },
    {"messages": [{"role": "user", "content": "Ünïcödé — 日本語 🙂 tabs\tand\nnewlines"}], "thinking": False},
]

TOKENIZER_CASES = [
    "Hello, world!",
    " leading space and  double  spaces",
    "Numbers: 1234567 and 3.14159, year 2026.",
    "def f(x):\n    return x * 2  # comment\n\n\n",
    "Ünïcödé — 日本語 🙂 emoji and accents: café naïve",
    "It's we're they'll I'd I'M",
    "<｜User｜>special tokens inline<｜Assistant｜></think>",
    "中文分词测试，日本語のテキスト。",
]


def f32(t: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(t.detach().to(torch.float32).cpu().numpy())


def load_encoding():
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO, "encoding/encoding_dsv4.py")
    spec = importlib.util.spec_from_file_location("encoding_dsv4", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tiny_config():
    from transformers import DeepseekV4Config

    return DeepseekV4Config(
        vocab_size=129280,
        hidden_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=6,
        num_attention_heads=8,
        num_key_value_heads=1,
        head_dim=512,
        qk_rope_head_dim=64,
        q_lora_rank=48,
        o_groups=2,
        o_lora_rank=32,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        scoring_func="sqrtsoftplus",
        norm_topk_prob=True,
        routed_scaling_factor=1.5,
        swiglu_limit=0.6,
        sliding_window=16,
        compress_ratios=[0, 4, 128, 4, 128, 0],
        num_hash_layers=2,
        compress_rope_theta=160000,
        rope_theta=10000,
        rope_parameters={"rope_type": "yarn", "factor": 16, "original_max_position_embeddings": 65536, "beta_fast": 32, "beta_slow": 1},
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=6,
        num_nextn_predict_layers=0,
        rms_norm_eps=1e-6,
        max_position_embeddings=1048576,
        tie_word_embeddings=False,
        bos_token_id=0,
        eos_token_id=1,
    )


def raw_config(cfg) -> dict:
    """config.json in the upstream (V4-Flash) key style, which both llama.cpp's converter and lapid read."""
    return {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "bos_token_id": 0,
        "eos_token_id": 1,
        "hc_eps": cfg.hc_eps,
        "hc_mult": cfg.hc_mult,
        "hc_sinkhorn_iters": cfg.hc_sinkhorn_iters,
        "head_dim": cfg.head_dim,
        "hidden_act": "silu",
        "hidden_size": cfg.hidden_size,
        "index_head_dim": cfg.index_head_dim,
        "index_n_heads": cfg.index_n_heads,
        "index_topk": cfg.index_topk,
        "max_position_embeddings": cfg.max_position_embeddings,
        "moe_intermediate_size": cfg.moe_intermediate_size,
        "n_routed_experts": cfg.n_routed_experts,
        "n_shared_experts": cfg.n_shared_experts,
        "norm_topk_prob": True,
        "num_attention_heads": cfg.num_attention_heads,
        "num_experts_per_tok": cfg.num_experts_per_tok,
        "num_hash_layers": 2,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_key_value_heads": 1,
        "num_nextn_predict_layers": 0,
        "o_groups": cfg.o_groups,
        "o_lora_rank": cfg.o_lora_rank,
        "q_lora_rank": cfg.q_lora_rank,
        "qk_rope_head_dim": 64,
        "rms_norm_eps": cfg.rms_norm_eps,
        "rope_scaling": {"beta_fast": 32, "beta_slow": 1, "factor": 16, "original_max_position_embeddings": 65536, "type": "yarn"},
        "rope_theta": 10000,
        "routed_scaling_factor": cfg.routed_scaling_factor,
        "scoring_func": "sqrtsoftplus",
        "sliding_window": cfg.sliding_window,
        "swiglu_limit": cfg.swiglu_limit,
        "tie_word_embeddings": False,
        "topk_method": "noaux_tc",
        "torch_dtype": "bfloat16",
        "vocab_size": cfg.vocab_size,
        "compress_rope_theta": 160000,
        "compress_ratios": [0, 4, 128, 4, 128, 0],
    }


UPSTREAM_LAYER = [
    ("input_layernorm.weight", "attn_norm.weight"),
    ("post_attention_layernorm.weight", "ffn_norm.weight"),
    ("attn_hc.fn", "hc_attn_fn"),
    ("attn_hc.base", "hc_attn_base"),
    ("attn_hc.scale", "hc_attn_scale"),
    ("ffn_hc.fn", "hc_ffn_fn"),
    ("ffn_hc.base", "hc_ffn_base"),
    ("ffn_hc.scale", "hc_ffn_scale"),
    ("self_attn.sinks", "attn.attn_sink"),
    ("self_attn.q_a_proj.weight", "attn.wq_a.weight"),
    ("self_attn.q_a_norm.weight", "attn.q_norm.weight"),
    ("self_attn.q_b_proj.weight", "attn.wq_b.weight"),
    ("self_attn.kv_proj.weight", "attn.wkv.weight"),
    ("self_attn.kv_norm.weight", "attn.kv_norm.weight"),
    ("self_attn.o_a_proj.weight", "attn.wo_a.weight"),
    ("self_attn.o_b_proj.weight", "attn.wo_b.weight"),
    ("self_attn.compressor.kv_proj.weight", "attn.compressor.wkv.weight"),
    ("self_attn.compressor.gate_proj.weight", "attn.compressor.wgate.weight"),
    ("self_attn.compressor.position_bias", "attn.compressor.ape"),
    ("self_attn.compressor.kv_norm.weight", "attn.compressor.norm.weight"),
    ("self_attn.compressor.indexer.kv_proj.weight", "attn.indexer.compressor.wkv.weight"),
    ("self_attn.compressor.indexer.gate_proj.weight", "attn.indexer.compressor.wgate.weight"),
    ("self_attn.compressor.indexer.position_bias", "attn.indexer.compressor.ape"),
    ("self_attn.compressor.indexer.kv_norm.weight", "attn.indexer.compressor.norm.weight"),
    ("self_attn.compressor.indexer.q_b_proj.weight", "attn.indexer.wq_b.weight"),
    ("self_attn.compressor.indexer.scorer.weights_proj.weight", "attn.indexer.weights_proj.weight"),
    ("mlp.gate.weight", "ffn.gate.weight"),
    ("mlp.gate.e_score_correction_bias", "ffn.gate.bias"),
    ("mlp.shared_experts.gate_proj.weight", "ffn.shared_experts.w1.weight"),
    ("mlp.shared_experts.down_proj.weight", "ffn.shared_experts.w2.weight"),
    ("mlp.shared_experts.up_proj.weight", "ffn.shared_experts.w3.weight"),
]


def upstream_state_dict(model, cfg) -> dict:
    """The tiny model under the tensor names of the upstream DeepSeek V4 checkpoints (what the
    real repos ship and llama.cpp's converter reads): bf16 weights, int32 hash routing tables."""
    sd = {k: v.detach() for k, v in model.state_dict().items()}
    out = {
        "embed.weight": sd["model.embed_tokens.weight"],
        "head.weight": sd["lm_head.weight"],
        "norm.weight": sd["model.norm.weight"],
        "hc_head_fn": sd["model.hc_head.hc_fn"],
        "hc_head_base": sd["model.hc_head.hc_base"],
        "hc_head_scale": sd["model.hc_head.hc_scale"],
    }
    inter = cfg.moe_intermediate_size
    for i in range(cfg.num_hidden_layers):
        p, q = f"model.layers.{i}.", f"layers.{i}."
        for hf, up in UPSTREAM_LAYER:
            if p + hf in sd:
                out[q + up] = sd[p + hf]
        if p + "mlp.gate.tid2eid" in sd:
            out[q + "ffn.gate.tid2eid"] = sd[p + "mlp.gate.tid2eid"].to(torch.int32)
            out.pop(q + "ffn.gate.bias", None)
        gate_up, down = sd[p + "mlp.experts.gate_up_proj"], sd[p + "mlp.experts.down_proj"]
        for e in range(gate_up.shape[0]):
            out[q + f"ffn.experts.{e}.w1.weight"] = gate_up[e, :inter]
            out[q + f"ffn.experts.{e}.w3.weight"] = gate_up[e, inter:]
            out[q + f"ffn.experts.{e}.w2.weight"] = down[e]
    return {k: (v.to(torch.bfloat16) if v.is_floating_point() else v).contiguous() for k, v in out.items()}


def tiny_model(out_dir: Path):
    from huggingface_hub import hf_hub_download
    from transformers import DeepseekV4ForCausalLM

    cfg = tiny_config()
    torch.manual_seed(1234)
    model = DeepseekV4ForCausalLM(cfg)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith("norm.weight"):
                p.normal_(1.0, 0.1)
            elif n.endswith(".sinks"):
                p.normal_(0.0, 1.0)
            elif n.endswith(".base") or n.endswith("hc_base"):
                p.normal_(0.0, 0.5)
            elif n.endswith(".scale") or n.endswith("hc_scale"):
                p.normal_(1.0, 0.2)
            elif n.endswith("position_bias"):
                p.normal_(0.0, 0.5)
            elif n.endswith("embed_tokens.weight"):
                p.normal_(0.0, 1.0)
            else:
                p.normal_(0.0, 0.08)
        for n, b in model.named_buffers():
            if n.endswith("e_score_correction_bias"):
                b.normal_(0.0, 0.1)
            elif n.endswith("tid2eid"):
                g = torch.Generator().manual_seed(7)
                e = cfg.n_routed_experts
                b.copy_(torch.argsort(torch.rand(b.shape[0], e, generator=g), dim=1)[:, : b.shape[1]])
        # round the weights to bf16 (the checkpoint dtype); RoPE inv_freq buffers stay fp32
        for p in model.parameters():
            p.copy_(p.to(torch.bfloat16).to(torch.float32))
        for n, b in model.named_buffers():
            if n.endswith("e_score_correction_bias"):
                b.copy_(b.to(torch.bfloat16).to(torch.float32))
    model = model.eval()
    ckpt = out_dir / "model"
    ckpt.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file as save_torch

    save_torch(upstream_state_dict(model, cfg), str(ckpt / "model.safetensors"), metadata={"format": "pt"})
    (ckpt / "config.json").write_text(json.dumps(raw_config(cfg), indent=2))
    for f in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        shutil.copy(hf_hub_download(REPO, f), ckpt / f)
    return model


def write_gguf(out_dir: Path):
    """llama.cpp's DeepSeek V4 converter on the tiny checkpoint, experts stacked unquantized."""
    sys.path.insert(0, str(LLAMA / "gguf-py"))
    sys.path.insert(0, str(LLAMA))
    import gguf
    from conversion.base import LazyTorchTensor
    from conversion.deepseek import DeepseekV4Model

    class Tiny(DeepseekV4Model):
        model_arch = gguf.MODEL_ARCH.DEEPSEEK4

        def _write_mxfp4_expert_tensor(self, bid, proj, tensor_key):
            n = self.hparams["n_routed_experts"]
            names = [f"layers.{bid}.ffn.experts.{e}.{proj}.weight" for e in range(n)]
            data = torch.stack([LazyTorchTensor.to_eager(self.model_tensors[x]()).to(torch.float32) for x in names])
            self.gguf_writer.add_tensor(self.format_tensor_name(tensor_key, bid), data.numpy().astype(np.float32))
            return names

        def prepare_tensors(self):
            super(DeepseekV4Model, self).prepare_tensors()

    src = out_dir / "model"
    bf16 = out_dir / "deepseek4-tiny-bf16.gguf"
    m = Tiny(src, gguf.LlamaFileType.MOSTLY_BF16, bf16, eager=True, model_name="deepseek4-tiny")
    m.write()
    quant = LLAMA / "build-cpu" / "bin" / "llama-quantize"
    for q in ("Q4_K_M", "Q8_0"):
        subprocess.run([str(quant), str(bf16), str(out_dir / f"deepseek4-tiny-{q}.gguf"), q], check=True)


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/reference/deepseek4/tiny", type=Path)
    ap.add_argument("--tiny", action="store_true", help="random tiny model (the only mode: real V4 checkpoints are 160+ GB)")
    ap.add_argument("--gguf", action="store_true")
    ap.add_argument("--max-new-tokens", default=40, type=int)
    ap.add_argument("--gen-logit-steps", default=12, type=int)
    ap.add_argument("--long-tail", default=8, type=int)
    args = ap.parse_args()
    if not args.tiny:
        print("only --tiny is supported (DeepSeek V4 checkpoints do not fit this machine)", file=sys.stderr)
        return 2

    from transformers import AutoTokenizer

    torch.set_grad_enabled(False)
    args.out.mkdir(parents=True, exist_ok=True)
    enc = load_encoding()
    model = tiny_model(args.out)
    tok = AutoTokenizer.from_pretrained(args.out / "model")
    model.config._attn_implementation = "eager"
    text = model.model
    eos = [1]

    tensors: dict[str, np.ndarray] = {}
    captured: dict[str, torch.Tensor] = {}

    def hook(key):
        def f(_m, _inp, out):
            captured[key] = out[0] if isinstance(out, tuple) else out

        return f

    handles = [text.embed_tokens.register_forward_hook(hook("embed"))]
    for i, layer in enumerate(text.layers):
        handles.append(layer.register_forward_hook(hook(f"layers.{i}")))
        handles.append(layer.self_attn.register_forward_hook(hook(f"layers.{i}.op")))
    handles.append(text.norm.register_forward_hook(hook("norm")))

    def flat(k, v):
        v = v[0]
        return v.reshape(v.shape[0], -1) if k.startswith("layers.") and not k.endswith(".op") else v

    cases = {}
    for cname, messages, thinking in (("short", SHORT_MESSAGES, False), ("long", LONG_MESSAGES, False)):
        prompt = enc.encode_messages(messages, thinking_mode="thinking" if thinking else "chat")
        ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids
        captured.clear()
        logits = model(input_ids=ids, use_cache=False).logits[0]
        if cname == "short":
            for k, v in captured.items():
                tensors[f"short.{k}"] = f32(flat(k, v))
            tensors["short.logits"] = f32(logits)
        else:
            for k, v in captured.items():
                if k.startswith("layers.") and not k.endswith(".op"):
                    tensors[f"long.{k}.tail"] = f32(flat(k, v)[-args.long_tail :])
            tensors["long.logits.last"] = f32(logits[-1:])
        gen, gen_logits = greedy(model, ids, args.max_new_tokens, eos)
        tensors[f"{cname}.gen.logits"] = f32(gen_logits[: args.gen_logit_steps])
        cases[cname] = {
            "messages_json": json.dumps(messages, ensure_ascii=False),
            "thinking": thinking,
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
            "thinking": c["thinking"],
            "text": enc.encode_messages(c["messages"], thinking_mode="thinking" if c["thinking"] else "chat"),
        }
        for c in TEMPLATE_CASES
    ]
    tokenizer_cases = [{"text": t, "ids": tok(t, add_special_tokens=False).input_ids} for t in TOKENIZER_CASES]

    save_file(tensors, str(args.out / "activations.safetensors"))
    import transformers

    result = {
        "model": "deepseek4-tiny (random)",
        "dtype": "fp32",
        "device": "cpu",
        "cases": cases,
        "template_cases": template_cases,
        "tokenizer_cases": tokenizer_cases,
        "special": {"eos": eos},
        "n_layers": len(text.layers),
        "layer_types": list(model.config.layer_types),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "python": sys.version.split()[0]},
    }
    (args.out / "result.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    if args.gguf:
        write_gguf(args.out)
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
