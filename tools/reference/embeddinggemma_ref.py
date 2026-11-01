# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.4",
#   "transformers>=4.57",
#   "sentence-transformers>=5.1",
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
"""Reference activations for lapid's EmbeddingGemma verification (sentence-transformers, fp32, CPU).

    uv run tools/reference/embeddinggemma_ref.py --model google/embeddinggemma-300m \
        --out data/reference/embeddinggemma

Each item is a text plus an optional prompt name ("query", "document", "STS", ...); the model
input is `prompt + text` tokenized with <bos> ... <eos> (sentence-transformers semantics).

With --bench: only times model.encode on the bench corpus (batch 1 / 32) and prints JSON.

Writes:
  <out>/activations.safetensors
      s{k}.embed            [n, 768]  scaled token embeddings (input of layer 0)
      s{k}.layers.{i}       [n, 768]  raw decoder-layer outputs
      s{k}.norm             [n, 768]  after the final RMSNorm (token embeddings)
      s{k}.pooled           [768]     mean pooling (prompt tokens included)
      s{k}.dense.0          [3072]    2_Dense
      s{k}.dense.1          [768]     3_Dense
      s{k}.embedding        [768]     L2-normalized sentence embedding
    for the first --layered items (each run alone, no padding), and
      embeddings            [N, 768]  model.encode() of every item (batched, padded)
  <out>/result.json         items (text, prompt_name, prompt, input_ids), prompts, layered,
                            sliding window, versions
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file

LONG_PARAGRAPH = (
    "The history of computing is a story of abstraction layered upon abstraction. Early machines were "
    "programmed by physically rewiring circuits, then by punched cards, then by assembly language and "
    "finally by high-level languages whose compilers hide the details of the hardware entirely. "
)

ITEMS: list[tuple[str, str | None]] = [
    ("Which planet is known as the Red Planet?", "query"),
    ("Mars, known for its reddish appearance, is often referred to as the Red Planet.", "document"),
    ("The quick brown fox jumps over the lazy dog.", None),
    ("Le chat dort sur le canapé pendant que la pluie tombe. 東京は日本の首都です。", "STS"),
    ("Venus is often called Earth's twin because of its similar size and proximity.", "document"),
    ("How do I bake sourdough bread at home?", "query"),
    ("emoji 🚀 and symbols ∑ ≈ ∞, numbers 12345.678", "Clustering"),
    # > 512 tokens: exercises the bidirectional sliding window (|i - j| < 257).
    (LONG_PARAGRAPH * 14, "document"),
]


# Same corpus as crates/lapid-hub/src/embeddinggemma/bench.rs (CORPUS), "document" prompt.
BENCH_CORPUS = [
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "Sourdough bread needs a lively starter, patience and a very hot oven.",
    "Rust guarantees memory safety without a garbage collector.",
    "The stock market closed higher today after a volatile trading session.",
    "Photosynthesis converts light energy into chemical energy in plants.",
    "A good night's sleep improves memory, mood and concentration.",
    "The museum's new exhibition features impressionist paintings from France.",
    "Electric cars are becoming cheaper as battery prices continue to fall.",
    "The recipe calls for two cups of flour, one egg and a pinch of salt.",
    "Mount Everest is the highest mountain above sea level on Earth.",
    "Machine translation quality has improved dramatically in the last decade.",
    "The library will be closed on Monday for the public holiday.",
]


def bench(model, runs: int, n: int) -> dict:
    """sentence-transformers fp32 on CPU: sentences/s at batch 1 and 32 (median of runs)."""
    import statistics
    import time

    texts = [BENCH_CORPUS[i % len(BENCH_CORPUS)] for i in range(n)]
    out = {"engine": "sentence-transformers (PyTorch fp32, cpu)", "threads": torch.get_num_threads(), "sentences": n}
    for bs in (1, 32):
        def one_pass() -> float:
            t = time.perf_counter()
            if bs == 1:
                for s in texts:
                    model.encode([s], prompt_name="document", batch_size=1)
            else:
                model.encode(texts, prompt_name="document", batch_size=bs)
            return (time.perf_counter() - t) * 1e3

        one_pass()
        med = statistics.median(one_pass() for _ in range(runs))
        out[f"batch_{bs}"] = {"total_ms": med, "ms_per_sentence": med / n, "sentences_per_s": n / (med / 1e3)}
    return out


def f32(t: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(t.detach().to(torch.float32).cpu().numpy())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/embeddinggemma-300m")
    ap.add_argument("--out", default="data/reference/embeddinggemma", type=Path)
    ap.add_argument("--layered", default=4, type=int, help="items with layer-by-layer activations")
    ap.add_argument("--long-layered", action="store_true", help="also trace the long (> 512 token) item")
    ap.add_argument("--bench", action="store_true", help="only benchmark (prints JSON), no reference data")
    ap.add_argument("--runs", default=3, type=int)
    args = ap.parse_args()

    import sentence_transformers
    import transformers
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(0)
    model = SentenceTransformer(args.model, device="cpu", model_kwargs={"torch_dtype": torch.float32})
    model.eval()
    if args.bench:
        with torch.no_grad():
            print(json.dumps(bench(model, args.runs, 64), indent=1))
        return 0
    tok = model.tokenizer
    prompts: dict[str, str] = dict(model.prompts)
    hf = model[0].auto_model
    cfg = hf.config

    acts: dict[str, np.ndarray] = {}
    items = []
    for k, (text, pname) in enumerate(ITEMS):
        prompt = prompts.get(pname, "") if pname else ""
        ids = tok(prompt + text, add_special_tokens=True)["input_ids"]
        items.append({"text": text, "prompt_name": pname, "prompt": prompt, "input_ids": ids, "n_tokens": len(ids)})

    traced = list(range(min(args.layered, len(items))))
    if args.long_layered:
        traced.append(len(items) - 1)
    with torch.no_grad():
        for k in traced:
            store: dict[str, torch.Tensor] = {}
            hooks = [hf.embed_tokens.register_forward_hook(lambda m, i, o: store.__setitem__("embed", o))]
            for li, layer in enumerate(hf.layers):
                hooks.append(
                    layer.register_forward_hook(
                        lambda m, i, o, li=li: store.__setitem__(f"layers.{li}", o[0] if isinstance(o, tuple) else o)
                    )
                )
            for mi, name in [(2, "dense.0"), (3, "dense.1")]:
                hooks.append(model[mi].register_forward_hook(lambda m, i, o, name=name: store.__setitem__(name, o["sentence_embedding"])))
            hooks.append(model[1].register_forward_hook(lambda m, i, o: store.__setitem__("pooled", o["sentence_embedding"])))
            ids = torch.tensor([items[k]["input_ids"]], dtype=torch.long)
            feats = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
            out = model(feats)
            for h in hooks:
                h.remove()
            for name, t in store.items():
                acts[f"s{k}.{name}"] = f32(t[0])
            acts[f"s{k}.norm"] = f32(out["token_embeddings"][0])
            acts[f"s{k}.embedding"] = f32(out["sentence_embedding"][0])

        # Batched encode exactly as users call it (grouped by prompt name).
        emb = np.zeros((len(items), cfg.hidden_size), dtype=np.float32)
        for pname in {it["prompt_name"] for it in items}:
            idx = [i for i, it in enumerate(items) if it["prompt_name"] == pname]
            texts = [items[i]["text"] for i in idx]
            e = model.encode(texts, prompt_name=pname, batch_size=8, convert_to_numpy=True)
            emb[idx] = e.astype(np.float32)
        acts["embeddings"] = emb

    for k in traced:
        d = float(np.dot(acts[f"s{k}.embedding"], emb[k]))
        print(f"item {k}: {items[k]['n_tokens']} tokens, cos(single, batched) = {d:.7f}", file=sys.stderr)

    args.out.mkdir(parents=True, exist_ok=True)
    save_file(acts, str(args.out / "activations.safetensors"))
    result = {
        "model": args.model,
        "items": items,
        "layered": traced,
        "n_layers": cfg.num_hidden_layers,
        "prompts": prompts,
        "sliding_window_exclusive": cfg.sliding_window,
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "sentence_transformers": sentence_transformers.__version__,
            "python": sys.version.split()[0],
        },
    }
    (args.out / "result.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
