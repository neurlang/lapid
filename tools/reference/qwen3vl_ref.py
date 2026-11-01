# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.6",
#   "transformers>=5.5",
#   "numpy",
#   "safetensors",
#   "pillow",
#   "torchvision",
#   "huggingface_hub",
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
"""Reference data for lapid's Qwen3-VL verification (HF transformers, fp32, CPU, eager, greedy).

    uv run tools/reference/qwen3vl_ref.py --model Qwen/Qwen3-VL-2B-Instruct --out data/reference/qwen3vl
    uv run tools/reference/qwen3vl_ref.py --tiny --out data/reference/qwen3vl/tiny

`--tiny` builds a random Qwen3-VL (4 vision blocks with 3 deepstack mergers, 4 text layers, real
head dims, M-RoPE sections, learned 48x48 position grid, real tokenizer / processor / template),
saves it to `<out>/model` (convertible with `lapid convert`) and records its reference.

Writes `<out>/test.png` (a deterministic synthetic picture), `activations.safetensors`
(`image.pixel_values` [n, 3*16*16] = the first temporal frame of HF's pixel_values,
`image.vision.blocks.{i}`, `image.vision.merged`, `image.vision.deepstack.{j}`, `image.embeds`
(merged text inputs), `image.layers.{i}`, `image.norm`, `image.logits`, `image.gen.logits`,
`text.*` for a text-only prompt) and `result.json` (prompts, token ids, M-RoPE positions,
grid, template / tokenizer cases, greedy tokens).
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors.numpy import save_file
from transformers import AutoProcessor

QUESTION = "Describe this image in one sentence."
TEXT_MESSAGES = [{"role": "user", "content": "What is the capital of France? Answer in one short sentence."}]
TEMPLATE_CASES = [
    {"messages": [{"role": "user", "content": "Hi there!"}], "add_generation_prompt": True},
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is this?"}]},
            {"role": "assistant", "content": "A picture."},
            {"role": "user", "content": [{"type": "text", "text": "Compare:"}, {"type": "image"}, {"type": "image"}]},
        ],
        "add_generation_prompt": True,
    },
    {"messages": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "ünïcödé ✓"}]}], "add_generation_prompt": False},
]
TOKENIZER_CASES = ["Hello world", "<|vision_start|><|image_pad|><|vision_end|>Describe.", "  leading spaces\n\ttabs", "数字 123 é"]


def test_image() -> Image.Image:
    """320x240: blue-to-cyan gradient, a red circle, a yellow square, a green bar."""
    w, h = 320, 240
    x = np.linspace(0, 1, w)[None, :, None]
    y = np.linspace(0, 1, h)[:, None, None]
    arr = np.concatenate([np.zeros((h, w, 1)) + 20, 80 + 120 * y + 0 * x, 200 + 40 * x + 0 * y], axis=2)
    img = Image.fromarray(arr.clip(0, 255).astype(np.uint8), "RGB")
    d = ImageDraw.Draw(img)
    d.ellipse([40, 50, 140, 150], fill=(220, 30, 30))
    d.rectangle([190, 60, 280, 150], fill=(250, 220, 40))
    d.rectangle([30, 190, 290, 215], fill=(40, 170, 60))
    return img


def f32(t: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(t.detach().to(torch.float32).cpu().numpy())


def tiny(out_dir: Path, repo: str):
    from huggingface_hub import hf_hub_download
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    real = json.loads(Path(hf_hub_download(repo, "config.json")).read_text())
    t = dict(real["text_config"])
    t.update(hidden_size=256, intermediate_size=512, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=128)
    v = dict(real["vision_config"])
    v.update(hidden_size=256, intermediate_size=512, depth=4, num_heads=4, out_hidden_size=256, deepstack_visual_indexes=[0, 1, 3])
    cfg = dict(real)
    cfg.update(text_config=t, vision_config=v)
    config = Qwen3VLConfig(**{k: x for k, x in cfg.items() if k not in ("architectures", "transformers_version")})
    for c in (config, config.text_config, config.vision_config):
        c.dtype = torch.float32  # the real config says bfloat16: build and run in f32
    torch.manual_seed(1234)
    model = Qwen3VLForConditionalGeneration(config).to(torch.float32)
    bufs = {n: b.dtype for n, b in model.named_buffers() if "inv_freq" in n}
    assert all(d == torch.float32 for d in bufs.values()), bufs
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "norm" in n and n.endswith("weight"):
                p.normal_(1.0, 0.1)
            elif n.endswith("bias"):
                p.normal_(0.0, 0.05)
            elif "pos_embed" in n:
                p.normal_(0.0, 0.5)
            else:
                p.normal_(0.0, 0.06)
        # bf16-representable weights; buffers (RoPE inv_freq) must stay exact f32, so the
        # module is never cast and the checkpoint is saved as f32.
        for p in model.parameters():
            p.copy_(p.to(torch.bfloat16).to(torch.float32))
    model = model.eval()
    ckpt = out_dir / "model"
    model.save_pretrained(ckpt)
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.json", "preprocessor_config.json", "generation_config.json"):
        shutil.copy(hf_hub_download(repo, f), ckpt / f)
    return model


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--out", default="data/reference/qwen3vl", type=Path)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--max-new-tokens", default=24, type=int)
    ap.add_argument("--gen-logit-steps", default=8, type=int)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    args.out.mkdir(parents=True, exist_ok=True)
    proc = AutoProcessor.from_pretrained(args.model)
    tok = proc.tokenizer
    if args.tiny:
        model = tiny(args.out, args.model)
        name = "qwen3vl-tiny (random)"
    else:
        from transformers import Qwen3VLForConditionalGeneration

        model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, dtype=torch.float32, attn_implementation="eager").eval()
        name = args.model
    model.config._attn_implementation = "eager"
    model.config.vision_config._attn_implementation = "eager"
    model.config.text_config._attn_implementation = "eager"
    visual = model.model.visual
    lm = model.model.language_model
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]

    img = test_image()
    img.save(args.out / "test.png")

    tensors: dict[str, np.ndarray] = {}
    captured: dict[str, torch.Tensor] = {}

    def hook(key):
        def f(_m, _inp, out):
            captured[key] = out[0] if isinstance(out, tuple) else out

        return f

    def pre_hook(_m, args_, kwargs):
        captured["embeds"] = kwargs.get("inputs_embeds")

    handles = [lm.register_forward_pre_hook(pre_hook, with_kwargs=True)]
    for i, layer in enumerate(lm.layers):
        handles.append(layer.register_forward_hook(hook(f"layers.{i}")))
    handles.append(lm.norm.register_forward_hook(hook("norm")))
    for i, blk in enumerate(visual.blocks):
        handles.append(blk.register_forward_hook(hook(f"vision.blocks.{i}")))
    handles.append(visual.merger.register_forward_hook(hook("vision.merged")))
    for j, m in enumerate(visual.deepstack_merger_list):
        handles.append(m.register_forward_hook(hook(f"vision.deepstack.{j}")))

    image_messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUESTION}]}]
    cases = {}
    for cname, messages, images in (("image", image_messages, [img]), ("text", TEXT_MESSAGES, None)):
        prompt = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = proc(text=[prompt], images=images, return_tensors="pt")
        ids = inputs["input_ids"]
        captured.clear()
        kw = {k: v for k, v in inputs.items() if k != "input_ids"}
        logits = model(input_ids=ids, use_cache=False, **kw).logits[0]
        for k, v in captured.items():
            if v is None:
                continue
            t = v[0] if (k.startswith("layers.") or k in ("norm", "embeds")) else v
            tensors[f"{cname}.{k}"] = f32(t)
        tensors[f"{cname}.logits"] = f32(logits)
        case = {"messages_json": json.dumps(messages, ensure_ascii=False), "prompt_text": prompt, "prompt_tokens": ids[0].tolist()}
        if images:
            pv = inputs["pixel_values"]
            n = pv.shape[0]
            tensors[f"{cname}.pixel_values"] = f32(pv.reshape(n, 3, 2, 16 * 16)[:, :, 0, :].reshape(n, -1))
            case["grid_thw"] = inputs["image_grid_thw"][0].tolist()
            mm = inputs.get("mm_token_type_ids")
            if mm is None:
                mm = (ids == model.config.image_token_id).int()
            pos, _ = model.model.get_rope_index(ids, mm, image_grid_thw=inputs["image_grid_thw"])
            case["positions"] = pos[:, 0, :].tolist()
        out = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens, do_sample=False, temperature=None, top_k=None, top_p=None,
            repetition_penalty=1.0, eos_token_id=eos, output_logits=True, return_dict_in_generate=True,
        )
        gen = out.sequences[0, ids.shape[1] :].tolist()
        tensors[f"{cname}.gen.logits"] = f32(torch.stack([l[0] for l in out.logits])[: args.gen_logit_steps])
        case["generated_tokens"] = gen
        case["generated_text"] = tok.decode(gen, skip_special_tokens=False)
        cases[cname] = case
        print(f"{cname}: {ids.shape[1]} prompt tokens -> {tok.decode(gen)!r}", file=sys.stderr)
    for h in handles:
        h.remove()

    template_cases = [
        {
            "messages_json": json.dumps(c["messages"], ensure_ascii=False),
            "add_generation_prompt": c["add_generation_prompt"],
            "text": proc.apply_chat_template(c["messages"], add_generation_prompt=c["add_generation_prompt"], tokenize=False),
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
        "eos": eos,
        "n_layers": len(lm.layers),
        "n_vision_blocks": len(visual.blocks),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "python": sys.version.split()[0]},
    }
    (args.out / "result.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
