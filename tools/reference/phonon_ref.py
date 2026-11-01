# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.4",
#   "transformers @ git+https://github.com/huggingface/transformers@66f51e76efd8542232b8bdd40ade7d1edbaac46f",
#   "numpy",
#   "safetensors",
#   "tokenizers",
#   "huggingface_hub",
#   "jinja2",
#   "soundfile",
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
"""Reference activations for lapid's Phonon verification (HF transformers port, fp32, CPU).

    uv run tools/reference/phonon_ref.py --model gradium/phonon-40m-en \
        --text "Hello from Phonon. This is a test of the lapid runtime." --seed 0 \
        --out data/reference/phonon-40m-en

The generation loop is written out (prefill, then one step per frame with the KV cache) so
the flow sampler's Gaussian noise can be saved; it is checked against `model.generate` with
the same seed. The codec decode is run stage by stage (non-streaming, causal).

Writes:
  <out>/activations.safetensors
     voice [F,512]                         default voice prompt latents
     lm.embeds [1,L,512]                   prompt input embeddings (voice | text | BOS)
     lm.layers.{i} [1,L,512]               prefill block outputs (all but the last block)
     lm.out [1,L,512]                      prefill hidden states after the final LayerNorm
     gen.noise [n,32]                      scaled noise of every frame
     gen.conditions [n,512]                backbone state that conditioned each frame
     gen.eos [n]                           end-of-speech logit of each frame
     gen.latents [n,32]                    sampled latents (normalized space)
     gen.audio_latents [n,32]              codec latents (latent * std + mean)
     mimi.latent_proj [1,512,n]  mimi.upsample [1,512,16n]  mimi.transformer [1,16n,512]
     mimi.seanet.{idx} [1,C,L]             every SEANet decoder layer output
     mimi.audio [1,1,1920n]                decoded audio (before fade-in / trimming)
     audio [N]                             processor.decode output (the final waveform)
  <out>/result.json  text, prompt, text_ids, frames_after_eos, n_frames, seed, temperature, ...
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import transformers
from huggingface_hub import hf_hub_download
from safetensors.numpy import save_file
from safetensors.torch import load_file
from transformers import AutoProcessor, GradiumPhononTTSForConditionalGeneration


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gradium/phonon-40m-en")
    ap.add_argument("--text", default="Hello from Phonon. This is a test of the lapid runtime.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    # oneDNN's aarch64 JIT crashes on the codec convolutions (Xbyak "label is redefined").
    torch.backends.mkldnn.enabled = False

    processor = AutoProcessor.from_pretrained(args.model)
    model = GradiumPhononTTSForConditionalGeneration.from_pretrained(args.model, dtype=torch.float32).eval()
    cfg, gen_cfg = model.config, model.generation_config
    temperature = gen_cfg.temperature if args.temperature is None else args.temperature

    voice = load_file(hf_hub_download(args.model, "default-voice.safetensors"))["speaker_wavs"][0].T.float()
    conversation = [{"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": args.text}]}]
    inputs = processor.apply_chat_template(
        conversation, tokenize=True, return_dict=True, add_generation_prompt=True,
        processor_kwargs={"audio_prompt_latents": [voice]},
    )
    input_ids = inputs["input_ids"]
    prompt_latents = inputs["audio_prompt_latents"]
    frames_after_eos = int(inputs["frames_after_eos"][0])
    placeholders = {cfg.audio_prompt_token_id, cfg.audio_bos_token_id, cfg.audio_latent_token_id}
    text_ids = [int(t) for t in input_ids[0] if int(t) not in placeholders]
    prepared = processor.tokenizer.decode(text_ids)
    max_frames = math.ceil((len(text_ids) / 3 + 2) * cfg.frame_rate)

    acts: dict[str, np.ndarray] = {"voice": voice.numpy()}

    # ---- prefill with hidden states
    out = model(
        input_ids=input_ids, audio_prompt_latents=prompt_latents, use_cache=True, output_hidden_states=True,
    )
    hs = out.hidden_states
    acts["lm.embeds"] = hs[0].numpy()
    # hidden_states = (embeddings, block 0 .. block n-2 outputs, final *normed* state): the
    # last block's raw output is not exposed (its normed version is lm.out).
    for i in range(cfg.num_hidden_layers - 1):
        acts[f"lm.layers.{i}"] = hs[i + 1].numpy()
    acts["lm.out"] = out.last_hidden_state.numpy()

    # ---- generation loop (HF `_sample` semantics, noise recorded)
    torch.manual_seed(args.seed)
    noise_scale = math.sqrt(temperature)
    past = out.past_key_values
    noises, conds, eoss, latents = [], [], [], []
    remaining = -1
    latent_id = torch.tensor([[cfg.audio_latent_token_id]])
    for frame in range(max_frames):
        cond = out.last_hidden_state[:, -1, :]
        eos = float(out.logits[0, -1, 0])
        noise = torch.randn(1, cfg.latent_dim) * noise_scale
        latent = model.sample_latent(cond, noise, gen_cfg.num_flow_steps)
        noises.append(noise[0]); conds.append(cond[0]); eoss.append(eos); latents.append(latent[0])
        if remaining < 0 and eos > gen_cfg.eos_threshold and frame >= gen_cfg.min_frames_before_eos:
            remaining = frames_after_eos
        if remaining == 0:
            break
        if remaining > 0:
            remaining -= 1
        out = model(input_ids=latent_id, audio_latents=latent[:, None, :], past_key_values=past, use_cache=True)
        past = out.past_key_values
    n = len(latents)
    lat = torch.stack(latents)
    audio_latents = model.denormalize_latents(lat)
    acts["gen.noise"] = torch.stack(noises).numpy()
    acts["gen.conditions"] = torch.stack(conds).numpy()
    acts["gen.eos"] = np.array(eoss, dtype=np.float32)
    acts["gen.latents"] = lat.numpy()
    acts["gen.audio_latents"] = audio_latents.numpy()

    # ---- cross-check with model.generate (same seed)
    torch.manual_seed(args.seed)
    g = model.generate(**inputs, do_sample=True, temperature=temperature)
    g_lat = g.audio_latents[0, : int(g.audio_latents_mask[0].sum())]
    generate_matches = bool(g_lat.shape[0] == n and torch.allclose(g_lat, audio_latents, atol=1e-5))

    # ---- codec decode, stage by stage
    tok = processor.audio_tokenizer.eval().float()
    dec = tok.decoder
    x = tok.latent_proj(audio_latents.T[None])
    acts["mimi.latent_proj"] = x.numpy()
    x = dec.upsample(x)
    acts["mimi.upsample"] = x.numpy()
    h = dec.decoder_transformer(x.transpose(1, 2)).last_hidden_state
    acts["mimi.transformer"] = h.numpy()
    x = h.transpose(1, 2)
    for idx, layer in enumerate(dec.decoder_seanet.layers):
        x = layer(x)
        acts[f"mimi.seanet.{idx}"] = x.numpy()
    acts["mimi.audio"] = x.numpy()
    wave = processor.decode(audio_latents)
    acts["audio"] = wave.numpy().astype(np.float32)
    decode_matches = bool(torch.allclose(wave, x[0, 0, : wave.shape[0]] if processor.fade_in_duration == 0 else wave))

    # The codec weights of the runtime package (what lapid converts) vs the processor's codec.
    st = load_file(hf_hub_download(args.model, "model.safetensors"))
    codec_diff = float((st["mimi.upsample.convtr.convtr.weight"] - dec.upsample.conv.weight).abs().max())

    acts = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in acts.items()}
    save_file(acts, str(args.out / "activations.safetensors"))
    result = {
        "model": args.model,
        "text": args.text,
        "prepared_text": prepared,
        "prompt": processor.tokenizer.decode(input_ids[0]),
        "text_ids": text_ids,
        "voice_frames": int(voice.shape[0]),
        "frames_after_eos": frames_after_eos,
        "max_frames": max_frames,
        "n_frames": n,
        "seed": args.seed,
        "temperature": temperature,
        "eos_threshold": gen_cfg.eos_threshold,
        "num_flow_steps": gen_cfg.num_flow_steps,
        "fade_in_duration": processor.fade_in_duration,
        "sample_rate": processor.feature_extractor.sampling_rate,
        "generate_matches_loop": generate_matches,
        "decode_matches_stages": decode_matches,
        "codec_upsample_weight_max_diff_vs_package": codec_diff,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
    }
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("prepared_text", "n_frames", "generate_matches_loop",
                                             "decode_matches_stages", "codec_upsample_weight_max_diff_vs_package")}))


if __name__ == "__main__":
    main()
