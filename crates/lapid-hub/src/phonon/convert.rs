//! Phonon checkpoints (Gradium's runtime package on the Hugging Face hub, e.g.
//! `gradium/phonon-40m-en`) -> one `.lapid` file through the generic converter
//! (`lapid_convert::engine`), and voice files -> `.lapid` extras.
//!
//! The model file holds three components (`flow_lm`: backbone, conditioners, flow head;
//! `mimi`: codec decoder; `voices`: voice-prompt latents `voices.<name>` [frames, 512], the
//! package's `default-voice.safetensors` as `voices.default`) and blobs: the tokenizer
//! (`tokenizer.json`, `tokenizer.model`) and the original JSON configs / chat template.
//! Names follow the checkpoint, except that the fused attention `in_proj` weights are split
//! into `q_proj` / `k_proj` / `v_proj` (rows in the original interleaved RoPE layout) and
//! training-only tensors (`flow.w_s_t.*`, `learnt_padding`) are dropped. Precision:
//! transformer / flow-head Linear weights follow the dtype; the text embedding at most int8;
//! convolutions and other tables f16; norms, biases, latent statistics and voices f32.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use lapid::{Error, Result, bail};
use lapid_convert::OutDType;
use lapid_convert::assets::HfFiles;
use lapid_convert::engine::{ConvertHooks, ConvertSpec, Emit, HfExtras, TensorSource};
use lapid_convert::mapping::{RoleRule, Rule};
use lapid_convert::policy::Role;
use lapid_convert::safetensors::SafeTensors;
use lapid_convert::writer::{ModelWriter, read_json};
use serde_json::{Value, json};

use super::config::{PhononConfig, from_hf};

pub(crate) const VOICE_PREFIX: &str = "voices.";
const VOICE_KEY: &str = "speaker_wavs";

pub static CONVERT: ConvertSpec = ConvertSpec {
    architecture: "phonon",
    license: "cc-by-nc-4.0",
    hf_model_types: &["gradium_phonon_tts"],
    gguf_architectures: &[],
    hf_names: &[
        Rule::drop("flow_lm.flow.w_s_t.*"),
        Rule::drop("*.learnt_padding"),
        Rule::drop("*num_ema_updates"),
        Rule::new("voices.{v}.speaker_wavs", "voices.{v}"),
    ],
    gguf_names: &[],
    roles: &[
        RoleRule::new("voices.*", Role::Param),
        RoleRule::new("*transcript_in_segment.embed.weight", Role::LinearInt8),
        RoleRule::new("*.embed.weight", Role::Dense),
    ],
    text_assets: false,
    hooks: &Hooks,
};

struct Hooks;

impl ConvertHooks for Hooks {
    fn hf_extras(&self, files: &HfFiles, cfg: &mut Value) -> Result<HfExtras> {
        let mut extras = HfExtras::default();
        let read = |p: &Path| std::fs::read(p).map_err(|e| Error::new("io", format!("{}: {e}", p.display())));
        let mut json_of = |name: &str| -> Result<Option<Value>> {
            match files.optional(name)? {
                Some(p) => {
                    extras.blobs.push((format!("config/{name}"), "application/json".into(), read(&p)?));
                    Ok(Some(read_json(&p)?))
                }
                None => Ok(None),
            }
        };
        let (generation, processor) = (json_of("generation_config.json")?, json_of("processor_config.json")?);
        let mut c = from_hf(cfg, generation.as_ref(), processor.as_ref())?;
        extras.blobs.push(("tokenizer/tokenizer.json".into(), "application/json".into(), read(&files.required("tokenizer.json")?)?));
        if let Some(p) = files.optional("tokenizer.model")? {
            c.assets.tokenizer_model = Some("tokenizer/tokenizer.model".into());
            extras.blobs.push(("tokenizer/tokenizer.model".into(), "application/x-sentencepiece".into(), read(&p)?));
        }
        if let Some(p) = files.optional("chat_template.jinja")? {
            extras.blobs.push(("config/chat_template.jinja".into(), "text/plain".into(), read(&p)?));
        }
        extras.blobs.push(("config/config.json".into(), "application/json".into(), read(&files.config)?));
        extras.weights.push((format!("{VOICE_PREFIX}default."), files.required("default-voice.safetensors")?));
        let license = card_license(files.optional("README.md")?.as_deref()).unwrap_or_else(|| CONVERT.license.to_string());
        extras.metadata.insert(
            "components".into(),
            json!({
                "flow_lm": {"architecture": "phonon-flow-lm", "config": "phonon", "license": license, "tensor_prefix": "flow_lm."},
                "mimi": {"architecture": "mimi-decoder", "config": "phonon", "license": license, "tensor_prefix": "mimi."},
                "voices": {"architecture": "phonon-voices", "config": "phonon", "license": license, "tensor_prefix": VOICE_PREFIX},
            }),
        );
        extras.metadata.insert("license".into(), json!(license));
        *cfg = serde_json::to_value(&c)?;
        Ok(extras)
    }

    fn expected(&self, cfg: &Value) -> Result<Vec<String>> {
        Ok(expected_tensors(&serde_json::from_value(cfg.clone())?))
    }

    /// Split fused `in_proj` weights into q / k / v; voices `[1, dim, frames]` -> `[frames, dim]`.
    fn transform(&self, name: &str, src: &dyn TensorSource, cfg: &Value) -> Result<Emit> {
        let shape = src.shape(name).unwrap_or_default();
        if let Some(voice) = name.strip_prefix(VOICE_PREFIX) {
            let (data, shape) = voice_latents(&src.f32(name)?, &shape, name)?;
            let dim = cfg.pointer("/lm/audio_prompt_dim").and_then(Value::as_u64).unwrap_or(0) as usize;
            if shape[1] != dim {
                bail!("shape_mismatch", "voice {voice:?} has width {}, model expects {dim}", shape[1]);
            }
            return Ok(Emit::Replace(vec![(name.to_string(), shape, data)]));
        }
        let Some(prefix) = name.strip_suffix(".in_proj.weight").or_else(|| name.strip_suffix(".in_proj_weight")) else {
            return Ok(Emit::Keep);
        };
        Ok(Emit::Replace(split_in_proj(prefix, &shape, &src.f32(name)?)?))
    }
}

/// `license:` from the model card front matter.
fn card_license(readme: Option<&Path>) -> Option<String> {
    let text = std::fs::read_to_string(readme?).ok()?;
    let front = text.strip_prefix("---")?.split("\n---").next()?;
    front.lines().find_map(|l| l.trim().strip_prefix("license:").map(|s| s.trim().to_string()))
}

/// (name, shape, data) of one written tensor.
type Mapped = (String, Vec<usize>, Vec<f32>);

/// Fused `{prefix}.in_proj.weight` [3d, d] -> q / k / v projections.
fn split_in_proj(prefix: &str, shape: &[usize], data: &[f32]) -> Result<Vec<Mapped>> {
    if shape.len() != 2 || !shape[0].is_multiple_of(3) {
        bail!("shape_mismatch", "{prefix}: fused in_proj must be [3*d, d], got {shape:?}");
    }
    let (rows, cols) = (shape[0] / 3, shape[1]);
    Ok(["q_proj", "k_proj", "v_proj"]
        .iter()
        .enumerate()
        .map(|(i, p)| (format!("{prefix}.{p}.weight"), vec![rows, cols], data[i * rows * cols..(i + 1) * rows * cols].to_vec()))
        .collect())
}

/// Voice latents `[1, dim, frames]` (or `[dim, frames]`) -> row-major `[frames, dim]`.
fn voice_latents(data: &[f32], shape: &[usize], what: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    let (dim, frames) = match shape {
        [1, d, t] | [d, t] => (*d, *t),
        s => bail!("shape_mismatch", "{what}: {VOICE_KEY} must be [1, dim, frames], got {s:?}"),
    };
    let mut out = vec![0.0f32; data.len()];
    for c in 0..dim {
        for t in 0..frames {
            out[t * dim + c] = data[c * frames + t];
        }
    }
    Ok((out, vec![frames, dim]))
}

/// Voice-prompt latents [frames, dim] from a voice safetensors (`speaker_wavs` [1, dim, frames]).
pub fn read_voice_safetensors(path: &Path) -> Result<(Vec<f32>, Vec<usize>)> {
    let st = SafeTensors::open(path)?;
    voice_latents(&st.to_f32(VOICE_KEY)?, &st.shape(VOICE_KEY)?, &path.display().to_string())
}

/// Tensors every Phonon checkpoint must provide (after mapping).
fn expected_tensors(cfg: &PhononConfig) -> Vec<String> {
    let mut v: Vec<String> = [
        "flow_lm.bos_emb",
        "flow_lm.emb_mean",
        "flow_lm.emb_std",
        "flow_lm.input_linear.weight",
        "flow_lm.out_norm.weight",
        "flow_lm.out_eos.weight",
        "flow_lm.condition_provider.conditioners.transcript_in_segment.embed.weight",
        "flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight",
        "flow_lm.flow_net.cond_embed.weight",
        "flow_lm.flow_net.input_proj.weight",
        "flow_lm.flow_net.final_layer.linear.weight",
        "mimi.quantizer.output_proj.weight",
        "mimi.upsample.convtr.convtr.weight",
        "voices.default",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    for i in 0..cfg.lm.n_layers {
        for p in
            ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj", "linear1", "linear2"]
        {
            v.push(format!("flow_lm.transformer.layers.{i}.{p}.weight"));
        }
    }
    for i in 0..cfg.mimi.transformer.n_layers {
        v.push(format!("mimi.decoder_transformer.transformer.layers.{i}.self_attn.q_proj.weight"));
        v.push(format!("mimi.decoder_transformer.transformer.layers.{i}.layer_scale_1.scale"));
    }
    for i in 0..cfg.flow.depth {
        v.push(format!("flow_lm.flow_net.res_blocks.{i}.mlp.0.weight"));
    }
    v
}

/// Write voices (each a `speaker_wavs` safetensors, e.g. from `gradium/phonon-voices`) into
/// a `.lapid` extras file (`kind: "extra"`, extends component `voices`); tensors
/// `voices.<name>`; `source_repo` is recorded as the source (e.g. "gradium/phonon-voices"). Load it with [`super::Phonon::add_voices`].
pub fn convert_voices(voices: &[(String, PathBuf)], source_repo: &str, license: &str, out: &Path) -> Result<()> {
    if voices.is_empty() {
        bail!("invalid_argument", "no voices to convert");
    }
    let source = json!({"repo": source_repo, "files": voices.iter().map(|(n, p)| json!({"voice": n, "file": p.display().to_string()})).collect::<Vec<_>>()});
    let mut meta = lapid_convert::writer::metadata("phonon-voices", "phonon", OutDType::F32, license, source);
    meta["kind"] = json!("extra");
    meta["extends"] = json!({"architecture": "phonon", "component": "voices"});
    let mut w = ModelWriter::create_shared(out, meta, OutDType::F32, CONVERT.role_arc(), voices.len())?;
    let mut seen = BTreeSet::new();
    let mut expected = Vec::new();
    for (name, path) in voices {
        if !seen.insert(name.clone()) {
            bail!("invalid_argument", "duplicate voice name {name:?}");
        }
        let (data, shape) = read_voice_safetensors(path)?;
        let t = format!("{VOICE_PREFIX}{name}");
        w.add_f32(&t, &shape, &data)?;
        expected.push(t);
    }
    w.finish(&expected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lapid_convert::mapping::{Renamed, rename};

    #[test]
    fn roles_and_names() {
        let role = |n: &str, s: &[usize]| CONVERT.role(n, s);
        assert_eq!(role("flow_lm.transformer.layers.0.linear1.weight", &[2048, 512]), Role::Linear);
        assert_eq!(role("flow_lm.out_norm.weight", &[512]), Role::Param);
        assert_eq!(role("voices.default", &[125, 512]), Role::Param);
        assert_eq!(role("mimi.decoder.model.0.conv.weight", &[512, 512, 7]), Role::Dense);
        assert_eq!(role("flow_lm.condition_provider.conditioners.num_speakers.embed.weight", &[32, 16]), Role::Dense);
        assert_eq!(role("flow_lm.condition_provider.conditioners.transcript_in_segment.embed.weight", &[4001, 512]), Role::LinearInt8);
        assert_eq!(rename(CONVERT.hf_names, "flow_lm.flow.w_s_t.0.bias"), Renamed::Drop);
        assert_eq!(rename(CONVERT.hf_names, "a.b.learnt_padding"), Renamed::Drop);
        assert_eq!(rename(CONVERT.hf_names, "voices.default.speaker_wavs"), Renamed::To("voices.default".into()));
    }

    #[test]
    fn in_proj_split_and_voices() {
        let data: Vec<f32> = (0..6 * 2).map(|x| x as f32).collect();
        let out = split_in_proj("a.self_attn", &[6, 2], &data).expect("split");
        assert_eq!(out.len(), 3);
        assert_eq!(out[1].0, "a.self_attn.k_proj.weight");
        assert_eq!(out[1].2, vec![4.0, 5.0, 6.0, 7.0]);
        let (v, s) = voice_latents(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3], "v").expect("voice");
        assert_eq!((v, s), (vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![3, 2]));
    }
}
