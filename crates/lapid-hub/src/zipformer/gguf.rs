//! GGUF source of the Zipformer2 converter: the k2/icefall export with
//! `general.architecture = "zipformer2"` (e.g. `vibe-app/zipformer2-66m-streaming-sk-gguf`).
//!
//! * Names: `embed.*` and `enc.{stack}.{layer}.*` become icefall's names (the ones the HF
//!   route and the loader use), see [`GGUF_NAMES`]. Shapes need no change: a GGUF tensor's
//!   `shape` is row-major (`ne` reversed), the layout of icefall's safetensors, so the
//!   `reshape` / `transform` hooks of [`super::convert`] apply as they are.
//! * Config: the `stt.*` keys. lapid's frontend, embedding stack and joiner are fixed, so the
//!   file's description of them is checked, not read. The file's own `chunk_size` /
//!   `left_context_frames` are -1 (full context, what offline decoders use); the streaming
//!   defaults come from its `eval_*` keys.
//! * Tokens: `tokenizer.ggml.tokens` becomes the `tokens.txt` blob.

use lapid::{Error, Result, bail};
use lapid_convert::engine::HfExtras;
use lapid_convert::gguf::{GgufFile, GgufValue};
use lapid_convert::mapping::Rule;
use serde_json::{Map, json};

use super::config::{TOKENS_BLOB, ZipformerConfig};
use super::convert::MODEL_TYPE;

/// GGUF name -> icefall name. Layers of a downsampled stack sit under `.encoder` (icefall's
/// `DownsampledZipformer2Encoder`), stack 0 never downsamples ([`config_from_gguf`] checks
/// that). Layer tensors have two to four name segments after the layer index; first match wins.
pub(crate) const GGUF_NAMES: &[Rule] = &[
    Rule::new("embed.{a}.{b}", "encoder_embed.{a}.{b}"),
    Rule::new("embed.{a}.{b}.{c}", "encoder_embed.{a}.{b}.{c}"),
    Rule::new("enc.downsample_output.bias", "encoder.downsample_output.bias"),
    Rule::new("enc.{i}.downsample.bias", "encoder.encoders.{i}.downsample.bias"),
    Rule::new("enc.{i}.out_combiner.bypass_scale", "encoder.encoders.{i}.out_combiner.bypass_scale"),
    Rule::new("enc.0.{j}.{a}.{b}", "encoder.encoders.0.layers.{j}.{a}.{b}"),
    Rule::new("enc.0.{j}.{a}.{b}.{c}", "encoder.encoders.0.layers.{j}.{a}.{b}.{c}"),
    Rule::new("enc.0.{j}.{a}.{b}.{c}.{d}", "encoder.encoders.0.layers.{j}.{a}.{b}.{c}.{d}"),
    Rule::new("enc.{i}.{j}.{a}.{b}", "encoder.encoders.{i}.encoder.layers.{j}.{a}.{b}"),
    Rule::new("enc.{i}.{j}.{a}.{b}.{c}", "encoder.encoders.{i}.encoder.layers.{j}.{a}.{b}.{c}"),
    Rule::new("enc.{i}.{j}.{a}.{b}.{c}.{d}", "encoder.encoders.{i}.encoder.layers.{j}.{a}.{b}.{c}.{d}"),
    Rule::new("decoder.{a}.{b}", "decoder.{a}.{b}"),
    Rule::new("joiner.{a}.{b}", "joiner.{a}.{b}"),
];

/// Typed reads of the GGUF metadata with errors that name the key.
struct Meta<'a>(&'a GgufFile);

impl<'a> Meta<'a> {
    fn get(&self, k: &str) -> Result<&'a GgufValue> {
        self.0.get(k).ok_or_else(|| Error::new("missing_metadata", format!("GGUF zipformer2: no {k}")))
    }

    fn bad(k: &str, what: &str) -> Error {
        Error::new("invalid_metadata", format!("GGUF zipformer2: {k} is not {what}"))
    }

    fn uint(&self, k: &str) -> Result<u64> {
        self.get(k)?.as_u64().ok_or_else(|| Self::bad(k, "a non-negative integer"))
    }

    fn list(&self, k: &str) -> Result<Vec<u64>> {
        let items = self.get(k)?.as_array().ok_or_else(|| Self::bad(k, "a list"))?;
        items.iter().map(|v| v.as_u64().ok_or_else(|| Self::bad(k, "a list of non-negative integers"))).collect()
    }

    fn text(&self, k: &str) -> Result<&'a str> {
        self.get(k)?.as_str().ok_or_else(|| Self::bad(k, "a string"))
    }

    fn float(&self, k: &str) -> Result<f64> {
        self.get(k)?.as_f64().ok_or_else(|| Self::bad(k, "a number"))
    }

    fn flag(&self, k: &str) -> Result<bool> {
        match self.get(k)? {
            GgufValue::Bool(b) => Ok(*b),
            _ => Err(Self::bad(k, "a bool")),
        }
    }
}

/// Fails when the file describes a frontend other than the one lapid's `Fbank` implements
/// (lhotse / kaldi: 25 ms Povey window, 10 ms shift, 512-point FFT, DC removal, pre-emphasis
/// 0.97, no dither, 20 Hz to Nyquist - 400 Hz, power spectrum, `snip_edges = false`).
fn check_frontend(m: &Meta, sample_rate: u64) -> Result<()> {
    let k = |s: &str| format!("stt.frontend.{s}");
    let near = |a: f64, b: f64| (a - b).abs() < 1e-4;
    let same = m.text(&k("type"))? == "kaldi_fbank"
        && m.text(&k("window"))? == "povey"
        && !m.flag(&k("snip_edges"))?
        && m.flag(&k("remove_dc_offset"))?
        && m.flag(&k("use_power"))?
        && (m.uint(&k("n_fft"))?, m.uint(&k("win_length"))?, m.uint(&k("hop_length"))?) == (512, 400, 160)
        && near(m.float(&k("pre_emphasis"))?, 0.97)
        && near(m.float(&k("dither"))?, 0.0)
        && near(m.float(&k("f_min"))?, 20.0)
        && near(m.float(&k("f_max"))?, sample_rate as f64 / 2.0 - 400.0)
        && near(m.float(&k("sample_scale"))?, 1.0);
    if !same {
        bail!("unsupported_model", "GGUF zipformer2: the stt.frontend.* keys describe a frontend other than lapid's kaldi fbank");
    }
    Ok(())
}

/// `ZipformerConfig` from the `stt.*` metadata of a `zipformer2` GGUF.
pub(crate) fn config_from_gguf(f: &GgufFile) -> Result<ZipformerConfig> {
    let m = Meta(f);
    let enc = |s: &str| format!("stt.zipformer.encoder.{s}");
    if m.text("stt.zipformer.head_kind")? != "transducer" || m.text("stt.zipformer.joiner.activation")? != "tanh" {
        bail!("unsupported_model", "GGUF zipformer2: only transducers with a tanh joiner are implemented");
    }
    if !m.flag(&enc("causal"))? {
        bail!("unsupported_model", "non-causal Zipformer2 checkpoints cannot stream (causal = false)");
    }
    // Fixed in lapid's embedding stack (`config::EMBED_*`) and final downsampling.
    if m.uint(&enc("output_downsampling_factor"))? != 2 || m.list(&enc("embed_channels"))? != [8, 32, 128] || m.uint(&enc("embed_convnext_kernel"))? != 7 {
        bail!("unsupported_model", "GGUF zipformer2: embedding stack other than 8/32/128 channels, ConvNeXt kernel 7, output downsampling 2");
    }
    let sample_rate = m.uint("stt.frontend.sample_rate")?;
    check_frontend(&m, sample_rate)?;
    let v = json!({
        "sample_rate": sample_rate,
        "feature_dim": m.uint("stt.frontend.num_mels")?,
        "num_encoder_layers": m.list(&enc("n_layers"))?,
        "downsampling_factor": m.list(&enc("downsampling_factor"))?,
        "feedforward_dim": m.list(&enc("feedforward_dim"))?,
        "num_heads": m.list(&enc("n_heads"))?,
        "encoder_dim": m.list(&enc("dim"))?,
        "cnn_module_kernel": m.list(&enc("cnn_module_kernel"))?,
        "query_head_dim": m.list(&enc("query_head_dim"))?,
        "value_head_dim": m.list(&enc("value_head_dim"))?,
        "pos_head_dim": m.list(&enc("pos_head_dim"))?,
        "pos_dim": m.uint(&enc("pos_dim"))?,
        "decoder_dim": m.uint("stt.zipformer.decoder.dim")?,
        "joiner_dim": m.uint("stt.zipformer.joiner.dim")?,
        "context_size": m.uint("stt.zipformer.decoder.context_size")?,
        "vocab_size": m.uint("stt.zipformer.decoder.vocab")?,
        "blank_id": f.get("tokenizer.ggml.blank_token_id").and_then(GgufValue::as_u64).unwrap_or(0),
        // icefall's streaming decode defaults (the file's chunk_size / left_context_frames are -1).
        "chunk_size": m.uint(&enc("eval_chunk_size")).unwrap_or(16),
        "left_context_frames": m.uint(&enc("eval_left_context_frames")).unwrap_or(128),
    });
    let cfg = ZipformerConfig::from_json(&v)?;
    if m.uint(&enc("n_stacks"))? as usize != cfg.num_stacks() {
        bail!("invalid_metadata", "GGUF zipformer2: n_stacks {} disagrees with the per-stack lists", m.uint(&enc("n_stacks"))?);
    }
    let ds = &cfg.downsampling_factor;
    if ds.first() != Some(&1) || ds[1..].contains(&1) {
        bail!("unsupported_model", "GGUF zipformer2: downsampling {ds:?} (stack 0 must be undownsampled and every other stack downsampled)");
    }
    Ok(cfg)
}

/// `tokens.txt` (`<piece> <id>` per line) from `tokenizer.ggml.tokens`.
fn tokens_txt(f: &GgufFile, vocab: usize) -> Result<String> {
    let tokens = f.get("tokenizer.ggml.tokens").and_then(GgufValue::as_array).ok_or_else(|| Error::new("tokenizer", "GGUF has no tokenizer.ggml.tokens"))?;
    if tokens.len() != vocab {
        bail!("tokenizer", "GGUF has {} tokens, the model vocabulary is {vocab}", tokens.len());
    }
    let mut text = String::new();
    for (id, t) in tokens.iter().enumerate() {
        let Some(piece) = t.as_str().filter(|p| !p.is_empty() && !p.contains(['\n', '\r'])) else {
            bail!("tokenizer", "tokenizer.ggml.tokens[{id}] is not a non-empty single-line string");
        };
        text.push_str(piece);
        text.push(' ');
        text.push_str(&id.to_string());
        text.push('\n');
    }
    Ok(text)
}

/// The `tokens.txt` blob; name and license from `general.*`.
pub(crate) fn extras(f: &GgufFile, cfg: &ZipformerConfig) -> Result<HfExtras> {
    let text = tokens_txt(f, cfg.vocab_size)?;
    let license = f.get("general.license").and_then(GgufValue::as_str).unwrap_or("unknown");
    let name = f.get("general.name").and_then(GgufValue::as_str).unwrap_or("zipformer");
    let mut metadata = Map::new();
    metadata.insert("name".into(), json!(name));
    metadata.insert("license".into(), json!(license));
    metadata.insert("components".into(), json!({"zipformer": {"architecture": MODEL_TYPE, "config": "zipformer", "license": license, "tensor_prefix": ""}}));
    Ok(HfExtras { blobs: vec![(TOKENS_BLOB.to_string(), "text/plain".to_string(), text.into_bytes())], metadata, ..HfExtras::default() })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use lapid_convert::mapping::{Renamed, rename};

    use super::super::convert::{expected_tensors, layer_prefix};
    use super::*;
    use crate::zipformer::config::tests::medium;

    /// The GGUF names of every tensor of `cfg`: only the layer prefix differs from icefall's.
    fn gguf_names_of(cfg: &ZipformerConfig) -> Vec<String> {
        let expected = expected_tensors(cfg);
        let layer0 = format!("{}.", layer_prefix(cfg, 0, 0));
        let rel: Vec<&str> = expected.iter().filter_map(|n| n.strip_prefix(&layer0)).collect();
        assert_eq!(rel.len(), 49);
        let mut v: Vec<String> = Vec::new();
        for c in ["conv.0", "conv.4", "conv.7", "convnext.depthwise_conv", "convnext.pointwise_conv1", "convnext.pointwise_conv2", "out"] {
            v.extend([format!("embed.{c}.weight"), format!("embed.{c}.bias")]);
        }
        v.extend(["embed.out_norm.log_scale".to_string(), "embed.out_norm.bias".to_string()]);
        for i in 0..cfg.num_stacks() {
            if cfg.downsampling_factor[i] > 1 {
                v.extend([format!("enc.{i}.downsample.bias"), format!("enc.{i}.out_combiner.bypass_scale")]);
            }
            for j in 0..cfg.num_encoder_layers[i] {
                v.extend(rel.iter().map(|r| format!("enc.{i}.{j}.{r}")));
            }
        }
        v.extend(["enc.downsample_output.bias".to_string(), "decoder.embedding.weight".to_string(), "decoder.conv.weight".to_string()]);
        for m in ["encoder_proj", "decoder_proj", "output_linear"] {
            v.extend([format!("joiner.{m}.weight"), format!("joiner.{m}.bias")]);
        }
        v
    }

    #[test]
    fn names_map_onto_expected_tensors() {
        let cfg = medium();
        let names = gguf_names_of(&cfg);
        assert_eq!(names.len(), 819);
        let to = |n: &str| match rename(GGUF_NAMES, n) {
            Renamed::To(o) => o,
            other => panic!("{n}: {other:?}"),
        };
        let mapped: BTreeSet<String> = names.iter().map(|n| to(n)).collect();
        assert_eq!(mapped.len(), 819, "names collide");
        assert_eq!(mapped, expected_tensors(&cfg).into_iter().collect::<BTreeSet<_>>());
        assert_eq!(to("enc.0.1.norm.bias"), "encoder.encoders.0.layers.1.norm.bias");
        assert_eq!(
            to("enc.3.2.conv_module1.depthwise_conv.causal_conv.weight"),
            "encoder.encoders.3.encoder.layers.2.conv_module1.depthwise_conv.causal_conv.weight"
        );
        assert_eq!(to("embed.convnext.pointwise_conv1.weight"), "encoder_embed.convnext.pointwise_conv1.weight");
        assert_eq!(rename(GGUF_NAMES, "enc.0.0.mystery"), Renamed::NoMatch);
        assert_eq!(rename(GGUF_NAMES, "output.weight"), Renamed::NoMatch);
    }

    /// Minimal GGUF v3 (metadata only) for the config tests.
    #[derive(Default)]
    struct Kv(Vec<u8>, u64);

    impl Kv {
        fn put(&mut self, key: &str, ty: u32, v: &[u8]) {
            self.0.extend((key.len() as u64).to_le_bytes());
            self.0.extend(key.as_bytes());
            self.0.extend(ty.to_le_bytes());
            self.0.extend(v);
            self.1 += 1;
        }
        fn string(v: &str) -> Vec<u8> {
            let mut b = (v.len() as u64).to_le_bytes().to_vec();
            b.extend(v.as_bytes());
            b
        }
        fn str(&mut self, k: &str, v: &str) {
            self.put(k, 8, &Self::string(v));
        }
        fn u32(&mut self, k: &str, v: u32) {
            self.put(k, 4, &v.to_le_bytes());
        }
        fn i32(&mut self, k: &str, v: i32) {
            self.put(k, 5, &v.to_le_bytes());
        }
        fn f32(&mut self, k: &str, v: f32) {
            self.put(k, 6, &v.to_le_bytes());
        }
        fn bool(&mut self, k: &str, v: bool) {
            self.put(k, 7, &[u8::from(v)]);
        }
        fn i32s(&mut self, k: &str, vs: &[i32]) {
            let mut b = 5u32.to_le_bytes().to_vec();
            b.extend((vs.len() as u64).to_le_bytes());
            vs.iter().for_each(|v| b.extend(v.to_le_bytes()));
            self.put(k, 9, &b);
        }
        fn strs(&mut self, k: &str, vs: &[String]) {
            let mut b = 8u32.to_le_bytes().to_vec();
            b.extend((vs.len() as u64).to_le_bytes());
            vs.iter().for_each(|v| b.extend(Self::string(v)));
            self.put(k, 9, &b);
        }
        fn build(&self) -> Vec<u8> {
            let mut b = b"GGUF".to_vec();
            b.extend(3u32.to_le_bytes());
            b.extend(0u64.to_le_bytes());
            b.extend(self.1.to_le_bytes());
            b.extend(&self.0);
            b
        }
    }

    /// The metadata of `vibe-app/zipformer2-66m-streaming-sk-gguf` (as `gguf-dump` shows it).
    fn kv_66m(edit: impl Fn(&mut Kv, &str) -> bool) -> Kv {
        let mut g = Kv::default();
        let z = |s: &str| format!("stt.zipformer.{s}");
        let ints = |g: &mut Kv, k: &str, v: i32| {
            if !edit(g, k) {
                g.u32(k, v as u32);
            }
        };
        g.str("general.architecture", "zipformer2");
        g.str("general.name", "Zipformer2 66M streaming Slovak");
        g.str("general.license", "cc-by-4.0");
        g.str("stt.zipformer.head_kind", "transducer");
        ints(&mut g, &z("encoder.n_stacks"), 6);
        let lists: [(&str, [i32; 6]); 10] = [
            ("n_layers", [2, 2, 3, 4, 3, 2]),
            ("dim", [192, 256, 384, 512, 384, 256]),
            ("downsampling_factor", [1, 2, 4, 8, 4, 2]),
            ("n_heads", [4, 4, 4, 8, 4, 4]),
            ("feedforward_dim", [512, 768, 1024, 1536, 1024, 768]),
            ("cnn_module_kernel", [31, 31, 15, 15, 15, 31]),
            ("query_head_dim", [32; 6]),
            ("value_head_dim", [12; 6]),
            ("pos_head_dim", [4; 6]),
            ("unmasked_dim", [192, 192, 256, 256, 256, 192]),
        ];
        for (k, v) in lists {
            g.i32s(&z(&format!("encoder.{k}")), &v);
        }
        for (k, v) in [("pos_dim", 48), ("output_dim", 512), ("output_downsampling_factor", 2), ("embed_convnext_kernel", 7), ("subsampling_factor", 4)] {
            ints(&mut g, &z(&format!("encoder.{k}")), v);
        }
        if !edit(&mut g, &z("encoder.causal")) {
            g.bool(&z("encoder.causal"), true);
        }
        g.i32s(&z("encoder.embed_channels"), &[8, 32, 128]);
        g.i32s(&z("encoder.train_chunk_size"), &[16, 32, 64, -1]);
        let chunks = [("eval_chunk_size", 16), ("eval_left_context_frames", 128), ("onnx_chunk_size", 64), ("onnx_left_context_frames", 256)];
        for (k, v) in chunks.into_iter().chain([("chunk_size", -1), ("left_context_frames", -1)]) {
            g.i32(&z(&format!("encoder.{k}")), v);
        }
        for (k, v) in [("decoder.dim", 512), ("decoder.context_size", 2), ("decoder.conv_groups", 128), ("decoder.vocab", 1483), ("joiner.dim", 512)] {
            ints(&mut g, &z(k), v);
        }
        if !edit(&mut g, &z("joiner.activation")) {
            g.str(&z("joiner.activation"), "tanh");
        }
        g.u32("stt.frontend.sample_rate", 16000);
        g.u32("stt.frontend.num_mels", 80);
        g.u32("stt.frontend.n_fft", 512);
        g.u32("stt.frontend.win_length", 400);
        g.u32("stt.frontend.hop_length", 160);
        g.str("stt.frontend.type", "kaldi_fbank");
        g.str("stt.frontend.window", "povey");
        for (k, v) in [("dither", 0.0), ("pre_emphasis", 0.97), ("f_min", 20.0), ("f_max", 7600.0), ("sample_scale", 1.0)] {
            g.f32(&format!("stt.frontend.{k}"), v);
        }
        for (k, v) in [("snip_edges", false), ("remove_dc_offset", true), ("use_power", true)] {
            g.bool(&format!("stt.frontend.{k}"), v);
        }
        g.u32("tokenizer.ggml.blank_token_id", 0);
        let mut tokens: Vec<String> = ["<blk>", "<sos/eos>", "<unk>", "S", "\u{2581}THE"].map(String::from).to_vec();
        tokens.extend((tokens.len()..1483).map(|i| format!("\u{2581}t{i}")));
        g.strs("tokenizer.ggml.tokens", &tokens);
        g
    }

    fn open(g: &Kv, tag: &str) -> GgufFile {
        let p = std::env::temp_dir().join(format!("lapid-zipformer-gguf-{}-{tag}.gguf", std::process::id()));
        std::fs::write(&p, g.build()).expect("write");
        let f = GgufFile::open(&p).expect("open");
        std::fs::remove_file(&p).ok();
        f
    }

    #[test]
    fn config_and_tokens_from_66m_metadata() {
        let f = open(&kv_66m(|_, _| false), "ok");
        assert_eq!(f.architecture(), Some("zipformer2"));
        let cfg = config_from_gguf(&f).expect("config");
        // streaming defaults come from eval_*; the file's own -1 (full context) is ignored
        let want = ZipformerConfig { vocab_size: 1483, ..medium() };
        assert_eq!(cfg, want);
        assert_eq!((cfg.chunk_size, cfg.left_context_frames), (16, 128));
        let x = extras(&f, &cfg).expect("extras");
        assert_eq!(x.blobs.len(), 1);
        let text = String::from_utf8(x.blobs[0].2.clone()).expect("utf8");
        assert!(text.starts_with("<blk> 0\n<sos/eos> 1\n<unk> 2\nS 3\n\u{2581}THE 4\n"));
        assert_eq!(super::super::tokens::Tokens::parse(&text).expect("parse").len(), 1483);
        assert_eq!(x.metadata["license"], "cc-by-4.0");
        assert_eq!(x.metadata["name"], "Zipformer2 66M streaming Slovak");
    }

    /// Opt-in smoke test on a real file: converts it, loads it (the loader checks every
    /// tensor shape) and streams audio through it.
    ///
    /// ```text
    /// LAPID_ZIPFORMER_GGUF=model-q8_0.gguf [LAPID_ZIPFORMER_WAV=speech.wav] [LAPID_DEVICE=cpu] \
    ///   cargo test -p lapid-hub zipformer::gguf::tests::real -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "needs LAPID_ZIPFORMER_GGUF"]
    fn real_gguf_converts_loads_and_streams() {
        let Some(src) = std::env::var_os("LAPID_ZIPFORMER_GGUF") else { return };
        let out = std::env::temp_dir().join(format!("lapid-zipformer-gguf-{}.lapid", std::process::id()));
        let t = std::time::Instant::now();
        super::super::convert::CONVERT.convert(src.to_str().expect("utf8"), lapid_convert::OutDType::Int8, &out).expect("convert");
        eprintln!("converted in {:?} -> {} bytes", t.elapsed(), std::fs::metadata(&out).expect("out").len());
        let device = lapid::Device::from_name(&std::env::var("LAPID_DEVICE").unwrap_or_else(|_| "cpu".into())).expect("device");
        let model = super::super::Zipformer::load(&out, &device).expect("load");
        assert_eq!(model.config().vocab_size, model.tokens().len());
        let audio = match std::env::var_os("LAPID_ZIPFORMER_WAV") {
            Some(w) => crate::audio::load_audio_16k_mono(std::path::Path::new(&w)).expect("wav"),
            None => (0..32_000).map(|i| ((i as f32) * 0.05).sin() * 0.1).collect(),
        };
        let t = std::time::Instant::now();
        let r = model.transcribe(&audio, None).expect("stream");
        eprintln!("streamed {:.1}s of audio in {:?}: {:?}", audio.len() as f64 / 16_000.0, t.elapsed(), r.text);
        std::fs::remove_file(&out).ok();
    }

    #[test]
    fn unsupported_variants_are_rejected() {
        let code = |edit: fn(&mut Kv, &str) -> bool| config_from_gguf(&open(&kv_66m(edit), "bad")).expect_err("rejected").code;
        assert_eq!(code(|g, k| { let hit = k == "stt.zipformer.encoder.causal"; if hit { g.bool(k, false); } hit }), "unsupported_model");
        assert_eq!(code(|g, k| { let hit = k == "stt.zipformer.joiner.activation"; if hit { g.str(k, "relu"); } hit }), "unsupported_model");
        assert_eq!(code(|g, k| { let hit = k == "stt.zipformer.encoder.n_stacks"; if hit { g.u32(k, 5); } hit }), "invalid_metadata");
        // a missing key names itself
        let g = Kv::default();
        assert_eq!(config_from_gguf(&open(&g, "empty")).expect_err("missing").code, "missing_metadata");
    }
}
