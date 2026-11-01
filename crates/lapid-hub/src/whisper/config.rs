//! Whisper hyper-parameters and special tokens, read from the `.lapid` metadata
//! (`metadata["whisper"]`, written by lapid-convert).

use std::collections::BTreeMap;

use lapid::{Error, Result};
use serde::{Deserialize, Deserializer};
use serde_json::Value;

use super::mel::{HOP_LENGTH, N_FRAMES, SAMPLE_RATE};

/// Ids of Whisper's special tokens (the aliases are the token names without `<|` `|>`).
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct Special {
    #[serde(alias = "startoftranscript")]
    pub sot: u32,
    #[serde(alias = "endoftext")]
    pub eot: u32,
    pub transcribe: u32,
    pub translate: u32,
    #[serde(alias = "notimestamps")]
    pub no_timestamps: u32,
    pub timestamp_begin: u32,
    #[serde(default, alias = "nospeech", alias = "nocaptions")]
    pub no_speech: Option<u32>,
    #[serde(alias = "sot_prev", alias = "startofprev")]
    pub prev: u32,
    #[serde(default, alias = "startoflm")]
    pub sot_lm: Option<u32>,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct WhisperConfig {
    pub n_mels: usize,
    pub n_audio_ctx: usize,
    pub n_audio_state: usize,
    pub n_audio_head: usize,
    pub n_audio_layer: usize,
    pub n_text_ctx: usize,
    pub n_text_state: usize,
    pub n_text_head: usize,
    pub n_text_layer: usize,
    pub n_vocab: usize,
    pub special: Special,
    /// (language code, token id), sorted by token id. Empty for English-only models.
    #[serde(default, deserialize_with = "languages")]
    pub languages: Vec<(String, u32)>,
    /// generation_config `suppress_tokens` (may contain -1 = "the standard non-speech set").
    #[serde(default)]
    pub suppress_tokens: Vec<i64>,
    /// generation_config `begin_suppress_tokens` (OpenAI: `encode(" ") + [eot]`).
    #[serde(default)]
    pub begin_suppress_tokens: Vec<u32>,
}

fn meta_err(msg: impl Into<String>) -> Error {
    Error::new("metadata", msg)
}

/// `{"en": 50259, "<|de|>": 50261, ...}` -> [("en", 50259), ("de", 50261), ...] by id.
fn languages<'de, D: Deserializer<'de>>(d: D) -> std::result::Result<Vec<(String, u32)>, D::Error> {
    let map = BTreeMap::<String, u32>::deserialize(d)?;
    let mut v: Vec<(String, u32)> = map.into_iter().map(|(k, id)| (strip_lang(&k).to_string(), id)).collect();
    v.sort_by_key(|(_, id)| *id);
    Ok(v)
}

/// "<|en|>" -> "en"; "en" -> "en".
pub(crate) fn strip_lang(code: &str) -> &str {
    code.strip_prefix("<|").and_then(|s| s.strip_suffix("|>")).unwrap_or(code)
}

impl WhisperConfig {
    pub fn from_metadata(metadata: &Value) -> Result<WhisperConfig> {
        let w = metadata
            .get("whisper")
            .ok_or_else(|| meta_err("not a whisper model: metadata has no 'whisper' object"))?;
        let cfg = WhisperConfig::deserialize(w).map_err(|e| meta_err(format!("whisper metadata: {e}")))?;
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        if self.n_audio_head == 0 || !self.n_audio_state.is_multiple_of(self.n_audio_head) {
            return Err(meta_err("n_audio_state must be divisible by n_audio_head"));
        }
        if self.n_text_head == 0 || !self.n_text_state.is_multiple_of(self.n_text_head) {
            return Err(meta_err("n_text_state must be divisible by n_text_head"));
        }
        if self.n_audio_ctx == 0 || !N_FRAMES.is_multiple_of(self.n_audio_ctx) {
            return Err(meta_err(format!("n_audio_ctx {} must divide {N_FRAMES} mel frames", self.n_audio_ctx)));
        }
        if self.n_mels == 0 || self.n_text_ctx < 4 {
            return Err(meta_err("invalid n_mels / n_text_ctx"));
        }
        if self.special.timestamp_begin as usize > self.n_vocab {
            return Err(meta_err("timestamp_begin is outside the vocabulary"));
        }
        Ok(())
    }

    pub fn is_multilingual(&self) -> bool {
        !self.languages.is_empty()
    }

    /// Token id of a language given as "en", "<|en|>" or an English name ("english").
    pub fn language_token(&self, lang: &str) -> Option<u32> {
        let code = strip_lang(lang.trim());
        let lower = code.to_ascii_lowercase();
        let code = language_code_from_name(&lower).unwrap_or(lower.as_str());
        self.languages.iter().find(|(c, _)| c == code).map(|(_, id)| *id)
    }

    pub fn language_code(&self, token: u32) -> Option<&str> {
        self.languages.iter().find(|(_, id)| *id == token).map(|(c, _)| c.as_str())
    }

    /// Seconds per timestamp step (0.02 for every released Whisper).
    pub fn time_precision(&self) -> f32 {
        self.input_stride() as f32 * HOP_LENGTH as f32 / SAMPLE_RATE as f32
    }

    /// Mel frames per encoder output position (2).
    pub fn input_stride(&self) -> usize {
        N_FRAMES / self.n_audio_ctx
    }
}

/// A few common English language names -> codes (OpenAI's TO_LANGUAGE_CODE subset).
fn language_code_from_name(name: &str) -> Option<&'static str> {
    const NAMES: &[(&str, &str)] = &[
        ("english", "en"),
        ("chinese", "zh"),
        ("german", "de"),
        ("spanish", "es"),
        ("russian", "ru"),
        ("korean", "ko"),
        ("french", "fr"),
        ("japanese", "ja"),
        ("portuguese", "pt"),
        ("turkish", "tr"),
        ("polish", "pl"),
        ("catalan", "ca"),
        ("dutch", "nl"),
        ("arabic", "ar"),
        ("swedish", "sv"),
        ("italian", "it"),
        ("indonesian", "id"),
        ("hindi", "hi"),
        ("finnish", "fi"),
        ("vietnamese", "vi"),
        ("hebrew", "he"),
        ("ukrainian", "uk"),
        ("greek", "el"),
        ("czech", "cs"),
        ("romanian", "ro"),
        ("danish", "da"),
        ("hungarian", "hu"),
        ("norwegian", "no"),
        ("thai", "th"),
        ("persian", "fa"),
    ];
    NAMES.iter().find(|(n, _)| *n == name).map(|(_, c)| *c)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Value {
        serde_json::json!({ "whisper": {
            "n_mels": 80, "n_audio_ctx": 1500, "n_audio_state": 768, "n_audio_head": 12,
            "n_audio_layer": 12, "n_text_ctx": 448, "n_text_state": 768, "n_text_head": 12,
            "n_text_layer": 12, "n_vocab": 51865,
            "special": { "sot": 50258, "eot": 50257, "transcribe": 50359, "translate": 50358,
                "no_timestamps": 50363, "timestamp_begin": 50364, "no_speech": 50362,
                "prev": 50361, "sot_lm": 50360 },
            "languages": { "en": 50259, "<|de|>": 50261 },
            "suppress_tokens": [1, 2, 7],
            "begin_suppress_tokens": [220, 50257]
        }})
    }

    #[test]
    fn parses_metadata() {
        let cfg = WhisperConfig::from_metadata(&sample()).unwrap();
        assert_eq!(cfg.n_mels, 80);
        assert_eq!(cfg.special.timestamp_begin, 50364);
        assert_eq!(cfg.language_token("en"), Some(50259));
        assert_eq!(cfg.language_token("German"), Some(50261));
        assert_eq!(cfg.language_code(50261), Some("de"));
        assert!((cfg.time_precision() - 0.02).abs() < 1e-7);
        assert_eq!(cfg.input_stride(), 2);
    }

    #[test]
    fn rejects_non_whisper() {
        let err = WhisperConfig::from_metadata(&serde_json::json!({})).unwrap_err();
        assert_eq!(err.code, "metadata");
        let mut m = sample();
        m["whisper"]["special"].as_object_mut().unwrap().remove("eot");
        assert_eq!(WhisperConfig::from_metadata(&m).unwrap_err().code, "metadata");
    }
}
