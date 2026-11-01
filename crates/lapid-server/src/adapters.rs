//! Adapters for the non-LLM hub models: speech recognition (Whisper, Parakeet), speech
//! synthesis (Phonon) and text embeddings (EmbeddingGemma).

use lapid::Result;
use lapid_hub::embeddinggemma::EmbedOptions;
use lapid_hub::whisper::{Task, TranscribeOptions};
use lapid_hub::{EmbeddingGemma, Parakeet, Phonon, SpeakOptions, Whisper};
use serde_json::{Value, json};

use crate::served::{
    AsrParams, AsrSegment, AsrWord, Capability, EmbedParams, Embeddings, ModelMeta, Served, TranscriptOut, TtsParams,
};

pub struct WhisperServed {
    pub model: Whisper,
    pub meta: ModelMeta,
}

impl WhisperServed {
    pub fn new(model: Whisper, mut meta: ModelMeta) -> WhisperServed {
        meta.capabilities = vec![Capability::Transcription];
        meta.extra.insert("sample_rate".into(), json!(16000));
        WhisperServed { model, meta }
    }
}

impl Served for WhisperServed {
    fn meta(&self) -> &ModelMeta {
        &self.meta
    }

    fn meta_mut(&mut self) -> &mut ModelMeta {
        &mut self.meta
    }

    fn transcribe(&mut self, samples: &[f32], opts: &AsrParams) -> Result<TranscriptOut> {
        let o = TranscribeOptions {
            language: opts.language.clone().filter(|l| !l.is_empty() && l != "auto"),
            task: if opts.translate { Task::Translate } else { Task::Transcribe },
            timestamps: true,
        };
        let t = self.model.transcribe(samples, &o)?;
        Ok(TranscriptOut {
            text: t.text.trim().to_string(),
            language: Some(t.language),
            segments: t
                .segments
                .into_iter()
                .map(|s| AsrSegment { start: s.start, end: s.end, text: s.text, tokens: s.tokens })
                .collect(),
            words: Vec::new(),
        })
    }
}

pub struct ParakeetServed {
    pub model: Parakeet,
    pub meta: ModelMeta,
}

impl ParakeetServed {
    pub fn new(model: Parakeet, mut meta: ModelMeta) -> ParakeetServed {
        meta.capabilities = vec![Capability::Transcription];
        meta.extra.insert("sample_rate".into(), json!(model.config().sample_rate));
        ParakeetServed { model, meta }
    }
}

impl Served for ParakeetServed {
    fn meta(&self) -> &ModelMeta {
        &self.meta
    }

    fn meta_mut(&mut self) -> &mut ModelMeta {
        &mut self.meta
    }

    fn transcribe(&mut self, samples: &[f32], _opts: &AsrParams) -> Result<TranscriptOut> {
        let sr = self.model.config().sample_rate;
        let resampled;
        let samples = if sr != 16_000 {
            resampled = lapid_hub::audio::resample(samples, 16_000, sr);
            &resampled[..]
        } else {
            samples
        };
        let t = self.model.transcribe(samples, &Default::default())?;
        Ok(TranscriptOut {
            text: t.text.trim().to_string(),
            language: None,
            segments: t
                .segments
                .into_iter()
                .map(|s| AsrSegment { start: s.start, end: s.end, text: s.text, tokens: s.tokens })
                .collect(),
            words: t.words.into_iter().map(|w| AsrWord { word: w.word, start: w.start, end: w.end }).collect(),
        })
    }
}

pub struct PhononServed {
    pub model: Phonon,
    pub meta: ModelMeta,
}

impl PhononServed {
    pub fn new(model: Phonon, mut meta: ModelMeta) -> PhononServed {
        meta.capabilities = vec![Capability::Speech];
        meta.extra.insert("sample_rate".into(), json!(model.sample_rate()));
        meta.extra.insert("voices".into(), json!(model.voices()));
        meta.extra.insert("default_voice".into(), json!(model.config().assets.default_voice));
        PhononServed { model, meta }
    }
}

impl Served for PhononServed {
    fn meta(&self) -> &ModelMeta {
        &self.meta
    }

    fn meta_mut(&mut self) -> &mut ModelMeta {
        &mut self.meta
    }

    fn speak(&mut self, text: &str, opts: &TtsParams, out: &mut dyn FnMut(&[f32]) -> bool) -> Result<()> {
        // OpenAI voice names ("alloy", ...) fall back to the model's default voice.
        let voice = opts.voice.clone().filter(|v| self.model.voices().contains(v));
        let o = SpeakOptions {
            voice,
            seed: opts.seed,
            temperature: opts.temperature,
            frames_per_chunk: Some(3),
            ..Default::default()
        };
        for pcm in self.model.speak_stream(text, &o)? {
            if !out(&pcm?) {
                break;
            }
        }
        Ok(())
    }

    fn sample_rate(&self) -> u32 {
        self.model.sample_rate()
    }
}

pub struct EmbedServed {
    pub model: EmbeddingGemma,
    pub meta: ModelMeta,
}

impl EmbedServed {
    pub fn new(model: EmbeddingGemma, mut meta: ModelMeta) -> EmbedServed {
        meta.capabilities = vec![Capability::Embedding];
        meta.extra.insert("embedding_dim".into(), json!(model.dim()));
        let prompts: Vec<Value> = model.prompts().keys().map(|k| json!(k)).collect();
        meta.extra.insert("prompts".into(), Value::Array(prompts));
        EmbedServed { model, meta }
    }
}

impl Served for EmbedServed {
    fn meta(&self) -> &ModelMeta {
        &self.meta
    }

    fn meta_mut(&mut self) -> &mut ModelMeta {
        &mut self.meta
    }

    fn embed(&mut self, texts: &[String], opts: &EmbedParams) -> Result<Embeddings> {
        let o = EmbedOptions { prompt_name: opts.prompt_name.clone(), dim: opts.dimensions, ..Default::default() };
        let prompt = self.model.prompt_for(&o)?;
        let mut prompt_tokens = 0;
        for t in texts {
            prompt_tokens += self.model.tokenize(t, &prompt)?.len();
        }
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let vectors = self.model.embed(&refs, &o)?;
        Ok(Embeddings { vectors, prompt_tokens })
    }
}
