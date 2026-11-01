//! Gemma 4 vision: image + text → text (`google/gemma-4-E2B-it`, the 26B-A4B MoE, ...).
//!
//! The Gemma 4 vision tower ([`vision`]) turns each image into soft tokens that replace the
//! `<|image|>` placeholders of the prompt; the text decoder is the gemma4moe one (dense or
//! MoE, per-layer embeddings use the pad id at image positions, and models with
//! `use_bidirectional_attention = "vision"` let image tokens see each other on sliding
//! layers). Preprocessing follows `Gemma4ImageProcessor` ([`preprocess`]); the chat template
//! renders `<|image|>` per image part and the processor expansion
//! `<|image>` + `<|image|>` * n + `<image|>` is done on the token ids. Resizing is
//! torchvision-exact (`vision::Image::resize`).
//!
//! Files: a `.lapid` written by the gemma4moe converter from a Hugging Face checkpoint (the
//! `model.vision_tower.*` / `model.embed_vision.*` tensors and the `vision_config` /
//! `image_processor` metadata are kept) or from a GGUF + mmproj pair ([`convert::add_mmproj`]).
//!
//! ```no_run
//! use lapid_hub::gemma4vl::{ChatOptions, Gemma4Vl};
//! use lapid_hub::vision::{Image, Message};
//! let device = lapid::Device::best();
//! let vlm = Gemma4Vl::from_hub("google/gemma-4-E2B-it", "int8", &device)?;
//! let image = Image::open("photo.jpg".as_ref())?;
//! let reply = vlm.chat(&[Message::user_with_images(vec![image], "Describe this image.")], &ChatOptions::default())?;
//! println!("{reply}");
//! # Ok::<(), lapid::Error>(())
//! ```

pub mod bench;
pub mod cmd;
pub mod config;
pub mod convert;
pub mod preprocess;
pub mod verify;
mod vision;

use std::path::Path;
use std::time::Instant;

use lapid::{Device, Error, LapidFile, Result, Tensor};
use lapid_nn::{Loader, SamplingParams, Tokenizer};
use serde_json::{Value, json};

use crate::gemma4moe::{Gemma4Moe, Gemma4MoeConfig, Injection};
use crate::model::{Bytes, Model, ModelInfo};
use crate::vision::{Image, Message, Part};
pub use crate::gemma4moe::{ChatOptions, GenerateOptions, Session, TokenStream};
pub use config::{ImageProcessorConfig, VisionConfig};
pub use lapid_nn::Precision;
pub use preprocess::PreparedImage;
use vision::VisionTower;

fn meta_err(m: impl Into<String>) -> Error {
    Error::new("metadata", format!("gemma4vl: {}", m.into()))
}

/// A multimodal prompt: token ids with the image soft tokens expanded, and the images (in
/// order) whose embeddings fill each `<|image|>` run.
pub struct Prompt {
    pub ids: Vec<u32>,
    pub images: Vec<PreparedImage>,
    /// Start of each image's soft-token run in `ids`.
    pub starts: Vec<usize>,
}

/// A loaded Gemma 4 vision-language model bound to a device.
pub struct Gemma4Vl {
    text: Gemma4Moe,
    tower: VisionTower,
    processor: ImageProcessorConfig,
    image_token: u32,
    boi: u32,
    eoi: u32,
}

impl Gemma4Vl {
    /// Load `repo` (e.g. "google/gemma-4-E2B-it") from the lapid cache, converting it (text
    /// decoder + vision tower) from the Hugging Face hub first if missing. The cache entry is
    /// `<repo>-vision` so it never collides with the text-only `Gemma4::from_hub` file.
    pub fn from_hub(repo: &str, dtype: &str, device: &Device) -> Result<Gemma4Vl> {
        let key = format!("{}-vision", repo.trim_end_matches('/'));
        let path = crate::model::hub_model_path(&key, dtype, |_, d, out| crate::gemma4moe::convert::convert_hf(repo, d, out))?;
        Gemma4Vl::load(&path, device)
    }

    pub fn load(path: &Path, device: &Device) -> Result<Gemma4Vl> {
        Gemma4Vl::load_with(path, device, Precision::Stored)
    }

    pub fn load_bytes(bytes: Bytes, device: &Device) -> Result<Gemma4Vl> {
        let text = Gemma4Moe::load_bytes(bytes.clone(), device)?;
        Gemma4Vl::with_text(text, &LapidFile::from_bytes(bytes)?, device, Precision::Stored)
    }

    /// Load with a weight precision (the fake-quant modes build verification references).
    pub fn load_with(path: &Path, device: &Device, precision: Precision) -> Result<Gemma4Vl> {
        let text = Gemma4Moe::load_with(path, device, precision)?;
        Gemma4Vl::with_text(text, &LapidFile::open(path)?, device, precision)
    }

    fn with_text(text: Gemma4Moe, file: &LapidFile, device: &Device, precision: Precision) -> Result<Gemma4Vl> {
        let cfg: &Gemma4MoeConfig = text.config();
        let vc = cfg.vision.as_ref().ok_or_else(|| meta_err("the file has no vision tower (convert with gemma4moe / a GGUF + mmproj pair)"))?;
        let vc = VisionConfig::from_json(vc)?;
        let processor = ImageProcessorConfig::from_json(cfg.image_processor.as_ref());
        if processor.patch_size != vc.patch_size || processor.pooling_kernel_size != vc.pooling_kernel_size {
            return Err(meta_err("image_processor patch / pooling sizes differ from vision_config"));
        }
        let tok = text.tokenizer();
        let id = |v: Option<u32>, s: &str| v.or_else(|| tok.token_to_id(s)).ok_or_else(|| meta_err(format!("no {s} token")));
        let it = cfg.image_tokens;
        let (image_token, boi, eoi) = (id(it.image, "<|image|>")?, id(it.boi, "<|image>")?, id(it.eoi, "<image|>")?);
        let tower = VisionTower::load(&Loader::new(file, device, precision), &vc, cfg.text.hidden_size)?;
        Ok(Gemma4Vl { text, tower, processor, image_token, boi, eoi })
    }

    /// The text model (tokenizer, template, text-only chat).
    pub fn text(&self) -> &Gemma4Moe {
        &self.text
    }

    pub fn vision_config(&self) -> &VisionConfig {
        &self.tower.cfg
    }

    pub fn processor(&self) -> &ImageProcessorConfig {
        &self.processor
    }

    /// Soft-token budget per image (HF `max_soft_tokens`: 70, 140, 280, 560 or 1120; more
    /// tokens = higher resolution). GGUF mmproj files do not record it (default 280).
    pub fn set_max_soft_tokens(&mut self, n: usize) -> Result<()> {
        if ![70, 140, 280, 560, 1120].contains(&n) {
            return Err(Error::new("invalid_argument", format!("gemma4vl: max_soft_tokens {n} (expected 70, 140, 280, 560 or 1120)")));
        }
        self.processor.max_soft_tokens = n;
        Ok(())
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        self.text.tokenizer()
    }

    pub fn recommended_sampling(&self) -> SamplingParams {
        self.text.recommended_sampling()
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        self.text.decode(ids, skip_special)
    }

    pub fn new_session(&self) -> Session {
        self.text.new_session()
    }

    /// Resize / rescale / patchify one image (HF `Gemma4ImageProcessor`).
    pub fn prepare_image(&self, image: &Image) -> Result<PreparedImage> {
        preprocess::prepare(image, &self.processor)
    }

    /// HF-style messages JSON for the chat template (`{"type": "image"}` parts).
    pub fn messages_json(messages: &[Message]) -> String {
        let msgs: Vec<Value> = messages
            .iter()
            .map(|m| {
                let text_only = m.content.iter().all(|p| matches!(p, Part::Text(_)));
                let content = if text_only {
                    json!(m.content.iter().filter_map(|p| if let Part::Text(t) = p { Some(t.as_str()) } else { None }).collect::<String>())
                } else {
                    Value::Array(
                        m.content
                            .iter()
                            .map(|p| match p {
                                Part::Text(t) => json!({"type": "text", "text": t}),
                                Part::Image(_) => json!({"type": "image"}),
                            })
                            .collect(),
                    )
                };
                json!({"role": m.role, "content": content})
            })
            .collect();
        Value::Array(msgs).to_string()
    }

    /// Replace each `<|image|>` of `ids` (one per image, in order) by
    /// `<|image>` + `<|image|>` * n_soft + `<image|>`.
    pub fn expand_image_tokens(&self, ids: &[u32], images: Vec<PreparedImage>) -> Result<Prompt> {
        let found = ids.iter().filter(|&&t| t == self.image_token).count();
        if found != images.len() {
            return Err(Error::new("invalid_argument", format!("gemma4vl: {found} image placeholders for {} images", images.len())));
        }
        let mut out = Vec::with_capacity(ids.len() + images.iter().map(|i| i.n_soft + 2).sum::<usize>());
        let mut starts = Vec::with_capacity(images.len());
        let mut it = images.iter();
        for &t in ids {
            if t != self.image_token {
                out.push(t);
                continue;
            }
            let img = it.next().ok_or_else(|| Error::new("internal", "gemma4vl: image count"))?;
            out.push(self.boi);
            starts.push(out.len());
            out.extend(std::iter::repeat_n(self.image_token, img.n_soft));
            out.push(self.eoi);
        }
        Ok(Prompt { ids: out, images, starts })
    }

    /// Render the template for `messages`, preprocess their images and expand the image tokens.
    pub fn prompt(&self, messages: &[Message], opts: &ChatOptions) -> Result<Prompt> {
        let text = self.text.render_chat_json(&Gemma4Vl::messages_json(messages), None, true, opts.enable_thinking)?;
        let ids = self.text.encode(&text)?;
        let images = crate::vision::collect_images(messages).into_iter().map(|i| self.prepare_image(i)).collect::<Result<Vec<_>>>()?;
        self.expand_image_tokens(&ids, images)
    }

    /// Soft-token embeddings [n_soft, D_text] of one prepared image (lazy graph).
    pub(crate) fn image_graph(&self, image: &PreparedImage, trace: Option<&mut Vec<(String, Tensor)>>) -> Result<Tensor> {
        self.tower.build(image, trace)
    }

    /// Realized soft-token embeddings of every image of `prompt`, as decoder injections.
    pub(crate) fn encode_images(&self, prompt: &Prompt) -> Result<Vec<Injection>> {
        let dev = Model::device(&self.text);
        let mut out = Vec::with_capacity(prompt.images.len());
        for (img, &start) in prompt.images.iter().zip(&prompt.starts) {
            let g = self.image_graph(img, None)?;
            let r = dev.eval(&[&g])?.into_iter().next().ok_or_else(|| Error::new("backend", "gemma4vl: no image embedding"))?;
            out.push(Injection { start, embeds: r });
        }
        Ok(out)
    }

    /// Stream the reply to an already built prompt.
    pub fn generate<'a>(&'a self, session: &'a mut Session, prompt: &Prompt, opts: GenerateOptions) -> Result<TokenStream<'a>> {
        let inject = self.encode_images(prompt)?;
        self.text.generate_with(session, &prompt.ids, inject, opts)
    }

    /// Stream the assistant reply to `messages` (text and image parts).
    pub fn chat_stream<'a>(&'a self, session: &'a mut Session, messages: &[Message], opts: &ChatOptions) -> Result<TokenStream<'a>> {
        let prompt = self.prompt(messages, opts)?;
        self.generate(session, &prompt, opts.generate.clone())
    }

    /// The full assistant reply (non-streaming convenience).
    pub fn chat(&self, messages: &[Message], opts: &ChatOptions) -> Result<String> {
        let mut session = self.new_session();
        let (text, _, _) = self.chat_stream(&mut session, messages, opts)?.collect_text()?;
        Ok(text)
    }

    /// Milliseconds to preprocess and encode `image` (benchmarks).
    pub fn time_image(&self, image: &Image) -> Result<(f64, f64, usize)> {
        let t = Instant::now();
        let p = self.prepare_image(image)?;
        let prep = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let g = self.image_graph(&p, None)?;
        let dev = Model::device(&self.text);
        let r = dev.eval(&[&g])?;
        dev.to_vec_f32(&r[0])?;
        Ok((prep, t.elapsed().as_secs_f64() * 1e3, p.n_soft))
    }
}

impl Model for Gemma4Vl {
    fn info(&self) -> ModelInfo {
        let mut i = self.text.info();
        i.task = "image-text-to-text".into();
        i
    }

    fn device(&self) -> &Device {
        Model::device(&self.text)
    }
}

#[cfg(test)]
mod tests;
