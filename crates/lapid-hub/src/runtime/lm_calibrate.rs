//! Calibrated int4 for any [`CausalLm`] (`lapid convert --dtype int4 --calibrate`):
//! calibration texts become token sequences, alternately raw text (`<bos>` + text) and a
//! chat exchange rendered with the model's template (first third as the user turn, the rest
//! as the assistant reply); each sequence is one uncached forward.

use std::path::Path;

use lapid::format::Reader;
use lapid::{Device, LapidFile, Result, Tensor};
use lapid_convert::calibrate::{CalibrationSource, Calibrator};
use lapid_convert::engine::ConvertSpec;
use lapid_convert::policy::Role;

use super::causal_lm::{CausalLm, Llm, forward_graph};
use super::chat::Message;

/// Tokens per calibration sequence (longer ones are truncated).
const MAX_TOKENS: usize = 1024;
/// Characters per calibration text chunk.
const CHUNK_CHARS: usize = 3000;
/// At most this many sequences.
const MAX_SAMPLES: usize = 64;

pub struct LmCalibrator<M: CausalLm> {
    /// Same image the model was loaded from: re-uploading a weight hits the device cache
    /// and returns the leaf the graphs use.
    file: LapidFile,
    device: Device,
    llm: Llm<M>,
    spec: &'static ConvertSpec,
    samples: Vec<Vec<u32>>,
}

impl<M: CausalLm> LmCalibrator<M> {
    /// Load the f32 `.lapid` at `path` on `device` and tokenize the calibration texts.
    pub fn open(path: &Path, device: &Device, source: &CalibrationSource, spec: &'static ConvertSpec) -> Result<LmCalibrator<M>> {
        let bytes = Reader::open(path)?.data();
        let file = LapidFile::from_bytes(bytes.clone())?;
        let llm = Llm::<M>::load_bytes(bytes, device)?;
        let mut samples = Vec::new();
        for (i, text) in source.texts(CHUNK_CHARS, MAX_SAMPLES)?.iter().enumerate() {
            let mut ids = if i % 2 == 0 {
                llm.tokenizer().encode(text, true)?
            } else {
                let n = text.chars().count();
                let cut = text.char_indices().nth(n / 3).map_or(text.len(), |(b, _)| b);
                let (q, a) = text.split_at(cut);
                llm.encode(&llm.render_chat(&[Message::user(q.trim()), Message::assistant(a.trim())], false, None)?)?
            };
            ids.truncate(MAX_TOKENS);
            if ids.len() >= 8 {
                samples.push(ids);
            }
        }
        Ok(LmCalibrator { file, device: device.clone(), llm, spec, samples })
    }
}

impl<M: CausalLm> Calibrator for LmCalibrator<M> {
    fn weights(&self) -> Result<Vec<(String, Tensor)>> {
        let mut out = Vec::new();
        for (name, t) in self.file.tensors.iter() {
            if self.spec.role(name, t.shape()) == Role::Linear && t.rank() == 2 {
                out.push((name.clone(), self.device.upload(t)?));
            }
        }
        Ok(out)
    }

    fn len(&self) -> usize {
        self.samples.len()
    }

    fn forward(&mut self, i: usize) -> Result<Vec<Tensor>> {
        Ok(vec![forward_graph(self.llm.model(), &self.samples[i])?])
    }

    fn eval(&self, outputs: &[&Tensor]) -> Result<Vec<Vec<f32>>> {
        let realized = self.device.eval(outputs)?;
        realized.iter().map(|t| self.device.to_vec_f32(t)).collect()
    }
}
