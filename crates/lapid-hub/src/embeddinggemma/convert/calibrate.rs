//! EmbeddingGemma's [`Calibrator`] for calibrated int4 (`lapid convert --dtype int4
//! --calibrate`): each calibration text, prefixed with one of the model's task prompts in
//! turn (and once without), is one unpadded single-item batch through the encoder, pooling
//! and Dense head.

use std::path::Path;

use lapid::format::Reader;
use lapid::{Device, LapidFile, Result, Tensor};
use lapid_convert::calibrate::{CalibrationSource, Calibrator};
use lapid_convert::policy::Role;

use super::role;
use crate::embeddinggemma::EmbeddingGemma;
use crate::embeddinggemma::model::Batch;

/// Characters per calibration text chunk.
const CHUNK_CHARS: usize = 1500;
/// At most this many sequences.
const MAX_SAMPLES: usize = 128;

pub(crate) struct GemmaCalibrator {
    /// Same image the model was loaded from (weights re-upload to the identical leaves).
    file: LapidFile,
    device: Device,
    model: EmbeddingGemma,
    samples: Vec<Vec<u32>>,
}

impl GemmaCalibrator {
    pub(crate) fn open(path: &Path, device: &Device, source: &CalibrationSource) -> Result<GemmaCalibrator> {
        let bytes = Reader::open(path)?.data();
        let file = LapidFile::from_bytes(bytes.clone())?;
        let model = EmbeddingGemma::load_bytes(bytes, device)?;
        let mut prompts: Vec<String> = vec![String::new()];
        prompts.extend(model.prompts().values().cloned());
        let mut samples = Vec::new();
        for (i, text) in source.texts(CHUNK_CHARS, MAX_SAMPLES)?.iter().enumerate() {
            samples.push(model.tokenize(text, &prompts[i % prompts.len()])?);
        }
        Ok(GemmaCalibrator { file, device: device.clone(), model, samples })
    }
}

impl Calibrator for GemmaCalibrator {
    fn weights(&self) -> Result<Vec<(String, Tensor)>> {
        let mut out = Vec::new();
        for (name, t) in self.file.tensors.iter() {
            if role(name, t.shape()) == Role::Linear && t.rank() == 2 {
                out.push((name.clone(), self.device.upload(t)?));
            }
        }
        Ok(out)
    }

    fn len(&self) -> usize {
        self.samples.len()
    }

    fn forward(&mut self, i: usize) -> Result<Vec<Tensor>> {
        let ids = &self.samples[i];
        let batch = Batch::new(&[ids.as_slice()], ids.len(), self.model.cfg.pad_token_id)?;
        Ok(vec![self.model.model.graph(&batch, false)?.embedding])
    }

    fn eval(&self, outputs: &[&Tensor]) -> Result<Vec<Vec<f32>>> {
        let realized = self.device.eval(outputs)?;
        realized.iter().map(|t| self.device.to_vec_f32(t)).collect()
    }
}
