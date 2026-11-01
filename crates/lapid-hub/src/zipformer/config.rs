//! Zipformer2 transducer hyper-parameters, read from the `.lapid` metadata
//! (`metadata["zipformer"]`, written by [`super::convert`] from the exported `config.json`).

use lapid::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
#[cfg(test)]
use serde_json::json;

/// Blob holding sherpa-onnx style `tokens.txt` (`<piece> <id>` per line).
pub(crate) const TOKENS_BLOB: &str = "tokenizer/tokens.txt";
/// Optional blob with the SentencePiece model the tokens come from.
pub(crate) const BPE_BLOB: &str = "tokenizer/bpe.model";

/// Fbank frames dropped by `Conv2dSubsampling` (7) plus the right context of its ConvNeXt
/// after the /2 (3 frames, i.e. 6 fbank frames): a chunk of `c` encoder frames reads
/// `2 c + PAD` fbank frames.
pub const PAD_FRAMES: usize = 7 + 2 * 3;
/// Tail padding (fbank frames of `LOG_EPS`) icefall's streaming decode appends to flush.
pub const TAIL_FRAMES: usize = 30;
/// `log(1e-10)`: the fbank value of padding frames.
pub const LOG_EPS: f32 = -23.025_850_929_940_457;

/// Lists may be given as "2,2,3" strings and integers as strings (icefall's command-line
/// form in exported `config.json` files); one-element lists apply to every stack.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ZipformerConfig {
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,
    #[serde(default = "default_feature_dim")]
    pub feature_dim: usize,
    /// Per encoder stack (all of the same length).
    pub num_encoder_layers: Vec<usize>,
    pub downsampling_factor: Vec<usize>,
    pub feedforward_dim: Vec<usize>,
    pub num_heads: Vec<usize>,
    pub encoder_dim: Vec<usize>,
    pub cnn_module_kernel: Vec<usize>,
    pub query_head_dim: Vec<usize>,
    pub value_head_dim: Vec<usize>,
    pub pos_head_dim: Vec<usize>,
    pub pos_dim: usize,
    pub decoder_dim: usize,
    pub joiner_dim: usize,
    pub context_size: usize,
    pub vocab_size: usize,
    pub blank_id: u32,
    /// Default streaming chunk, encoder frames at 50 Hz (16 = 320 ms).
    pub chunk_size: usize,
    /// Default left context, encoder frames at 50 Hz.
    pub left_context_frames: usize,
}

/// Encoder output frames per second after the final /2 (fbank 100 Hz / 4).
pub const OUTPUT_FRAME_SECONDS: f64 = 0.04;
/// Channels of the `Conv2dSubsampling` ConvNeXt (fixed in icefall).
pub const EMBED_CHANNELS: usize = 128;
/// Layer 1 / 2 channels of `Conv2dSubsampling`.
pub const EMBED_LAYER1: usize = 8;
pub const EMBED_LAYER2: usize = 32;

impl ZipformerConfig {
    pub fn num_stacks(&self) -> usize {
        self.encoder_dim.len()
    }

    pub fn max_dim(&self) -> usize {
        self.encoder_dim.iter().copied().max().unwrap_or(0)
    }

    pub fn total_layers(&self) -> usize {
        self.num_encoder_layers.iter().sum()
    }

    /// Frequency width after `Conv2dSubsampling`: ((F - 1) / 2 - 1) / 2 (19 for 80 bins).
    pub fn embed_width(&self) -> usize {
        ((self.feature_dim - 1) / 2 - 1) / 2
    }

    /// Hidden channels of the nonlinear attention of stack `i`.
    pub fn nonlin_dim(&self, i: usize) -> usize {
        3 * self.encoder_dim[i] / 4
    }

    /// The metadata object (`metadata["zipformer"]`): the fields plus the tokens blob name.
    pub fn to_json(&self) -> Value {
        let mut v = serde_json::to_value(self).unwrap_or_default();
        v["tokens_blob"] = Value::from(TOKENS_BLOB);
        v
    }

    pub fn from_metadata(metadata: &Value) -> Result<ZipformerConfig> {
        let z = metadata
            .get("zipformer")
            .ok_or_else(|| meta_err("not a zipformer model: metadata has no 'zipformer' object"))?;
        ZipformerConfig::from_json(z)
    }

    /// From the metadata object or an exported `config.json`.
    pub fn from_json(z: &Value) -> Result<ZipformerConfig> {
        ZipformerConfig::deserialize(numeric(z)).map_err(|e| meta_err(format!("zipformer config: {e}")))?.normalized()
    }

    /// Broadcast one-element lists to every stack (icefall `_to_tuple`) and validate.
    fn normalized(mut self) -> Result<ZipformerConfig> {
        let n = self.downsampling_factor.len();
        for v in [
            &mut self.num_encoder_layers,
            &mut self.feedforward_dim,
            &mut self.num_heads,
            &mut self.encoder_dim,
            &mut self.cnn_module_kernel,
            &mut self.query_head_dim,
            &mut self.value_head_dim,
            &mut self.pos_head_dim,
        ] {
            if v.len() == 1 && n > 1 {
                *v = vec![v[0]; n];
            }
        }
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<()> {
        let bad = |m: String| Err(meta_err(format!("zipformer config: {m}")));
        let n = self.downsampling_factor.len();
        if n == 0 {
            return bad("no encoder stacks".into());
        }
        let lists = [
            &self.num_encoder_layers,
            &self.feedforward_dim,
            &self.num_heads,
            &self.encoder_dim,
            &self.cnn_module_kernel,
            &self.query_head_dim,
            &self.value_head_dim,
            &self.pos_head_dim,
        ];
        if lists.iter().any(|l| l.len() != n) {
            return bad(format!("per-stack lists must all have {n} entries"));
        }
        if self.cnn_module_kernel.iter().any(|k| k % 2 == 0) {
            return bad(format!("conv kernels {:?} must be odd", self.cnn_module_kernel));
        }
        if self.encoder_dim.iter().any(|d| d % 4 != 0) || self.downsampling_factor.contains(&0) {
            return bad(format!("encoder dims {:?} / downsampling {:?}", self.encoder_dim, self.downsampling_factor));
        }
        if self.feature_dim < 7 || self.pos_dim % 2 != 0 || self.context_size == 0 {
            return bad(format!("feature dim {} / pos dim {} / context {}", self.feature_dim, self.pos_dim, self.context_size));
        }
        if self.context_size > 1 && self.decoder_dim % 4 != 0 {
            return bad(format!("decoder dim {} not divisible by 4 (group size)", self.decoder_dim));
        }
        if self.blank_id as usize >= self.vocab_size {
            return bad(format!("blank id {} outside vocabulary {}", self.blank_id, self.vocab_size));
        }
        self.check_stream(self.chunk_size, self.left_context_frames)
    }

    /// A streaming configuration (encoder frames at 50 Hz) this model can run: the chunk is a
    /// multiple of every stack's downsampling (and of the output /2), each stack sees at
    /// least 2 frames per chunk, and the left context is a multiple of every downsampling.
    pub fn check_stream(&self, chunk: usize, left: usize) -> Result<()> {
        let bad = |m: String| Err(Error::new("invalid_argument", format!("zipformer stream: {m}")));
        if chunk == 0 || chunk % 2 != 0 {
            return bad(format!("chunk size {chunk} must be a positive even number of 50 Hz frames"));
        }
        for &ds in &self.downsampling_factor {
            if chunk % ds != 0 || chunk / ds < 2 {
                return bad(format!("chunk size {chunk} must be a multiple of {ds} with at least 2 frames per stack"));
            }
            if left % ds != 0 || left == 0 {
                return bad(format!("left context {left} must be a positive multiple of {ds}"));
            }
        }
        Ok(())
    }
}

fn default_sample_rate() -> u32 {
    16_000
}

fn default_feature_dim() -> usize {
    80
}

/// Per-stack list fields.
const LISTS: [&str; 9] = [
    "num_encoder_layers", "downsampling_factor", "feedforward_dim", "num_heads", "encoder_dim",
    "cnn_module_kernel", "query_head_dim", "value_head_dim", "pos_head_dim",
];

/// icefall's command-line forms as JSON: "2,2,3" strings (and single integers) as lists for
/// the per-stack fields, integer strings as integers.
fn numeric(z: &Value) -> Value {
    let mut z = z.clone();
    for (k, v) in z.as_object_mut().into_iter().flatten() {
        let list = LISTS.contains(&k.as_str());
        match v {
            Value::String(s) => {
                if let Some(n) = s.split(',').map(|x| x.trim().parse::<u64>().ok()).collect::<Option<Vec<u64>>>() {
                    *v = if list || n.len() != 1 { Value::from(n) } else { Value::from(n[0]) };
                }
            }
            Value::Number(_) if list => *v = Value::Array(vec![v.clone()]),
            _ => {}
        }
    }
    z
}

fn meta_err(msg: impl Into<String>) -> Error {
    Error::new("metadata", msg)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// icefall's 66M streaming model (train.py defaults, --causal 1).
    pub(crate) fn medium() -> ZipformerConfig {
        let v = json!({
            "num_encoder_layers": "2,2,3,4,3,2", "downsampling_factor": "1,2,4,8,4,2",
            "feedforward_dim": "512,768,1024,1536,1024,768", "num_heads": "4,4,4,8,4,4",
            "encoder_dim": "192,256,384,512,384,256", "cnn_module_kernel": "31,31,15,15,15,31",
            "query_head_dim": "32", "value_head_dim": "12", "pos_head_dim": "4", "pos_dim": 48,
            "decoder_dim": 512, "joiner_dim": 512, "context_size": 2, "vocab_size": 500,
            "blank_id": 0, "chunk_size": 16, "left_context_frames": 128
        });
        ZipformerConfig::from_json(&v).expect("config")
    }

    #[test]
    fn parse_roundtrip_and_checks() {
        let c = medium();
        assert_eq!(c.query_head_dim, vec![32; 6]);
        assert_eq!(c.total_layers(), 16);
        assert_eq!(c.max_dim(), 512);
        assert_eq!(c.embed_width(), 19);
        let back = ZipformerConfig::from_metadata(&json!({ "zipformer": c.to_json() })).expect("roundtrip");
        assert_eq!(back, c);
        assert!(c.check_stream(16, 128).is_ok());
        assert!(c.check_stream(32, 256).is_ok());
        assert!(c.check_stream(8, 128).is_err());
        assert!(c.check_stream(16, 100).is_err());
        assert_eq!(ZipformerConfig::from_metadata(&json!({})).unwrap_err().code, "metadata");
    }
}
