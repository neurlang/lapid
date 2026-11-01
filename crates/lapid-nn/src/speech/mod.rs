//! Speech building blocks shared by the audio models in `lapid-hub` (speech recognition,
//! voice activity, speech synthesis). Each piece is a general layer or a helper at least two
//! models use; weight names stay with the models.

pub mod conformer;
pub mod conv_transpose;
pub mod lstm;
pub mod rel_pos;
pub mod streaming_transformer;
pub mod taps;

pub use conformer::{ConformerBlock, ConvModule};
pub use lstm::{HostLstmCell, HostLstmState, Lstm, LstmLayer, LstmState};
pub use rel_pos::{RelPosAttention, rel_shift, sinusoidal_rel_positions};
pub use taps::{DepthwiseConv, kernel_taps, strided, taps_1d, taps_2d, weighted_sum};
