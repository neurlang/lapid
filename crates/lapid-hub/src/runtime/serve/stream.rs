//! Streaming transcription through [`AnyModel`](super::AnyModel): a recognizer that works
//! incrementally opens *streams* that live on its worker thread (never leaving it, so they
//! need no locking); the server feeds them audio chunk by chunk and turns the returned
//! [`StreamUpdate`]s into protocol events. Model-specific choices (chunk sizes, search) stay
//! with the model: it maps the generic [`Delay`] levels onto what it can do.

use lapid::Result;

use crate::runtime::endpoint::EndpointConfig;

/// How much latency a streaming recognizer may spend on accuracy (the realtime API's
/// `delay`). A model maps each level to what it supports, clamping a level it lacks to its
/// nearest one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Delay {
    /// The most latency-sensitive interactions.
    Minimal,
    /// Low-latency live captions.
    #[default]
    Low,
    /// A balanced latency / accuracy tradeoff.
    Medium,
    /// Accuracy matters more than immediate display.
    High,
    /// The most delay for the most context.
    XHigh,
}

impl Delay {
    pub const ALL: [Delay; 5] = [Delay::Minimal, Delay::Low, Delay::Medium, Delay::High, Delay::XHigh];

    pub fn name(self) -> &'static str {
        match self {
            Delay::Minimal => "minimal",
            Delay::Low => "low",
            Delay::Medium => "medium",
            Delay::High => "high",
            Delay::XHigh => "xhigh",
        }
    }

    pub fn parse(s: &str) -> Option<Delay> {
        Delay::ALL.into_iter().find(|d| d.name() == s)
    }
}

/// Options of one stream.
#[derive(Clone, Debug, Default)]
pub struct StreamParams {
    pub delay: Delay,
    /// Close segments at endpoints (server-side turn detection); `None` = one segment for the
    /// whole stream, closed by [`AsrStream::finish`].
    pub endpoint: Option<EndpointConfig>,
}

/// State of one segment after some audio; `text` is the whole segment so far, so successive
/// updates of a segment extend each other.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StreamUpdate {
    /// Segment index within the stream (increments at every closed segment).
    pub segment: usize,
    pub text: String,
    /// The segment is closed: an endpoint was reached or the input finished.
    pub is_final: bool,
    /// Segment start, in seconds of audio since the stream began.
    pub start: f32,
}

/// One live recognizer over a 16 kHz mono signal (see [`AnyModel::stream_open`](super::AnyModel::stream_open)).
pub trait AsrStream {
    /// Feed samples; updates for the segments that changed, in order.
    fn accept(&mut self, samples_16k: &[f32]) -> Result<Vec<StreamUpdate>>;
    /// End of input: flush the audio and close the open segment.
    fn finish(&mut self) -> Result<Vec<StreamUpdate>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delay_names_round_trip() {
        for d in Delay::ALL {
            assert_eq!(Delay::parse(d.name()), Some(d));
        }
        assert_eq!(Delay::parse("fast"), None);
        assert_eq!(Delay::parse("XHigh"), None);
        assert_eq!(Delay::default(), Delay::Low);
    }
}
