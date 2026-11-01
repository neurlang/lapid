//! Endpoint detection of streaming recognizers (sherpa-onnx / Kaldi rules): when a
//! segment closes, from the decoded time of the segment and the trailing silence (time
//! since the last emitted token, e.g. `transducer::ChunkOutput::trailing`).

/// One endpoint rule (seconds), sherpa-onnx / Kaldi: fires when
/// `(contains_nonsilence || !must_contain_nonsilence) && trailing_silence >= min_trailing_silence
/// && utterance_length >= min_utterance_length`; trailing silence is the time since the last
/// emitted token, `contains_nonsilence = utterance_length > trailing_silence`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EndpointRule {
    pub must_contain_nonsilence: bool,
    pub min_trailing_silence: f32,
    pub min_utterance_length: f32,
}

impl EndpointRule {
    fn fires(&self, utterance: f32, trailing: f32) -> bool {
        (utterance > trailing || !self.must_contain_nonsilence) && trailing >= self.min_trailing_silence && utterance >= self.min_utterance_length
    }
}

/// Endpoint rules of a streaming recognizer (sherpa-onnx `OnlineRecognizerConfig` defaults).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EndpointConfig {
    /// 2.4 s of silence, even with nothing decoded.
    pub rule1: EndpointRule,
    /// 1.2 s of silence after some speech.
    pub rule2: EndpointRule,
    /// Utterances longer than 20 s.
    pub rule3: EndpointRule,
}

impl Default for EndpointConfig {
    fn default() -> EndpointConfig {
        let rule = |must_contain_nonsilence, min_trailing_silence, min_utterance_length| EndpointRule { must_contain_nonsilence, min_trailing_silence, min_utterance_length };
        EndpointConfig { rule1: rule(false, 2.4, 0.0), rule2: rule(true, 1.2, 0.0), rule3: rule(false, 0.0, 20.0) }
    }
}

impl EndpointConfig {
    /// `utterance` (decoded time of the segment) and `trailing` silence in seconds.
    pub fn is_endpoint(&self, utterance: f32, trailing: f32) -> bool {
        [self.rule1, self.rule2, self.rule3].iter().any(|r| r.fires(utterance, trailing))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_rules() {
        let e = EndpointConfig::default();
        assert!(!e.is_endpoint(1.0, 0.5));
        assert!(e.is_endpoint(3.0, 1.2)); // rule 2
        assert!(!e.is_endpoint(1.2, 1.2)); // silence only, < 2.4 s
        assert!(e.is_endpoint(2.4, 2.4)); // rule 1
        assert!(e.is_endpoint(20.0, 0.0)); // rule 3
    }
}
