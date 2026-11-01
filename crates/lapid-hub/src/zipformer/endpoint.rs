//! Endpoint detection with sherpa-onnx's (Kaldi's) rules. A rule fires when
//! `(contains_nonsilence || !must_contain_nonsilence) && trailing_silence >= min_trailing_silence
//! && utterance_length >= min_utterance_length`, where trailing silence is the time since the
//! last emitted token (blank frames) and the utterance length is the time decoded in the
//! current segment; `contains_nonsilence = utterance_length > trailing_silence`.

/// One endpoint rule (seconds).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EndpointRule {
    pub must_contain_nonsilence: bool,
    pub min_trailing_silence: f32,
    pub min_utterance_length: f32,
}

/// sherpa-onnx `OnlineRecognizerConfig` endpoint defaults.
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
        EndpointConfig {
            rule1: EndpointRule { must_contain_nonsilence: false, min_trailing_silence: 2.4, min_utterance_length: 0.0 },
            rule2: EndpointRule { must_contain_nonsilence: true, min_trailing_silence: 1.2, min_utterance_length: 0.0 },
            rule3: EndpointRule { must_contain_nonsilence: false, min_trailing_silence: 0.0, min_utterance_length: 20.0 },
        }
    }
}

impl EndpointRule {
    fn fires(&self, utterance: f32, trailing: f32) -> bool {
        let contains_nonsilence = utterance > trailing;
        (contains_nonsilence || !self.must_contain_nonsilence)
            && trailing >= self.min_trailing_silence
            && utterance >= self.min_utterance_length
    }
}

impl EndpointConfig {
    /// `utterance` / `trailing` in seconds.
    pub fn is_endpoint(&self, utterance: f32, trailing: f32) -> bool {
        [self.rule1, self.rule2, self.rule3].iter().any(|r| r.fires(utterance, trailing))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_rules() {
        let e = EndpointConfig::default();
        assert!(!e.is_endpoint(1.0, 0.5));
        assert!(e.is_endpoint(3.0, 1.2)); // rule 2
        assert!(!e.is_endpoint(1.2, 1.2)); // silence only, < 2.4 s
        assert!(e.is_endpoint(2.4, 2.4)); // rule 1
        assert!(e.is_endpoint(20.0, 0.0)); // rule 3
    }
}
