//! TDT greedy decoding (NeMo `GreedyTDTInfer` semantics) as a host loop over device graphs,
//! plus token / word / segment assembly.
//!
//! Each step evaluates one small graph: the joint on encoder frame `t` with the current
//! prediction output, preceded by an LSTM step when the previous step emitted a token. The
//! token is the argmax over vocabulary + blank, the duration the argmax over the duration
//! logits. A blank with duration 0 advances one frame; after `max_symbols_per_step`
//! symbols on one frame the decoder advances one frame.

use lapid::{Error, Result, Tensor};
use lapid_nn::Tokenizer;
use serde::Serialize;

use super::Parakeet;
use super::model::decoder::LstmState;

/// One decoder step (blank steps included).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Step {
    pub frame: usize,
    pub token: u32,
    /// Predicted duration in frames (before the blank / max-symbols adjustments).
    pub duration: usize,
}

/// How tokens are chosen.
pub(crate) enum Path<'a> {
    Greedy,
    /// Follow given steps (verification: every stage sees the reference path).
    Replay(&'a [Step]),
}

pub(crate) struct Decoded {
    pub steps: Vec<Step>,
    /// Joint logits per step, when requested.
    pub logits: Vec<Vec<f32>>,
}

impl Decoded {
    /// Emitted (non-blank) steps.
    pub fn emitted(&self, blank: u32) -> Vec<Step> {
        self.steps.iter().copied().filter(|s| s.token != blank).collect()
    }
}

/// First index of the maximum (torch.argmax order).
fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

/// Decode projected encoder frames `enc` [1, T, H] (realized on `device`), `frames <= T`.
pub(crate) fn decode(p: &Parakeet, enc: &Tensor, frames: usize, path: Path<'_>, keep_logits: bool) -> Result<Decoded> {
    let (device, cfg, pred, joint) = (&p.device, &p.cfg, &p.prediction, &p.joint);
    let blank = cfg.blank_id;
    let v = cfg.vocab_size;
    let hdim = enc.shape().get(2).copied().unwrap_or(0);
    let mut state: LstmState = pred.zero_state();
    let mut dec_out: Option<Tensor> = None;
    // The first step runs the prediction network on blank (the start symbol).
    let mut pending: Option<u32> = Some(blank);
    let (mut t, mut symbols) = (0usize, 0usize);
    let mut out = Decoded { steps: Vec::new(), logits: Vec::new() };
    let max_steps = frames * (cfg.max_symbols_per_step + 1) + 1;

    while t < frames {
        if out.steps.len() > max_steps {
            return Err(Error::new("internal", "tdt decode: step limit exceeded"));
        }
        let enc_t = enc.slice(1, t, t + 1)?.reshape(&[1, hdim])?;
        let (dec, next) = match pending.take() {
            Some(tok) => {
                let (d, s) = pred.step(tok, &state)?;
                (d, Some(s))
            }
            None => (dec_out.clone().ok_or_else(|| Error::new("internal", "tdt decode: no decoder state"))?, None),
        };
        let logits = joint.logits(&enc_t, &dec)?;
        let mut outs: Vec<&Tensor> = vec![&logits];
        if let Some(s) = &next {
            outs.push(&dec);
            outs.extend(s.tensors());
        }
        let mut realized = device.eval(&outs)?.into_iter();
        let logits_r = realized.next().ok_or_else(|| Error::new("backend", "device.eval returned no outputs"))?;
        if next.is_some() {
            dec_out = realized.next();
            state = LstmState::from_tensors(realized.collect(), pred.n_layers());
        }
        let host = device.to_vec_f32(&logits_r)?;
        if host.len() != cfg.joint_dim() {
            return Err(Error::new("shape_mismatch", format!("joint gave {} logits, expected {}", host.len(), cfg.joint_dim())));
        }

        let (token, duration) = match &path {
            Path::Greedy => (argmax(&host[..v]) as u32, cfg.durations[argmax(&host[v..])]),
            Path::Replay(steps) => {
                let s = steps.get(out.steps.len()).ok_or_else(|| Error::new("reference", "replay: reference path ended early"))?;
                if s.frame != t {
                    return Err(Error::new("reference", format!("replay: step {} at frame {t}, reference at {}", out.steps.len(), s.frame)));
                }
                (s.token, s.duration)
            }
        };
        out.steps.push(Step { frame: t, token, duration });
        if keep_logits {
            out.logits.push(host);
        }

        let mut adv = duration;
        if token != blank {
            pending = Some(token);
            symbols += 1;
        } else if adv == 0 {
            adv = 1;
        }
        if adv == 0 && symbols >= cfg.max_symbols_per_step {
            adv = 1;
        }
        if adv > 0 {
            symbols = 0;
        }
        t += adv;
    }
    Ok(out)
}

/// NeMo's punctuation set for TDT timestamp refinement.
const PUNCTUATION: [&str; 11] = ["?", "'", "¡", "¿", "-", ":", ",", "%", "/", ".", "!"];
/// Segment separators (NeMo `segment_seperators`).
const SEGMENT_END: [char; 3] = ['.', '?', '!'];

#[derive(Clone, Debug, Serialize)]
pub struct Token {
    pub id: u32,
    /// The SentencePiece piece (`▁` marks a word start).
    pub piece: String,
    pub start: f32,
    pub end: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct Word {
    pub word: String,
    pub start: f32,
    pub end: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct Segment {
    pub start: f32,
    pub end: f32,
    pub text: String,
    pub tokens: Vec<u32>,
}

/// Token offsets in seconds (`start = frame`, `end = frame + duration`); punctuation
/// tokens take the previous token's end (NeMo `_refine_timestamps_tdt`). Blank and special
/// tokens are skipped.
pub(crate) fn tokens(tok: &Tokenizer, emitted: &[Step], frame_s: f64, blank: u32) -> Vec<Token> {
    let mut out: Vec<Token> = Vec::new();
    for s in emitted {
        if s.token == blank || tok.is_special(s.token) {
            continue;
        }
        let piece = tok.id_to_token(s.token).unwrap_or("").to_string();
        let (mut start, mut end) = ((s.frame as f64 * frame_s) as f32, ((s.frame + s.duration) as f64 * frame_s) as f32);
        if let Some(prev) = out.last().filter(|_| PUNCTUATION.contains(&piece.as_str())) {
            start = prev.end;
            end = prev.end;
        }
        out.push(Token { id: s.token, piece, start, end });
    }
    out
}

/// Pieces → text (Metaspace decoding: `▁` is a space, the leading one dropped).
pub(crate) fn text(tokens: &[Token]) -> String {
    let s: String = tokens.iter().map(|t| t.piece.as_str()).collect();
    s.replace('▁', " ").trim().to_string()
}

/// Words: a new word starts at every piece beginning with `▁`.
pub(crate) fn words(tokens: &[Token]) -> Vec<Word> {
    let mut out: Vec<Word> = Vec::new();
    for t in tokens {
        let body = t.piece.trim_start_matches('▁');
        match out.last_mut() {
            Some(w) if !t.piece.starts_with('▁') => {
                w.word.push_str(body);
                w.end = t.end;
            }
            _ => out.push(Word { word: body.to_string(), start: t.start, end: t.end }),
        }
    }
    out.retain(|w| !w.word.is_empty());
    out
}

/// Segments ending at words whose text ends with `.`, `?` or `!`.
pub(crate) fn segments(tokens: &[Token]) -> Vec<Segment> {
    let mut out = Vec::new();
    let mut cur: Vec<&Token> = Vec::new();
    for (i, t) in tokens.iter().enumerate() {
        cur.push(t);
        let next_starts_word = tokens.get(i + 1).is_none_or(|n| n.piece.starts_with('▁'));
        let ends = t.piece.trim_end().ends_with(SEGMENT_END);
        if (ends && next_starts_word) || i + 1 == tokens.len() {
            let owned: Vec<Token> = cur.drain(..).cloned().collect();
            let first = owned.first().map_or(0.0, |t| t.start);
            out.push(Segment {
                start: first,
                end: owned.last().map_or(first, |t| t.end),
                text: text(&owned),
                tokens: owned.iter().map(|t| t.id).collect(),
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tk(piece: &str, start: f32, end: f32) -> Token {
        Token { id: 0, piece: piece.into(), start, end }
    }

    #[test]
    fn words_and_segments() {
        let t = vec![
            tk("▁And", 0.24, 0.4),
            tk("▁so", 0.56, 0.8),
            tk(",", 0.8, 0.8),
            tk("▁ask", 1.0, 1.2),
            tk("▁no", 1.2, 1.3),
            tk("t", 1.3, 1.5),
            tk(".", 1.5, 1.5),
            tk("▁Yes", 2.0, 2.3),
        ];
        assert_eq!(text(&t), "And so, ask not. Yes");
        let w = words(&t);
        assert_eq!(w.iter().map(|w| w.word.as_str()).collect::<Vec<_>>(), ["And", "so,", "ask", "not.", "Yes"]);
        assert_eq!((w[3].start, w[3].end), (1.2, 1.5));
        let s = segments(&t);
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].text, "And so, ask not.");
        assert_eq!((s[1].start, s[1].end), (2.0, 2.3));
        assert_eq!(argmax(&[0.0, 2.0, 2.0, 1.0]), 1);
    }
}
