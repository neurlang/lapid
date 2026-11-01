//! Streaming text filters for generated output: reasoning ("thinking") separation and
//! OpenAI stop strings. Both hold back the shortest tail that could still become a marker,
//! so streamed deltas never contain partial markers.

use crate::served::Delta;

/// Length (bytes) of the longest suffix of `s` that is a proper prefix of one of `markers`.
fn partial_suffix(s: &str, markers: &[&str]) -> usize {
    let mut best = 0;
    for m in markers {
        let max = m.len().saturating_sub(1).min(s.len());
        for k in (best + 1..=max).rev() {
            let start = s.len() - k;
            if s.is_char_boundary(start) && m.starts_with(&s[start..]) {
                best = k;
                break;
            }
        }
    }
    best
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Phase {
    /// Nothing emitted yet: the output may open with the reasoning marker.
    Start,
    Thinking,
    Content,
}

/// Splits `open … close` reasoning from the answer.
pub struct ThinkSplitter {
    open: String,
    close: String,
    phase: Phase,
    buf: String,
    /// Drop whitespace at the start of the answer (after the reasoning block).
    trim_content: bool,
}

impl ThinkSplitter {
    /// `prompt_open`: the prompt already ends with the open marker (generation starts inside
    /// the reasoning block).
    pub fn new(open: &str, close: &str, prompt_open: bool) -> ThinkSplitter {
        ThinkSplitter {
            open: open.to_string(),
            close: close.to_string(),
            phase: if prompt_open { Phase::Thinking } else { Phase::Start },
            buf: String::new(),
            trim_content: false,
        }
    }

    /// No reasoning separation (everything is content).
    pub fn passthrough() -> ThinkSplitter {
        ThinkSplitter { open: String::new(), close: String::new(), phase: Phase::Content, buf: String::new(), trim_content: false }
    }

    pub fn push(&mut self, text: &str, out: &mut Vec<Delta>) {
        self.buf.push_str(text);
        loop {
            match self.phase {
                Phase::Start => {
                    let t = self.buf.trim_start();
                    if t.starts_with(&self.open) {
                        let rest = t[self.open.len()..].to_string();
                        self.buf = rest;
                        self.phase = Phase::Thinking;
                        continue;
                    }
                    if !t.is_empty() && !self.open.starts_with(t) || t.len() >= self.open.len() {
                        self.phase = Phase::Content;
                        continue;
                    }
                    return; // all whitespace or a partial open marker
                }
                Phase::Thinking => {
                    if let Some(i) = self.buf.find(&self.close) {
                        let reasoning = self.buf[..i].to_string();
                        let rest = self.buf[i + self.close.len()..].to_string();
                        if !reasoning.is_empty() {
                            out.push(Delta::Reasoning(reasoning));
                        }
                        self.buf = rest;
                        self.phase = Phase::Content;
                        self.trim_content = true;
                        continue;
                    }
                    let keep = partial_suffix(&self.buf, &[&self.close]);
                    let emit: String = self.buf.drain(..self.buf.len() - keep).collect();
                    if !emit.is_empty() {
                        out.push(Delta::Reasoning(emit));
                    }
                    return;
                }
                Phase::Content => {
                    if self.trim_content {
                        let t = self.buf.trim_start();
                        if t.is_empty() {
                            self.buf.clear();
                            return;
                        }
                        self.buf = t.to_string();
                        self.trim_content = false;
                    }
                    if !self.buf.is_empty() {
                        out.push(Delta::Content(std::mem::take(&mut self.buf)));
                    }
                    return;
                }
            }
        }
    }

    /// End of generation: flush what is held back.
    pub fn finish(&mut self, out: &mut Vec<Delta>) {
        let rest = std::mem::take(&mut self.buf);
        if rest.is_empty() {
            return;
        }
        match self.phase {
            Phase::Thinking => out.push(Delta::Reasoning(rest)),
            _ => out.push(Delta::Content(rest)),
        }
    }
}

/// OpenAI `stop` strings over the answer text.
pub struct StopMatcher {
    stops: Vec<String>,
    buf: String,
    stopped: bool,
}

impl StopMatcher {
    pub fn new(stops: &[String]) -> StopMatcher {
        StopMatcher { stops: stops.iter().filter(|s| !s.is_empty()).cloned().collect(), buf: String::new(), stopped: false }
    }

    pub fn stopped(&self) -> bool {
        self.stopped
    }

    /// Returns the text that is safe to emit.
    pub fn push(&mut self, text: &str) -> String {
        if self.stopped {
            return String::new();
        }
        if self.stops.is_empty() {
            return text.to_string();
        }
        self.buf.push_str(text);
        let first = self.stops.iter().filter_map(|s| self.buf.find(s.as_str())).min();
        if let Some(i) = first {
            self.stopped = true;
            let out = self.buf[..i].to_string();
            self.buf.clear();
            return out;
        }
        let markers: Vec<&str> = self.stops.iter().map(String::as_str).collect();
        let keep = partial_suffix(&self.buf, &markers);
        self.buf.drain(..self.buf.len() - keep).collect()
    }

    pub fn finish(&mut self) -> String {
        std::mem::take(&mut self.buf)
    }
}

/// Reasoning split + stop strings, as one filter over raw token text.
pub struct OutputFilter {
    think: ThinkSplitter,
    stop: StopMatcher,
}

impl OutputFilter {
    pub fn new(think: ThinkSplitter, stops: &[String]) -> OutputFilter {
        OutputFilter { think, stop: StopMatcher::new(stops) }
    }

    pub fn stopped(&self) -> bool {
        self.stop.stopped()
    }

    fn apply_stop(&mut self, deltas: Vec<Delta>) -> Vec<Delta> {
        let mut out = Vec::new();
        for d in deltas {
            match d {
                Delta::Content(t) => {
                    let t = self.stop.push(&t);
                    if !t.is_empty() {
                        out.push(Delta::Content(t));
                    }
                }
                r => {
                    if !self.stop.stopped() {
                        out.push(r);
                    }
                }
            }
        }
        out
    }

    pub fn push(&mut self, text: &str) -> Vec<Delta> {
        let mut d = Vec::new();
        self.think.push(text, &mut d);
        self.apply_stop(d)
    }

    pub fn finish(&mut self) -> Vec<Delta> {
        let mut d = Vec::new();
        self.think.finish(&mut d);
        let mut out = self.apply_stop(d);
        if !self.stop.stopped() {
            let rest = self.stop.finish();
            if !rest.is_empty() {
                out.push(Delta::Content(rest));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(f: &mut OutputFilter, pieces: &[&str]) -> (String, String) {
        let (mut c, mut r) = (String::new(), String::new());
        let mut all: Vec<Delta> = pieces.iter().flat_map(|p| f.push(p)).collect();
        all.extend(f.finish());
        for d in all {
            match d {
                Delta::Content(t) => c.push_str(&t),
                Delta::Reasoning(t) => r.push_str(&t),
            }
        }
        (c, r)
    }

    #[test]
    fn splits_thinking() {
        let mut f = OutputFilter::new(ThinkSplitter::new("<think>", "</think>", false), &[]);
        assert_eq!(run(&mut f, &["<thi", "nk>hm", "m</th", "ink>\n\nHi", " there"]), ("Hi there".into(), "hmm".into()));
        let mut f = OutputFilter::new(ThinkSplitter::new("<think>", "</think>", true), &[]);
        assert_eq!(run(&mut f, &["a", "b</think>c"]), ("c".into(), "ab".into()));
        let mut f = OutputFilter::new(ThinkSplitter::new("<think>", "</think>", false), &[]);
        assert_eq!(run(&mut f, &["<", "b>"]), ("<b>".into(), "".into()));
    }

    #[test]
    fn stop_strings() {
        let mut f = OutputFilter::new(ThinkSplitter::passthrough(), &["END".into()]);
        assert_eq!(run(&mut f, &["abc E", "N", "Dxyz"]).0, "abc ");
        let mut f = OutputFilter::new(ThinkSplitter::passthrough(), &["END".into()]);
        assert_eq!(run(&mut f, &["abc E", "N"]).0, "abc EN");
        let mut f = OutputFilter::new(ThinkSplitter::passthrough(), &["ü".into()]);
        assert_eq!(run(&mut f, &["aé", "üb"]).0, "aé");
    }
}
