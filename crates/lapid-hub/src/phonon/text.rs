//! Text preparation, tokenization and chunking for Phonon.
//!
//! `prepare` is the language-independent part of the reference runtime's text preparation
//! (the normalizer the HF port writes into its tokenizer): typographic quotes / dashes
//! folded, bullets, arrows, emoji and control characters to spaces, `…` to `.`, `;`, `(`,
//! `)` and a free-standing `:` to `, `, punctuation before `.` / `,` dropped, whitespace
//! collapsed, closing quotes stripped, first letter upper-cased, a final `.` after an
//! alphanumeric end. Not done (gaps): accent folding of letters missing from the vocabulary
//! (the tokenizer falls back to bytes) and the language-specific normalization of the
//! runtime (numbers, amounts, symbols read aloud).

use lapid::{Result, bail};
use lapid_nn::Tokenizer;

fn is_dropped_char(c: char) -> bool {
    matches!(c, '–' | '*' | '—' | '[' | ']' | '{' | '}' | '•' | '‣' | '◦' | '·' | '→' | '←' | '↑' | '↓' | '➡' | '➜')
        || matches!(c as u32, 0x1F000..=0x1FAFF | 0x2600..=0x27BF | 0x2B00..=0x2BFF | 0xFE0F | 0x200D)
        || c.is_control()
        || c.is_whitespace()
}

fn fold_char(c: char) -> char {
    match c {
        '»' | '«' | '‹' | '›' | '“' | '”' | '„' | '‟' | '″' | '‶' | '⹂' | '❝' | '❞' | '❠' | '〝' | '〞' | '〟'
        | '＂' => '"',
        '‘' | '’' | '‚' | '‛' | '′' | '‵' | '❛' | '❜' | '❟' | '＇' | 'ʼ' | 'ʻ' | 'ʹ' | '´' | '`' => {
            '\''
        }
        '‐' | '‑' | '‒' | '―' => '-',
        '₀'..='₉' => char::from_digit(c as u32 - '₀' as u32, 10).unwrap_or(c),
        _ => c,
    }
}

/// Punctuation a following `.` or `,` swallows (with whitespace).
fn swallowed(c: char) -> bool {
    c.is_whitespace() || "!()*,-./:;<=>?[\\]^_`{|}~".contains(c)
}

/// Runtime text preparation (see the module docs).
pub fn prepare(text: &str) -> String {
    // 1. fold characters; runs of dropped characters become one space; … -> .
    let mut s = String::with_capacity(text.len());
    let mut in_drop = false;
    for c in text.chars().map(fold_char) {
        if is_dropped_char(c) {
            if !in_drop {
                s.push(' ');
            }
            in_drop = true;
            continue;
        }
        in_drop = false;
        if c == '…' {
            s.push('.');
        } else {
            s.push(c);
        }
    }
    // 2. `;`, `(`, `)` and a `:` not between two non-space characters -> ", "
    let chars: Vec<char> = s.chars().collect();
    let mut t = String::with_capacity(s.len());
    for (i, &c) in chars.iter().enumerate() {
        let prev_ns = i > 0 && !chars[i - 1].is_whitespace();
        let next_ns = chars.get(i + 1).is_some_and(|n| !n.is_whitespace());
        if matches!(c, ';' | '(' | ')') || (c == ':' && !(prev_ns && next_ns)) {
            t.push_str(", ");
        } else {
            t.push(c);
        }
    }
    // 3. a `.` or `,` swallows the run of spaces / punctuation before it (a `$` too, unless
    //    it follows a digit)
    let s = drop_before_period(&t);
    // 4. collapse whitespace, strip closing quotes
    let mut s = s.split_whitespace().collect::<Vec<_>>().join(" ");
    while s.ends_with('"') {
        s.pop();
    }
    if s.ends_with('\'') && s[..s.len() - 1].ends_with(['.', '!', '?', ',', ';']) {
        s.pop();
    }
    let mut s = s.trim().to_string();
    // 5. upper-case the first letter, end alphanumeric text with a period
    if let Some(first) = s.chars().next()
        && first.is_lowercase()
    {
        let upper: String = first.to_uppercase().collect();
        s.replace_range(..first.len_utf8(), &upper);
    }
    if s.chars().last().is_some_and(char::is_alphanumeric) {
        s.push('.');
    }
    s
}

/// Within each run of swallowable characters, drop everything before the run's last `.` /
/// `,` (regex `(?:\s|[punct]|(?<![0-9])\$)+(?=[.,])` → "").
fn drop_before_period(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let sw = |k: usize| swallowed(chars[k]) || (chars[k] == '$' && !(k > 0 && chars[k - 1].is_ascii_digit()));
    let mut out = Vec::with_capacity(chars.len());
    let mut i = 0;
    while i < chars.len() {
        if sw(i) {
            let mut j = i;
            while j < chars.len() && sw(j) {
                j += 1;
            }
            match chars[i..j].iter().rposition(|c| matches!(c, '.' | ',')) {
                Some(p) => out.extend_from_slice(&chars[i + p..j]),
                None => out.extend_from_slice(&chars[i..j]),
            }
            i = j;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out.into_iter().collect()
}

/// Number of words (whitespace-separated) of prepared text.
pub fn word_count(prepared: &str) -> usize {
    prepared.split_whitespace().count()
}

/// Token ids of prepared text (no special tokens).
pub fn encode(tok: &Tokenizer, prepared: &str) -> Result<Vec<u32>> {
    let ids = tok.encode(prepared, false)?;
    if ids.is_empty() {
        bail!("invalid_argument", "text {prepared:?} has no tokens");
    }
    Ok(ids)
}

/// Splits prepared text into chunks of at most `max_tokens` tokens at sentence ends (after
/// `.`, `!`, `?` followed by a space), then clause ends (`,` `;` `:`) for longer sentences,
/// grouping consecutive pieces greedily (HF `PocketTTSProcessor.split_text`).
pub fn split(tok: &Tokenizer, prepared: &str, max_tokens: usize) -> Result<Vec<String>> {
    let pieces_at = |s: &str, ends: &[char]| -> Vec<String> {
        let mut out = Vec::new();
        let mut cur = String::new();
        let chars: Vec<char> = s.chars().collect();
        for (i, &c) in chars.iter().enumerate() {
            cur.push(c);
            let boundary = ends.contains(&c) && chars.get(i + 1).is_some_and(|n| n.is_whitespace());
            // not a decimal point
            let decimal = c == '.'
                && i > 0
                && chars[i - 1].is_ascii_digit()
                && chars.get(i + 1).is_some_and(char::is_ascii_digit);
            if boundary && !decimal {
                out.push(std::mem::take(&mut cur).trim().to_string());
            }
        }
        if !cur.trim().is_empty() {
            out.push(cur.trim().to_string());
        }
        out
    };
    let mut segments: Vec<(usize, String)> = Vec::new();
    for sentence in pieces_at(prepared, &['.', '!', '?']) {
        let n = tok.encode(&sentence, false)?.len();
        if n > max_tokens {
            for clause in pieces_at(&sentence, &[',', ';', ':']) {
                segments.push((tok.encode(&clause, false)?.len(), clause));
            }
        } else {
            segments.push((n, sentence));
        }
    }
    let mut chunks = Vec::new();
    let (mut chunk, mut len) = (String::new(), 0usize);
    for (n, seg) in segments {
        if !chunk.is_empty() && len + n > max_tokens {
            chunks.push(std::mem::take(&mut chunk));
            len = 0;
        }
        if !chunk.is_empty() {
            chunk.push(' ');
        }
        chunk.push_str(&seg);
        len += n;
    }
    if !chunk.is_empty() {
        chunks.push(chunk);
    }
    Ok(chunks)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepare_basics() {
        assert_eq!(prepare("hello world"), "Hello world.");
        assert_eq!(prepare("  Hello   from\nPhonon.  "), "Hello from Phonon.");
        assert_eq!(prepare("“quoted” — text…"), "\"quoted\" text.");
        assert_eq!(prepare("a (b) c"), "A, b, c.");
        assert_eq!(prepare("wait , what ?"), "Wait, what ?");
        assert_eq!(prepare("time: 10:30"), "Time, 10:30.");
        assert_eq!(prepare("He said \"hi.\""), "He said \"hi.");
        assert_eq!(prepare("ok!"), "Ok!");
    }
}
