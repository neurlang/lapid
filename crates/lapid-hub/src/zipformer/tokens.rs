//! Token table (sherpa-onnx `tokens.txt`: `<piece> <id>` per line) and detokenization of
//! SentencePiece pieces: `▁` starts a word, `<0xHH>` pieces are raw bytes.

use lapid::{Result, bail};

#[derive(Clone, Debug)]
pub struct Tokens {
    pieces: Vec<String>,
}

impl Tokens {
    pub fn parse(text: &str) -> Result<Tokens> {
        let mut pairs = Vec::new();
        for (n, line) in text.lines().enumerate() {
            let line = line.trim_end_matches('\r');
            if line.trim().is_empty() {
                continue;
            }
            // The piece may itself be a space-like character; the id is the last field.
            let Some((piece, id)) = line.rsplit_once(' ') else {
                bail!("tokenizer", "tokens.txt line {}: expected '<piece> <id>'", n + 1)
            };
            let Ok(id) = id.trim().parse::<usize>() else { bail!("tokenizer", "tokens.txt line {}: bad id {id:?}", n + 1) };
            pairs.push((id, piece.to_string()));
        }
        let n = pairs.iter().map(|p| p.0 + 1).max().unwrap_or(0);
        let mut pieces = vec![String::new(); n];
        for (id, p) in pairs {
            pieces[id] = p;
        }
        Ok(Tokens { pieces })
    }

    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    pub fn piece(&self, id: u32) -> &str {
        self.pieces.get(id as usize).map_or("", String::as_str)
    }

    /// Id of `piece` (e.g. "<unk>").
    pub fn id(&self, piece: &str) -> Option<u32> {
        self.pieces.iter().position(|p| p == piece).map(|i| i as u32)
    }

    /// Text of a token sequence (leading / trailing spaces trimmed).
    pub fn text(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            let p = self.piece(id);
            if let Some(b) = byte_piece(p) {
                bytes.push(b);
            } else {
                bytes.extend(p.replace('\u{2581}', " ").as_bytes());
            }
        }
        String::from_utf8_lossy(&bytes).trim().to_string()
    }
}

fn byte_piece(p: &str) -> Option<u8> {
    let hex = p.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() != 2 {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_and_detokenize() {
        let t = Tokens::parse("<blk> 0\n<sos/eos> 1\n<unk> 2\n\u{2581}HELLO 3\nS 4\n\u{2581}W 5\n<0x41> 6\n").expect("tokens");
        assert_eq!(t.len(), 7);
        assert_eq!(t.text(&[3, 4, 5, 6]), "HELLOS WA");
        assert_eq!(t.id("<unk>"), Some(2));
        assert!(Tokens::parse("nope\n").is_err());
    }
}
