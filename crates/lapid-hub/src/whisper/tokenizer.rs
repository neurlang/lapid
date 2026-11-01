//! Whisper's tokenizer: a decode-only byte-level BPE vocabulary (`lapid_nn::Tokenizer`)
//! where every id >= eot is special (sot, languages, tasks, timestamps), plus OpenAI's
//! non-speech token set.

use lapid::Result;
use lapid_nn::Tokenizer;

/// Tokenizer from the id-ordered byte-level token strings of the model file.
pub(crate) fn from_strings<S: AsRef<str>>(tokens: &[S], eot: u32) -> Result<Tokenizer> {
    let mut tok = Tokenizer::byte_level(tokens, &[])?;
    tok.mark_special(eot..tokens.len() as u32);
    Ok(tok)
}

/// Id of the longest text token that is a prefix of `bytes` (the first BPE token in
/// practice for the symbols used below).
fn longest_prefix_token(tok: &Tokenizer, bytes: &[u8]) -> Option<u32> {
    (1..=bytes.len()).rev().find_map(|n| tok.token_for_bytes(&bytes[..n]))
}

/// OpenAI `Tokenizer.non_speech_tokens`, computed from the vocabulary: symbols (and
/// " " + symbol) that are a single token, plus the first token of the musical symbols,
/// plus " -" and " '".
pub(crate) fn non_speech_tokens(tok: &Tokenizer) -> Vec<u32> {
    let mut symbols: Vec<String> = "\"#()*+/:;<=>@[\\]^_`{|}~「」『』".chars().map(String::from).collect();
    symbols.extend(
        "<< >> <<< >>> -- --- -( -[ (' (\" (( )) ((( ))) [[ ]] {{ }} ♪♪ ♪♪♪"
            .split(' ')
            .map(String::from),
    );
    let misc: Vec<String> = "♩♪♫♬♭♮♯".chars().map(String::from).collect();

    let mut result: Vec<u32> = Vec::new();
    result.extend(longest_prefix_token(tok, b" -"));
    result.extend(longest_prefix_token(tok, b" '"));
    for s in &symbols {
        for variant in [s.clone(), format!(" {s}")] {
            result.extend(tok.token_for_bytes(variant.as_bytes()));
        }
    }
    for s in &misc {
        for variant in [s.clone(), format!(" {s}")] {
            result.extend(longest_prefix_token(tok, variant.as_bytes()));
        }
    }
    result.sort_unstable();
    result.dedup();
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use lapid_nn::tokenizer::bytes_to_unicode;

    #[test]
    fn whisper_specials_and_non_speech() {
        let bytes = "♪".as_bytes(); // e2 99 aa
        let table = bytes_to_unicode();
        let a: String = bytes[..2].iter().map(|&b| table[b as usize]).collect();
        let b: String = bytes[2..].iter().map(|&b| table[b as usize]).collect();
        let tok = from_strings(&[a, b, "<|endoftext|>".to_string(), "<|0.00|>".to_string()], 2).unwrap();
        assert_eq!(tok.decode(&[0, 1, 2, 3], true), "♪");
        assert!(tok.is_special(2) && tok.is_special(3) && !tok.is_special(1));
        // musical symbol: first token is the longest prefix
        assert_eq!(non_speech_tokens(&tok), vec![0]);
    }
}
