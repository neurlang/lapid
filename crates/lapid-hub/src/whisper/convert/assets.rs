//! Whisper's non-tensor assets in a `.lapid` file, and the loader-side boundary for them.
//!
//! Format 1.1 files store the token list as the blob [`TOKENS_BLOB`]
//! (`application/json`: a JSON array of `n_vocab` strings, id-ordered, GPT-2 byte-level
//! BPE form, special tokens verbatim); metadata `whisper.tokens_blob` names it. Format 1.0
//! files kept the same array inline as metadata `whisper.tokens`.

use lapid::{Error, LapidFile, Result};
use serde_json::Value;

/// Blob holding the id-ordered token strings.
pub(crate) const TOKENS_BLOB: &str = "tokenizer/tokens.json";
pub(crate) const TOKENS_MEDIA_TYPE: &str = "application/json";

/// Serialized [`TOKENS_BLOB`] contents.
pub(crate) fn tokens_blob(tokens: &[String]) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(tokens)?)
}

/// Id-ordered token strings (byte-level BPE form) of a Whisper `.lapid` file: the
/// `whisper.tokens_blob` blob (default [`TOKENS_BLOB`]), or for format 1.0 files the
/// metadata array `whisper.tokens`. Checks the length against `whisper.n_vocab` and that
/// `whisper.special.eot` is inside the list. Error code "metadata".
pub fn load_token_strings(file: &LapidFile) -> Result<Vec<String>> {
    let w = file.metadata.get("whisper");
    let meta_err = |m: String| Error::new("metadata", format!("whisper: {m}"));
    let tokens: Vec<String> = match w.and_then(|w| w.get("tokens")) {
        Some(Value::Array(arr)) => arr
            .iter()
            .enumerate()
            .map(|(i, v)| v.as_str().map(str::to_string).ok_or_else(|| meta_err(format!("token {i} is not a string"))))
            .collect::<Result<_>>()?,
        Some(_) => return Err(meta_err("metadata 'tokens' is not an array".into())),
        None => {
            let name = w.and_then(|w| w.get("tokens_blob")).and_then(Value::as_str).unwrap_or(TOKENS_BLOB);
            let bytes = file.blob(name).ok_or_else(|| meta_err(format!("no token list (blob {name:?} missing)")))?;
            serde_json::from_slice(bytes).map_err(|e| meta_err(format!("blob {name:?}: {e}")))?
        }
    };
    if let Some(n) = w.and_then(|w| w.get("n_vocab")).and_then(Value::as_u64)
        && tokens.len() as u64 != n {
            return Err(meta_err(format!("{} token strings, n_vocab is {n}", tokens.len())));
        }
    if let Some(eot) = w.and_then(|w| w.pointer("/special/eot")).and_then(Value::as_u64)
        && eot as usize > tokens.len() {
            return Err(meta_err(format!("eot id {eot} outside the token list ({} tokens)", tokens.len())));
        }
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lapid::format::Writer;
    use serde_json::json;

    fn write(tag: &str, whisper: Value, blob: Option<&[u8]>) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("lapid-hub-assets-{}-{tag}.lapid", std::process::id()));
        let m = json!({"name": "w", "architecture": "whisper", "dtype": "fp32", "license": "mit",
                       "source": {"repo": "x/y"}, "converter": "t", "whisper": whisper});
        let mut w = Writer::create(&p, m).expect("create");
        if let Some(b) = blob {
            w.add_blob(TOKENS_BLOB, TOKENS_MEDIA_TYPE, b).expect("blob");
        }
        w.finish().expect("finish");
        p
    }

    #[test]
    fn blob_and_legacy_metadata() {
        let toks = vec!["Ġthe".to_string(), "<|endoftext|>".to_string()];
        let p = write("blob", json!({"n_vocab": 2, "special": {"eot": 1}}), Some(&tokens_blob(&toks).expect("json")));
        assert_eq!(load_token_strings(&LapidFile::open(&p).expect("open")).expect("tokens"), toks);
        std::fs::remove_file(&p).ok();

        let p = write("legacy", json!({"n_vocab": 2, "tokens": toks}), None);
        assert_eq!(load_token_strings(&LapidFile::open(&p).expect("open")).expect("tokens"), toks);
        std::fs::remove_file(&p).ok();

        let p = write("none", json!({"n_vocab": 3}), Some(&tokens_blob(&toks).expect("json")));
        let e = load_token_strings(&LapidFile::open(&p).expect("open")).err().expect("n_vocab mismatch");
        assert_eq!(e.code, "metadata");
        std::fs::remove_file(&p).ok();
    }
}
