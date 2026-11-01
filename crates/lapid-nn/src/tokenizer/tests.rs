use super::*;

fn byte_str(s: &str) -> String {
    let table = bytes_to_unicode();
    s.bytes().map(|b| table[b as usize]).collect()
}

/// 256 byte-level single-char pieces (id = byte) followed by `extra`.
fn byte_vocab(extra: &[&str]) -> Vec<String> {
    let table = bytes_to_unicode();
    let mut v: Vec<String> = table.iter().map(|c| c.to_string()).collect();
    v.extend(extra.iter().map(|s| s.to_string()));
    v
}

#[test]
fn byte_level_decode_whisper_style() {
    let toks = vec![
        "ĠAnd".to_string(),
        "Ġso".to_string(),
        ",".to_string(),
        byte_str(" é"),
        "<|endoftext|>".to_string(),
        "<|0.00|>".to_string(),
    ];
    let mut tok = Tokenizer::byte_level(&toks, &[]).unwrap();
    tok.mark_special(4..6);
    assert_eq!(tok.decode(&[0, 1, 2], true), " And so,");
    assert_eq!(tok.decode(&[0, 3, 4, 5, 99], true), " And é");
    assert_eq!(tok.decode(&[0, 4], false), " And<|endoftext|>");
    assert_eq!(tok.token_for_bytes(b" so"), Some(1));
    assert_eq!(tok.token_for_bytes(b"<|endoftext|>"), None, "special ids are not text tokens");
    assert_eq!(tok.token_to_id("<|endoftext|>"), Some(4));
    assert_eq!(tok.id_to_token(1), Some("Ġso"));
    assert_eq!(tok.special_ids(), vec![4, 5]);
    assert!(tok.is_special(5) && !tok.is_special(0));
    assert_eq!(tok.vocab_size(), 6);
}

#[test]
fn split_utf8_across_tokens() {
    let bytes = "♪".as_bytes(); // e2 99 aa
    let table = bytes_to_unicode();
    let a: String = bytes[..2].iter().map(|&b| table[b as usize]).collect();
    let b: String = bytes[2..].iter().map(|&b| table[b as usize]).collect();
    let tok = Tokenizer::byte_level(&[a, b], &[]).unwrap();
    assert_eq!(tok.decode(&[0, 1], true), "♪");
    assert_eq!(tok.decode(&[0], true), "\u{FFFD}");
    assert_eq!(tok.decode_bytes(&[1], true), vec![0xaa]);
}

#[test]
fn byte_level_bpe_encode_roundtrip() {
    let pieces = byte_vocab(&["he", "ll", "hell", "hello", "Ġw", "Ġwo", "or"]);
    let merges: Vec<(String, String)> = [("h", "e"), ("l", "l"), ("he", "ll"), ("hell", "o"), ("Ġ", "w"), ("o", "r"), ("Ġw", "o")]
        .iter()
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect();
    let tok = Tokenizer::byte_level(&pieces, &merges).unwrap();
    let ids = tok.encode("hello world", false).unwrap();
    // "hello" | "Ġworld": Ġw (rank 4), then "or" (5) beats "Ġwo" (6); then l d
    assert_eq!(ids, vec![259, 260, 262, b'l' as u32, b'd' as u32]);
    assert_eq!(tok.decode(&ids, true), "hello world");
    let text = "Ünïcödé + tabs\t\nand 1234!";
    assert_eq!(tok.decode(&tok.encode(text, false).unwrap(), true), text);
    let bad: Vec<(String, String)> = vec![("x".into(), "yz".into())];
    assert!(Tokenizer::byte_level(&pieces, &bad).is_err());
}

fn bpe_json_byte_level() -> String {
    let mut vocab = serde_json::Map::new();
    for (i, p) in byte_vocab(&["he", "ll", "hello", "hell"]).iter().enumerate() {
        vocab.insert(p.clone(), (i as u64).into());
    }
    serde_json::json!({
        "added_tokens": [
            {"id": 260, "content": "<|bos|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 261, "content": "<|eos|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
        ],
        "normalizer": null,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": false},
            {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}
        ]},
        "post_processor": {"type": "TemplateProcessing", "pair": [],
            "single": [{"SpecialToken": {"id": "<|bos|>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": "<|eos|>", "type_id": 0}}],
            "special_tokens": {"<|bos|>": {"id": "<|bos|>", "ids": [260], "tokens": ["<|bos|>"]}, "<|eos|>": {"id": "<|eos|>", "ids": [261], "tokens": ["<|eos|>"]}}},
        "decoder": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false},
        "model": {"type": "BPE", "vocab": vocab, "merges": ["h e", "l l", "he ll", "hell o"], "ignore_merges": true}
    })
    .to_string()
}

#[test]
fn tokenizer_json_byte_level_bpe() {
    let tok = Tokenizer::from_tokenizer_json(&bpe_json_byte_level()).unwrap();
    assert_eq!(tok.vocab_size(), 262);
    let ids = tok.encode("hello<|eos|>12345", true).unwrap();
    // ignore_merges: "hello" is a vocab piece; digits split 3 + 2 (llama3 pattern)
    assert_eq!(ids, vec![260, 258, 261, b'1' as u32, b'2' as u32, b'3' as u32, b'4' as u32, b'5' as u32, 261]);
    assert_eq!(tok.decode(&ids, true), "hello12345");
    assert_eq!(tok.decode(&ids, false), "<|bos|>hello<|eos|>12345<|eos|>");
    assert!(Tokenizer::from_tokenizer_json("{}").is_err());
    assert!(Tokenizer::from_tokenizer_json("not json").is_err());
}

#[test]
fn tokenizer_json_sentencepiece_bpe_byte_fallback() {
    let mut vocab = serde_json::Map::new();
    let mut add = |p: &str| {
        let id = vocab.len() as u64;
        vocab.insert(p.to_string(), id.into());
    };
    for p in ["<pad>", "<eos>", "<bos>", "<unk>"] {
        add(p);
    }
    for b in 0..256 {
        add(&format!("<0x{b:02X}>"));
    }
    for p in ["▁", "h", "i", "▁h", "▁hi"] {
        add(p);
    }
    let json = serde_json::json!({
        "added_tokens": [
            {"id": 0, "content": "<pad>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}, {"id": 1, "content": "<eos>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 2, "content": "<bos>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}, {"id": 3, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
        ],
        "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "▁"},
        "pre_tokenizer": null,
        "post_processor": {"type": "TemplateProcessing", "pair": [],
            "single": [{"SpecialToken": {"id": "<bos>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}],
            "special_tokens": {"<bos>": {"id": "<bos>", "ids": [2], "tokens": ["<bos>"]}}},
        "decoder": {"type": "Sequence", "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "}, {"type": "ByteFallback"}, {"type": "Fuse"}]},
        "model": {"type": "BPE", "vocab": vocab, "merges": [["▁", "h"], ["▁h", "i"]], "byte_fallback": true, "unk_token": "<unk>"}
    })
    .to_string();
    let tok = Tokenizer::from_tokenizer_json(&json).unwrap();
    let ids = tok.encode(" hi é", true).unwrap();
    assert_eq!(ids, vec![2, 264, 260, 4 + 0xC3, 4 + 0xA9]);
    assert_eq!(tok.decode(&ids, true), " hi é");
    assert_eq!(tok.encode("hi", false).unwrap(), vec![261, 262]);
}

#[test]
fn tokenizer_json_unigram_metaspace() {
    let json = serde_json::json!({
        "added_tokens": [{"id": 0, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}],
        "normalizer": null,
        "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true},
        "decoder": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true},
        "model": {"type": "Unigram", "unk_id": 0, "vocab": [
            ["<unk>", 0.0], ["▁", -2.0], ["▁hello", -1.0], ["▁world", -1.5], ["hel", -3.0], ["lo", -3.0], ["▁wor", -4.0], ["ld", -4.0]
        ]}
    })
    .to_string();
    let tok = Tokenizer::from_tokenizer_json(&json).unwrap();
    assert_eq!(tok.encode("hello world", false).unwrap(), vec![2, 3]);
    assert_eq!(tok.encode("hello zz", false).unwrap(), vec![2, 1, 0], "unknown run fuses into one <unk>");
    assert_eq!(tok.decode(&[2, 3], true), "hello world");
}

