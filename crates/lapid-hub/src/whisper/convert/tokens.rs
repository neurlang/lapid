//! Whisper tokenizer tables shared by the converters: GPT-2 byte-level encoding, the
//! language list (token order), and the default suppress lists.

/// Whisper language codes in token order (`<|en|>` = sot + 1, ...). `yue` exists only in
/// 100-language models (large-v3).
pub(crate) const LANGUAGES: [&str; 100] = [
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi",
    "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml",
    "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs",
    "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am",
    "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln",
    "ha", "ba", "jw", "su", "yue",
];

/// Non-speech suppress list (ids < eot) of the multilingual Whisper tokenizer, as in HF
/// `generation_config.json` (identical for tiny..large-v3).
const SUPPRESS_MULTILINGUAL: [u32; 82] = [
    1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93, 359, 503, 522, 542, 873, 893,
    902, 918, 922, 931, 1350, 1853, 1982, 2460, 2627, 3246, 3253, 3268, 3536, 3846, 3961, 4183, 4667, 6585, 6647, 7273,
    9061, 9383, 10428, 10929, 11938, 12033, 12331, 12562, 13793, 14157, 14635, 15265, 15618, 16553, 16604, 18362,
    18956, 20075, 21675, 22520, 26130, 26161, 26435, 28279, 29464, 31650, 32302, 32470, 36865, 42863, 47425, 49870,
    50254,
];

/// Same for the English-only (GPT-2) tokenizer (`*.en` models).
const SUPPRESS_ENGLISH: [u32; 84] = [
    1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93, 357, 366, 438, 532, 685, 705,
    796, 930, 1058, 1220, 1267, 1279, 1303, 1343, 1377, 1391, 1635, 1782, 1875, 2162, 2361, 2488, 3467, 4008, 4211,
    4600, 4808, 5299, 5855, 6329, 7203, 9609, 9959, 10563, 10786, 11420, 11709, 11907, 13163, 13697, 13700, 14808,
    15306, 16410, 16791, 17992, 19203, 19510, 20724, 22305, 22935, 27007, 30109, 30420, 33409, 34949, 40283, 40493,
    40549, 47282, 49146,
];

/// GPT-2 `bytes_to_unicode`: byte → char used in byte-level BPE token strings.
pub(crate) fn byte_to_char_table() -> [char; 256] {
    let printable = |b: u32| (0x21..=0x7e).contains(&b) || (0xa1..=0xac).contains(&b) || (0xae..=0xff).contains(&b);
    let mut table = ['\0'; 256];
    let mut n = 0u32;
    for b in 0..256u32 {
        let cp = if printable(b) {
            b
        } else {
            n += 1;
            255 + n
        };
        table[b as usize] = char::from_u32(cp).unwrap_or('\u{fffd}');
    }
    table
}

/// Raw token bytes → byte-level BPE string (HF `vocab.json` form).
pub(crate) fn bytes_to_bpe_string(table: &[char; 256], bytes: &[u8]) -> String {
    bytes.iter().map(|&b| table[b as usize]).collect()
}

/// Special-token ids of a Whisper vocabulary.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Specials {
    pub eot: u32,
    pub sot: u32,
    pub n_languages: u32,
    pub translate: u32,
    pub transcribe: u32,
    pub sot_lm: u32,
    pub prev: u32,
    pub no_speech: u32,
    pub no_timestamps: u32,
    pub timestamp_begin: u32,
}

impl Specials {
    /// The canonical layout used by every OpenAI Whisper model (as in whisper.cpp):
    /// multilingual iff n_vocab >= 51865; languages = n_vocab - 51765 - multilingual.
    pub(crate) fn from_n_vocab(n_vocab: u32) -> Specials {
        let multilingual = n_vocab >= 51865;
        let eot = if multilingual { 50257 } else { 50256 };
        let sot = eot + 1;
        let n_languages = n_vocab.saturating_sub(51765 + multilingual as u32).min(LANGUAGES.len() as u32);
        let translate = sot + 1 + n_languages;
        Specials {
            eot,
            sot,
            n_languages,
            translate,
            transcribe: translate + 1,
            sot_lm: translate + 2,
            prev: translate + 3,
            no_speech: translate + 4,
            no_timestamps: translate + 5,
            timestamp_begin: translate + 6,
        }
    }

    pub(crate) fn is_multilingual(&self) -> bool {
        self.eot == 50257
    }

    /// Token strings for every special id (HF spelling), keyed by id.
    pub(crate) fn token_strings(&self, n_vocab: u32) -> Vec<(u32, String)> {
        let mut v = vec![(self.eot, "<|endoftext|>".to_string()), (self.sot, "<|startoftranscript|>".to_string())];
        for (i, code) in LANGUAGES.iter().take(self.n_languages as usize).enumerate() {
            v.push((self.sot + 1 + i as u32, format!("<|{code}|>")));
        }
        v.push((self.translate, "<|translate|>".into()));
        v.push((self.transcribe, "<|transcribe|>".into()));
        v.push((self.sot_lm, "<|startoflm|>".into()));
        v.push((self.prev, "<|startofprev|>".into()));
        let nosp = if self.n_languages >= 100 { "<|nospeech|>" } else { "<|nocaptions|>" };
        v.push((self.no_speech, nosp.into()));
        v.push((self.no_timestamps, "<|notimestamps|>".into()));
        for id in self.timestamp_begin..n_vocab {
            let i = id - self.timestamp_begin;
            v.push((id, format!("<|{}.{:02}|>", i / 50, (i % 50) * 2)));
        }
        v
    }

    /// Default `suppress_tokens` (as in HF generation_config).
    pub(crate) fn suppress_tokens(&self) -> Vec<u32> {
        let base: &[u32] = if self.is_multilingual() { &SUPPRESS_MULTILINGUAL } else { &SUPPRESS_ENGLISH };
        let mut v = base.to_vec();
        v.extend([self.sot, self.translate, self.transcribe, self.sot_lm, self.prev, self.no_speech]);
        v
    }

    pub(crate) fn begin_suppress_tokens(&self) -> Vec<u32> {
        vec![220, self.eot]
    }

    pub(crate) fn languages(&self) -> Vec<(&'static str, u32)> {
        LANGUAGES.iter().take(self.n_languages as usize).enumerate().map(|(i, c)| (*c, self.sot + 1 + i as u32)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_level_table() {
        let t = byte_to_char_table();
        assert_eq!(t[b'a' as usize], 'a');
        assert_eq!(t[b' ' as usize], 'Ġ');
        assert_eq!(t[b'\n' as usize], 'Ċ');
        assert_eq!(t[0], 'Ā');
        assert_eq!(t[0xad], 'Ń');
        assert_eq!(bytes_to_bpe_string(&t, b" the"), "Ġthe");
    }

    #[test]
    fn whisper_small_specials() {
        let s = Specials::from_n_vocab(51865);
        assert_eq!((s.eot, s.sot, s.translate, s.transcribe), (50257, 50258, 50358, 50359));
        assert_eq!((s.sot_lm, s.prev, s.no_speech, s.no_timestamps, s.timestamp_begin), (50360, 50361, 50362, 50363, 50364));
        assert_eq!(s.languages()[0], ("en", 50259));
        assert_eq!(s.languages().last().copied(), Some(("su", 50357)));
        let strs = s.token_strings(51865);
        assert_eq!(strs.last().map(|t| t.1.as_str()), Some("<|30.00|>"));
        assert!(strs.contains(&(50364, "<|0.00|>".to_string())));
        assert!(strs.contains(&(50365, "<|0.02|>".to_string())));
        let sup = s.suppress_tokens();
        assert_eq!(&sup[sup.len() - 6..], &[50258, 50358, 50359, 50360, 50361, 50362]);
    }

    #[test]
    fn english_and_v3_specials() {
        let en = Specials::from_n_vocab(51864);
        assert_eq!((en.eot, en.sot, en.translate, en.no_timestamps, en.timestamp_begin), (50256, 50257, 50357, 50362, 50363));
        let v3 = Specials::from_n_vocab(51866);
        assert_eq!((v3.translate, v3.transcribe, v3.no_timestamps), (50359, 50360, 50364));
        assert_eq!(v3.languages().last().copied(), Some(("yue", 50358)));
    }
}
