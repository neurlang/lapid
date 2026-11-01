//! `Whisper::load_bytes` (in-memory image) transcribes exactly like `Whisper::load` (mmap).
//! Skipped unless the converted model is cached and `data/jfk.wav` exists.

use std::path::Path;
use std::sync::Arc;

use lapid::Device;
use lapid_hub::{Model, TranscribeOptions, Whisper};

#[test]
fn load_bytes_matches_load_on_cpu() {
    let path = lapid_convert::default_model_path("openai/whisper-small", lapid_convert::OutDType::Int8);
    let audio = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/jfk.wav");
    if !path.is_file() || !audio.is_file() {
        eprintln!("skipped: {} or {} missing", path.display(), audio.display());
        return;
    }
    let dev = Device::cpu();
    let opts = TranscribeOptions::default();
    let mapped = Whisper::load(&path, &dev).expect("load");
    let image = std::fs::read(&path).expect("read");
    let in_memory = Whisper::load_bytes(Arc::new(image), &dev).expect("load_bytes");
    assert_eq!(mapped.info(), in_memory.info());
    assert_eq!(in_memory.info().architecture, "whisper");
    let a = mapped.transcribe_file(&audio, &opts).expect("transcribe mmap");
    let b = in_memory.transcribe_file(&audio, &opts).expect("transcribe bytes");
    assert_eq!(a.text, b.text);
    let toks = |t: &lapid_hub::Transcript| t.segments.iter().flat_map(|s| s.tokens.clone()).collect::<Vec<_>>();
    assert_eq!(toks(&a), toks(&b));
    assert!(b.text.contains("ask not what your country can do for you"), "{}", b.text);
}
