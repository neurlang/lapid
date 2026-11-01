//! Sentence embeddings with EmbeddingGemma, library first: embed a few sentences in one
//! batched call and print their cosine-similarity matrix.
//!
//! ```text
//! cargo run --release -p lapid-hub --example embeddinggemma -- [sentences ...]
//!     --model=<repo | model.lapid>   default google/embeddinggemma-300m
//!     --dtype=<fp16 | int8 | ...>   default fp16
//!     --prompt=<name>               prompt name: query, document, STS, Clustering, ... (default STS)
//!     --dim=<768 | 512 | 256 | 128> Matryoshka size (default 768)
//!     --query="..."                 rank the sentences (as documents) against this query instead
//! ```
//! `LAPID_DEVICE=cpu|cuda|vulkan|metal` picks the device (default: best available).

use std::path::Path;
use std::time::Instant;

use lapid::Device;
use lapid_hub::embeddinggemma::{EmbedOptions, EmbeddingGemma, similarity_matrix};
use lapid_hub::Model;

const DEFAULT_SENTENCES: [&str; 5] = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
    "Mars is often called the Red Planet.",
    "Which planet is known for its reddish color?",
];

fn flag<'a>(flags: &'a [String], name: &str) -> Option<&'a str> {
    flags.iter().find_map(|f| f.strip_prefix(&format!("--{name}=")))
}

fn short(s: &str) -> String {
    let t: String = s.chars().take(40).collect();
    if t.len() < s.len() { format!("{t}…") } else { t }
}

fn main() -> lapid::Result<()> {
    let (flags, args): (Vec<String>, Vec<String>) = std::env::args().skip(1).partition(|a| a.starts_with("--"));
    let model = flag(&flags, "model").unwrap_or(lapid_hub::embeddinggemma::DEFAULT_REPO);
    let dtype = flag(&flags, "dtype").unwrap_or("fp16");
    let dim = flag(&flags, "dim")
        .map(|d| d.parse::<usize>().map_err(|_| lapid::Error::new("invalid_argument", format!("--dim={d} is not a number"))))
        .transpose()?;
    let sentences: Vec<&str> =
        if args.is_empty() { DEFAULT_SENTENCES.to_vec() } else { args.iter().map(String::as_str).collect() };

    let device = Device::best();
    let t = Instant::now();
    let eg = if Path::new(model).is_file() {
        EmbeddingGemma::load(Path::new(model), &device)?
    } else {
        EmbeddingGemma::from_hub(model, dtype, &device)?
    };
    eprintln!("loaded {} ({}) on {} in {:.0} ms", eg.info().name, eg.info().dtype, device.description(), t.elapsed().as_secs_f64() * 1e3);

    if let Some(query) = flag(&flags, "query") {
        let t = Instant::now();
        let q = eg.embed(&[query], &EmbedOptions { dim, ..EmbedOptions::query() })?;
        let docs = eg.embed(&sentences, &EmbedOptions { dim, ..EmbedOptions::document() })?;
        eprintln!("embedded {} texts in {:.1} ms", sentences.len() + 1, t.elapsed().as_secs_f64() * 1e3);
        let scores = &similarity_matrix(&q, &docs)[0];
        let mut ranked: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        println!("query: {query}");
        for (i, s) in ranked {
            println!("  {s:.4}  {}", sentences[i]);
        }
        return Ok(());
    }

    let opts = EmbedOptions { dim, ..EmbedOptions::named(flag(&flags, "prompt").unwrap_or("STS")) };
    let t = Instant::now();
    let emb = eg.embed(&sentences, &opts)?;
    eprintln!(
        "embedded {} sentences in {:.1} ms (dim {})",
        sentences.len(),
        t.elapsed().as_secs_f64() * 1e3,
        emb.first().map_or(0, Vec::len)
    );
    let m = similarity_matrix(&emb, &emb);
    for (i, row) in m.iter().enumerate() {
        let cells: Vec<String> = row.iter().map(|v| format!("{v:6.3}")).collect();
        println!("{}  [{i}] {}", cells.join(" "), short(sentences[i]));
    }
    Ok(())
}
