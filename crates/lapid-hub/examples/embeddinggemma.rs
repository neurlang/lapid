//! Sentence embeddings with EmbeddingGemma: rank sentences against a query (retrieval
//! prompts), then print their similarity matrix at a Matryoshka size.
//!
//! ```text
//! cargo run --release -p lapid-hub --example embeddinggemma -- ["query"] [sentences ...]
//! ```
//! `LAPID_DEVICE=cpu|cuda|vulkan|metal` picks the device (default: best available).

use lapid::Device;
use lapid_hub::embeddinggemma::{DEFAULT_REPO, EmbedOptions, EmbeddingGemma, similarity_matrix};

const SENTENCES: [&str; 4] = [
    "Mars is often called the Red Planet.",
    "Venus is often called Earth's twin.",
    "The weather is lovely today.",
    "He drove to the stadium.",
];

fn main() -> lapid::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let query = args.first().map_or("Which planet is known for its reddish color?", String::as_str);
    let docs: Vec<&str> = if args.len() > 1 { args[1..].iter().map(String::as_str).collect() } else { SENTENCES.to_vec() };

    let device = Device::best();
    let model = EmbeddingGemma::from_hub(DEFAULT_REPO, "fp16", &device)?;

    // Retrieval: the query and the documents use their own task prompts.
    let q = model.embed(&[query], &EmbedOptions::query())?;
    let d = model.embed(&docs, &EmbedOptions::document())?;
    let mut ranked: Vec<(f32, &str)> = similarity_matrix(&q, &d)[0].iter().copied().zip(docs.iter().copied()).collect();
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
    println!("query: {query}");
    for (score, doc) in &ranked {
        println!("  {score:.4}  {doc}");
    }

    // Similarity of the documents at 256 dimensions (truncated and renormalized).
    let e = model.embed(&docs, &EmbedOptions::named("STS").with_dim(256))?;
    println!("\nsimilarity (dim {}):", e[0].len());
    for (row, doc) in similarity_matrix(&e, &e).iter().zip(&docs) {
        let cells: Vec<String> = row.iter().map(|v| format!("{v:6.3}")).collect();
        println!("{}  {doc}", cells.join(" "));
    }
    Ok(())
}
