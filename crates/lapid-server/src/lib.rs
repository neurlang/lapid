//! lapid-server: an OpenAI-compatible HTTP server with an embedded web UI for lapid hub
//! models (`lapid serve`), and the interactive terminal runner (`lapid run`).
//!
//! Each loaded model lives on its own worker thread ([`worker::Worker`]) as a
//! `lapid_hub::AnyModel` (see `lapid_hub::load`); requests to one model run one at a time in
//! arrival order.
//!
//! Endpoints: `GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/completions`,
//! `POST /v1/embeddings`, `POST /v1/audio/transcriptions`, `POST /v1/audio/translations`,
//! `POST /v1/audio/speech`, `GET /health`, `GET /api/info`, and the web UI on every other path.

pub mod api;
mod assets;
pub mod cli;
pub mod format;
pub mod hf;
pub mod load;
pub mod run;
pub mod worker;

use std::sync::Arc;
use std::time::Instant;

use lapid::{Device, Error, Result};

pub use load::{LoadOptions, ModelSpec};
pub use run::{RunOptions, run};

/// `lapid serve` options.
#[derive(Clone, Debug)]
pub struct ServeOptions {
    /// Model arguments (`[id=]source`).
    pub models: Vec<String>,
    pub host: String,
    pub port: u16,
    /// Device name (`None` = best / `$LAPID_DEVICE`).
    pub device: Option<String>,
    pub load: LoadOptions,
}

impl Default for ServeOptions {
    fn default() -> Self {
        ServeOptions { models: Vec::new(), host: "127.0.0.1".into(), port: 8080, device: None, load: LoadOptions::default() }
    }
}

pub fn device(name: Option<&str>) -> Result<Device> {
    match name {
        Some(n) => Device::from_name(n),
        None => Ok(Device::best()),
    }
}

/// Load every model (in parallel) and serve until Ctrl-C.
pub fn serve(opts: ServeOptions) -> Result<()> {
    if opts.models.is_empty() {
        return Err(Error::new("invalid_argument", "give at least one model (HF repo, .gguf or .lapid file)"));
    }
    let filter = tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).with_writer(std::io::stderr).try_init();
    let device = device(opts.device.as_deref())?;
    eprintln!("lapid: device {}", device.description());

    // Resolve (convert if needed) sequentially, then load on worker threads in parallel.
    let mut resolved = Vec::new();
    for arg in &opts.models {
        let spec = ModelSpec::parse(arg);
        let r = lapid_hub::runtime::resolve::resolve(&spec.source, opts.load.dtype.as_deref())?;
        resolved.push((r.path, spec.id.unwrap_or(r.id)));
    }
    let mut ids: Vec<&str> = resolved.iter().map(|(_, id)| id.as_str()).collect();
    ids.sort_unstable();
    if ids.windows(2).any(|w| w[0] == w[1]) {
        return Err(Error::new("invalid_argument", "two models have the same id; name them with id=source"));
    }
    let workers: Vec<Result<worker::Worker>> = std::thread::scope(|s| {
        let handles: Vec<_> = resolved
            .iter()
            .map(|(path, id)| {
                let (path, id, device, lo) = (path.clone(), id.clone(), device.clone(), opts.load.clone());
                s.spawn(move || {
                    let t0 = Instant::now();
                    let w = worker::Worker::spawn(path, id.clone(), device, lo);
                    if w.is_ok() {
                        eprintln!("lapid: loaded {id} in {:.1} s", t0.elapsed().as_secs_f64());
                    }
                    w
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(Error::new("internal", "model loader panicked")))).collect()
    });
    let workers = workers.into_iter().collect::<Result<Vec<_>>>()?;
    let state = Arc::new(api::AppState { workers, started: Instant::now(), device: device.description(), devices: Device::list() });

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| Error::new("io", format!("tokio runtime: {e}")))?;
    rt.block_on(async move {
        let addr = format!("{}:{}", opts.host, opts.port);
        let listener =
            tokio::net::TcpListener::bind(&addr).await.map_err(|e| Error::new("io", format!("cannot listen on {addr}: {e}")))?;
        for w in &state.workers {
            let m = w.meta();
            let caps: Vec<String> = m.capabilities.iter().map(|c| format!("{c:?}").to_lowercase()).collect();
            eprintln!("lapid: model {} ({}, {}, {})", m.id, m.architecture, m.dtype, caps.join("+"));
        }
        eprintln!("lapid: listening on http://{addr} (web UI and OpenAI API at /v1)");
        axum::serve(listener, api::router(state))
            .with_graceful_shutdown(async {
                let _ = tokio::signal::ctrl_c().await;
                eprintln!("lapid: shutting down");
            })
            .await
            .map_err(|e| Error::new("io", format!("server: {e}")))
    })
}
