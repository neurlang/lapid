//! `GET /health`, `GET /api/info`, `GET /v1/models[/{id}]` and `POST /v1/embeddings`.

use axum::Json;
use axum::body::Bytes;
use axum::extract::{Path, State};
use base64::Engine;
use serde::Deserialize;
use serde_json::{Value, json};

use super::{ApiError, ApiResult, Shared, now_secs, parse_json};
use crate::served::{Capability, EmbedParams, Embeddings};
use crate::worker::Worker;

pub async fn health() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

fn model_entry(w: &Worker, created: u64) -> Value {
    let m = w.meta();
    json!({
        "id": m.id, "object": "model", "created": created, "owned_by": "lapid",
        "task": m.task, "architecture": m.architecture, "dtype": m.dtype, "capabilities": m.capabilities,
    })
}

fn created(st: &Shared) -> u64 {
    now_secs().saturating_sub(st.started.elapsed().as_secs())
}

pub async fn models(State(st): State<Shared>) -> Json<Value> {
    let c = created(&st);
    let data: Vec<Value> = st.workers.iter().map(|w| model_entry(w, c)).collect();
    Json(json!({"object": "list", "data": data}))
}

pub async fn model(State(st): State<Shared>, Path(id): Path<String>) -> ApiResult<Json<Value>> {
    let w = st
        .workers
        .iter()
        .find(|w| w.meta().id == id)
        .ok_or_else(|| ApiError::not_found("model_not_found", format!("model {id:?} is not loaded")))?;
    Ok(Json(model_entry(w, created(&st))))
}

pub async fn info(State(st): State<Shared>) -> Json<Value> {
    let models: Vec<Value> = st
        .workers
        .iter()
        .map(|w| {
            let mut v = serde_json::to_value(w.meta()).unwrap_or(Value::Null);
            v["queued"] = json!(w.queued());
            v["busy"] = json!(w.busy());
            v
        })
        .collect();
    Json(json!({
        "name": "lapid",
        "version": env!("CARGO_PKG_VERSION"),
        "uptime_s": st.started.elapsed().as_secs_f64(),
        "device": st.device,
        "devices": st.devices,
        "models": models,
    }))
}

#[derive(Deserialize)]
#[serde(untagged)]
enum EmbedInput {
    One(String),
    Many(Vec<String>),
    Tokens(#[allow(dead_code)] Value),
}

#[derive(Deserialize)]
struct EmbedRequest {
    model: Option<String>,
    input: EmbedInput,
    dimensions: Option<usize>,
    encoding_format: Option<String>,
    /// Extension: prompt name ("query", "document", ...) of models with a prompt table.
    prompt_name: Option<String>,
}

pub async fn embeddings(State(st): State<Shared>, body: Bytes) -> ApiResult<Json<Value>> {
    let req: EmbedRequest = parse_json(&body)?;
    let texts = match req.input {
        EmbedInput::One(s) => vec![s],
        EmbedInput::Many(v) => v,
        EmbedInput::Tokens(_) => return Err(ApiError::bad_request("\"input\" must be a string or an array of strings")),
    };
    if texts.is_empty() {
        return Err(ApiError::bad_request("\"input\" must not be empty"));
    }
    let base64 = match req.encoding_format.as_deref() {
        None | Some("float") => false,
        Some("base64") => true,
        Some(other) => return Err(ApiError::bad_request(format!("encoding_format {other:?} (use float or base64)"))),
    };
    let worker = st.find(req.model.as_deref(), Capability::Embedding)?;
    let model = worker.meta().id.clone();
    let opts = EmbedParams { dimensions: req.dimensions, prompt_name: req.prompt_name };
    let e: Embeddings = worker.call(move |s| s.embed(&texts, &opts)).await?;
    let data: Vec<Value> = e
        .vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let embedding = if base64 {
                let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                json!(base64::engine::general_purpose::STANDARD.encode(bytes))
            } else {
                json!(v)
            };
            json!({"object": "embedding", "index": i, "embedding": embedding})
        })
        .collect();
    Ok(Json(json!({
        "object": "list", "data": data, "model": model,
        "usage": {"prompt_tokens": e.prompt_tokens, "total_tokens": e.prompt_tokens},
    })))
}
