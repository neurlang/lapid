//! `POST /v1/audio/transcriptions`, `/v1/audio/translations` (multipart WAV upload) and
//! `POST /v1/audio/speech` (streamed WAV / PCM, or SSE audio deltas).

use std::convert::Infallible;

use axum::body::{Body, Bytes};
use axum::extract::{Multipart, State};
use axum::http::header;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use base64::Engine;
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use super::{ApiError, ApiResult, Shared, parse_json};
use crate::format;
use crate::served::{AsrParams, Capability, TranscriptOut, TtsParams};

#[derive(Default)]
struct TranscriptionForm {
    file: Option<Bytes>,
    model: Option<String>,
    language: Option<String>,
    response_format: Option<String>,
    granularities: Vec<String>,
}

async fn read_form(mut mp: Multipart) -> ApiResult<TranscriptionForm> {
    let mut f = TranscriptionForm::default();
    while let Some(field) = mp.next_field().await.map_err(|e| ApiError::bad_request(format!("multipart: {e}")))? {
        let name = field.name().unwrap_or("").to_string();
        let bytes = field.bytes().await.map_err(|e| ApiError::bad_request(format!("multipart field {name:?}: {e}")))?;
        let text = || String::from_utf8_lossy(&bytes).trim().to_string();
        match name.as_str() {
            "file" => f.file = Some(bytes.clone()),
            "model" => f.model = Some(text()),
            "language" => f.language = Some(text()),
            "response_format" => f.response_format = Some(text()),
            "timestamp_granularities[]" | "timestamp_granularities" => f.granularities.push(text()),
            _ => {} // prompt, temperature, ... are accepted and ignored
        }
    }
    Ok(f)
}

/// WAV bytes -> 16 kHz mono samples.
fn decode_wav(bytes: &[u8]) -> ApiResult<Vec<f32>> {
    let wav = lapid_hub::audio::parse_wav(bytes).map_err(|e| ApiError::bad_request(format!("{} (upload 16-bit or float WAV)", e.message)))?;
    let mono = lapid_hub::audio::mixdown(&wav.samples, wav.channels as usize);
    Ok(lapid_hub::audio::resample(&mono, wav.sample_rate, 16_000))
}

async fn transcribe(st: Shared, mp: Multipart, translate: bool) -> ApiResult<Response> {
    let form = read_form(mp).await?;
    let file = form.file.ok_or_else(|| ApiError::bad_request("multipart field \"file\" is required"))?;
    let worker = st.find(form.model.as_deref(), Capability::Transcription)?;
    let samples = tokio::task::spawn_blocking(move || decode_wav(&file))
        .await
        .map_err(|e| ApiError::from(lapid::Error::new("internal", e.to_string())))??;
    let duration = samples.len() as f32 / 16_000.0;
    let opts = AsrParams { language: form.language.filter(|l| !l.is_empty()), translate };
    let t: TranscriptOut = worker.call(move |s| s.transcribe(&samples, &opts)).await?;
    let fmt = form.response_format.as_deref().unwrap_or("json");
    let text_response = |body: String, mime: &'static str| ([(header::CONTENT_TYPE, mime)], body).into_response();
    Ok(match fmt {
        "json" => axum::Json(json!({"text": t.text})).into_response(),
        "text" => text_response(format!("{}\n", t.text), "text/plain; charset=utf-8"),
        "srt" => text_response(format::srt(&t.segments), "application/x-subrip; charset=utf-8"),
        "vtt" => text_response(format::vtt(&t.segments), "text/vtt; charset=utf-8"),
        "verbose_json" => {
            let segments: Vec<Value> = t
                .segments
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    json!({
                        "id": i, "seek": 0, "start": s.start, "end": s.end, "text": s.text, "tokens": s.tokens,
                        "temperature": 0.0, "avg_logprob": 0.0, "compression_ratio": 0.0, "no_speech_prob": 0.0,
                    })
                })
                .collect();
            let mut v = json!({
                "task": if translate { "translate" } else { "transcribe" },
                "language": t.language.clone().unwrap_or_else(|| "unknown".into()),
                "duration": duration,
                "text": t.text,
                "segments": segments,
            });
            let want_words = form.granularities.iter().any(|g| g == "word");
            if want_words && !t.words.is_empty() {
                v["words"] = json!(t.words);
            }
            axum::Json(v).into_response()
        }
        other => return Err(ApiError::bad_request(format!("response_format {other:?} (use json, text, srt, verbose_json or vtt)"))),
    })
}

pub async fn transcriptions(State(st): State<Shared>, mp: Multipart) -> ApiResult<Response> {
    transcribe(st, mp, false).await
}

pub async fn translations(State(st): State<Shared>, mp: Multipart) -> ApiResult<Response> {
    transcribe(st, mp, true).await
}

#[derive(Deserialize)]
struct SpeechRequest {
    model: Option<String>,
    input: String,
    voice: Option<String>,
    response_format: Option<String>,
    stream_format: Option<String>,
    seed: Option<u64>,
    temperature: Option<f32>,
}

enum Audio {
    Pcm(Vec<f32>),
    Error(lapid::Error),
}

pub async fn speech(State(st): State<Shared>, body: Bytes) -> ApiResult<Response> {
    let req: SpeechRequest = parse_json(&body)?;
    if req.input.trim().is_empty() {
        return Err(ApiError::bad_request("\"input\" must not be empty"));
    }
    let worker = st.find(req.model.as_deref(), Capability::Speech)?;
    let rate = worker.meta().extra.get("sample_rate").and_then(Value::as_u64).unwrap_or(24_000) as u32;
    let fmt = req.response_format.clone().unwrap_or_else(|| "wav".into());
    if fmt != "wav" && fmt != "pcm" {
        return Err(ApiError::bad_request(format!("response_format {fmt:?} is not supported (use wav or pcm)")));
    }
    let sse = req.stream_format.as_deref() == Some("sse");
    let opts = TtsParams { voice: req.voice.clone(), seed: req.seed, temperature: req.temperature };
    let input = req.input;
    let (tx, rx) = mpsc::channel::<Audio>(64);
    worker.submit(move |s| {
        if tx.is_closed() {
            return;
        }
        let r = s.speak(&input, &opts, &mut |pcm| tx.blocking_send(Audio::Pcm(pcm.to_vec())).is_ok());
        if let Err(e) = r {
            let _ = tx.blocking_send(Audio::Error(e));
        }
    })?;
    if sse {
        let events = ReceiverStream::new(rx)
            .map(|a| {
                let v = match a {
                    Audio::Pcm(p) => {
                        json!({"type": "speech.audio.delta", "audio": base64::engine::general_purpose::STANDARD.encode(format::pcm16(&p))})
                    }
                    Audio::Error(e) => ApiError::from(e).body(),
                };
                Ok::<_, Infallible>(Event::default().data(v.to_string()))
            })
            .chain(futures_util::stream::once(async { Ok(Event::default().data(json!({"type": "speech.audio.done"}).to_string())) }));
        return Ok(Sse::new(events).keep_alive(KeepAlive::default()).into_response());
    }
    // Wait for the first chunk so an immediate failure (unknown voice, empty text) becomes
    // a proper error response instead of a broken audio stream.
    let mut rx = rx;
    let first = match rx.recv().await {
        Some(Audio::Pcm(p)) => format::pcm16(&p),
        Some(Audio::Error(e)) => return Err(e.into()),
        None => Vec::new(),
    };
    let mut head = if fmt == "wav" { format::wav_header(rate, None) } else { Vec::new() };
    head.extend_from_slice(&first);
    let rest = ReceiverStream::new(rx).filter_map(|a| async move {
        match a {
            Audio::Pcm(p) => Some(Ok::<_, Infallible>(Bytes::from(format::pcm16(&p)))),
            Audio::Error(e) => {
                eprintln!("lapid: speech failed mid-stream: {}", e.message);
                None
            }
        }
    });
    let stream = futures_util::stream::once(async move { Ok::<_, Infallible>(Bytes::from(head)) }).chain(rest);
    let mime = if fmt == "wav" { "audio/wav" } else { "audio/pcm" };
    Ok(([(header::CONTENT_TYPE, mime)], Body::from_stream(stream)).into_response())
}
