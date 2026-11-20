//! `GET /v1/realtime?intent=transcription`: realtime transcription over a WebSocket, speaking
//! the transcription-session subset of the OpenAI Realtime API (see `docs/REALTIME.md`). The
//! protocol lives in [`session`] (independent of sockets and models), the wire types in
//! [`events`], and the worker-thread streams in [`backend`]; this file is the socket loop.
//!
//! Events are handled one at a time: while a chunk is being recognized the next frame is not
//! read, so a client sending faster than real time is slowed by TCP, not buffered.

mod backend;
mod events;
mod session;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Deserialize;

use super::{ApiError, Shared, new_id};
use backend::AppModels;
use events::{EventError, ServerEvent};
use session::Session;

/// Largest frame or message: one `append` of several minutes of audio.
const MAX_MESSAGE: usize = 16 << 20;

#[derive(Deserialize)]
pub struct RealtimeQuery {
    /// Only `transcription` is served (the default).
    intent: Option<String>,
    /// The model; may also be set later with `session.update`.
    model: Option<String>,
}

/// The HTTP error of a connection that cannot be set up (before the upgrade).
fn refuse(e: &EventError) -> Response {
    let status = if e.code == "model_not_found" { StatusCode::NOT_FOUND } else { StatusCode::BAD_REQUEST };
    ApiError { status, kind: e.kind, code: e.code.clone(), message: e.message.clone() }.into_response()
}

pub async fn realtime(State(state): State<Shared>, Query(q): Query<RealtimeQuery>, ws: WebSocketUpgrade) -> Response {
    if let Some(intent) = q.intent.as_deref().filter(|i| *i != "transcription") {
        return ApiError::bad_request(format!("intent {intent:?} is not supported: this server only serves \"transcription\"")).into_response();
    }
    let prefix = new_id("s").split_once('-').map_or_else(String::new, |(_, t)| t.to_string());
    let session = match Session::new(AppModels { state }, prefix, q.model.as_deref()) {
        Ok(s) => s,
        Err(e) => return refuse(&e),
    };
    ws.max_message_size(MAX_MESSAGE).max_frame_size(MAX_MESSAGE).on_upgrade(move |socket| connection(socket, session))
}

/// Send `events` in order; false when the client is gone.
async fn send(socket: &mut WebSocket, events: &[ServerEvent]) -> bool {
    for e in events {
        if socket.send(Message::Text(e.to_json().into())).await.is_err() {
            return false;
        }
    }
    true
}

async fn connection(mut socket: WebSocket, mut session: Session<AppModels>) {
    let created = session.created();
    if !send(&mut socket, &[created]).await {
        return;
    }
    tracing::debug!("realtime: session opened");
    while let Some(Ok(msg)) = socket.recv().await {
        let events = match msg {
            Message::Text(text) => session.handle_text(text.as_str()).await,
            Message::Binary(_) => vec![session.protocol_error("binary_frame", "send JSON text events, not binary frames")],
            Message::Close(_) => break,
            Message::Ping(_) | Message::Pong(_) => continue,
        };
        if !send(&mut socket, &events).await {
            break;
        }
    }
    // Dropping the session closes its stream on the model's worker thread.
    tracing::debug!("realtime: session closed");
}
