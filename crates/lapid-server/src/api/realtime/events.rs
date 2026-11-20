//! Wire types of the realtime transcription protocol (the OpenAI Realtime API's transcription
//! sessions): the events a client sends, the events the server answers with, and the
//! protocol error that becomes an `error` event. Unknown fields are ignored so clients may
//! send the full OpenAI session object.

use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

/// A protocol error reported to the client as an `error` event; the session stays open.
#[derive(Clone, Debug, PartialEq)]
pub struct EventError {
    /// `invalid_request_error` (the client's doing) or `server_error`.
    pub kind: &'static str,
    pub code: String,
    pub message: String,
    /// The client event that caused it, when it had an `event_id`.
    pub event_id: Option<String>,
}

impl EventError {
    pub fn invalid(code: &str, message: impl Into<String>) -> EventError {
        EventError { kind: "invalid_request_error", code: code.into(), message: message.into(), event_id: None }
    }

    pub fn server(code: &str, message: impl Into<String>) -> EventError {
        EventError { kind: "server_error", code: code.into(), message: message.into(), event_id: None }
    }

    pub fn at(mut self, event_id: &Option<String>) -> EventError {
        self.event_id.clone_from(event_id);
        self
    }
}

/// Events a client sends.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum ClientEvent {
    #[serde(rename = "session.update")]
    SessionUpdate {
        #[serde(default)]
        event_id: Option<String>,
        session: SessionUpdate,
    },
    #[serde(rename = "input_audio_buffer.append")]
    Append {
        #[serde(default)]
        event_id: Option<String>,
        /// Base64 of 16-bit little-endian mono PCM.
        audio: String,
    },
    #[serde(rename = "input_audio_buffer.commit")]
    Commit {
        #[serde(default)]
        event_id: Option<String>,
    },
    #[serde(rename = "input_audio_buffer.clear")]
    Clear {
        #[serde(default)]
        event_id: Option<String>,
    },
}

const CLIENT_EVENTS: [&str; 4] = ["session.update", "input_audio_buffer.append", "input_audio_buffer.commit", "input_audio_buffer.clear"];

impl ClientEvent {
    /// The client's `event_id`, echoed in errors.
    pub fn event_id(&self) -> &Option<String> {
        match self {
            ClientEvent::SessionUpdate { event_id, .. }
            | ClientEvent::Append { event_id, .. }
            | ClientEvent::Commit { event_id }
            | ClientEvent::Clear { event_id } => event_id,
        }
    }
}

/// Parse one text frame into an event.
pub fn parse(text: &str) -> Result<ClientEvent, EventError> {
    let v: Value = serde_json::from_str(text).map_err(|e| EventError::invalid("invalid_json", format!("invalid JSON: {e}")))?;
    let event_id = v.get("event_id").and_then(Value::as_str).map(String::from);
    let Some(kind) = v.get("type").and_then(Value::as_str) else {
        return Err(EventError::invalid("missing_type", "the event has no \"type\"").at(&event_id));
    };
    if !CLIENT_EVENTS.contains(&kind) {
        return Err(EventError::invalid("unknown_event", format!("unknown event type {kind:?} (supported: {})", CLIENT_EVENTS.join(", "))).at(&event_id));
    }
    serde_json::from_value(v).map_err(|e| EventError::invalid("invalid_event", e.to_string()).at(&event_id))
}

/// `session.update`'s session object (transcription sessions).
#[derive(Debug, Default, Deserialize)]
pub struct SessionUpdate {
    #[serde(rename = "type")]
    pub kind: Option<String>,
    pub audio: Option<AudioConfig>,
}

#[derive(Debug, Default, Deserialize)]
pub struct AudioConfig {
    pub input: Option<InputConfig>,
}

#[derive(Debug, Default, Deserialize)]
pub struct InputConfig {
    pub format: Option<FormatConfig>,
    pub transcription: Option<TranscriptionConfig>,
    /// Absent = unchanged, `null` = manual turns, object = server turn detection.
    #[serde(default, deserialize_with = "present")]
    pub turn_detection: Option<Option<TurnDetection>>,
}

#[derive(Debug, Default, Deserialize)]
pub struct FormatConfig {
    #[serde(rename = "type")]
    pub kind: Option<String>,
    pub rate: Option<u32>,
}

#[derive(Debug, Default, Deserialize)]
pub struct TranscriptionConfig {
    pub model: Option<String>,
    pub language: Option<String>,
    /// `minimal`, `low`, `medium`, `high` or `xhigh`.
    pub delay: Option<String>,
    // `prompt` and `keywords` are accepted and ignored (unknown fields are skipped): transducer
    // models take no prompt and no keyword biasing is implemented.
}

#[derive(Debug, Default, Deserialize)]
pub struct TurnDetection {
    #[serde(rename = "type")]
    pub kind: Option<String>,
    pub silence_duration_ms: Option<u32>,
}

/// Distinguishes an absent field (outer `None`) from an explicit `null` (`Some(None)`).
fn present<'de, D: Deserializer<'de>, T: Deserialize<'de>>(d: D) -> Result<Option<Option<T>>, D::Error> {
    Option::<T>::deserialize(d).map(Some)
}

/// The body of an `error` event.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ErrorBody {
    #[serde(rename = "type")]
    pub kind: String,
    pub code: String,
    pub message: String,
    pub param: Option<String>,
    /// The client event that caused the error.
    pub event_id: Option<String>,
}

/// Events the server sends.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum ServerEvent {
    #[serde(rename = "session.created")]
    SessionCreated { event_id: String, session: Value },
    #[serde(rename = "session.updated")]
    SessionUpdated { event_id: String, session: Value },
    #[serde(rename = "input_audio_buffer.committed")]
    Committed { event_id: String, item_id: String, previous_item_id: Option<String> },
    #[serde(rename = "input_audio_buffer.cleared")]
    Cleared { event_id: String },
    #[serde(rename = "input_audio_buffer.speech_started")]
    SpeechStarted { event_id: String, audio_start_ms: u64, item_id: String },
    #[serde(rename = "input_audio_buffer.speech_stopped")]
    SpeechStopped { event_id: String, audio_end_ms: u64, item_id: String },
    #[serde(rename = "conversation.item.input_audio_transcription.delta")]
    Delta { event_id: String, item_id: String, content_index: u32, delta: String, logprobs: Option<Value> },
    #[serde(rename = "conversation.item.input_audio_transcription.completed")]
    Completed { event_id: String, item_id: String, content_index: u32, transcript: String, usage: Value, logprobs: Option<Value> },
    #[serde(rename = "conversation.item.input_audio_transcription.failed")]
    Failed { event_id: String, item_id: String, content_index: u32, error: ErrorBody },
    #[serde(rename = "error")]
    Error { event_id: String, error: ErrorBody },
}

impl ServerEvent {
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{\"type\":\"error\"}".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_four_client_events() {
        let ev = parse(r#"{"type":"input_audio_buffer.append","event_id":"e1","audio":"AAA="}"#).expect("append");
        assert!(matches!(&ev, ClientEvent::Append { audio, .. } if audio == "AAA="));
        assert_eq!(ev.event_id().as_deref(), Some("e1"));
        assert!(matches!(parse(r#"{"type":"input_audio_buffer.commit"}"#), Ok(ClientEvent::Commit { event_id: None })));
        assert!(matches!(parse(r#"{"type":"input_audio_buffer.clear"}"#), Ok(ClientEvent::Clear { .. })));
        assert!(matches!(parse(r#"{"type":"session.update","session":{}}"#), Ok(ClientEvent::SessionUpdate { .. })));
    }

    #[test]
    fn bad_events_name_their_problem() {
        assert_eq!(parse("{nope").expect_err("json").code, "invalid_json");
        assert_eq!(parse(r#"{"event_id":"x"}"#).expect_err("type").code, "missing_type");
        let e = parse(r#"{"type":"response.create","event_id":"x"}"#).expect_err("unknown");
        assert_eq!((e.code.as_str(), e.event_id.as_deref()), ("unknown_event", Some("x")));
        // a known type with a missing field
        assert_eq!(parse(r#"{"type":"input_audio_buffer.append"}"#).expect_err("audio").code, "invalid_event");
    }

    #[test]
    fn turn_detection_tells_absent_from_null() {
        let td = |json: &str| match parse(json).expect("update") {
            ClientEvent::SessionUpdate { session, .. } => session.audio.and_then(|a| a.input).expect("input").turn_detection,
            _ => unreachable!(),
        };
        assert!(td(r#"{"type":"session.update","session":{"audio":{"input":{}}}}"#).is_none());
        assert!(matches!(td(r#"{"type":"session.update","session":{"audio":{"input":{"turn_detection":null}}}}"#), Some(None)));
        let on = td(r#"{"type":"session.update","session":{"audio":{"input":{"turn_detection":{"type":"server_vad","silence_duration_ms":400,"threshold":0.5}}}}}"#);
        assert_eq!(on.and_then(|t| t).and_then(|t| t.silence_duration_ms), Some(400));
    }

    #[test]
    fn server_events_serialize_like_openai() {
        let e = ServerEvent::Committed { event_id: "event_1".into(), item_id: "item_1".into(), previous_item_id: None };
        assert_eq!(serde_json::from_str::<Value>(&e.to_json()).expect("json"), serde_json::json!({"type": "input_audio_buffer.committed", "event_id": "event_1", "item_id": "item_1", "previous_item_id": null}));
        let d = ServerEvent::Delta { event_id: "event_2".into(), item_id: "item_1".into(), content_index: 0, delta: "ahoj".into(), logprobs: None };
        let v: Value = serde_json::from_str(&d.to_json()).expect("json");
        assert_eq!(v["type"], "conversation.item.input_audio_transcription.delta");
        assert_eq!(v["delta"], "ahoj");
    }
}
