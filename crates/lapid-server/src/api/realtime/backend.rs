//! The production [`Backend`]: streams live on a model's worker thread (the model never
//! leaves it), a session reaches them through queued jobs; and the choice of model.

use lapid::{Error, Result};
use lapid_hub::runtime::serve::{Capability, StreamParams, StreamUpdate};

use super::events::EventError;
use super::session::{Backend, Models};
use crate::api::Shared;
use crate::worker::Worker;

/// Streaming recognizers of one served model.
pub struct WorkerBackend {
    worker: Worker,
}

/// An open stream on a worker thread; dropping it (a finished turn, a cleared buffer, a closed
/// socket) queues its close.
pub struct WorkerStream {
    worker: Worker,
    id: u64,
}

impl Drop for WorkerStream {
    fn drop(&mut self) {
        let id = self.id;
        let _ = self.worker.submit(move |m| m.stream_close(id));
    }
}

impl Backend for WorkerBackend {
    type Handle = WorkerStream;

    async fn open(&self, params: StreamParams) -> Result<WorkerStream> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.worker.submit(move |m| {
            let opened = m.stream_open(&params);
            // The session went away while this ran: do not leak the stream.
            if let Err(Ok(id)) = tx.send(opened) {
                m.stream_close(id);
            }
        })?;
        let id = rx.await.map_err(|_| Error::new("internal", "the request was dropped by the model worker"))??;
        Ok(WorkerStream { worker: self.worker.clone(), id })
    }

    async fn accept(&self, stream: &WorkerStream, samples: Vec<f32>) -> Result<Vec<StreamUpdate>> {
        let id = stream.id;
        self.worker.call(move |m| m.stream_accept(id, &samples)).await
    }

    async fn finish(&self, stream: &WorkerStream) -> Result<Vec<StreamUpdate>> {
        let id = stream.id;
        self.worker.call(move |m| m.stream_finish(id)).await
    }
}

/// A loaded model as far as choosing one is concerned.
struct Candidate<'a> {
    id: &'a str,
    name: &'a str,
    streaming: bool,
}

fn not_found(message: String) -> EventError {
    EventError::invalid("model_not_found", message)
}

/// The model a client asked for. Like the other endpoints, a server with a single model that can
/// stream accepts any name (clients send names such as `gpt-4o-transcribe`); a named model that
/// cannot stream is an error, not a fallback.
fn choose(models: &[Candidate], requested: Option<&str>) -> std::result::Result<usize, EventError> {
    let able: Vec<usize> = (0..models.len()).filter(|&i| models[i].streaming).collect();
    let list = || able.iter().map(|&i| models[i].id).collect::<Vec<_>>().join(", ");
    if let Some(name) = requested.filter(|m| !m.is_empty()) {
        if let Some(i) = models.iter().position(|m| m.id == name || m.name == name) {
            return if models[i].streaming {
                Ok(i)
            } else {
                Err(EventError::invalid("unsupported_model", format!("model {name:?} does not support streaming transcription (models that do: {})", list())))
            };
        }
        return match able.as_slice() {
            [i] => Ok(*i),
            _ => Err(not_found(format!("model {name:?} is not loaded (models with streaming transcription: {})", list()))),
        };
    }
    match able.as_slice() {
        [i] => Ok(*i),
        [] => Err(not_found("no loaded model supports streaming transcription".into())),
        _ => Err(EventError::invalid("model_required", format!("\"model\" is required when several models can stream ({})", list()))),
    }
}

/// The models of the running server.
pub struct AppModels {
    pub state: Shared,
}

impl Models for AppModels {
    type Backend = WorkerBackend;

    fn resolve(&self, model: Option<&str>) -> std::result::Result<(WorkerBackend, String), EventError> {
        let candidates: Vec<Candidate> = self
            .state
            .workers
            .iter()
            .map(|w| {
                let m = w.meta();
                Candidate { id: &m.id, name: &m.name, streaming: m.has(Capability::Transcription) && m.extra.get("streaming").and_then(|v| v.as_bool()) == Some(true) }
            })
            .collect();
        let worker = &self.state.workers[choose(&candidates, model)?];
        Ok((WorkerBackend { worker: worker.clone() }, worker.meta().id.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c<'a>(id: &'a str, name: &'a str, streaming: bool) -> Candidate<'a> {
        Candidate { id, name, streaming }
    }

    #[test]
    fn a_single_streaming_model_is_the_default_and_takes_any_name() {
        let models = [c("whisper", "Whisper", false), c("zipformer", "Zipformer", true)];
        assert_eq!(choose(&models, None).ok(), Some(1));
        assert_eq!(choose(&models, Some("")).ok(), Some(1));
        assert_eq!(choose(&models, Some("zipformer")).ok(), Some(1));
        assert_eq!(choose(&models, Some("Zipformer")).ok(), Some(1), "by display name too");
        assert_eq!(choose(&models, Some("gpt-4o-transcribe")).ok(), Some(1));
    }

    #[test]
    fn a_named_model_that_cannot_stream_is_an_error() {
        let models = [c("whisper", "Whisper", false), c("zipformer", "Zipformer", true)];
        let e = choose(&models, Some("whisper")).expect_err("whisper cannot stream");
        assert_eq!(e.code, "unsupported_model");
        assert!(e.message.contains("zipformer"), "{}", e.message);
    }

    #[test]
    fn several_streaming_models_need_a_name() {
        let models = [c("a", "A", true), c("b", "B", true)];
        assert_eq!(choose(&models, None).expect_err("ambiguous").code, "model_required");
        assert_eq!(choose(&models, Some("b")).ok(), Some(1));
        assert_eq!(choose(&models, Some("c")).expect_err("unknown").code, "model_not_found");
    }

    #[test]
    fn no_streaming_model() {
        assert_eq!(choose(&[c("whisper", "Whisper", false)], None).expect_err("none").code, "model_not_found");
        assert_eq!(choose(&[], Some("x")).expect_err("empty").code, "model_not_found");
    }
}
