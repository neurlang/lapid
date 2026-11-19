//! One realtime transcription session: the protocol state machine, independent of sockets and
//! models. A [`Session`] turns client events into server events, driving streaming recognizers
//! through a [`Backend`] (the worker thread in production, a script in tests).
//!
//! A *turn* is one input buffer: in manual mode (no `turn_detection`) it runs from the first
//! append to `commit` or `clear`, and its `item_id` exists from the first append, so deltas
//! stream before the commit. With server turn detection the recognizer's endpoints close
//! segments, each an item, and the stream runs until `commit` or `clear`.

use std::future::Future;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use lapid_hub::runtime::endpoint::EndpointConfig;
use lapid_hub::runtime::serve::{Delay, StreamParams, StreamUpdate};
use serde_json::{Value, json};

use super::events::{ClientEvent, ErrorBody, EventError, FormatConfig, ServerEvent, SessionUpdate, TurnDetection, parse};

use audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, Resampler, FixedSync};

/// The rate recognizers operate at: 16-bit little-endian mono PCM at 16 kHz. Input audio
/// arriving at 24 kHz is downsampled to this rate before it reaches the recognizer.
pub const SAMPLE_RATE: u32 = 16_000;

/// The (only) supported non-native input rate.
pub const INPUT_RATE_24K: u32 = 24_000;

/// Recognizers of one model.
pub trait Backend: Send + Sync {
    /// An open stream; dropping it closes the stream.
    type Handle: Send;

    fn open(&self, params: StreamParams) -> impl Future<Output = lapid::Result<Self::Handle>> + Send;
    fn accept(&self, stream: &Self::Handle, samples: Vec<f32>) -> impl Future<Output = lapid::Result<Vec<StreamUpdate>>> + Send;
    fn finish(&self, stream: &Self::Handle) -> impl Future<Output = lapid::Result<Vec<StreamUpdate>>> + Send;
}

/// The models a connection may choose from.
pub trait Models: Send {
    type Backend: Backend;

    /// The backend and id of `model` (`None` = the default model).
    fn resolve(&self, model: Option<&str>) -> Result<(Self::Backend, String), EventError>;
}

/// Server turn detection: segments close at the recognizer's endpoints.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Vad {
    pub silence_ms: Option<u32>,
}

impl Vad {
    /// Endpoint rules: the model's defaults, with `silence_duration_ms` as the silence that
    /// closes a segment after speech.
    fn endpoint(self) -> EndpointConfig {
        let mut e = EndpointConfig::default();
        if let Some(ms) = self.silence_ms {
            let secs = ms as f32 / 1000.0;
            e.rule2.min_trailing_silence = secs;
            e.rule1.min_trailing_silence = e.rule1.min_trailing_silence.max(secs);
        }
        e
    }
}

/// What `session.update` configures.
#[derive(Clone, Debug, Default)]
pub struct Settings {
    pub language: Option<String>,
    pub delay: Delay,
    pub vad: Option<Vad>,
}

impl Settings {
    fn params(&self) -> StreamParams {
        StreamParams { delay: self.delay, endpoint: self.vad.map(Vad::endpoint) }
    }
}

/// Event and item ids, unique within a connection.
pub struct Ids {
    prefix: String,
    n: u64,
}

impl Ids {
    fn next(&mut self, kind: &str) -> String {
        self.n += 1;
        format!("{kind}_{}{:05}", self.prefix, self.n)
    }

    fn event(&mut self) -> String {
        self.next("event")
    }

    fn item(&mut self) -> String {
        self.next("item")
    }
}

struct Item {
    id: String,
    /// Text already sent as deltas.
    emitted: String,
}

/// An open recognizer stream and the item it is currently transcribing.
struct Live<H> {
    handle: H,
    item: Option<Item>,
    /// Samples fed so far.
    samples: u64,
}

/// Turn bookkeeping, apart from the backend so both can be borrowed together.
struct Turns<H> {
    ids: Ids,
    live: Option<Live<H>>,
    last_item: Option<String>,
    /// An odd trailing byte of the previous append.
    carry: Option<u8>,
}

fn error_body(e: &EventError) -> ErrorBody {
    ErrorBody { kind: e.kind.into(), code: e.code.clone(), message: e.message.clone(), param: None, event_id: e.event_id.clone() }
}

fn backend_error(e: &lapid::Error) -> EventError {
    EventError::server(e.code, e.message.clone())
}

fn completed(ids: &mut Ids, item_id: &str, transcript: String, seconds: f64) -> ServerEvent {
    let seconds = (seconds.max(0.0) * 100.0).round() / 100.0;
    ServerEvent::Completed {
        event_id: ids.event(),
        item_id: item_id.to_string(),
        content_index: 0,
        transcript,
        usage: json!({"type": "duration", "seconds": seconds}),
        logprobs: None,
    }
}

impl<H> Turns<H> {
    fn error(&mut self, e: EventError) -> ServerEvent {
        ServerEvent::Error { event_id: self.ids.event(), error: error_body(&e) }
    }

    fn new_item(&mut self) -> Item {
        Item { id: self.ids.item(), emitted: String::new() }
    }

    /// Base64 of 16-bit little-endian PCM to samples; an odd byte waits for the next append.
    fn decode(&mut self, audio: &str) -> Result<Vec<f32>, EventError> {
        let mut bytes = STANDARD.decode(audio.trim()).map_err(|_| EventError::invalid("invalid_value", "audio is not valid base64"))?;
        if let Some(b) = self.carry.take() {
            bytes.insert(0, b);
        }
        if bytes.len() % 2 == 1 {
            self.carry = bytes.pop();
        }
        Ok(bytes.as_chunks::<2>().0.iter().map(|p| f32::from(i16::from_le_bytes(*p)) / 32768.0).collect())
    }

    /// The turn failed: report its item as failed, drop the stream, then the error itself.
    fn abort(&mut self, e: EventError) -> Vec<ServerEvent> {
        let mut out = Vec::new();
        if let Some(item) = self.live.take().and_then(|l| l.item) {
            out.push(ServerEvent::Failed { event_id: self.ids.event(), item_id: item.id, content_index: 0, error: error_body(&e) });
        }
        out.push(self.error(e));
        out
    }

    /// Events for recognizer updates: a delta for text that extends what was sent, and at the
    /// end of a segment `completed` (with `speech_stopped` and `committed` first when the
    /// server detects turns; manual turns send `committed` at the commit).
    fn apply(&mut self, vad: bool, updates: Vec<StreamUpdate>) -> Vec<ServerEvent> {
        let mut out = Vec::new();
        let Some(live) = self.live.as_mut() else { return out };
        let fed = live.samples as f64 / f64::from(SAMPLE_RATE);
        for u in updates {
            if live.item.is_none() {
                if u.is_final && u.text.is_empty() {
                    continue; // a flush of silence
                }
                let id = self.ids.item();
                if vad {
                    out.push(ServerEvent::SpeechStarted { event_id: self.ids.event(), audio_start_ms: (u.start * 1000.0) as u64, item_id: id.clone() });
                }
                live.item = Some(Item { id, emitted: String::new() });
            }
            let Some(item) = live.item.as_mut() else { continue };
            // Greedy decoding only appends; text that does not extend what was sent (a partly
            // decoded character) waits for `completed`, which carries the whole transcript.
            if let Some(rest) = u.text.strip_prefix(item.emitted.as_str()).filter(|r| !r.is_empty()) {
                out.push(ServerEvent::Delta { event_id: self.ids.event(), item_id: item.id.clone(), content_index: 0, delta: rest.to_string(), logprobs: None });
                item.emitted.clone_from(&u.text);
            }
            if u.is_final {
                let Some(item) = live.item.take() else { continue };
                if vad {
                    out.push(ServerEvent::SpeechStopped { event_id: self.ids.event(), audio_end_ms: (fed * 1000.0) as u64, item_id: item.id.clone() });
                    out.push(ServerEvent::Committed { event_id: self.ids.event(), item_id: item.id.clone(), previous_item_id: self.last_item.clone() });
                }
                out.push(completed(&mut self.ids, &item.id, u.text, fed - f64::from(u.start)));
                self.last_item = Some(item.id);
            }
        }
        out
    }
}

/// 24 kHz → 16 kHz downsampler (ratio 3:2). Buffers internally so callers pass chunks of any size.
///
/// Uses `rubato`'s sinc-interpolation resampler, which has only a sinc-window of latency
/// (a few ms) rather than a full chunk, so it stays roughly proportional to input length.
pub struct Downsampler24to16 {
    inner: Fft<f32>,
}

impl Downsampler24to16 {
    pub fn new() -> Self {
        // `chunk_size` only sizes rubato's internal working buffer; input length is free.
        let inner = Fft::<f32>::new(24_000, 16_000, 1024, 1, FixedSync::Both)
            .expect("24 kHz -> 16 kHz resampler parameters are valid");
        Self { inner }
    }

    /// Resample `input` (24 kHz) to 16 kHz. Returns 0 samples on the first few calls until the
    /// sinc window is primed.
    pub fn process(&mut self, input: Vec<f32>) -> Result<Vec<f32>, lapid::Error> {
        if input.is_empty() {
            return Ok(Vec::new());
        }
        let frames = input.len();               // mono => frames == samples
        let input = InterleavedSlice::new(&input, 1, frames)
            .map_err(|e| lapid::Error::new("resample", format!("{e}")))?;
        let out = self
            .inner
            .process(&input, None)
            .map_err(|e| lapid::Error::new("resample", format!("{e}")))?;
        Ok(out.take_data())                     // InterleavedOwned<f32> -> Vec<f32>
    }

    /// Feed trailing silence to flush the sinc window; the returned tail ends in silence but
    /// the recognizer sees the correct last samples plus a short trailing quiet, which is
    /// harmless (and often helps VAD fire the endpoint).
    pub fn flush(&mut self) -> Result<Vec<f32>, lapid::Error> {
        let silence = vec![0.0f32; 512];
        let frames = silence.len();
        let input = InterleavedSlice::new(&silence, 1, frames)
            .map_err(|e| lapid::Error::new("resample", format!("{e}")))?;
        let out = self
            .inner
            .process(&input, None)
            .map_err(|e| lapid::Error::new("resample", format!("{e}")))?;
        Ok(out.take_data())
    }
}

/// One connection's session.
pub struct Session<M: Models> {
    models: M,
    backend: Option<(M::Backend, String)>,
    settings: Settings,
    session_id: String,
    turns: Turns<<M::Backend as Backend>::Handle>,
    /// Native input rate; `SAMPLE_RATE` or `INPUT_RATE_24K`.
    input_rate: u32,
    /// `Some` iff `input_rate != SAMPLE_RATE` (i.e. the client is sending 24 kHz audio).
    resampler: Option<Downsampler24to16>,
}

fn no_model() -> EventError {
    EventError::invalid("no_model", "no model is selected: set audio.input.transcription.model in session.update")
}

fn check_format(f: &FormatConfig) -> Result<u32, EventError> {
    if let Some(k) = f.kind.as_deref().filter(|k| *k != "audio/pcm") {
        return Err(EventError::invalid(
            "unsupported_audio_format",
            format!("input audio format {k:?} is not supported: send \"audio/pcm\" (16-bit little-endian mono)"),
        ));
    }
    let rate = f.rate.unwrap_or(SAMPLE_RATE);
    if !matches!(rate, SAMPLE_RATE | INPUT_RATE_24K) {
        return Err(EventError::invalid(
            "unsupported_audio_format",
            format!(
                "sample rate {rate} is not supported: send {SAMPLE_RATE} Hz or {INPUT_RATE_24K} Hz audio"
            ),
        ));
    }
    Ok(rate)
}

fn parse_language(l: &str) -> Result<Option<String>, EventError> {
    if l.is_empty() {
        return Ok(None);
    }
    if l.len() > 35 || !l.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
        return Err(EventError::invalid("invalid_value", "language must be an ISO 639 code or a regional tag such as \"sk\" or \"zh-TW\""));
    }
    Ok(Some(l.to_string()))
}

fn parse_vad(td: &TurnDetection) -> Result<Vad, EventError> {
    if let Some(k) = td.kind.as_deref().filter(|k| !matches!(*k, "server_vad" | "semantic_vad")) {
        return Err(EventError::invalid("invalid_value", format!("turn_detection type {k:?} is not supported (use \"server_vad\", or null for manual commits)")));
    }
    if let Some(ms) = td.silence_duration_ms.filter(|ms| !(100..=10_000).contains(ms)) {
        return Err(EventError::invalid("invalid_value", format!("silence_duration_ms {ms} must be between 100 and 10000")));
    }
    Ok(Vad { silence_ms: td.silence_duration_ms })
}

impl<M: Models> Session<M> {
    pub fn new(models: M, prefix: String, model: Option<&str>) -> Result<Session<M>, EventError> {
        let backend = match model {
            Some(m) => Some(models.resolve(Some(m))?),
            None => models.resolve(None).ok(),
        };
        Ok(Session {
            models,
            backend,
            settings: Settings::default(),
            session_id: format!("sess_{prefix}"),
            turns: Turns { ids: Ids { prefix, n: 0 }, live: None, last_item: None, carry: None },
            input_rate: SAMPLE_RATE,
            resampler: None,
        })
    }

    /// The session as clients see it: only what is applied (no `prompt`, no `keywords`).
    fn session_json(&self) -> Value {
        let turn_detection = self.settings.vad.map(|v| json!({"type": "server_vad", "silence_duration_ms": (v.endpoint().rule2.min_trailing_silence * 1000.0).round() as u32}));
        json!({
            "id": self.session_id,
            "object": "realtime.transcription_session",
            "type": "transcription",
            "audio": {"input": {
                "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                "transcription": {
                    "model": self.backend.as_ref().map(|b| b.1.as_str()),
                    "language": self.settings.language,
                    "delay": self.settings.delay.name(),
                },
                "turn_detection": turn_detection,
            }},
        })
    }

    /// The `session.created` event sent on connect.
    pub fn created(&mut self) -> ServerEvent {
        ServerEvent::SessionCreated { event_id: self.turns.ids.event(), session: self.session_json() }
    }

    /// A protocol error outside any event (e.g. a binary frame).
    pub fn protocol_error(&mut self, code: &str, message: &str) -> ServerEvent {
        self.turns.error(EventError::invalid(code, message))
    }

    /// Handle one text frame; the events to send back, in order.
    pub async fn handle_text(&mut self, text: &str) -> Vec<ServerEvent> {
        let event = match parse(text) {
            Ok(e) => e,
            Err(e) => return vec![self.turns.error(e)],
        };
        let id = event.event_id().clone();
        let result = match event {
            ClientEvent::SessionUpdate { session, .. } => self.update(session).map(|()| vec![ServerEvent::SessionUpdated { event_id: self.turns.ids.event(), session: self.session_json() }]),
            ClientEvent::Append { audio, .. } => self.append(&audio).await,
            ClientEvent::Commit { .. } => self.commit().await,
            ClientEvent::Clear { .. } => Ok(self.clear()),
        };
        result.unwrap_or_else(|e| vec![self.turns.error(e.at(&id))])
    }

    /// Validate the whole update, then apply it.
    fn update(&mut self, upd: SessionUpdate) -> Result<(), EventError> {
        if upd.kind.as_deref().is_some_and(|k| k != "transcription") {
            return Err(EventError::invalid(
                "invalid_value",
                "only transcription sessions are supported (\"type\": \"transcription\")",
            ));
        }
        let mut next = self.settings.clone();
        let mut model = None;
        let mut new_rate = self.input_rate;
        if let Some(input) = upd.audio.and_then(|a| a.input) {
            if let Some(f) = &input.format {
                new_rate = check_format(f)?;
            }
            if let Some(t) = &input.transcription {
                if let Some(d) = &t.delay {
                    next.delay = Delay::parse(d).ok_or_else(|| {
                        EventError::invalid(
                            "invalid_value",
                            format!("delay {d:?} is not one of minimal, low, medium, high, xhigh"),
                        )
                    })?;
                }
                if let Some(l) = &t.language {
                    next.language = parse_language(l)?;
                }
                if let Some(m) = t.model.as_deref().filter(|m| !m.is_empty()) {
                    let (backend, id) = self.models.resolve(Some(m))?;
                    if self.backend.as_ref().map(|b| b.1.as_str()) != Some(id.as_str()) {
                        model = Some((backend, id));
                    }
                }
            }
            match input.turn_detection {
                None => {}
                Some(None) => next.vad = None,
                Some(Some(td)) => next.vad = Some(parse_vad(&td)?),
            }
        }
        if self.turns.live.is_some()
            && (next.delay != self.settings.delay
                || next.vad != self.settings.vad
                || model.is_some()
                || new_rate != self.input_rate)
        {
            return Err(EventError::invalid(
                "session_busy",
                "delay, turn detection, model and input format cannot change while audio is buffered: \
                 commit or clear the input buffer first",
            ));
        }
        self.settings = next;
        if model.is_some() {
            self.backend = model;
        }
        if new_rate != self.input_rate {
            self.input_rate = new_rate;
            self.resampler = if new_rate == SAMPLE_RATE {
                None
            } else {
                Some(Downsampler24to16::new())
            };
        }
        Ok(())
    }

    async fn append(&mut self, audio: &str) -> Result<Vec<ServerEvent>, EventError> {
        let samples = self.turns.decode(audio)?;
        if samples.is_empty() {
            return Ok(Vec::new());
        }
        // Downsample if the client is sending 24 kHz audio.
        let samples = match &mut self.resampler {
            Some(rs) => rs.process(samples).map_err(|e| backend_error(&e))?,
            None => samples,
        };
        if samples.is_empty() {
            // Resampler is priming its sinc window; nothing to feed the recognizer yet.
            return Ok(Vec::new());
        }
        let backend = self.backend.as_ref().map(|b| &b.0).ok_or_else(no_model)?;
        let vad = self.settings.vad.is_some();
        if self.turns.live.is_none() {
            let handle = backend
                .open(self.settings.params())
                .await
                .map_err(|e| backend_error(&e))?;
            let item = (!vad).then(|| self.turns.new_item());
            self.turns.live = Some(Live { handle, item, samples: 0 });
        }
        let live = self
            .turns
            .live
            .as_mut()
            .ok_or_else(|| EventError::server("internal_error", "the stream is missing"))?;
        live.samples += samples.len() as u64;
        Ok(match backend.accept(&live.handle, samples).await {
            Ok(updates) => self.turns.apply(vad, updates),
            Err(e) => self.turns.abort(backend_error(&e)),
        })
    }

    async fn commit(&mut self) -> Result<Vec<ServerEvent>, EventError> {
        let backend = self.backend.as_ref().map(|b| &b.0).ok_or_else(no_model)?;
        let vad = self.settings.vad.is_some();
        let mut out = Vec::new();

        // Flush the resampler's sinc tail into the recognizer *before* finishing, so the last
        // few milliseconds of audio aren't dropped. May open the stream if the resampler had
        // buffered everything so far (e.g. a sub-sinc-window turn).
        if let Some(rs) = &mut self.resampler {
            let flushed = rs.flush().map_err(|e| backend_error(&e))?;
            if !flushed.is_empty() {
                if self.turns.live.is_none() {
                    let handle = backend
                        .open(self.settings.params())
                        .await
                        .map_err(|e| backend_error(&e))?;
                    let item = (!vad).then(|| self.turns.new_item());
                    self.turns.live = Some(Live { handle, item, samples: 0 });
                }
                let live = self
                    .turns
                    .live
                    .as_mut()
                    .ok_or_else(|| EventError::server("internal_error", "the stream is missing"))?;
                live.samples += flushed.len() as u64;
                let result = backend.accept(&live.handle, flushed).await;
                match result {
                    Ok(updates) => out.extend(self.turns.apply(vad, updates)),
                    Err(e) => {
                        out.extend(self.turns.abort(backend_error(&e)));
                        return Ok(out);
                    }
                }
            }
        }

        let Some(live) = self.turns.live.as_ref() else {
            return Err(EventError::invalid(
                "input_audio_buffer_commit_empty",
                "Error committing input audio buffer: the buffer is empty.",
            ));
        };
        if !vad && let Some(item) = &live.item {
            out.push(ServerEvent::Committed {
                event_id: self.turns.ids.event(),
                item_id: item.id.clone(),
                previous_item_id: self.turns.last_item.clone(),
            });
        }
        match backend.finish(&live.handle).await {
            Ok(updates) => out.extend(self.turns.apply(vad, updates)),
            Err(e) => {
                out.extend(self.turns.abort(backend_error(&e)));
                return Ok(out);
            }
        }
        if let Some(live) = self.turns.live.as_mut()
            && let Some(item) = live.item.take()
        {
            let seconds = live.samples as f64 / f64::from(SAMPLE_RATE);
            out.push(completed(&mut self.turns.ids, &item.id, item.emitted, seconds));
            self.turns.last_item = Some(item.id);
        }
        self.turns.live = None;
        // Fresh resampler for the next turn — the tail we just flushed belongs to the
        // stream we just closed.
        if self.resampler.is_some() {
            self.resampler = Some(Downsampler24to16::new());
        }
        Ok(out)
    }

    fn clear(&mut self) -> Vec<ServerEvent> {
        self.turns.live = None;
        self.turns.carry = None;
        if self.resampler.is_some() {
            self.resampler = Some(Downsampler24to16::new());
        }
        vec![ServerEvent::Cleared { event_id: self.turns.ids.event() }]
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use super::*;

    #[derive(Default)]
    struct Script {
        accept: VecDeque<Vec<StreamUpdate>>,
        finish: VecDeque<Vec<StreamUpdate>>,
        params: Vec<StreamParams>,
        fed: Vec<usize>,
        fail_accept: bool,
    }

    /// A scripted recognizer: each call pops the next prepared reply.
    #[derive(Clone, Default)]
    struct Fake {
        script: Arc<Mutex<Script>>,
        closed: Arc<AtomicUsize>,
    }

    struct Handle(Arc<AtomicUsize>);

    impl Drop for Handle {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl Backend for Fake {
        type Handle = Handle;

        fn open(&self, params: StreamParams) -> impl Future<Output = lapid::Result<Handle>> + Send {
            self.script.lock().expect("script").params.push(params);
            let closed = self.closed.clone();
            async move { Ok(Handle(closed)) }
        }

        fn accept(&self, _: &Handle, samples: Vec<f32>) -> impl Future<Output = lapid::Result<Vec<StreamUpdate>>> + Send {
            let mut s = self.script.lock().expect("script");
            s.fed.push(samples.len());
            let r = if s.fail_accept { Err(lapid::Error::new("internal", "boom")) } else { Ok(s.accept.pop_front().unwrap_or_default()) };
            async move { r }
        }

        fn finish(&self, _: &Handle) -> impl Future<Output = lapid::Result<Vec<StreamUpdate>>> + Send {
            let r = Ok(self.script.lock().expect("script").finish.pop_front().unwrap_or_default());
            async move { r }
        }
    }

    /// Models "zipformer" (the default) and "other".
    struct Pair(Fake, Fake);

    impl Models for Pair {
        type Backend = Fake;

        fn resolve(&self, model: Option<&str>) -> Result<(Fake, String), EventError> {
            match model {
                None | Some("zipformer") => Ok((self.0.clone(), "zipformer".into())),
                Some("other") => Ok((self.1.clone(), "other".into())),
                Some(m) => Err(EventError::invalid("model_not_found", format!("model {m:?} is not loaded"))),
            }
        }
    }

    fn upd(text: &str, is_final: bool, segment: usize) -> StreamUpdate {
        StreamUpdate { segment, text: text.into(), is_final, start: 0.0 }
    }

    fn session(fake: &Fake) -> Session<Pair> {
        Session::new(Pair(fake.clone(), Fake::default()), "t".into(), None).expect("session")
    }

    fn append(samples: usize) -> String {
        json!({"type": "input_audio_buffer.append", "audio": STANDARD.encode(vec![0u8; samples * 2])}).to_string()
    }

    fn cmd(kind: &str) -> String {
        json!({"type": kind}).to_string()
    }

    fn kinds(events: &[ServerEvent]) -> Vec<String> {
        events.iter().map(|e| serde_json::to_value(e).expect("json")["type"].as_str().expect("type").replace("conversation.item.input_audio_transcription.", "item.")).collect()
    }

    fn field(e: &ServerEvent, key: &str) -> Value {
        serde_json::to_value(e).expect("json")[key].clone()
    }

    #[tokio::test]
    async fn manual_turn_streams_deltas_then_completes() {
        let fake = Fake::default();
        {
            let mut s = fake.script.lock().expect("script");
            s.accept = [vec![upd("Ahoj", false, 0)], vec![upd("Ahoj svet", false, 0)]].into();
            s.finish = [vec![upd("Ahoj svet.", true, 0)]].into();
        }
        let mut s = session(&fake);
        assert_eq!(field(&s.created(), "session")["audio"]["input"]["transcription"]["model"], "zipformer");

        let e1 = s.handle_text(&append(1600)).await;
        assert_eq!(kinds(&e1), ["item.delta"]);
        assert_eq!(field(&e1[0], "delta"), "Ahoj");
        let e2 = s.handle_text(&append(1600)).await;
        assert_eq!(field(&e2[0], "delta"), " svet");

        let e3 = s.handle_text(&cmd("input_audio_buffer.commit")).await;
        assert_eq!(kinds(&e3), ["input_audio_buffer.committed", "item.delta", "item.completed"]);
        assert_eq!(field(&e3[1], "delta"), ".");
        assert_eq!(field(&e3[2], "transcript"), "Ahoj svet.");
        assert_eq!(field(&e3[2], "usage")["seconds"], 0.2);
        let id = field(&e1[0], "item_id");
        assert!([&e2[0], &e3[0], &e3[1], &e3[2]].iter().all(|e| field(e, "item_id") == id));
        assert_eq!(field(&e3[0], "previous_item_id"), Value::Null);
        assert_eq!(fake.closed.load(Ordering::SeqCst), 1, "the stream is closed after the commit");
    }

    #[tokio::test]
    async fn the_next_turn_chains_previous_item_id() {
        let fake = Fake::default();
        fake.script.lock().expect("script").finish = [vec![upd("jedna", true, 0)], vec![upd("dva", true, 0)]].into();
        let mut s = session(&fake);
        s.handle_text(&append(800)).await;
        let first = s.handle_text(&cmd("input_audio_buffer.commit")).await;
        s.handle_text(&append(800)).await;
        let second = s.handle_text(&cmd("input_audio_buffer.commit")).await;
        let (a, b) = (field(&first[0], "item_id"), field(&second[0], "item_id"));
        assert_ne!(a, b);
        assert_eq!(field(&second[0], "previous_item_id"), a);
        assert_eq!(fake.script.lock().expect("script").params.len(), 2, "one stream per turn");
    }

    #[tokio::test]
    async fn commit_and_clear_without_audio() {
        let fake = Fake::default();
        let mut s = session(&fake);
        let e = s.handle_text(&json!({"type": "input_audio_buffer.commit", "event_id": "c1"}).to_string()).await;
        assert_eq!(field(&e[0], "error")["code"], "input_audio_buffer_commit_empty");
        assert_eq!(field(&e[0], "error")["event_id"], "c1");
        s.handle_text(&append(800)).await;
        let cleared = s.handle_text(&cmd("input_audio_buffer.clear")).await;
        assert_eq!(kinds(&cleared), ["input_audio_buffer.cleared"]);
        assert_eq!(fake.closed.load(Ordering::SeqCst), 1);
        assert_eq!(field(&s.handle_text(&cmd("input_audio_buffer.commit")).await[0], "error")["code"], "input_audio_buffer_commit_empty");
    }

    #[tokio::test]
    async fn server_turn_detection_makes_an_item_per_segment() {
        let fake = Fake::default();
        {
            let mut s = fake.script.lock().expect("script");
            s.accept = [vec![upd("Ahoj", false, 0)], vec![upd("Ahoj.", true, 0), upd("Dobry", false, 1)]].into();
            s.finish = [vec![upd("Dobry den", true, 1)]].into();
        }
        let mut s = session(&fake);
        let up = json!({"type": "session.update", "session": {"type": "transcription", "audio": {"input": {"turn_detection": {"type": "server_vad", "silence_duration_ms": 800}}}}});
        let applied = s.handle_text(&up.to_string()).await;
        assert_eq!(field(&applied[0], "session")["audio"]["input"]["turn_detection"]["silence_duration_ms"], 800);

        let a = s.handle_text(&append(1600)).await;
        assert_eq!(kinds(&a), ["input_audio_buffer.speech_started", "item.delta"]);
        let b = s.handle_text(&append(1600)).await;
        assert_eq!(
            kinds(&b),
            ["item.delta", "input_audio_buffer.speech_stopped", "input_audio_buffer.committed", "item.completed", "input_audio_buffer.speech_started", "item.delta"]
        );
        let c = s.handle_text(&cmd("input_audio_buffer.commit")).await;
        assert_eq!(kinds(&c), ["item.delta", "input_audio_buffer.speech_stopped", "input_audio_buffer.committed", "item.completed"]);
        let (first, second) = (field(&a[0], "item_id"), field(&b[5], "item_id"));
        assert_ne!(first, second);
        assert_eq!(field(&b[2], "previous_item_id"), Value::Null);
        assert_eq!(field(&c[2], "previous_item_id"), first);
        assert_eq!(field(&c[3], "transcript"), "Dobry den");
        let ep = fake.script.lock().expect("script").params[0].endpoint.expect("endpoint");
        assert_eq!(ep.rule2.min_trailing_silence, 0.8);
    }

    #[tokio::test]
    async fn text_that_does_not_extend_waits_for_completed() {
        let fake = Fake::default();
        {
            let mut s = fake.script.lock().expect("script");
            s.accept = [vec![upd("ab", false, 0)], vec![upd("xyz", false, 0)]].into();
            s.finish = [vec![upd("xyz!", true, 0)]].into();
        }
        let mut s = session(&fake);
        assert_eq!(kinds(&s.handle_text(&append(100)).await), ["item.delta"]);
        assert!(s.handle_text(&append(100)).await.is_empty(), "no delta for a rewrite");
        let end = s.handle_text(&cmd("input_audio_buffer.commit")).await;
        assert_eq!(field(end.last().expect("completed"), "transcript"), "xyz!");
    }

    #[tokio::test]
    async fn a_recognizer_that_never_closes_the_item_still_completes_it() {
        let fake = Fake::default();
        fake.script.lock().expect("script").accept = [vec![upd("Ahoj", false, 0)]].into();
        let mut s = session(&fake);
        s.handle_text(&append(100)).await;
        let end = s.handle_text(&cmd("input_audio_buffer.commit")).await;
        assert_eq!(kinds(&end), ["input_audio_buffer.committed", "item.completed"]);
        assert_eq!(field(&end[1], "transcript"), "Ahoj");
    }

    #[tokio::test]
    async fn backend_failure_fails_the_item_and_frees_the_stream() {
        let fake = Fake::default();
        fake.script.lock().expect("script").fail_accept = true;
        let mut s = session(&fake);
        let e = s.handle_text(&append(100)).await;
        assert_eq!(kinds(&e), ["item.failed", "error"]);
        assert_eq!(field(&e[1], "error")["type"], "server_error");
        assert_eq!(fake.closed.load(Ordering::SeqCst), 1);
        assert_eq!(field(&s.handle_text(&cmd("input_audio_buffer.commit")).await[0], "error")["code"], "input_audio_buffer_commit_empty");
    }

    #[tokio::test]
    async fn audio_decoding_carries_odd_bytes_and_rejects_garbage() {
        let fake = Fake::default();
        let mut s = session(&fake);
        let raw = |bytes: &[u8]| json!({"type": "input_audio_buffer.append", "audio": STANDARD.encode(bytes)}).to_string();
        s.handle_text(&raw(&[1, 0, 2])).await; // one sample and a stray byte
        s.handle_text(&raw(&[0])).await; // completes the second sample
        assert_eq!(fake.script.lock().expect("script").fed, [1, 1]);
        let bad = s.handle_text(&json!({"type": "input_audio_buffer.append", "audio": "***"}).to_string()).await;
        assert_eq!(field(&bad[0], "error")["code"], "invalid_value");
        let junk = s.handle_text("not json").await;
        assert_eq!(field(&junk[0], "error")["code"], "invalid_json");
    }

    #[tokio::test]
    async fn session_update_validates_before_applying() {
        let fake = Fake::default();
        let mut s = session(&fake);
        let up = |input: Value| json!({"type": "session.update", "event_id": "u1", "session": {"type": "transcription", "audio": {"input": input}}}).to_string();

        let bad = s.handle_text(&up(json!({"transcription": {"delay": "fast"}}))).await;
        assert_eq!((field(&bad[0], "error")["code"].as_str(), field(&bad[0], "error")["event_id"].as_str()), (Some("invalid_value"), Some("u1")));
let rate = s.handle_text(&up(json!({"format": {"type": "audio/pcm", "rate": 22050}}))).await;
assert_eq!(field(&rate[0], "error")["code"], "unsupported_audio_format");
        let unknown = s.handle_text(&up(json!({"transcription": {"model": "nope"}}))).await;
        assert_eq!(field(&unknown[0], "error")["code"], "model_not_found");
        let still = s.handle_text(&up(json!({"transcription": {"language": "sk", "delay": "fast"}}))).await;
        assert_eq!(kinds(&still), ["error"]);

        let ok = s.handle_text(&up(json!({"transcription": {"model": "other", "language": "sk", "delay": "high", "prompt": "x", "keywords": ["a"]}}))).await;
        let session = field(&ok[0], "session");
        let t = &session["audio"]["input"]["transcription"];
        assert_eq!((t["model"].as_str(), t["language"].as_str(), t["delay"].as_str()), (Some("other"), Some("sk"), Some("high")));
        assert!(t.get("prompt").is_none() && t.get("keywords").is_none(), "unapplied options are not echoed");
        assert_eq!(session["audio"]["input"]["turn_detection"], Value::Null);
    }

    #[tokio::test]
    async fn settings_are_locked_while_audio_is_buffered() {
        let fake = Fake::default();
        let mut s = session(&fake);
        let up = |input: Value| json!({"type": "session.update", "session": {"audio": {"input": input}}}).to_string();
        s.handle_text(&up(json!({"transcription": {"delay": "minimal"}}))).await;
        s.handle_text(&append(100)).await;
        assert_eq!(field(&s.handle_text(&up(json!({"transcription": {"delay": "high"}}))).await[0], "error")["code"], "session_busy");
        assert_eq!(field(&s.handle_text(&up(json!({"transcription": {"model": "other"}}))).await[0], "error")["code"], "session_busy");
        assert_eq!(kinds(&s.handle_text(&up(json!({"transcription": {"delay": "minimal", "language": "cs"}}))).await), ["session.updated"]);
        s.handle_text(&cmd("input_audio_buffer.clear")).await;
        s.handle_text(&up(json!({"transcription": {"delay": "high"}}))).await;
        s.handle_text(&append(100)).await;
        let params = fake.script.lock().expect("script").params.clone();
        assert_eq!((params[0].delay, params[1].delay), (Delay::Minimal, Delay::High));
    }

    #[tokio::test]
    async fn only_transcription_sessions() {
        let mut s = session(&Fake::default());
        for session in [json!({"type": "realtime"}), json!({"audio": {"input": {"turn_detection": {"type": "magic"}}}})] {
            let e = s.handle_text(&json!({"type": "session.update", "session": session}).to_string()).await;
            assert_eq!(field(&e[0], "error")["code"], "invalid_value");
        }
    }

    #[tokio::test]
    async fn accepts_24khz_and_downsamples() {
        let fake = Fake::default();
        fake.script.lock().expect("script").finish = [vec![upd("Ahoj", true, 0)]].into();
        let mut s = session(&fake);
        let up = json!({
            "type": "session.update",
            "session": {"audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000}}}},
        });
        let applied = s.handle_text(&up.to_string()).await;
        assert_eq!(kinds(&applied), ["session.updated"]);

        // 720 samples @ 24 kHz => ~480 samples @ 16 kHz (plus sinc tail flushed on commit).
        s.handle_text(&append(720)).await;
        let end = s.handle_text(&cmd("input_audio_buffer.commit")).await;

        let total: usize = fake.script.lock().expect("script").fed.iter().sum();
        assert!(total >= 480, "expected >= 480 samples at 16 kHz, got {total}");
        assert!(total <= 720, "expected <= 720 samples at 16 kHz, got {total}");
        assert_eq!(field(end.last().expect("completed"), "transcript"), "Ahoj");
    }

    #[tokio::test]
    async fn rate_cannot_change_while_audio_is_buffered() {
        let fake = Fake::default();
        let mut s = session(&fake);
        s.handle_text(&append(100)).await;
        let up = json!({
            "type": "session.update",
            "session": {"audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000}}}},
        });
        assert_eq!(field(&s.handle_text(&up.to_string()).await[0], "error")["code"], "session_busy");
    }
}
