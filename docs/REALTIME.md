# Realtime transcription

`lapid serve` speaks the transcription-session subset of the OpenAI Realtime API over a
WebSocket, for models that can stream (the Zipformer transducers):

```
ws://localhost:8080/v1/realtime?intent=transcription[&model=ID]
```

Serve it behind a reverse proxy for `wss://`. `intent` must be `transcription` (the default);
`model` may be omitted when one loaded model can stream (any name is then accepted, as at the
other endpoints) and can also be set by `session.update`. A model that cannot stream is
refused (`/v1/models` lists what is loaded; streaming models have `"streaming": true` in
`/api/info`).

## Audio

`input_audio_buffer.append` carries base64 of **16-bit little-endian mono PCM at 16 kHz**
(`{"type": "audio/pcm", "rate": 16000}`). Other formats and rates are rejected with
`unsupported_audio_format`; resample on the client. Chunks may have any size and may split a
sample. Events are handled one at a time, so a client that sends faster than real time is
slowed down by TCP.

## Session

On connect the server sends `session.created`. `session.update` configures the session; the
whole update is validated before any of it applies, and `session.updated` echoes only what is
applied:

```json
{"type": "session.update", "session": {"type": "transcription", "audio": {"input": {
  "format": {"type": "audio/pcm", "rate": 16000},
  "transcription": {"model": "zipformer", "language": "sk", "delay": "low"},
  "turn_detection": null
}}}}
```

| field | meaning |
|---|---|
| `transcription.model` | A model that can stream. |
| `transcription.language` | An ISO 639 code or regional tag (`sk`, `zh-TW`). Recorded; a model recognizes the language it was trained on. |
| `transcription.delay` | `minimal`, `low` (default), `medium`, `high`, `xhigh`: how much latency the model may spend on accuracy. |
| `transcription.prompt`, `keywords` | Accepted and ignored (transducers take no prompt; no keyword biasing). |
| `turn_detection` | Omitted or `null`: manual turns. `{"type": "server_vad", "silence_duration_ms": 100..10000}`: the recognizer's endpoints close turns. |

`delay`, `turn_detection` and the model cannot change while audio is buffered (`session_busy`):
commit or clear first. Zipformer streams in chunks of 16, 32 or 64 encoder frames
(0.32 / 0.64 / 1.28 s) for `minimal`, `low` and `medium`; `high` and `xhigh` are `medium`, the
largest chunk the model is trained with. The first text can appear one chunk after the speech.

## Turns and events

A turn is one input buffer. Its `item_id` exists from the first append, so deltas stream
before the commit; use `item_id` to order and reconcile, and `previous_item_id` for the order
of completed items.

Manual turns (no `turn_detection`):

1. `input_audio_buffer.append` (any number of times) → `conversation.item.input_audio_transcription.delta`
   with the new text of the item.
2. `input_audio_buffer.commit` → `input_audio_buffer.committed` (`item_id`, `previous_item_id`),
   the remaining deltas, then `conversation.item.input_audio_transcription.completed` with the
   whole `transcript` and `usage` (`{"type": "duration", "seconds": …}`). A commit with no
   audio is the error `input_audio_buffer_commit_empty`.
3. `input_audio_buffer.clear` drops the buffer → `input_audio_buffer.cleared`.

With `server_vad` each closed segment is an item: `input_audio_buffer.speech_started`, deltas,
`speech_stopped`, `committed`, `completed`; the stream keeps running until `commit` (which
flushes and closes the open segment) or `clear`. Segments close on the recognizer's endpoint
rules (silence since the last decoded token), not on acoustic energy.

Deltas only append: greedy decoding never revises text. Text that does not extend what was sent
(a partly decoded character) is held back, and `completed` always carries the full transcript,
which is the authoritative result. Timestamps, speaker labels and confidence are not sent.

## Errors

An `error` event (`type`, `code`, `message`, `event_id` of the offending client event) leaves
the session open. Codes: `invalid_json`, `unknown_event`, `invalid_event`, `invalid_value`,
`unsupported_audio_format`, `session_busy`, `no_model`, `model_not_found`,
`model_required`, `unsupported_model`, `input_audio_buffer_commit_empty`, `binary_frame`, and
`server_error` codes from the model (`too_many_streams`: a model keeps 64 streams open). A turn
that fails after its item exists also sends
`conversation.item.input_audio_transcription.failed`. A closed socket closes its stream.

## Example

Convert any audio to the wire format with `ffmpeg -i speech.wav -f s16le -ar 16000 -ac 1 speech.pcm`.

```python
import asyncio, base64, json, websockets

async def main(pcm: bytes):  # 16 kHz mono s16le
    async with websockets.connect("ws://localhost:8080/v1/realtime?intent=transcription") as ws:
        await ws.send(json.dumps({"type": "session.update", "session": {"type": "transcription",
                                  "audio": {"input": {"transcription": {"delay": "low"}}}}}))
        for i in range(0, len(pcm), 6400):  # 200 ms
            await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                      "audio": base64.b64encode(pcm[i:i + 6400]).decode()}))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        async for message in ws:
            event = json.loads(message)
            if event["type"].endswith(".delta"):
                print(event["delta"], end="", flush=True)
            elif event["type"].endswith(".completed"):
                print("\n" + event["transcript"])
                break

asyncio.run(main(open("speech.pcm", "rb").read()))
```
