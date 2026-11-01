import { createParser } from "eventsource-parser";
import { ApiError, type ChatChunk, type ChatRequest, toApiError } from "@/lib/api";

/** POSTs a streaming chat completion and calls `onChunk` for every parsed SSE `data:` JSON payload. */
export async function streamChat(body: ChatRequest, onChunk: (chunk: ChatChunk) => void, signal: AbortSignal): Promise<void> {
  let res: Response;
  try {
    res = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(body),
      signal,
    });
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") throw e;
    throw new ApiError("Cannot reach the lapid server", 0, "network_error");
  }
  if (!res.ok) throw await toApiError(res);
  if (!res.body) throw new ApiError("Empty response body", res.status);

  let done = false;
  let streamError: ApiError | null = null;
  const parser = createParser({
    onEvent(ev) {
      if (done) return;
      if (ev.data === "[DONE]") {
        done = true;
        return;
      }
      let parsed: ChatChunk & { error?: { message?: string; type?: string } };
      try {
        parsed = JSON.parse(ev.data);
      } catch {
        return;
      }
      if (parsed.error) {
        streamError = new ApiError(parsed.error.message ?? "Stream error", 500, parsed.error.type);
        done = true;
        return;
      }
      onChunk(parsed);
    },
  });

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  try {
    while (!done) {
      const { value, done: eof } = await reader.read();
      if (eof) break;
      parser.feed(value);
    }
  } finally {
    reader.cancel().catch(() => {});
  }
  if (streamError) throw streamError;
}
