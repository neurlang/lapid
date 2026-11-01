import type { TranscriptionSegment } from "@/lib/api";

function stamp(t: number, sep: "," | "."): string {
  const ms = Math.max(0, Math.round(t * 1000));
  const h = Math.floor(ms / 3600000);
  const m = Math.floor((ms % 3600000) / 60000);
  const s = Math.floor((ms % 60000) / 1000);
  const r = ms % 1000;
  const p = (n: number, w = 2) => String(n).padStart(w, "0");
  return `${p(h)}:${p(m)}:${p(s)}${sep}${p(r, 3)}`;
}

export function toSrt(segments: TranscriptionSegment[]): string {
  return segments
    .map((s, i) => `${i + 1}\n${stamp(s.start, ",")} --> ${stamp(s.end, ",")}\n${s.text.trim()}\n`)
    .join("\n");
}

export function toVtt(segments: TranscriptionSegment[]): string {
  return `WEBVTT\n\n${segments.map((s) => `${stamp(s.start, ".")} --> ${stamp(s.end, ".")}\n${s.text.trim()}\n`).join("\n")}`;
}
