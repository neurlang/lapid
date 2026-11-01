import { MicIcon, SquareIcon } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { formatClock } from "@/lib/utils";

/** Mic capture via MediaRecorder with a live level meter (AnalyserNode RMS). */
export function Recorder({ onRecorded, disabled }: { onRecorded: (blob: Blob) => void; disabled?: boolean }) {
  const [recording, setRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [levels, setLevels] = useState<number[]>(() => Array(24).fill(0));
  const cleanup = useRef<(() => void) | null>(null);
  const recorder = useRef<MediaRecorder | null>(null);

  useEffect(() => () => cleanup.current?.(), []);

  const start = async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      toast.error("Microphone capture needs a secure context (https or localhost).");
      return;
    }
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
    } catch {
      toast.error("Microphone permission denied.");
      return;
    }
    const ctx = new AudioContext();
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 1024;
    ctx.createMediaStreamSource(stream).connect(analyser);
    const buf = new Float32Array(analyser.fftSize);
    const t0 = performance.now();
    let raf = 0;
    const tick = () => {
      analyser.getFloatTimeDomainData(buf);
      let sum = 0;
      for (const v of buf) sum += v * v;
      const rms = Math.sqrt(sum / buf.length);
      const level = Math.min(1, Math.sqrt(rms) * 1.8);
      setLevels((l) => [...l.slice(1), level]);
      setElapsed((performance.now() - t0) / 1000);
      raf = requestAnimationFrame(tick);
    };
    tick();

    const chunks: Blob[] = [];
    const rec = new MediaRecorder(stream);
    rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    rec.onstop = () => {
      const blob = new Blob(chunks, { type: rec.mimeType || "audio/webm" });
      if (blob.size) onRecorded(blob);
    };
    rec.start(250);
    recorder.current = rec;
    cleanup.current = () => {
      cancelAnimationFrame(raf);
      if (rec.state !== "inactive") rec.stop();
      stream.getTracks().forEach((t) => t.stop());
      void ctx.close();
      cleanup.current = null;
    };
    setElapsed(0);
    setRecording(true);
  };

  const stop = () => {
    cleanup.current?.();
    setRecording(false);
    setLevels(Array(24).fill(0));
  };

  return (
    <div className="flex items-center gap-3">
      <Button
        type="button"
        variant={recording ? "destructive" : "outline"}
        onClick={recording ? stop : () => void start()}
        disabled={disabled && !recording}
        aria-pressed={recording}
      >
        {recording ? <SquareIcon className="fill-current" /> : <MicIcon />}
        {recording ? "Stop" : "Record"}
      </Button>
      {recording && (
        <div className="flex items-center gap-3" aria-live="polite">
          <div className="flex h-8 items-center gap-0.5" aria-hidden>
            {levels.map((l, i) => (
              <span key={i} className="w-1 rounded-full bg-destructive/80 transition-[height] duration-75" style={{ height: `${Math.max(8, l * 100)}%` }} />
            ))}
          </div>
          <span className="font-mono text-xs text-muted-foreground tabular-nums">{formatClock(elapsed)}</span>
        </div>
      )}
    </div>
  );
}
