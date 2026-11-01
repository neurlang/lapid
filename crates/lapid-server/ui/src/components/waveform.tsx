import { PauseIcon, PlayIcon } from "lucide-react";
import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import { Button } from "@/components/ui/button";
import { useResolvedTheme } from "@/hooks/use-theme";
import { cn, formatClock } from "@/lib/utils";

export interface WaveformHandle {
  seek: (seconds: number) => void;
  play: () => void;
}

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/** Waveform + transport for a Blob or URL. Theme-aware colors, keyboard: Space toggles when focused. */
export const Waveform = forwardRef<
  WaveformHandle,
  { src: Blob | string; height?: number; className?: string; onTime?: (t: number) => void; autoplay?: boolean }
>(function Waveform({ src, height = 72, className, onTime, autoplay }, ref) {
  const container = useRef<HTMLDivElement>(null);
  const ws = useRef<WaveSurfer | null>(null);
  const onTimeRef = useRef(onTime);
  onTimeRef.current = onTime;
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const theme = useResolvedTheme();

  useImperativeHandle(ref, () => ({
    seek: (s) => ws.current?.setTime(s),
    play: () => void ws.current?.play(),
  }));

  useEffect(() => {
    if (!container.current) return;
    const w = WaveSurfer.create({
      container: container.current,
      height,
      waveColor: cssVar("--wave"),
      progressColor: cssVar("--wave-progress"),
      cursorColor: cssVar("--foreground"),
      cursorWidth: 1,
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      normalize: true,
      dragToSeek: true,
      autoplay,
    });
    ws.current = w;
    const unsubs = [
      w.on("play", () => setPlaying(true)),
      w.on("pause", () => setPlaying(false)),
      w.on("finish", () => setPlaying(false)),
      w.on("ready", (d) => setDuration(d)),
      w.on("timeupdate", (t) => {
        setTime(t);
        onTimeRef.current?.(t);
      }),
    ];
    setTime(0);
    setDuration(0);
    if (typeof src === "string") void w.load(src).catch(() => {});
    else void w.loadBlob(src).catch(() => {});
    return () => {
      unsubs.forEach((u) => u());
      w.destroy();
      ws.current = null;
    };
  }, [src, height, autoplay]);

  useEffect(() => {
    ws.current?.setOptions({ waveColor: cssVar("--wave"), progressColor: cssVar("--wave-progress"), cursorColor: cssVar("--foreground") });
  }, [theme]);

  return (
    <div className={cn("flex items-center gap-3 rounded-lg border bg-card p-3", className)}>
      <Button
        type="button"
        size="icon"
        variant="secondary"
        className="rounded-full"
        aria-label={playing ? "Pause" : "Play"}
        onClick={() => void ws.current?.playPause()}
      >
        {playing ? <PauseIcon /> : <PlayIcon />}
      </Button>
      <div ref={container} className="min-w-0 flex-1 cursor-pointer" />
      <span className="shrink-0 text-right whitespace-nowrap font-mono text-xs text-muted-foreground tabular-nums">
        {formatClock(time)} / {formatClock(duration)}
      </span>
    </div>
  );
});
