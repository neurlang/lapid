import { useMutation } from "@tanstack/react-query";
import { AudioLinesIcon, DownloadIcon, FileAudioIcon, LoaderCircleIcon, UploadIcon, XIcon } from "lucide-react";
import { motion } from "motion/react";
import { useEffect, useRef, useState } from "react";
import { useDropzone } from "react-dropzone";
import { NoModel, PageToolbar } from "@/components/layout/empty-state";
import { ModelPicker, useSelectedModel } from "@/components/layout/model-picker";
import { Waveform, type WaveformHandle } from "@/components/waveform";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { CopyButton } from "@/features/chat/code-block";
import { api, type Transcription } from "@/lib/api";
import { toWav16k } from "@/lib/audio";
import { toSrt, toVtt } from "@/lib/subtitles";
import { cn, downloadBlob, formatBytes, formatClock } from "@/lib/utils";
import { Recorder } from "./recorder";

interface AudioInput {
  blob: Blob;
  name: string;
}

type Phase = "idle" | "converting" | "transcribing";

function useElapsed(active: boolean) {
  const [t, setT] = useState(0);
  useEffect(() => {
    if (!active) return;
    const t0 = performance.now();
    setT(0);
    const id = setInterval(() => setT((performance.now() - t0) / 1000), 100);
    return () => clearInterval(id);
  }, [active]);
  return t;
}

export function TranscribePage() {
  const { model, isLoading, isError } = useSelectedModel("transcription");
  const [input, setInput] = useState<AudioInput | null>(null);
  const [language, setLanguage] = useState("");
  const [words, setWords] = useState(false);
  const [phase, setPhase] = useState<Phase>("idle");
  const [result, setResult] = useState<{ data: Transcription; ms: number; base: string } | null>(null);
  const [now, setNow] = useState(0);
  const wave = useRef<WaveformHandle>(null);
  const abort = useRef<AbortController | null>(null);
  const elapsed = useElapsed(phase !== "idle");

  const dz = useDropzone({
    accept: { "audio/*": [".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac", ".webm"], "video/webm": [".webm"], "video/mp4": [".mp4"] },
    multiple: false,
    noClick: true,
    onDrop: (files) => {
      const f = files[0];
      if (f) {
        setInput({ blob: f, name: f.name });
        setResult(null);
      }
    },
  });

  const mutation = useMutation({
    mutationFn: async () => {
      if (!input || !model) throw new Error("Nothing to transcribe");
      const controller = new AbortController();
      abort.current = controller;
      const t0 = performance.now();
      setPhase("converting");
      let wav: Blob;
      try {
        wav = (await toWav16k(input.blob)).wav;
      } catch {
        throw new Error("Could not decode this audio file in the browser.");
      }
      setPhase("transcribing");
      const data = await api.transcribe({ model: model.id, wav, language, words, signal: controller.signal });
      return { data, ms: performance.now() - t0, base: input.name.replace(/\.[^.]+$/, "") || "transcript" };
    },
    onSuccess: (r) => setResult(r),
    onSettled: () => {
      setPhase("idle");
      abort.current = null;
    },
  });

  if (!model && !isLoading) {
    return (
      <div className="h-full overflow-y-auto">
        <NoModel icon={AudioLinesIcon} capability="transcription" offline={isError} />
      </div>
    );
  }

  const segments = result?.data.segments ?? [];
  const activeSeg = segments.findIndex((s) => now >= s.start && now < s.end);

  return (
    <div className="flex h-full flex-col">
      <PageToolbar>
        <ModelPicker capability="transcription" />
      </PageToolbar>
      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto flex max-w-4xl flex-col gap-5 p-4 sm:p-6">
          <div
            {...dz.getRootProps()}
            className={cn(
              "flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed p-6 text-center transition-colors sm:p-8",
              dz.isDragActive ? "border-ring bg-accent/50" : "border-border",
            )}
          >
            <input {...dz.getInputProps()} aria-label="Audio file" />
            <UploadIcon className="size-8 text-muted-foreground" />
            <div>
              <p className="font-medium">Drop an audio file here</p>
              <p className="text-sm text-muted-foreground">WAV, MP3, FLAC, OGG, M4A, WebM. Converted to 16 kHz mono in your browser.</p>
            </div>
            <div className="flex flex-wrap items-center justify-center gap-2">
              <Button type="button" variant="outline" onClick={dz.open}>
                <FileAudioIcon /> Choose file
              </Button>
              <Recorder
                disabled={phase !== "idle"}
                onRecorded={(blob) => {
                  setInput({ blob, name: `recording-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}` });
                  setResult(null);
                }}
              />
            </div>
          </div>

          {input && (
            <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} className="space-y-3">
              <div className="flex items-center gap-2 text-sm">
                <FileAudioIcon className="size-4 text-muted-foreground" />
                <span className="truncate font-medium">{input.name}</span>
                <span className="text-muted-foreground">{formatBytes(input.blob.size)}</span>
                <Button variant="ghost" size="icon-xs" className="ml-auto" aria-label="Remove audio" onClick={() => (setInput(null), setResult(null))}>
                  <XIcon />
                </Button>
              </div>
              <Waveform ref={wave} src={input.blob} onTime={setNow} />
              <div className="flex flex-wrap items-end gap-4">
                <div className="space-y-1.5">
                  <Label htmlFor="language">Language</Label>
                  <Input id="language" placeholder="auto" value={language} onChange={(e) => setLanguage(e.target.value)} className="w-32" />
                </div>
                <div className="flex h-9 items-center gap-2">
                  <Switch id="words" checked={words} onCheckedChange={setWords} />
                  <Label htmlFor="words">Word timestamps</Label>
                </div>
                <div className="ml-auto flex gap-2">
                  {phase !== "idle" && (
                    <Button variant="ghost" onClick={() => abort.current?.abort()}>
                      Cancel
                    </Button>
                  )}
                  <Button onClick={() => mutation.mutate()} disabled={!model || phase !== "idle"}>
                    {phase !== "idle" && <LoaderCircleIcon className="animate-spin" />}
                    {phase === "converting" ? "Converting..." : phase === "transcribing" ? `Transcribing ${elapsed.toFixed(1)}s` : "Transcribe"}
                  </Button>
                </div>
              </div>
            </motion.div>
          )}

          {result && (
            <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}>
              <Card>
                <CardHeader className="flex-row flex-wrap items-center gap-2">
                  <CardTitle>Transcript</CardTitle>
                  <span className="text-xs text-muted-foreground">
                    {[result.data.language, result.data.duration != null && `${result.data.duration.toFixed(1)} s audio`, `${(result.ms / 1000).toFixed(1)} s`]
                      .filter(Boolean)
                      .join(" · ")}
                  </span>
                  <div className="ml-auto flex items-center gap-1">
                    <CopyButton text={result.data.text.trim()} label="Copy text" />
                    <Button
                      variant="outline"
                      size="sm"
                      disabled={!segments.length}
                      onClick={() => downloadBlob(new Blob([toSrt(segments)], { type: "application/x-subrip" }), `${result.base}.srt`)}
                    >
                      <DownloadIcon /> SRT
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      disabled={!segments.length}
                      onClick={() => downloadBlob(new Blob([toVtt(segments)], { type: "text/vtt" }), `${result.base}.vtt`)}
                    >
                      <DownloadIcon /> VTT
                    </Button>
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  <p className="leading-relaxed whitespace-pre-wrap">{result.data.text.trim() || <span className="text-muted-foreground">(no speech detected)</span>}</p>
                  {segments.length > 0 && (
                    <ol className="divide-y rounded-lg border" aria-label="Segments">
                      {segments.map((s, i) => (
                        <li key={s.id ?? i}>
                          <button
                            type="button"
                            onClick={() => {
                              wave.current?.seek(s.start);
                              wave.current?.play();
                            }}
                            className={cn(
                              "flex w-full gap-3 px-3 py-2 text-left text-sm outline-none transition-colors hover:bg-accent/60 focus-visible:bg-accent",
                              i === activeSeg && "bg-accent",
                            )}
                          >
                            <span className="shrink-0 font-mono text-xs leading-5 text-muted-foreground tabular-nums">
                              {formatClock(s.start)}
                            </span>
                            <span>{s.text.trim()}</span>
                          </button>
                        </li>
                      ))}
                    </ol>
                  )}
                </CardContent>
              </Card>
            </motion.div>
          )}
        </div>
      </div>
    </div>
  );
}
