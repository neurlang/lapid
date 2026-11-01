import { useMutation } from "@tanstack/react-query";
import { DownloadIcon, LoaderCircleIcon, PlayIcon, Trash2Icon, Volume2Icon } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { useEffect, useRef, useState } from "react";
import { NoModel, PageToolbar } from "@/components/layout/empty-state";
import { ModelPicker, useSelectedModel } from "@/components/layout/model-picker";
import { Waveform } from "@/components/waveform";
import { Button } from "@/components/ui/button";
import { Input, Textarea } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { api } from "@/lib/api";
import { cn, downloadBlob } from "@/lib/utils";
import { type Clip, useClips } from "@/stores/clips";

const SAMPLE = "Hello! This voice was generated locally by lapid, without sending a single byte to the cloud.";

function clipName(c: Clip) {
  return `speech-${c.voice ?? "default"}-${new Date(c.createdAt).toISOString().slice(0, 19).replace(/[:T]/g, "-")}.wav`;
}

export function SpeakPage() {
  const { model, isLoading, isError } = useSelectedModel("speech");
  const [text, setText] = useState(SAMPLE);
  const [voice, setVoice] = useState<string>("");
  const [seed, setSeed] = useState("");
  const [temperature, setTemperature] = useState("");
  const clips = useClips((s) => s.clips);
  const addClip = useClips((s) => s.add);
  const removeClip = useClips((s) => s.remove);
  const [currentId, setCurrentId] = useState<string | null>(clips[0]?.id ?? null);
  const [autoplay, setAutoplay] = useState(false);
  const abort = useRef<AbortController | null>(null);
  const voices = model?.voices ?? [];

  useEffect(() => {
    if (!model) return;
    if (!voice || !voices.includes(voice)) setVoice(model.default_voice ?? voices[0] ?? "");
  }, [model, voices, voice]);

  const mutation = useMutation({
    mutationFn: async () => {
      if (!model) throw new Error("No speech model");
      const controller = new AbortController();
      abort.current = controller;
      const t0 = performance.now();
      const seedN = seed.trim() === "" ? undefined : Number(seed);
      const tempN = temperature.trim() === "" ? undefined : Number(temperature);
      const blob = await api.speech(
        {
          model: model.id,
          input: text.trim(),
          ...(voice ? { voice } : {}),
          ...(seedN !== undefined && Number.isFinite(seedN) ? { seed: seedN } : {}),
          ...(tempN !== undefined && Number.isFinite(tempN) ? { temperature: tempN } : {}),
        },
        controller.signal,
      );
      return addClip({ blob, text: text.trim(), voice: voice || undefined, model: model.id, elapsedMs: performance.now() - t0 });
    },
    onSuccess: (clip) => {
      setAutoplay(true);
      setCurrentId(clip.id);
    },
    onSettled: () => {
      abort.current = null;
    },
  });

  if (!model && !isLoading) {
    return (
      <div className="h-full overflow-y-auto">
        <NoModel icon={Volume2Icon} capability="speech" offline={isError} />
      </div>
    );
  }

  const current = clips.find((c) => c.id === currentId) ?? null;
  const canGenerate = !!model && text.trim().length > 0 && !mutation.isPending;

  return (
    <div className="flex h-full flex-col">
      <PageToolbar>
        <ModelPicker capability="speech" />
      </PageToolbar>
      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto flex max-w-3xl flex-col gap-5 p-4 sm:p-6">
          <form
            className="space-y-4"
            onSubmit={(e) => {
              e.preventDefault();
              if (canGenerate) mutation.mutate();
            }}
          >
            <div className="space-y-1.5">
              <Label htmlFor="tts-text">Text</Label>
              <Textarea
                id="tts-text"
                value={text}
                onChange={(e) => setText(e.target.value)}
                className="min-h-32 text-[15px]"
                placeholder="Type something to say"
                onKeyDown={(e) => {
                  if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && canGenerate) {
                    e.preventDefault();
                    mutation.mutate();
                  }
                }}
              />
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>Ctrl+Enter to generate</span>
                <span className="tabular-nums">{text.length} chars</span>
              </div>
            </div>
            {model?.prompts && model.prompts.length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {model.prompts.slice(0, 6).map((p) => (
                  <button
                    key={p}
                    type="button"
                    onClick={() => setText(p)}
                    className="max-w-full truncate rounded-full border px-3 py-1 text-xs text-muted-foreground outline-none hover:bg-accent hover:text-foreground focus-visible:ring-[3px] focus-visible:ring-ring/50"
                  >
                    {p}
                  </button>
                ))}
              </div>
            )}
            <div className="flex flex-wrap items-end gap-3">
              <div className="flex flex-col gap-1.5">
                <Label>Voice</Label>
                <Select value={voice} onValueChange={setVoice} disabled={voices.length === 0}>
                  <SelectTrigger className="w-44" aria-label="Voice">
                    <SelectValue placeholder="Default" />
                  </SelectTrigger>
                  <SelectContent>
                    {voices.map((v) => (
                      <SelectItem key={v} value={v}>
                        {v}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="tts-seed">Seed</Label>
                <Input id="tts-seed" inputMode="numeric" placeholder="random" value={seed} onChange={(e) => setSeed(e.target.value.replace(/\D/g, ""))} className="w-28" />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="tts-temp">Temperature</Label>
                <Input
                  id="tts-temp"
                  type="number"
                  step={0.05}
                  min={0}
                  max={2}
                  placeholder="default"
                  value={temperature}
                  onChange={(e) => setTemperature(e.target.value)}
                  className="w-28"
                />
              </div>
              <div className="ml-auto flex gap-2">
                {mutation.isPending && (
                  <Button type="button" variant="ghost" onClick={() => abort.current?.abort()}>
                    Cancel
                  </Button>
                )}
                <Button type="submit" disabled={!canGenerate}>
                  {mutation.isPending ? <LoaderCircleIcon className="animate-spin" /> : <Volume2Icon />}
                  {mutation.isPending ? "Generating..." : "Generate"}
                </Button>
              </div>
            </div>
          </form>

          {current && (
            <motion.div key={current.id} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} className="space-y-2">
              <Waveform src={current.url} height={80} autoplay={autoplay} />
              <div className="flex items-center gap-2">
                <p className="line-clamp-1 flex-1 text-sm text-muted-foreground">{current.text}</p>
                <Button variant="outline" size="sm" onClick={() => downloadBlob(current.blob, clipName(current))}>
                  <DownloadIcon /> WAV
                </Button>
              </div>
            </motion.div>
          )}

          {clips.length > 0 && (
            <section aria-label="History" className="space-y-2">
              <h2 className="text-sm font-medium text-muted-foreground">This session</h2>
              <ul className="divide-y rounded-lg border">
                <AnimatePresence initial={false}>
                  {clips.map((c) => (
                    <motion.li key={c.id} layout initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className={cn("flex items-center gap-2 px-2 py-1.5", c.id === currentId && "bg-accent/60")}>
                      <Button
                        variant="ghost"
                        size="icon-sm"
                        aria-label="Load clip"
                        onClick={() => {
                          setAutoplay(true);
                          setCurrentId(c.id);
                        }}
                      >
                        <PlayIcon />
                      </Button>
                      <div className="min-w-0 flex-1">
                        <p className="truncate text-sm">{c.text}</p>
                        <p className="text-xs text-muted-foreground">
                          {[c.voice, `${(c.elapsedMs / 1000).toFixed(1)} s`, new Date(c.createdAt).toLocaleTimeString()].filter(Boolean).join(" · ")}
                        </p>
                      </div>
                      <Button variant="ghost" size="icon-sm" aria-label="Download clip" onClick={() => downloadBlob(c.blob, clipName(c))}>
                        <DownloadIcon />
                      </Button>
                      <Button variant="ghost" size="icon-sm" aria-label="Remove clip" onClick={() => removeClip(c.id)}>
                        <Trash2Icon />
                      </Button>
                    </motion.li>
                  ))}
                </AnimatePresence>
              </ul>
            </section>
          )}
        </div>
      </div>
    </div>
  );
}
