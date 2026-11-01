import { ArrowDownIcon, ArrowUpIcon, MessageSquareIcon, SparklesIcon, SquareIcon } from "lucide-react";
import { motion } from "motion/react";
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { NoModel, PageToolbar } from "@/components/layout/empty-state";
import { ModelPicker, useSelectedModel } from "@/components/layout/model-picker";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import { useConversations, useStream } from "@/stores/conversations";
import { editAndResend, regenerate, sendMessage } from "./generate";
import { MessageView } from "./message";
import { SettingsSheet } from "./settings-sheet";

const EXAMPLES = [
  { title: "Explain a concept", prompt: "Explain how attention works in transformers, with the key equation in LaTeX." },
  { title: "Write code", prompt: "Write a Rust function that parses a WAV header and returns sample rate and channel count." },
  { title: "Compare options", prompt: "Compare SQLite, Postgres and DuckDB for a local analytics app as a markdown table." },
  { title: "Brainstorm", prompt: "Give me five names for a small, fast, local inference server, each with a one line pitch." },
];

/** Keeps the view pinned to the bottom unless the user scrolled up. */
function useStickToBottom(dep: unknown) {
  const ref = useRef<HTMLDivElement>(null);
  const [stuck, setStuck] = useState(true);
  const stuckRef = useRef(true);
  const onScroll = useCallback(() => {
    const el = ref.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
    stuckRef.current = atBottom;
    setStuck(atBottom);
  }, []);
  useLayoutEffect(() => {
    const el = ref.current;
    if (el && stuckRef.current) el.scrollTop = el.scrollHeight;
  }, [dep]);
  const scrollToBottom = useCallback(() => {
    const el = ref.current;
    if (!el) return;
    stuckRef.current = true;
    setStuck(true);
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, []);
  return { ref, stuck, onScroll, scrollToBottom };
}

function Composer({ disabled, onSend }: { disabled: boolean; onSend: (text: string) => void }) {
  const [text, setText] = useState("");
  const streaming = useStream((s) => s.controller !== null);
  const stop = useStream((s) => s.stop);
  const ref = useRef<HTMLTextAreaElement>(null);

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 240)}px`;
  }, [text]);

  const submit = () => {
    const t = text.trim();
    if (!t || disabled || streaming) return;
    onSend(t);
    setText("");
  };

  return (
    <form
      className="mx-auto w-full max-w-3xl px-3 pb-3 sm:px-4 sm:pb-4"
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
    >
      <div className="flex items-end gap-2 rounded-2xl border bg-background p-2 shadow-sm focus-within:border-ring focus-within:ring-[3px] focus-within:ring-ring/30 dark:bg-input/20">
        <Textarea
          ref={ref}
          rows={1}
          value={text}
          autoFocus
          aria-label="Message"
          placeholder={disabled ? "Load a chat model to start" : "Message (Enter to send, Shift+Enter for a newline)"}
          className="max-h-60 min-h-9 resize-none border-0 bg-transparent px-2 py-1.5 shadow-none focus-visible:ring-0 dark:bg-transparent"
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              submit();
            } else if (e.key === "Escape" && streaming) {
              e.preventDefault();
              stop();
            }
          }}
        />
        {streaming ? (
          <Button type="button" size="icon" className="rounded-xl" onClick={stop} aria-label="Stop generating" title="Stop (Esc)">
            <SquareIcon className="fill-current" />
          </Button>
        ) : (
          <Button type="submit" size="icon" className="rounded-xl" disabled={disabled || !text.trim()} aria-label="Send">
            <ArrowUpIcon />
          </Button>
        )}
      </div>
    </form>
  );
}

export function ChatPage() {
  const { model, isLoading, isError } = useSelectedModel("chat");
  const activeId = useConversations((s) => s.activeId);
  const hydrated = useConversations((s) => s.hydrated);
  const conv = useConversations((s) => s.conversations.find((c) => c.id === s.activeId));
  const streamMsg = useStream((s) => s.msgId);
  const streaming = streamMsg !== null;
  const messages = conv?.messages ?? [];
  const last = messages.at(-1);
  const { ref, stuck, onScroll, scrollToBottom } = useStickToBottom(`${messages.length}:${last?.content.length}:${last?.reasoning?.length}`);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && useStream.getState().controller) useStream.getState().stop();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const canAct = !!model && !streaming;
  const send = (text: string) => model && void sendMessage(text, model.id);

  return (
    <div className="flex h-full flex-col">
      <PageToolbar>
        <ModelPicker capability="chat" />
        <div className="ml-auto flex items-center gap-2">
          <SettingsSheet convId={activeId} />
        </div>
      </PageToolbar>

      <div className="relative min-h-0 flex-1">
        <div ref={ref} onScroll={onScroll} className="h-full overflow-y-auto">
          {!hydrated ? (
            <div className="mx-auto max-w-3xl space-y-4 p-6">
              <Skeleton className="ml-auto h-10 w-1/2" />
              <Skeleton className="h-24 w-full" />
            </div>
          ) : messages.length === 0 ? (
            !model && !isLoading ? (
              <NoModel icon={MessageSquareIcon} capability="chat" offline={isError} />
            ) : (
              <div className="mx-auto flex max-w-3xl flex-col items-center px-4 pt-[12vh] pb-8">
                <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="mb-8 text-center">
                  <div className="mx-auto mb-4 flex size-12 items-center justify-center rounded-2xl bg-muted">
                    <SparklesIcon className="size-6 text-brand" />
                  </div>
                  <h2 className="text-xl font-semibold tracking-tight">What can I help with?</h2>
                  <p className="mt-1 text-sm text-muted-foreground">
                    Running {model?.name || model?.id || "..."} locally{model?.device ? ` on ${model.device}` : ""}.
                  </p>
                </motion.div>
                <div className="grid w-full gap-2 sm:grid-cols-2">
                  {EXAMPLES.map((ex, i) => (
                    <motion.button
                      key={ex.title}
                      type="button"
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.05 * i }}
                      disabled={!model}
                      onClick={() => send(ex.prompt)}
                      className="rounded-xl border p-3 text-left text-sm outline-none transition-colors hover:bg-accent focus-visible:ring-[3px] focus-visible:ring-ring/50 disabled:opacity-50"
                    >
                      <div className="font-medium">{ex.title}</div>
                      <div className="mt-0.5 line-clamp-2 text-muted-foreground">{ex.prompt}</div>
                    </motion.button>
                  ))}
                </div>
              </div>
            )
          ) : (
            <div className="mx-auto flex max-w-3xl flex-col gap-6 px-4 py-6">
              {messages.map((m, i) => (
                <MessageView
                  key={m.id}
                  message={m}
                  streaming={m.id === streamMsg}
                  isLast={i === messages.length - 1}
                  canAct={canAct}
                  onRegenerate={() => conv && model && void regenerate(conv.id, model.id)}
                  onEdit={(text) => conv && model && void editAndResend(conv.id, m.id, text, model.id)}
                />
              ))}
            </div>
          )}
        </div>
        {!stuck && messages.length > 0 && (
          <Button
            variant="outline"
            size="icon-sm"
            onClick={scrollToBottom}
            aria-label="Scroll to bottom"
            className={cn("absolute bottom-3 left-1/2 -translate-x-1/2 rounded-full shadow-md")}
          >
            <ArrowDownIcon />
          </Button>
        )}
      </div>

      <Composer disabled={!model} onSend={send} />
    </div>
  );
}
