import { BrainIcon, ChevronRightIcon, PencilIcon, RefreshCwIcon } from "lucide-react";
import { motion } from "motion/react";
import { memo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Textarea } from "@/components/ui/input";
import { Tooltip } from "@/components/ui/tooltip";
import { splitThink } from "@/lib/math";
import { cn } from "@/lib/utils";
import type { Message as Msg, MessageStats } from "@/stores/conversations";
import { CopyButton } from "./code-block";
import { Markdown } from "./markdown";

function Stats({ stats, model }: { stats: MessageStats; model?: string }) {
  const parts: string[] = [];
  const gen = stats.generated_tokens ?? stats.completion_tokens;
  if (gen != null) parts.push(`${gen} tokens`);
  if (stats.decode_tps != null) parts.push(`${stats.decode_tps.toFixed(1)} tok/s`);
  if (stats.ttft_ms != null) parts.push(`TTFT ${stats.ttft_ms < 1000 ? `${Math.round(stats.ttft_ms)} ms` : `${(stats.ttft_ms / 1000).toFixed(2)} s`}`);
  const detail = [
    model && `Model: ${model}`,
    stats.prompt_tokens != null && `Prompt tokens: ${stats.prompt_tokens}${stats.cached_tokens ? ` (${stats.cached_tokens} cached)` : ""}`,
    stats.prefill_tps != null && `Prefill: ${stats.prefill_tps.toFixed(1)} tok/s`,
    stats.elapsed_ms != null && `Wall time: ${(stats.elapsed_ms / 1000).toFixed(2)} s`,
    stats.finish_reason && `Finish: ${stats.finish_reason}`,
  ].filter(Boolean);
  if (!parts.length) return null;
  return (
    <Tooltip content={<div className="space-y-0.5">{detail.map((d) => <div key={String(d)}>{d}</div>)}</div>}>
      <span tabIndex={0} className="cursor-default rounded text-xs text-muted-foreground tabular-nums outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50">
        {parts.join(" \u00b7 ")}
        {stats.finish_reason === "length" && <span className="ml-2 text-warning">truncated (max tokens)</span>}
      </span>
    </Tooltip>
  );
}

function Thinking({ text, live }: { text: string; live: boolean }) {
  const [open, setOpen] = useState(false);
  return (
    <Collapsible open={open} onOpenChange={setOpen} className="mb-2">
      <CollapsibleTrigger className="group/think flex items-center gap-1.5 rounded-md py-1 pr-2 text-sm text-muted-foreground outline-none hover:text-foreground focus-visible:ring-[3px] focus-visible:ring-ring/50">
        <ChevronRightIcon className={cn("size-3.5 transition-transform", open && "rotate-90")} />
        <BrainIcon className="size-3.5" />
        <span className={cn(live && "animate-pulse")}>{live ? "Thinking..." : "Thinking"}</span>
        {!live && <span className="text-xs opacity-70">({text.trim().split(/\s+/).length} words)</span>}
      </CollapsibleTrigger>
      <CollapsibleContent className="data-[state=open]:animate-in data-[state=open]:fade-in-0">
        <div className="mt-1 max-h-96 overflow-y-auto border-l-2 pl-3 text-sm whitespace-pre-wrap text-muted-foreground">{text.trim()}</div>
      </CollapsibleContent>
    </Collapsible>
  );
}

export const MessageView = memo(function MessageView({
  message,
  streaming,
  isLast,
  canAct,
  onRegenerate,
  onEdit,
}: {
  message: Msg;
  streaming: boolean;
  isLast: boolean;
  canAct: boolean;
  onRegenerate: () => void;
  onEdit: (text: string) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(message.content);

  if (message.role === "user") {
    return (
      <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} className="group flex flex-col items-end gap-1">
        {editing ? (
          <div className="w-full max-w-[85%] space-y-2">
            <Textarea
              autoFocus
              value={draft}
              aria-label="Edit message"
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  setEditing(false);
                  if (draft.trim()) onEdit(draft.trim());
                } else if (e.key === "Escape") setEditing(false);
              }}
              className="min-h-20"
            />
            <div className="flex justify-end gap-2">
              <Button size="sm" variant="ghost" onClick={() => setEditing(false)}>
                Cancel
              </Button>
              <Button
                size="sm"
                disabled={!draft.trim()}
                onClick={() => {
                  setEditing(false);
                  onEdit(draft.trim());
                }}
              >
                Send
              </Button>
            </div>
          </div>
        ) : (
          <div className="max-w-[85%] rounded-2xl rounded-br-md bg-secondary px-4 py-2.5 text-[15px] whitespace-pre-wrap text-secondary-foreground">
            {message.content}
          </div>
        )}
        {!editing && (
          <div className="flex gap-0.5 opacity-0 transition-opacity group-hover:opacity-100 focus-within:opacity-100 max-md:opacity-100">
            <CopyButton text={message.content} label="Copy message" />
            {canAct && (
              <Button
                variant="ghost"
                size="icon-xs"
                aria-label="Edit and resend"
                title="Edit and resend"
                className="text-muted-foreground"
                onClick={() => {
                  setDraft(message.content);
                  setEditing(true);
                }}
              >
                <PencilIcon className="size-3.5" />
              </Button>
            )}
          </div>
        )}
      </motion.div>
    );
  }

  const inline = !message.reasoning ? splitThink(message.content) : null;
  const reasoning = message.reasoning || inline?.reasoning || "";
  const answer = inline ? inline.answer : message.content;
  const thinkingLive = streaming && !answer;

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="group flex flex-col gap-1">
      {reasoning && <Thinking text={reasoning} live={thinkingLive} />}
      {answer ? (
        <Markdown text={answer} />
      ) : streaming && !reasoning ? (
        <div className="flex h-6 items-center gap-1" aria-label="Generating">
          {[0, 1, 2].map((i) => (
            <span key={i} className="size-1.5 animate-bounce rounded-full bg-muted-foreground/60" style={{ animationDelay: `${i * 120}ms` }} />
          ))}
        </div>
      ) : null}
      {message.error && <div className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">{message.error}</div>}
      {!streaming && (
        <div className="mt-1 flex min-h-7 flex-wrap items-center gap-1">
          {answer && <CopyButton text={answer} label="Copy answer" />}
          {isLast && canAct && (
            <Button variant="ghost" size="icon-xs" aria-label="Regenerate" title="Regenerate" className="text-muted-foreground" onClick={onRegenerate}>
              <RefreshCwIcon className="size-3.5" />
            </Button>
          )}
          {message.stopped && <span className="text-xs text-muted-foreground">stopped</span>}
          {message.stats && <Stats stats={message.stats} model={message.model} />}
        </div>
      )}
    </motion.div>
  );
});
