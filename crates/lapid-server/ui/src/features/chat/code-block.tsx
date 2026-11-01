import { CheckIcon, CopyIcon } from "lucide-react";
import { memo, useEffect, useState } from "react";
import { highlight } from "@/lib/highlighter";
import { copyText } from "@/lib/utils";

export function CopyButton({ text, label = "Copy", className }: { text: string; label?: string; className?: string }) {
  const [done, setDone] = useState(false);
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      className={
        className ??
        "inline-flex size-7 items-center justify-center rounded-md text-muted-foreground outline-none hover:bg-accent hover:text-foreground focus-visible:ring-[3px] focus-visible:ring-ring/50"
      }
      onClick={async () => {
        if (await copyText(text)) {
          setDone(true);
          setTimeout(() => setDone(false), 1500);
        }
      }}
    >
      {done ? <CheckIcon className="size-3.5" /> : <CopyIcon className="size-3.5" />}
    </button>
  );
}

export const CodeBlock = memo(function CodeBlock({ code, lang }: { code: string; lang?: string }) {
  const [html, setHtml] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    // Debounce while streaming: highlight after the text stops changing for a moment.
    const t = setTimeout(() => {
      highlight(code, lang)
        .then((h) => alive && setHtml(h))
        .catch(() => alive && setHtml(null));
    }, 120);
    return () => {
      alive = false;
      clearTimeout(t);
    };
  }, [code, lang]);

  return (
    <div className="group/code my-3 overflow-hidden rounded-lg border bg-muted/40">
      <div className="flex h-8 items-center justify-between border-b bg-muted/60 pr-1 pl-3 text-xs text-muted-foreground">
        <span className="font-mono">{lang || "text"}</span>
        <CopyButton text={code} label="Copy code" />
      </div>
      {html ? (
        <div className="overflow-x-auto p-3 font-mono text-[13px] leading-relaxed [&_pre]:m-0" dangerouslySetInnerHTML={{ __html: html }} />
      ) : (
        <pre className="overflow-x-auto p-3 font-mono text-[13px] leading-relaxed">
          <code>{code}</code>
        </pre>
      )}
    </div>
  );
});
