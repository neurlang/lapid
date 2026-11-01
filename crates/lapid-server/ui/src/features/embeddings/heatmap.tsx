import { useState } from "react";
import { cn } from "@/lib/utils";

/** Similarity matrix as a CSS grid. Colors interpolate between theme tokens --heat-lo and --heat-hi. */
export function Heatmap({ matrix, labels }: { matrix: number[][]; labels: string[] }) {
  const [hover, setHover] = useState<{ r: number; c: number } | null>(null);
  const n = matrix.length;
  let lo = 1;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i !== j) lo = Math.min(lo, matrix[i]![j]!);
  if (n < 2 || lo >= 1) lo = 0;
  const range = Math.max(1e-6, 1 - lo);

  return (
    <div className="space-y-3">
      <div className="overflow-x-auto pb-1">
        <div
          role="grid"
          aria-label="Cosine similarity matrix"
          className="grid w-max min-w-full gap-1"
          style={{ gridTemplateColumns: `2.5rem repeat(${n}, minmax(3rem, 1fr))` }}
          onMouseLeave={() => setHover(null)}
        >
          <div />
          {labels.map((_, j) => (
            <div key={j} className={cn("py-1 text-center text-xs font-medium text-muted-foreground transition-colors", hover?.c === j && "text-foreground")}>
              S{j + 1}
            </div>
          ))}
          {matrix.map((row, i) => (
            <div key={i} role="row" className="contents">
              <div
                title={labels[i]}
                className={cn("flex items-center justify-end pr-2 text-xs font-medium text-muted-foreground transition-colors", hover?.r === i && "text-foreground")}
              >
                S{i + 1}
              </div>
              {row.map((v, j) => {
                const p = Math.min(1, Math.max(0, (v - lo) / range));
                const inLine = hover && (hover.r === i || hover.c === j);
                return (
                  <div
                    key={j}
                    role="gridcell"
                    tabIndex={0}
                    aria-label={`S${i + 1} vs S${j + 1}: ${v.toFixed(3)}`}
                    title={`${labels[i]}\n${labels[j]}\n${v.toFixed(4)}`}
                    onMouseEnter={() => setHover({ r: i, c: j })}
                    onFocus={() => setHover({ r: i, c: j })}
                    className={cn(
                      "flex aspect-square min-h-10 items-center justify-center rounded-md text-xs font-medium tabular-nums outline-none transition-[opacity,box-shadow] focus-visible:ring-[3px] focus-visible:ring-ring",
                      hover && !inLine && "opacity-45",
                      hover?.r === i && hover?.c === j && "ring-2 ring-foreground",
                    )}
                    style={{
                      background: `color-mix(in oklch, var(--heat-hi) ${(p * 100).toFixed(1)}%, var(--heat-lo))`,
                      color: p > 0.55 ? "var(--heat-text-hi)" : "var(--heat-text-lo)",
                    }}
                  >
                    {v.toFixed(2)}
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </div>
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <span className="tabular-nums">{lo.toFixed(2)}</span>
        <div className="h-2 w-40 rounded-full" style={{ background: "linear-gradient(to right, var(--heat-lo), var(--heat-hi))" }} />
        <span>1.00</span>
        <span className="ml-2">cosine similarity</span>
      </div>
    </div>
  );
}
