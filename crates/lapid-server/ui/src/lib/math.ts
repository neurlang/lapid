/**
 * Normalizes LaTeX delimiters for remark-math: `\[..\]` -> `$$..$$` and `\(..\)` -> `$..$`,
 * leaving fenced and inline code untouched.
 */
export function normalizeMath(src: string): string {
  if (!src.includes("\\[") && !src.includes("\\(")) return src;
  const parts = src.split(/(```[\s\S]*?(?:```|$)|`[^`\n]*`)/g);
  return parts
    .map((part, i) => {
      if (i % 2 === 1) return part; // code
      return part
        .replace(/\\\[([\s\S]*?)\\\]/g, (_, m: string) => `\n$$\n${m.trim()}\n$$\n`)
        .replace(/\\\(([\s\S]*?)\\\)/g, (_, m: string) => `$${m.trim()}$`);
    })
    .join("");
}

/** Splits `<think>...</think>` out of content for models that inline their reasoning. */
export function splitThink(content: string): { reasoning: string; answer: string } | null {
  const t = content.trimStart();
  if (!t.startsWith("<think>")) return null;
  const end = t.indexOf("</think>");
  if (end < 0) return { reasoning: t.slice(7), answer: "" };
  return { reasoning: t.slice(7, end), answer: t.slice(end + 8).trimStart() };
}
