import type { HighlighterCore } from "shiki/core";

// Fine-grained Shiki: JS regex engine (no wasm), a handful of grammars, two themes. Loaded on first use.
const LANGS = {
  typescript: () => import("shiki/langs/typescript.mjs"),
  javascript: () => import("shiki/langs/javascript.mjs"),
  tsx: () => import("shiki/langs/tsx.mjs"),
  json: () => import("shiki/langs/json.mjs"),
  bash: () => import("shiki/langs/bash.mjs"),
  python: () => import("shiki/langs/python.mjs"),
  rust: () => import("shiki/langs/rust.mjs"),
  c: () => import("shiki/langs/c.mjs"),
  cpp: () => import("shiki/langs/cpp.mjs"),
  go: () => import("shiki/langs/go.mjs"),
  java: () => import("shiki/langs/java.mjs"),
  html: () => import("shiki/langs/html.mjs"),
  css: () => import("shiki/langs/css.mjs"),
  sql: () => import("shiki/langs/sql.mjs"),
  yaml: () => import("shiki/langs/yaml.mjs"),
  markdown: () => import("shiki/langs/markdown.mjs"),
  diff: () => import("shiki/langs/diff.mjs"),
};

const ALIASES: Record<string, keyof typeof LANGS> = {
  ts: "typescript", js: "javascript", jsx: "tsx", mjs: "javascript", cjs: "javascript",
  sh: "bash", shell: "bash", zsh: "bash", console: "bash", py: "python", rs: "rust",
  "c++": "cpp", cc: "cpp", h: "c", hpp: "cpp", golang: "go", xml: "html", svg: "html",
  yml: "yaml", md: "markdown", patch: "diff", jsonc: "json", json5: "json",
};

let highlighter: Promise<HighlighterCore> | null = null;
const loading = new Map<string, Promise<void>>();

function getHighlighter(): Promise<HighlighterCore> {
  highlighter ??= (async () => {
    const [{ createHighlighterCore }, { createJavaScriptRegexEngine }, light, dark] = await Promise.all([
      import("shiki/core"),
      import("shiki/engine/javascript"),
      import("shiki/themes/github-light.mjs"),
      import("shiki/themes/github-dark.mjs"),
    ]);
    return createHighlighterCore({
      themes: [light.default, dark.default],
      langs: [],
      engine: createJavaScriptRegexEngine({ forgiving: true }),
    });
  })();
  return highlighter;
}

export function resolveLang(lang: string | undefined): keyof typeof LANGS | null {
  if (!lang) return null;
  const l = lang.toLowerCase();
  if (l in LANGS) return l as keyof typeof LANGS;
  return ALIASES[l] ?? null;
}

/** Returns dual-theme HTML (colors as --shiki-light/--shiki-dark vars), or null for unknown languages. */
export async function highlight(code: string, lang: string | undefined): Promise<string | null> {
  const id = resolveLang(lang);
  if (!id) return null;
  const hl = await getHighlighter();
  if (!hl.getLoadedLanguages().includes(id)) {
    let p = loading.get(id);
    if (!p) {
      p = LANGS[id]().then((m) => hl.loadLanguage(m.default));
      loading.set(id, p);
    }
    await p;
  }
  return hl.codeToHtml(code, {
    lang: id,
    themes: { light: "github-light", dark: "github-dark" },
    defaultColor: false,
  });
}
