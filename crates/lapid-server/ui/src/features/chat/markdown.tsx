import { isValidElement, memo, type ReactElement, type ReactNode, useEffect, useState } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { normalizeMath } from "@/lib/math";
import { CodeBlock } from "./code-block";

function textOf(node: ReactNode): string {
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(textOf).join("");
  if (isValidElement(node)) return textOf((node.props as { children?: ReactNode }).children);
  return "";
}

const components: Components = {
  pre({ children }) {
    const child = (Array.isArray(children) ? children[0] : children) as ReactElement<{ className?: string; children?: ReactNode }> | undefined;
    const className = child?.props?.className ?? "";
    const lang = /language-([\w+#.-]+)/.exec(className)?.[1];
    const code = textOf(child?.props?.children).replace(/\n$/, "");
    return <CodeBlock code={code} lang={lang} />;
  },
  a({ children, href }) {
    return (
      <a href={href} target="_blank" rel="noreferrer noopener">
        {children}
      </a>
    );
  },
};

type Plugins = { remark: unknown[]; rehype: unknown[] };
const BASE: Plugins = { remark: [remarkGfm], rehype: [] };

// KaTeX (+ its CSS and fonts) is loaded only once a message actually contains math.
let mathPlugins: Plugins | null = null;
let mathLoading: Promise<Plugins> | null = null;
function loadMath(): Promise<Plugins> {
  mathLoading ??= Promise.all([import("remark-math"), import("rehype-katex"), import("katex/dist/katex.min.css")]).then(([rm, rk]) => {
    mathPlugins = {
      remark: [remarkGfm, [rm.default, { singleDollarTextMath: true }]],
      rehype: [[rk.default, { throwOnError: false, strict: "ignore" }]],
    };
    return mathPlugins;
  });
  return mathLoading;
}

const MATH_HINT = /\$|\\\(|\\\[|\\begin\{/;

function usePlugins(text: string): Plugins {
  const needs = MATH_HINT.test(text);
  const [plugins, setPlugins] = useState<Plugins>(() => (needs && mathPlugins) || BASE);
  useEffect(() => {
    if (!needs || plugins !== BASE) return;
    if (mathPlugins) setPlugins(mathPlugins);
    else void loadMath().then(setPlugins);
  }, [needs, plugins]);
  return plugins;
}

export const Markdown = memo(function Markdown({ text }: { text: string }) {
  const plugins = usePlugins(text);
  return (
    <div className="prose-chat text-[15px]">
      <ReactMarkdown remarkPlugins={plugins.remark as never} rehypePlugins={plugins.rehype as never} components={components}>
        {plugins === BASE ? text : normalizeMath(text)}
      </ReactMarkdown>
    </div>
  );
});
