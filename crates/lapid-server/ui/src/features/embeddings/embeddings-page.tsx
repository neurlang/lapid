import { useMutation } from "@tanstack/react-query";
import { LoaderCircleIcon, NetworkIcon, PlusIcon, XIcon } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { useState } from "react";
import { NoModel, PageToolbar } from "@/components/layout/empty-state";
import { ModelPicker, useSelectedModel } from "@/components/layout/model-picker";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { api } from "@/lib/api";
import { uid } from "@/lib/utils";
import { Heatmap } from "./heatmap";

const EXAMPLES = [
  "The cat sat quietly on the warm windowsill.",
  "A kitten was napping in the sunshine by the window.",
  "Quarterly revenue grew by twelve percent year over year.",
  "The company reported strong earnings this quarter.",
  "How do I bake sourdough bread at home?",
];

interface Row {
  id: string;
  text: string;
}

function dot(a: number[], b: number[]) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!;
  return s;
}

export function EmbeddingsPage() {
  const { model, isLoading, isError } = useSelectedModel("embedding");
  const [rows, setRows] = useState<Row[]>(() => EXAMPLES.map((text) => ({ id: uid(), text })));
  const [dims, setDims] = useState("");

  const mutation = useMutation({
    mutationFn: async () => {
      if (!model) throw new Error("No embedding model");
      const input = rows.map((r) => r.text.trim()).filter(Boolean);
      const d = dims.trim() ? Number(dims) : undefined;
      const t0 = performance.now();
      const res = await api.embeddings({ model: model.id, input, ...(d ? { dimensions: d } : {}) });
      const vecs = [...res.data].sort((a, b) => a.index - b.index).map((x) => x.embedding);
      return {
        labels: input,
        matrix: vecs.map((a) => vecs.map((b) => dot(a, b))),
        dim: vecs[0]?.length ?? 0,
        tokens: res.usage?.prompt_tokens,
        ms: performance.now() - t0,
      };
    },
  });

  if (!model && !isLoading) {
    return (
      <div className="h-full overflow-y-auto">
        <NoModel icon={NetworkIcon} capability="embedding" offline={isError} />
      </div>
    );
  }

  const update = (id: string, text: string) => setRows((rs) => rs.map((r) => (r.id === id ? { ...r, text } : r)));
  const filled = rows.filter((r) => r.text.trim()).length;
  const result = mutation.data;

  return (
    <div className="flex h-full flex-col">
      <PageToolbar>
        <ModelPicker capability="embedding" />
        {model?.embedding_dim != null && <span className="text-xs text-muted-foreground">native dim {model.embedding_dim}</span>}
      </PageToolbar>
      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto grid max-w-6xl gap-5 p-4 sm:p-6 lg:grid-cols-[minmax(0,5fr)_minmax(0,6fr)]">
          <Card>
            <CardHeader>
              <CardTitle>Sentences</CardTitle>
              <CardDescription>Edit, add or remove. Similarity is the dot product of normalized vectors.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-2">
              <ol className="space-y-2">
                <AnimatePresence initial={false}>
                  {rows.map((r, i) => (
                    <motion.li key={r.id} layout initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }} className="flex items-center gap-2">
                      <span className="w-7 shrink-0 text-right text-xs font-medium text-muted-foreground">S{i + 1}</span>
                      <Input
                        value={r.text}
                        aria-label={`Sentence ${i + 1}`}
                        onChange={(e) => update(r.id, e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") {
                            e.preventDefault();
                            setRows((rs) => [...rs.slice(0, i + 1), { id: uid(), text: "" }, ...rs.slice(i + 1)]);
                          }
                        }}
                      />
                      <Button variant="ghost" size="icon-sm" aria-label={`Remove sentence ${i + 1}`} disabled={rows.length <= 1} onClick={() => setRows((rs) => rs.filter((x) => x.id !== r.id))}>
                        <XIcon />
                      </Button>
                    </motion.li>
                  ))}
                </AnimatePresence>
              </ol>
              <div className="flex flex-wrap items-end gap-2 pt-2">
                <Button variant="outline" size="sm" onClick={() => setRows((rs) => [...rs, { id: uid(), text: "" }])}>
                  <PlusIcon /> Add
                </Button>
                <div className="ml-auto flex items-end gap-2">
                  <div className="space-y-1">
                    <Label htmlFor="dims" className="text-xs">
                      Dimensions
                    </Label>
                    <Input id="dims" inputMode="numeric" placeholder="full" value={dims} onChange={(e) => setDims(e.target.value.replace(/\D/g, ""))} className="h-8 w-24" />
                  </div>
                  <Button size="sm" onClick={() => mutation.mutate()} disabled={!model || filled < 1 || mutation.isPending}>
                    {mutation.isPending && <LoaderCircleIcon className="animate-spin" />}
                    Compute
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Similarity</CardTitle>
              <CardDescription>
                {result
                  ? [`${result.dim} dimensions`, result.tokens != null && `${result.tokens} tokens`, `${Math.round(result.ms)} ms`].filter(Boolean).join(" · ")
                  : "Compute to see the cosine similarity matrix. Hover a cell to highlight its row and column."}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {result ? (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                  <Heatmap matrix={result.matrix} labels={result.labels} />
                </motion.div>
              ) : (
                <div className="flex h-48 items-center justify-center rounded-lg border border-dashed text-sm text-muted-foreground">No result yet</div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
