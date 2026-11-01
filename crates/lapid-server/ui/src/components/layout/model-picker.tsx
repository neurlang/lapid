import { useEffect } from "react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useModels } from "@/hooks/use-server";
import type { Capability, ModelInfo } from "@/lib/api";
import { cn } from "@/lib/utils";
import { useSettings } from "@/stores/settings";

/** Returns the selected model for a capability, falling back to the first available one. */
export function useSelectedModel(cap: Capability): { model: ModelInfo | undefined; models: ModelInfo[]; isLoading: boolean; isError: boolean } {
  const { models, isLoading, isError } = useModels(cap);
  const stored = useSettings((s) => s.models[cap]);
  const setModel = useSettings((s) => s.setModel);
  const model = models.find((m) => m.id === stored) ?? models[0];
  useEffect(() => {
    if (model && model.id !== stored) setModel(cap, model.id);
  }, [model, stored, cap, setModel]);
  return { model, models, isLoading, isError };
}

export function ModelPicker({ capability, className }: { capability: Capability; className?: string }) {
  const { model, models } = useSelectedModel(capability);
  const setModel = useSettings((s) => s.setModel);
  return (
    <Select value={model?.id ?? ""} onValueChange={(v) => setModel(capability, v)} disabled={models.length === 0}>
      <SelectTrigger className={cn("w-auto max-w-72 min-w-44", className)} aria-label={`${capability} model`}>
        <SelectValue placeholder="No model loaded" />
      </SelectTrigger>
      <SelectContent>
        {models.map((m) => (
          <SelectItem key={m.id} value={m.id}>
            <span className="truncate">{m.name || m.id}</span>
            {m.dtype && <span className="ml-1 text-xs text-muted-foreground">{m.dtype}</span>}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
