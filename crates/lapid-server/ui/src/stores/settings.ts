import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { Capability } from "@/lib/api";

export type Theme = "light" | "dark" | "system";

export interface Sampling {
  temperature: number;
  top_p: number;
  top_k: number;
  max_tokens: number;
  repetition_penalty: number;
  seed: string; // empty = random
  stop: string; // one sequence per line
  enable_thinking: boolean;
}

export const DEFAULT_SAMPLING: Sampling = {
  temperature: 0.7,
  top_p: 0.95,
  top_k: 40,
  max_tokens: 2048,
  repetition_penalty: 1,
  seed: "",
  stop: "",
  enable_thinking: true,
};

interface SettingsState {
  theme: Theme;
  sidebarOpen: boolean;
  sampling: Sampling;
  models: Partial<Record<Capability, string>>;
  setTheme: (t: Theme) => void;
  setSidebarOpen: (open: boolean) => void;
  setSampling: (s: Sampling) => void;
  setModel: (cap: Capability, id: string) => void;
}

export const useSettings = create<SettingsState>()(
  persist(
    (set) => ({
      theme: "system",
      sidebarOpen: true,
      sampling: DEFAULT_SAMPLING,
      models: {},
      setTheme: (theme) => set({ theme }),
      setSidebarOpen: (sidebarOpen) => set({ sidebarOpen }),
      setSampling: (sampling) => set({ sampling }),
      setModel: (cap, id) => set((s) => ({ models: { ...s.models, [cap]: id } })),
    }),
    {
      name: "lapid.settings",
      version: 1,
      merge: (persisted, current) => {
        const p = (persisted ?? {}) as Partial<SettingsState>;
        return { ...current, ...p, sampling: { ...DEFAULT_SAMPLING, ...p.sampling } };
      },
    },
  ),
);
