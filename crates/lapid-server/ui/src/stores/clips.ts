import { create } from "zustand";
import { uid } from "@/lib/utils";

export interface Clip {
  id: string;
  url: string;
  blob: Blob;
  text: string;
  voice?: string;
  model: string;
  elapsedMs: number;
  createdAt: number;
}

const MAX = 12;

/** Session-only history of generated speech clips (object URLs are revoked on eviction). */
export const useClips = create<{
  clips: Clip[];
  add: (c: Omit<Clip, "id" | "url" | "createdAt">) => Clip;
  remove: (id: string) => void;
}>()((set, get) => ({
  clips: [],
  add: (c) => {
    const clip: Clip = { ...c, id: uid(), url: URL.createObjectURL(c.blob), createdAt: Date.now() };
    const next = [clip, ...get().clips];
    for (const old of next.slice(MAX)) URL.revokeObjectURL(old.url);
    set({ clips: next.slice(0, MAX) });
    return clip;
  },
  remove: (id) => {
    const c = get().clips.find((x) => x.id === id);
    if (c) URL.revokeObjectURL(c.url);
    set({ clips: get().clips.filter((x) => x.id !== id) });
  },
}));
