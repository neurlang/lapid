import { useQuery } from "@tanstack/react-query";
import { api, type Capability, type ModelInfo } from "@/lib/api";

export function useServerInfo(refetchInterval = 5000) {
  return useQuery({
    queryKey: ["info"],
    queryFn: ({ signal }) => api.info(signal),
    refetchInterval,
    retry: false,
  });
}

export function useHealth() {
  return useQuery({
    queryKey: ["health"],
    queryFn: ({ signal }) => api.health(signal),
    refetchInterval: 5000,
    retry: false,
  });
}

export function useModels(cap: Capability): { models: ModelInfo[]; isLoading: boolean; isError: boolean } {
  const q = useServerInfo();
  return {
    models: (q.data?.models ?? []).filter((m) => m.capabilities?.includes(cap)),
    isLoading: q.isLoading,
    isError: q.isError,
  };
}
