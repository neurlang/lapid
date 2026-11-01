import { Tooltip } from "@/components/ui/tooltip";
import { useHealth, useServerInfo } from "@/hooks/use-server";
import { cn, formatDuration } from "@/lib/utils";

export function StatusDot() {
  const health = useHealth();
  const info = useServerInfo();
  const ok = health.isSuccess && health.data?.status === "ok" && !health.isError;
  const state = health.isLoading ? "connecting" : ok ? "online" : "offline";
  const busy = info.data?.models.some((m) => m.busy) ?? false;
  const label =
    state === "online"
      ? `Server online${info.data ? `, lapid ${info.data.version}, up ${formatDuration(info.data.uptime_s)}` : ""}${busy ? ", busy" : ""}`
      : state === "connecting"
        ? "Connecting to server"
        : "Server unreachable";
  return (
    <Tooltip content={label} side="bottom">
      <button type="button" className="flex h-8 items-center gap-2 rounded-md px-2 text-xs text-muted-foreground outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50" aria-label={label}>
        <span className="relative flex size-2.5">
          {state === "online" && busy && <span className="absolute inline-flex size-full animate-ping rounded-full bg-success opacity-60" />}
          <span
            className={cn(
              "relative inline-flex size-2.5 rounded-full",
              state === "online" ? "bg-success" : state === "connecting" ? "bg-warning" : "bg-destructive",
            )}
          />
        </span>
        <span className="hidden sm:inline">{state === "online" ? (busy ? "busy" : "online") : state}</span>
      </button>
    </Tooltip>
  );
}
