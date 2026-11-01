import type * as React from "react";
import { BoxesIcon, ClockIcon, CpuIcon, HardDriveIcon, ServerIcon } from "lucide-react";
import { motion } from "motion/react";
import { EmptyState } from "@/components/layout/empty-state";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { useServerInfo } from "@/hooks/use-server";
import type { ModelInfo } from "@/lib/api";
import { formatBytes, formatDuration } from "@/lib/utils";

function Field({ label, value, mono }: { label: string; value?: React.ReactNode; mono?: boolean }) {
  if (value == null || value === "") return null;
  return (
    <div className="min-w-0">
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd className={mono ? "truncate font-mono text-xs" : "truncate text-sm"} title={typeof value === "string" ? value : undefined}>
        {value}
      </dd>
    </div>
  );
}

function ModelCard({ m, index }: { m: ModelInfo; index: number }) {
  const queued = m.queued ?? 0;
  return (
    <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: index * 0.04 }}>
      <Card className="h-full">
        <CardHeader>
          <div className="flex items-start gap-2">
            <div className="min-w-0 flex-1">
              <CardTitle className="truncate" title={m.name || m.id}>
                {m.name || m.id}
              </CardTitle>
              {m.name && m.name !== m.id && <CardDescription className="truncate font-mono text-xs">{m.id}</CardDescription>}
            </div>
            {m.busy ? (
              <Badge variant="warning">
                <span className="size-1.5 animate-pulse rounded-full bg-warning" /> busy{queued ? `, ${queued} queued` : ""}
              </Badge>
            ) : (
              <Badge variant="success">idle</Badge>
            )}
          </div>
          <div className="flex flex-wrap gap-1 pt-1">
            {m.capabilities?.map((c) => (
              <Badge key={c} variant="secondary">
                {c}
              </Badge>
            ))}
          </div>
        </CardHeader>
        <CardContent>
          <dl className="grid grid-cols-2 gap-x-4 gap-y-3">
            <Field label="Architecture" value={m.architecture} />
            <Field label="Task" value={m.task} />
            <Field label="Dtype" value={m.dtype} />
            <Field label="Size" value={m.file_size != null ? formatBytes(m.file_size) : undefined} />
            <Field label="Device" value={m.device} />
            <Field label="Load time" value={m.load_ms != null ? `${(m.load_ms / 1000).toFixed(2)} s` : undefined} />
            <Field label="Sample rate" value={m.sample_rate ? `${m.sample_rate} Hz` : undefined} />
            <Field label="Embedding dim" value={m.embedding_dim} />
            <Field label="License" value={m.license} />
            <Field label="Source" value={m.source} mono />
          </dl>
          {m.path && (
            <p className="mt-3 truncate font-mono text-xs text-muted-foreground" title={m.path}>
              {m.path}
            </p>
          )}
          {m.voices && m.voices.length > 0 && (
            <div className="mt-3">
              <p className="mb-1 text-xs text-muted-foreground">Voices</p>
              <div className="flex flex-wrap gap-1">
                {m.voices.map((v) => (
                  <Badge key={v} variant="outline" className={v === m.default_voice ? "border-ring" : undefined}>
                    {v}
                  </Badge>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
}

export function ModelsPage() {
  const info = useServerInfo(3000);

  if (info.isLoading) {
    return (
      <div className="grid gap-4 p-6 md:grid-cols-2">
        <Skeleton className="h-56" />
        <Skeleton className="h-56" />
      </div>
    );
  }
  if (!info.data) {
    return (
      <EmptyState icon={ServerIcon} title="Server unreachable">
        Could not load <code className="font-mono text-xs">/api/info</code>. Retrying every few seconds.
      </EmptyState>
    );
  }
  const d = info.data;

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto flex max-w-6xl flex-col gap-6 p-4 sm:p-6">
        <div className="grid gap-4 sm:grid-cols-3">
          <Card className="gap-1 py-4">
            <CardContent className="flex items-center gap-3">
              <ServerIcon className="size-5 text-muted-foreground" />
              <div>
                <p className="text-xs text-muted-foreground">Server</p>
                <p className="font-medium">
                  {d.name} {d.version}
                </p>
              </div>
            </CardContent>
          </Card>
          <Card className="gap-1 py-4">
            <CardContent className="flex items-center gap-3">
              <ClockIcon className="size-5 text-muted-foreground" />
              <div>
                <p className="text-xs text-muted-foreground">Uptime</p>
                <p className="font-medium tabular-nums">{formatDuration(d.uptime_s)}</p>
              </div>
            </CardContent>
          </Card>
          <Card className="gap-1 py-4">
            <CardContent className="flex items-center gap-3">
              <CpuIcon className="size-5 text-muted-foreground" />
              <div className="min-w-0">
                <p className="text-xs text-muted-foreground">Device</p>
                <p className="truncate font-medium" title={d.device}>
                  {d.device ?? "-"}
                </p>
              </div>
            </CardContent>
          </Card>
        </div>

        <section className="space-y-3">
          <h2 className="flex items-center gap-2 text-sm font-medium">
            <BoxesIcon className="size-4" /> Loaded models <span className="text-muted-foreground">({d.models.length})</span>
          </h2>
          {d.models.length === 0 ? (
            <EmptyState icon={BoxesIcon} title="No models loaded" className="rounded-xl border py-10" />
          ) : (
            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
              {d.models.map((m, i) => (
                <ModelCard key={m.id} m={m} index={i} />
              ))}
            </div>
          )}
        </section>

        {d.devices && d.devices.length > 0 && (
          <section className="space-y-3">
            <h2 className="flex items-center gap-2 text-sm font-medium">
              <HardDriveIcon className="size-4" /> Devices
            </h2>
            <ul className="divide-y rounded-xl border">
              {d.devices.map((dev) => (
                <li key={dev} className="px-4 py-2.5 font-mono text-xs">
                  {dev}
                </li>
              ))}
            </ul>
          </section>
        )}
      </div>
    </div>
  );
}
