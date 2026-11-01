import type { LucideIcon } from "lucide-react";
import type * as React from "react";
import { cn } from "@/lib/utils";

export function EmptyState({
  icon: Icon,
  title,
  children,
  className,
}: {
  icon: LucideIcon;
  title: string;
  children?: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex flex-col items-center justify-center gap-3 px-6 py-16 text-center", className)}>
      <div className="flex size-12 items-center justify-center rounded-full bg-muted">
        <Icon className="size-6 text-muted-foreground" />
      </div>
      <h2 className="text-base font-semibold">{title}</h2>
      {children && <div className="max-w-md text-sm text-muted-foreground">{children}</div>}
    </div>
  );
}

/** Shown when the server has no model with the capability the page needs. */
export function NoModel({ icon, capability, offline }: { icon: LucideIcon; capability: string; offline?: boolean }) {
  return (
    <EmptyState icon={icon} title={offline ? "Server unreachable" : `No ${capability} model loaded`}>
      {offline ? (
        <>Start <code className="rounded bg-muted px-1 py-0.5 font-mono text-xs">lapid serve</code> and this page will connect automatically.</>
      ) : (
        <>
          Start <code className="rounded bg-muted px-1 py-0.5 font-mono text-xs">lapid serve</code> with a model that supports {capability}. Loaded models
          are listed on the Models page.
        </>
      )}
    </EmptyState>
  );
}

export function PageToolbar({ children, className }: { children: React.ReactNode; className?: string }) {
  return <div className={cn("flex flex-wrap items-center gap-2 border-b px-4 py-2.5", className)}>{children}</div>;
}
