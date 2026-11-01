import { Link, Outlet, useRouterState } from "@tanstack/react-router";
import { AudioLinesIcon, BoxesIcon, type LucideIcon, MessageSquareIcon, NetworkIcon, PanelLeftIcon, Volume2Icon } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { useEffect, useState } from "react";
import { StatusDot } from "@/components/layout/status-dot";
import { ThemeToggle } from "@/components/layout/theme-toggle";
import { Button } from "@/components/ui/button";
import { ConversationList } from "@/features/chat/conversation-list";
import { useMediaQuery } from "@/hooks/use-media";
import { useServerInfo } from "@/hooks/use-server";
import { cn } from "@/lib/utils";
import { useSettings } from "@/stores/settings";

const NAV: { to: string; label: string; icon: LucideIcon }[] = [
  { to: "/", label: "Chat", icon: MessageSquareIcon },
  { to: "/transcribe", label: "Transcribe", icon: AudioLinesIcon },
  { to: "/speak", label: "Speak", icon: Volume2Icon },
  { to: "/embeddings", label: "Embeddings", icon: NetworkIcon },
  { to: "/models", label: "Models", icon: BoxesIcon },
];

function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const info = useServerInfo();
  return (
    <div className="flex h-full w-64 flex-col bg-sidebar">
      <div className="flex h-12 items-center gap-2 px-4">
        <img src="/favicon.svg" alt="" className="size-6" />
        <span className="font-semibold tracking-tight">lapid</span>
        {info.data && <span className="ml-auto text-xs text-muted-foreground">v{info.data.version}</span>}
      </div>
      <nav className="flex flex-col gap-0.5 px-2 py-2" aria-label="Main">
        {NAV.map(({ to, label, icon: Icon }) => {
          const active = to === "/" ? pathname === "/" : pathname.startsWith(to);
          return (
            <Link
              key={to}
              to={to}
              onClick={onNavigate}
              aria-current={active ? "page" : undefined}
              className={cn(
                "flex h-9 items-center gap-3 rounded-md px-3 text-sm outline-none transition-colors focus-visible:ring-[3px] focus-visible:ring-ring/50",
                active ? "bg-accent font-medium text-accent-foreground" : "text-muted-foreground hover:bg-accent/60 hover:text-foreground",
              )}
            >
              <Icon className="size-4" />
              {label}
            </Link>
          );
        })}
      </nav>
      {pathname === "/" && <ConversationList onSelect={onNavigate} />}
    </div>
  );
}

export function AppShell() {
  const isDesktop = useMediaQuery("(min-width: 768px)");
  const desktopOpen = useSettings((s) => s.sidebarOpen);
  const setDesktopOpen = useSettings((s) => s.setSidebarOpen);
  const [mobileOpen, setMobileOpen] = useState(false);
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const title = NAV.find((n) => (n.to === "/" ? pathname === "/" : pathname.startsWith(n.to)))?.label ?? "lapid";

  useEffect(() => {
    document.title = title === "lapid" ? "lapid" : `${title} | lapid`;
  }, [title]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "b") {
        e.preventDefault();
        if (isDesktop) setDesktopOpen(!desktopOpen);
        else setMobileOpen((o) => !o);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [isDesktop, desktopOpen, setDesktopOpen]);

  const toggle = () => (isDesktop ? setDesktopOpen(!desktopOpen) : setMobileOpen(!mobileOpen));

  return (
    <div className="flex h-dvh overflow-hidden">
      {isDesktop ? (
        <aside
          className={cn("shrink-0 overflow-hidden border-r transition-[width] duration-200", desktopOpen ? "w-64" : "w-0 border-r-0")}
          aria-hidden={!desktopOpen}
          inert={!desktopOpen}
        >
          <Sidebar />
        </aside>
      ) : (
        <AnimatePresence>
          {mobileOpen && (
            <>
              <motion.div
                className="fixed inset-0 z-40 bg-black/40"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                onClick={() => setMobileOpen(false)}
              />
              <motion.aside
                className="fixed inset-y-0 left-0 z-50 border-r shadow-xl"
                initial={{ x: -280 }}
                animate={{ x: 0 }}
                exit={{ x: -280 }}
                transition={{ type: "tween", duration: 0.2 }}
                onKeyDown={(e) => e.key === "Escape" && setMobileOpen(false)}
              >
                <Sidebar onNavigate={() => setMobileOpen(false)} />
              </motion.aside>
            </>
          )}
        </AnimatePresence>
      )}
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-12 shrink-0 items-center gap-2 border-b px-2 sm:px-3">
          <Button variant="ghost" size="icon-sm" onClick={toggle} aria-label="Toggle sidebar" title="Toggle sidebar (Ctrl+B)">
            <PanelLeftIcon />
          </Button>
          <h1 className="text-sm font-medium">{title}</h1>
          <div className="ml-auto flex items-center gap-1">
            <StatusDot />
            <ThemeToggle />
          </div>
        </header>
        <main className="min-h-0 flex-1 overflow-hidden">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
