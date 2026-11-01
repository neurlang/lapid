import { createRootRoute, createRoute, createRouter, lazyRouteComponent } from "@tanstack/react-router";
import { AppShell } from "@/components/layout/app-shell";
import { EmptyState } from "@/components/layout/empty-state";
import { CompassIcon } from "lucide-react";

const rootRoute = createRootRoute({
  component: AppShell,
  notFoundComponent: () => (
    <EmptyState icon={CompassIcon} title="Page not found">
      Nothing lives at this address. Pick a page from the sidebar.
    </EmptyState>
  ),
});

const routes = [
  createRoute({ getParentRoute: () => rootRoute, path: "/", component: lazyRouteComponent(() => import("@/features/chat/chat-page"), "ChatPage") }),
  createRoute({
    getParentRoute: () => rootRoute,
    path: "/transcribe",
    component: lazyRouteComponent(() => import("@/features/transcribe/transcribe-page"), "TranscribePage"),
  }),
  createRoute({ getParentRoute: () => rootRoute, path: "/speak", component: lazyRouteComponent(() => import("@/features/speak/speak-page"), "SpeakPage") }),
  createRoute({
    getParentRoute: () => rootRoute,
    path: "/embeddings",
    component: lazyRouteComponent(() => import("@/features/embeddings/embeddings-page"), "EmbeddingsPage"),
  }),
  createRoute({ getParentRoute: () => rootRoute, path: "/models", component: lazyRouteComponent(() => import("@/features/models/models-page"), "ModelsPage") }),
];

export const router = createRouter({
  routeTree: rootRoute.addChildren(routes),
  defaultPreload: "intent",
  defaultPendingMinMs: 0,
});

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
