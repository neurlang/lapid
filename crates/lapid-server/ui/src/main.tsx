import { MutationCache, QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { RouterProvider } from "@tanstack/react-router";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { Toaster, toast } from "sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { useApplyTheme } from "@/hooks/use-theme";
import { errorMessage } from "@/lib/api";
import { router } from "@/router";
import "@/index.css";

const queryClient = new QueryClient({
  defaultOptions: { queries: { refetchOnWindowFocus: true, staleTime: 2000 } },
  mutationCache: new MutationCache({
    onError: (err) => {
      if (err instanceof DOMException && err.name === "AbortError") return;
      toast.error(errorMessage(err));
    },
  }),
});

function App() {
  const theme = useApplyTheme();
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider delayDuration={300}>
        <RouterProvider router={router} />
        <Toaster theme={theme} richColors closeButton position="bottom-right" />
      </TooltipProvider>
    </QueryClientProvider>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
