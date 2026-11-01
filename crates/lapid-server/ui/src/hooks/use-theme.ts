import { useEffect, useState } from "react";
import { useSettings } from "@/stores/settings";

const media = () => window.matchMedia("(prefers-color-scheme: dark)");

/** Applies the theme class to <html> and returns the resolved theme. */
export function useApplyTheme(): "light" | "dark" {
  const theme = useSettings((s) => s.theme);
  const [systemDark, setSystemDark] = useState(() => media().matches);
  useEffect(() => {
    const m = media();
    const on = () => setSystemDark(m.matches);
    m.addEventListener("change", on);
    return () => m.removeEventListener("change", on);
  }, []);
  const resolved = theme === "system" ? (systemDark ? "dark" : "light") : theme;
  useEffect(() => {
    document.documentElement.classList.toggle("dark", resolved === "dark");
  }, [resolved]);
  return resolved;
}

/** Resolved theme without side effects (reads the class on <html>). */
export function useResolvedTheme(): "light" | "dark" {
  const theme = useSettings((s) => s.theme);
  const [, force] = useState(0);
  useEffect(() => {
    const obs = new MutationObserver(() => force((n) => n + 1));
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    return () => obs.disconnect();
  }, []);
  void theme;
  return document.documentElement.classList.contains("dark") ? "dark" : "light";
}
