import { readdirSync, readFileSync, statSync, unlinkSync, writeFileSync } from "node:fs";
import path from "node:path";
import { gzipSync } from "node:zlib";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vite";

const COMPRESSIBLE = /\.(js|mjs|css|html|svg|json|wasm|txt|map)$/;

function walk(dir: string): string[] {
  return readdirSync(dir)
    .sort()
    .flatMap((name) => {
      const p = path.join(dir, name);
      return statSync(p).isDirectory() ? walk(p) : [p];
    });
}

/**
 * Replaces each compressible asset > 1 KB with a deterministic `.gz` (level 9). Only the
 * compressed copy is embedded in the binary; the server sends it as is (every browser accepts
 * gzip) and decompresses on the fly for the rare client that doesn't.
 */
function precompress(): Plugin {
  let outDir = "";
  return {
    name: "lapid-precompress",
    apply: "build",
    configResolved(config) {
      outDir = path.resolve(config.root, config.build.outDir);
    },
    closeBundle() {
      for (const file of walk(outDir)) {
        if (!COMPRESSIBLE.test(file)) continue;
        const data = readFileSync(file);
        if (data.length <= 1024) continue;
        const gz = gzipSync(data, { level: 9 });
        gz[9] = 3; // pin the gzip OS byte (unix) so output is identical across platforms
        writeFileSync(`${file}.gz`, gz);
        unlinkSync(file);
      }
    },
  };
}

/** KaTeX ships woff2 + woff + ttf for every font; keep only woff2 (all target browsers support it). */
function katexWoff2Only(): Plugin {
  return {
    name: "lapid-katex-woff2-only",
    enforce: "pre",
    transform(code, id) {
      if (!/katex(\.min)?\.css$/.test(id.split("?")[0] ?? "")) return null;
      return code.replace(/,\s*url\([^)]*\.(woff|ttf)\)\s*format\("(woff|truetype)"\)/g, "");
    },
  };
}

const target = process.env.LAPID_SERVER ?? "http://127.0.0.1:8080";

export default defineConfig({
  base: "/",
  plugins: [katexWoff2Only(), react(), tailwindcss(), precompress()],
  resolve: {
    alias: { "@": path.resolve(import.meta.dirname, "src") },
  },
  server: {
    proxy: {
      "/v1": { target, changeOrigin: true },
      "/api": { target, changeOrigin: true },
      "/health": { target, changeOrigin: true },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    assetsDir: "assets",
    sourcemap: false,
    chunkSizeWarningLimit: 800,
  },
});
