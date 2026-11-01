# lapid web UI

The browser UI served by `lapid serve`: chat, transcription, speech, embeddings and a models overview.
React 19, Vite, TypeScript, Tailwind v4 and shadcn-style components.

## Develop

```sh
pnpm install
pnpm dev
```

The dev server proxies `/v1`, `/api` and `/health` to a running `lapid serve` at
`http://127.0.0.1:8080`. Point it elsewhere with `LAPID_SERVER=http://host:port pnpm dev`.

## Build

```sh
pnpm build
```

This type-checks and writes `dist/`, including precompressed `.br` and `.gz` siblings for
compressible assets over 1 KB. The build is deterministic.

`dist/` is committed. `cargo build` embeds it into the `lapid` binary, so building the Rust
crates never needs Node or pnpm. After changing anything under `ui/`, run `pnpm build` and commit
the updated `dist/`. `tools/check_ui_dist.sh` verifies that the committed `dist/` matches the sources.
