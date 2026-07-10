# AGENTS.md

Guidance for coding agents (and humans) working in this repo. Keep changes small,
match the surrounding style, and verify before claiming done.

## What this is

`local-llm-acp` is a Rust ACP-to-OpenAI stdio shim: it speaks the Agent Client
Protocol (ACP) over stdin/stdout and translates to OpenAI `/v1/chat/completions`.
The module layout and design decisions are documented in `STRUCTURE.md`.

## Build / test / verify

```bash
cargo build --all-targets
cargo test --bins
LOCAL_LLM_USE_CONNECTOR=1 cargo test --bins golden
cargo deny check advisories licenses
```

Definition of done: the four commands above pass, and any behavior change is
reflected in `CHANGELOG.md`.

## Invariants (do not break)

- **`rustls`-only.** Never add `native-tls` or a dependency that pulls it in — it
  breaks the `aarch64-pc-windows-msvc` target.
- **The ACP seam is `src/acp/server.rs`** — the dispatch loop is the only consumer of
  `acp::frame` and `acp::messages`.
- **Don't hand-edit golden snapshots** (`*.snap`); regenerate with `cargo insta review`.
- **Public env-var ABI:** the `NWIRO_LOCAL_LLM_*` variables are a stable contract the
  host bridge consumes — don't rename them without a deliberate, versioned migration.
- **Tool-arg coercion strict gate:** `coerce_args_to_schema` (`src/bridge/tools.rs`)
  only ever coerces a stringified value whose schema property declares exactly ONE
  non-string JSON type (`array`/`object`/`boolean`/`number`/`integer`). Never loosen it
  to touch string-typed, union-typed (`type: [..]`, `oneOf`/`anyOf`), or schema-less
  properties — the host plugin owns validation and rejection.

## Scope boundary

This repo is the **shim only**. The UE5 host (the "Nwiro Integration Kit") is a
separate, private codebase. Do not add references to the host's internal source
files, line numbers, or commit hashes here — describe protocol/wire behavior in
neutral terms ("the host bridge ...") instead.
