# local-llm-acp

A first-party Rust binary that bridges the Nwiro UE5 Integration Kit to any
OpenAI-compatible local LLM endpoint (Ollama, LM Studio, llama.cpp, vLLM, or
a remote OpenAI-compatible service). It runs as a child process of
`NwiroIKBridge`, communicates over inherited stdin/stdout using the Agent
Client Protocol (ACP) JSON-RPC framing, and translates ACP session lifecycle
requests into OpenAI `/v1/chat/completions` HTTP calls.

## ACP implementation choice

This crate uses a **from-scratch ACP implementation** in `src/acp/` rather than
the official [`agent-client-protocol`](https://docs.rs/agent-client-protocol/latest/agent_client_protocol/)
Rust SDK. Rationale:

- ACP surface is narrow (8 message types, line-delimited JSON-RPC framing) — ~200 lines.
- The SDK's transitive dependencies risk pulling in `native-tls`, which breaks
  cross-compilation to `aarch64-pc-windows-msvc`. From-scratch keeps the
  dependency closure minimal and `rustls`-only.
- Versions of the SDK weren't verified at the time of writing; pinning a
  specific version without testing is unsafe.

If a future implementer wants to migrate to the SDK, the seam is `src/acp/server.rs`
— the dispatch loop is the only consumer of `acp::frame` and `acp::messages`.

## Build

```bash
cargo build --release
```

Cross-compile is handled by `.github/workflows/release.yml` — see
[`RELEASING.md`](RELEASING.md) for the release-asset naming contract that the
host app's auto-update resolver matches.

## Security notes

- **API key delivery**: `NWIRO_LOCAL_LLM_API_KEY_localllm` env var only. Set
  by the host UE5 bridge immediately before it launches the shim. Never
  appears in CLI args, never logged.
- **`ApiKey` newtype**: Manual `Debug` impl prints `[REDACTED]`. All functions
  taking `ApiKey` are decorated with `#[tracing::instrument(skip(api_key))]`.
- **The bridge is the security boundary**: the host bridge enforces
  `clientCapabilities.terminal: false` unconditionally and rejects
  `terminal/create` with JSON-RPC -32002. The shim's
  `safety.blockCommandExecution` check is defense-in-depth only.
- **Logging**: `tracing-subscriber` reads `RUST_LOG` env var; default `info`.
  Log records never include the API key, even in error paths.
- **No HTTP redirects (anti-SSRF, SEC-KEY-2)**: the HTTP client is built with
  `reqwest::redirect::Policy::none()`, so a backend that answers `/chat/completions`
  with a 3xx is **not** followed — it degrades to a clean refusal, and the prompt, tool
  schemas, and bearer token never reach a redirect target.
- **Bounded streams (anti-DoS)**: a runaway or wedged backend cannot hang or OOM the
  editor. The prompt round is bounded by a per-attempt pre-stream timeout
  (`NWIRO_LOCAL_LLM_PROMPT_PRESTREAM_TIMEOUT_SECS`); the stream by a per-token
  inactivity timeout (`NWIRO_LOCAL_LLM_INACTIVITY_TIMEOUT_SECS`) and a wall-clock turn
  deadline (`NWIRO_LOCAL_LLM_MAX_TURN_DURATION_SECS`); and the accumulated response by a
  hard size ceiling (`NWIRO_LOCAL_LLM_MAX_RESPONSE_BYTES`). Each fires a clean refusal
  with a diagnosable `errorKind`, never a hang or a raw `-32000`.

## Usage

This binary is spawned by `NwiroIKBridge::EnsureProcess()` as a child process
with inherited stdio. It is **not intended to be run interactively** — there is
no CLI argument parsing, and it expects ACP `initialize` as the first message.

**For testers and integrators** running the binary manually against any local
LLM provider (Ollama, LM Studio, llama.cpp, vLLM, or any OpenAI-compatible
endpoint) on Windows, macOS, or Linux: see **[`docs/RUNNING.md`](docs/RUNNING.md)**.
That doc covers binary downloads, per-OS provider install, env vars, smoke
tests, and common gotchas.

**For running larger models** (Kimi-K2, Qwen3, GLM with specific quantizations,
TDR registry tuning, partial CPU offload) — recommended hardware is a
Blackwell-class GPU with 48GB+ VRAM: see
**[`docs/MODEL-SETUP.md`](docs/MODEL-SETUP.md)**.

For the minimal one-line smoke test:

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  | NWIRO_LOCAL_LLM_BASE_URL=http://localhost:11434/v1 \
    NWIRO_LOCAL_LLM_MODEL=qwen3:14b \
    ./target/release/local-llm-acp
```

The shim falls back to `NWIRO_LOCAL_LLM_BASE_URL` and `NWIRO_LOCAL_LLM_MODEL`
env vars when the ACP `initialize` request omits `localLlm.baseUrl` /
`localLlm.model` (e.g. before P1-006 lands in the C++ bridge).

> **Picking the model matters.** The example uses `qwen3:14b` deliberately —
> it is the validated tool-ready reference (Native tool calls at the full
> 220-tool registry). A small or chat-only model put here will *run* but may
> not emit tool calls at scale; see **[Model support](#model-support)** and
> **[`docs/MODEL-COMPATIBILITY.md`](docs/MODEL-COMPATIBILITY.md)** before
> choosing one.
>
> **Base URL per provider.** Ollama listens on `:11434` (shown above). For
> **LM Studio**, the OpenAI-compat server defaults to `:1234`, so use
> `NWIRO_LOCAL_LLM_BASE_URL=http://localhost:1234/v1` (enable it under
> *Developer → Local Server* first). llama.cpp's `llama-server` defaults to
> `:8080`. On Windows the shim normalizes `localhost → 127.0.0.1` internally to
> avoid an IPv6 connect stall.

## Model support

Local models vary widely in their ability to drive Nwiro's ~220-tool registry.
The shim **detects** each model's tool tier and **fails cleanly** when a model is
out of its depth — it never streams garbage to the editor — but it cannot make a
weak model competent (model fitness is out of scope; the fix for a weak model is a
better model).

- **Tool-heavy work:** use **`qwen3:14b` or larger** — the validated 🟢 reference
  (Native tool calls at the full 220-tool registry).
- **Small / chat-only models** (e.g. GLM-4-9B-Chat) run, but **collapse above ~30
  tools** — a 9B model can't hold ~220 tool schemas (~25K tokens) in context, so
  the shim returns one clean refusal instead. They're fine for chat, not for tool
  calling at scale.

The full per-model verdict table (🟢/🟡/🔴, tool ceilings, backends) lives in
**[`docs/MODEL-COMPATIBILITY.md`](docs/MODEL-COMPATIBILITY.md)**; the methodology
and the worked example of *why* a model collapses (Finding J) are in
**[`docs/MODEL-TEST-PLAN.md`](docs/MODEL-TEST-PLAN.md)**.

### Schema-bleed guard & tool-count warning

When a model exceeds its tool ceiling it tends to **schema-bleed** — echo the tool
schema back as text (`"object"`, `"type"`, `"properties"`) instead of emitting a
call. The shim detects this at the post-stream buffer-flush, suppresses the garbage,
and returns exactly one clean refusal line (`stopReason: "refusal"`). Tool
invocation is **model-agnostic** — behaviour keys on the MCP `isError` signal and
the probed tool tier, never on a model's name. A one-shot `tracing::warn!` also
fires when a non-Native model is handed more tools than
`NWIRO_LOCAL_LLM_HIGH_TOOL_WARN` (default 50) — a breadcrumb that a small model may
collapse. The guard is shape-based, so a model that *legitimately* emits a
JSON-schema answer could be suppressed; disable it with
`NWIRO_LOCAL_LLM_BLEED_GUARD=off`.

### Test harness & compatibility data

Model fitness is measured, not guessed. Two layers (see
[`docs/MODEL-TEST-PLAN.md`](docs/MODEL-TEST-PLAN.md)):

- **`scripts/tool-curve.py`** — the shim-free tool-count curve. Hits
  `/v1/chat/completions` directly (no shim, no MCP) and walks
  `1 / 30 / 80 / 150 / 220` tools, recording where each model collapses into
  schema-bleed. Isolates raw **model** capability from shim logic.
- **`scripts/model-matrix.py`** — the end-to-end matrix, driving the shim over
  ACP with an MCP responder so tool round-trips are asserted (C1-C8 pass/fail).

For backend install, `n_ctx` sizing, and per-family chat-template fixes
(especially GLM), see [`docs/MODEL-SETUP.md`](docs/MODEL-SETUP.md).

### Observability

To see exactly what a model passed to a tool and what came back, enable opt-in
tool-I/O logging (off by default; never logs the API key):

```bash
NWIRO_LOCAL_LLM_LOG_TOOL_IO=full       # log every tool call's args + response
NWIRO_LOCAL_LLM_LOG_TOOL_IO=failures  # only calls that errored (or look anomalous)
```

Records are emitted on the `tracing` target `tool_io` (so `RUST_LOG=tool_io=info`
isolates them). Each call logs the **raw** argument string and the tool response;
per-field output is capped by `NWIRO_LOCAL_LLM_LOG_TOOL_IO_MAX_BYTES` (truncation
is always marked). On the wire, a tool failure also carries a typed advisory at
`result._meta.errorKind` so the client can react to *why* a call failed, not just
that it did.

## Limitations

- **Per-model tool ceiling.** Small / chat-only models collapse above their tool
  ceiling (GLM-4-9B ≈ 30). The shim detects the resulting schema-bleed and emits
  one clean refusal — it never streams garbage — but it cannot raise the ceiling.
  The fix is **`qwen3:14b` or larger**, or **filter the tool array** down to the
  model's ceiling (Nwiro-side ToolSelector). See
  [`docs/MODEL-COMPATIBILITY.md`](docs/MODEL-COMPATIBILITY.md) for per-model
  ceilings and [`docs/MODEL-TEST-PLAN.md`](docs/MODEL-TEST-PLAN.md) for the
  mechanism.
- **Concurrent sessions are serialized.** The dispatcher awaits each
  `session/prompt` inline, so one session cannot process a second's prompt while
  streaming or awaiting an MCP round-trip. True concurrency is deferred by design —
  Nwiro is a single-session UE5 editor, so there is no consumer for it and no
  correctness bug (MCP ids are a process-global `AtomicU64`; routing is exact-key).
- **Large tool array needs a large `n_ctx`.** The bridge pushes the full tool
  registry on every prompt (~25-30K tokens for 100+ tools). Backends loaded with a
  small `n_ctx` (LM Studio default 4096) refuse with `n_keep >= n_ctx`. Load with
  `n_ctx ≥ 65536`; see [`docs/MODEL-SETUP.md`](docs/MODEL-SETUP.md).
- **No vision / mixed-content prompts.** `session/prompt` blocks of a type other
  than `text` are silently dropped. Image support is future work, not implemented.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for how to build, test, and submit
changes, and [`SECURITY.md`](SECURITY.md) for the trust model and how to report
a vulnerability.

## License

Licensed under the Apache License, Version 2.0 — see [`LICENSE`](LICENSE).
Third-party dependency licenses are listed in
[`THIRD-PARTY-LICENSES.md`](THIRD-PARTY-LICENSES.md).
