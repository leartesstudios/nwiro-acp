# local-llm-acp — Crate Structure (Phase 2 contract)

> **Crate structure and module contract** for `local-llm-acp`.

## Overview

`local-llm-acp` is a statically-linked Rust binary that runs as a child process of the Nwiro UE5 bridge (`NwiroIKBridge`). It speaks **Agent Client Protocol (ACP)** as a JSON-RPC server on stdin/stdout and translates requests to any **OpenAI-compatible HTTP endpoint** (Ollama, LM Studio, llama.cpp, vLLM, remote OpenAI-compat).

It handles the full ACP session lifecycle (`initialize`, `session/new`, `session/prompt`, `session/cancel`, `session/set_config_option`), streams SSE responses back as `session/update` notifications, and drives stateful MCP tool execution round-trips via `mcp/connect` + `mcp/message`.

Targets 6 platforms (win-x64, win-arm64, mac-x64, mac-arm64, linux-x64, linux-arm64) using `rustls-tls` for cross-compile compatibility.

## Architectural decision: starting point

**DECISION (Q1 resolved):** Use the official `agent-client-protocol` Rust SDK ([crates.io](https://crates.io/crates/agent-client-protocol), [docs.rs](https://docs.rs/agent-client-protocol/latest/agent_client_protocol/)) as the foundation. It exports an `Agent` type (re-exported from `role::acp`) and a `ByteStreams` type for stdio transport. Use `Agent::builder()` + `ConnectTo` patterns to wire the agent to stdin/stdout.

**FALLBACK PATH:** If the SDK's agent-side stdio API turns out to be unsuitable (unstable signature, missing handlers, blocked by a version mismatch), fall back to a from-scratch implementation using `serde_json` + custom line-delimited JSON-RPC framing. The ACP surface is narrow (8 message types) so from-scratch is ~150 lines, low-risk.

The implementer must verify the SDK's agent stdio API on docs.rs / by `cargo add agent-client-protocol --features agent` before committing to it. Document the chosen path in `README.md`.

## Module layout

```
.
├── Cargo.toml                   # workspace metadata + deps + cross-compile profile
├── README.md                    # what the shim is, how to build, security notes
├── RELEASING.md                 # filename convention contract (for CI workflow)
├── .github/workflows/
│   └── release.yml              # cross-compile matrix on push tag v*
├── src/
│   ├── main.rs                  # entry point: read env var, build agent, serve stdio
│   ├── error.rs                 # ShimError thiserror enum
│   ├── acp/
│   │   ├── mod.rs               # public ACP API
│   │   ├── frame.rs             # stdio JSON-RPC framing (line-delimited JSON, fallback if SDK insufficient)
│   │   ├── messages.rs          # ACP request/response types matching bridge surface
│   │   └── server.rs            # request dispatch loop (or thin wrapper around SDK Agent if used)
│   ├── openai/
│   │   ├── mod.rs               # public OpenAI client API
│   │   ├── client.rs            # reqwest-based HTTP client; POST /v1/chat/completions
│   │   ├── messages.rs          # OpenAI request/response types (Chat, Tool, etc.)
│   │   └── stream.rs            # SSE chunk parser using eventsource-stream or manual
│   └── bridge/
│       ├── mod.rs               # ACP ↔ OpenAI translation orchestrator
│       └── tools.rs             # stateful MCP round-trip loop (mcp/connect + mcp/message)
└── tests/
    └── smoke.rs                 # mock-server-based smoke test (optional, defer to follow-up)
```

## Cargo.toml dependencies

Pin to latest stable as of 2026-05-08. **Implementer must run `cargo search` to verify exact current versions before committing the file.**

```toml
[package]
name = "local-llm-acp"
version = "0.1.0"
edition = "2021"
license = "Apache-2.0"

[dependencies]
agent-client-protocol = "<latest>"            # if using the SDK; otherwise omit
tokio = { version = "1", features = ["rt", "macros", "io-std", "io-util", "sync", "time"] }
reqwest = { version = "0.12", default-features = false, features = ["rustls-tls", "json", "stream"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
eventsource-stream = "0.2"  # OR roll SSE parsing inline
thiserror = "1"
anyhow = "1"
futures-util = "0.3"

[dev-dependencies]
mockito = "1"
tokio = { version = "1", features = ["test-util"] }
```

**Cross-compile gotchas the implementer MUST honor:**
- `rustls-tls` feature is **required**; never enable `native-tls` — breaks `aarch64-pc-windows-msvc`.
- `tokio` `current_thread` runtime is sufficient and simpler — use `#[tokio::main(flavor = "current_thread")]` unless a parallel use case emerges (none expected for a single-user local LLM proxy).
- `agent-client-protocol` may itself depend on TLS internals; verify it doesn't pull native-tls transitively.

## Key types (Rust signatures)

The implementer must hit these types exactly so multiple modules stay consistent.

```rust
// src/error.rs
#[derive(Debug, thiserror::Error)]
pub enum ShimError {
    #[error("ACP framing error: {0}")] AcpFraming(String),
    #[error("OpenAI HTTP error: {0}")] OpenAiHttp(String),
    #[error("Configuration error: {0}")] Config(String),
    #[error("MCP round-trip error: {0}")] McpRoundtrip(String),
    #[error("Cancelled by client")] Cancelled,
}
pub type Result<T> = std::result::Result<T, ShimError>;

// src/main.rs (sketch)
#[derive(Clone)]
pub struct ApiKey(String);
impl std::fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ApiKey([REDACTED])")
    }
}
impl ApiKey {
    pub fn from_env() -> Option<Self> {
        std::env::var("NWIRO_LOCAL_LLM_API_KEY_localllm").ok().map(Self)
    }
    /// SAFETY: only call from openai::client to construct the Authorization header.
    pub(crate) fn as_bearer(&self) -> &str { &self.0 }
}

// src/acp/messages.rs (subset — full ACP surface in §"ACP message inventory" below)
#[derive(serde::Deserialize)] pub struct InitializeRequest { /* … */ }
#[derive(serde::Serialize)]   pub struct InitializeResponse { /* serverCapabilities, serverInfo */ }
#[derive(serde::Deserialize)] pub struct SessionNewRequest { /* sessionId */ }
#[derive(serde::Deserialize)] pub struct SessionPromptRequest { /* sessionId, content, tools? */ }
#[derive(serde::Deserialize)] pub struct SessionCancelNotification { /* sessionId */ }
#[derive(serde::Deserialize)] pub struct SetConfigOptionRequest { /* sessionId, configId, value */ }
#[derive(serde::Serialize)]   pub struct SessionUpdateNotification { /* sessionId, content_chunk */ }

// src/openai/messages.rs (subset)
#[derive(serde::Serialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub stream: bool,
    pub tools: Option<Vec<Tool>>,
}

// src/bridge/mod.rs
pub struct SessionState {
    pub session_id: String,
    pub current_model: String,
    pub history: Vec<ChatMessage>,
    pub cancel_token: tokio_util::sync::CancellationToken,
}

pub async fn handle_session_prompt(
    req: SessionPromptRequest,
    state: &mut SessionState,
    client: &openai::Client,
    write_update: impl Fn(SessionUpdateNotification),
) -> Result<()>;
```

## ACP message inventory (the host bridge wire contract)

**Bridge → shim** (shim must handle):
| Method | Direction | Notes |
|---|---|---|
| `initialize` | request | Shim reads `localLlm.baseUrl` and `localLlm.model` from context (requires P1-006); responds with capabilities EXCLUDING `terminal` when `safety.blockCommandExecution=true` |
| `session/new` | request | Creates `SessionState` for `sessionId` |
| `session/set_config_option` | request | Sent before each prompt when model changes (DoSendPrompt:1240); update `SessionState.current_model` |
| `session/prompt` | request | Translate to OpenAI; stream `session/update` back |
| `session/cancel` | notification (no id) | Call `cancel_token.cancel()`; remove session from registry |

**Shim → bridge** (shim originates):
| Method | Direction | Notes |
|---|---|---|
| `session/update` | notification (no id) | One per SSE chunk; payload includes content delta |
| `mcp/connect` | request | Get `connectionId` for MCP tool calls |
| `mcp/message` | request | Single tool call execution; response goes back into OpenAI message history as `role: tool` |

## Security implementation plan

### a. Env var read path

`src/main.rs`:
```rust
let api_key = ApiKey::from_env(); // None when local endpoint has no auth
let client = openai::Client::new(base_url, model, api_key);
// api_key is moved into client; nothing else holds it
```

### b. Terminal capability rejection

`src/acp/server.rs` `handle_initialize`:
```rust
let advertise_terminal = !req.client_capabilities.safety.block_command_execution;
let mut caps = ServerCapabilities::default_for_chat();
if !advertise_terminal { caps.terminal = false; }
```
**Note:** Bridge enforces this unconditionally via `DoInitialize:1074` and `HandleMethod:1815` (rejects `terminal/create` with -32002). Shim's check is **defense-in-depth**, not the security boundary.

### c. No-log discipline

- `ApiKey` newtype with manual `Debug` → `[REDACTED]`. No `#[derive(Debug)]` on any struct containing the raw key.
- All functions taking `ApiKey` decorated with `#[tracing::instrument(skip(api_key))]`.
- `ShimError::Display` impls never include raw key values.
- `tracing` filter in `main.rs` ensures span attrs that contain `api_key` field are stripped.

## Cross-compile readiness

| Target | Build approach | Gotcha |
|---|---|---|
| `x86_64-pc-windows-msvc` | Native Windows runner | None (with rustls) |
| `aarch64-pc-windows-msvc` | Native or `cargo-zigbuild` from Linux | native-tls breaks here — rustls only |
| `x86_64-apple-darwin` | Native macOS runner | None |
| `aarch64-apple-darwin` | Native macOS runner (M-series) | None |
| `x86_64-unknown-linux-gnu` | Native Linux runner with `cargo-zigbuild` | Pin to `glibc 2.28` for max compat |
| `aarch64-unknown-linux-gnu` | `cargo-zigbuild` from Linux | None (with rustls) |

## Deployment / CI

- GitHub Actions matrix workflow in `.github/workflows/release.yml`
- Triggers on `push: tags: 'v*'`
- Produces assets named exactly per `RELEASING.md` (matched by the host app's auto-update resolver)
- macOS runners: native build for x86_64 and aarch64
- Linux/Windows targets: `cargo-zigbuild` for cross-compile from a single Linux runner

## Open questions deferred to implementer

1. **Q1 — RESOLVED**: SDK exists; try it first, fall back to from-scratch if API surface insufficient. Document chosen path in `README.md`.
2. **Q2 — HARD BLOCKER for E2E (not for code-write)**: Phase 1 needs **P1-006** — extract `localLlm.baseUrl` and `localLlm.model` in C++ `SetAdapterContext()` and inject them into `DoInitialize()` ACP message. Tracked in PLAN.md as a follow-up task. Shim implementation can proceed without it; E2E test can't.
3. **Q3 — Tokio runtime**: `current_thread` recommended (single-user local proxy, simpler). Implementer may choose `multi_thread` if there's a concrete reason.
4. **Q4 — Stdout buffering**: Flush after every line (no BufWriter) to prevent deadlock if buffer fills while awaiting bridge response.
5. **Q5 — `session/set_config_option` timing**: Bridge sends it before prompt (DoSendPrompt:1240). Apply atomically; stored model is read at next prompt start, not mid-prompt.
6. **Q6 — Transient OpenAI HTTP retries**: For local endpoints, retries are rarely useful (server is down or up). Surface 5xx as ACP error immediately. No retry logic.
7. **Q7 — Cargo version pins**: All version numbers in this doc are approximate. Implementer must run `cargo search` or check `crates.io` directly before committing `Cargo.toml`.

## Implementation breakdown

The work splits naturally into three units:

- **ACP layer**: `Cargo.toml`, `src/main.rs`, `src/error.rs`, `src/acp/*`, `README.md`
- **Bridge layer**: `src/openai/*`, `src/bridge/*`
- **CI**: `.github/workflows/release.yml`, `RELEASING.md`

All three follow THIS document as their contract.
