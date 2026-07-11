# Changelog

All notable changes to `local-llm-acp` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

`local-llm-acp` is a LOCAL-LLM-ONLY ACP→OpenAI stdio shim whose only client is the
nwiro UE5 plugin. This file is the human-readable release summary; for the full
per-version trail see the git history.

## [Unreleased]

### Fixed
- **`session/cancel` destroyed the whole session**, wedging host chats with
  `-32000 "ACP framing error: unknown session: <id>"` on every subsequent
  `session/prompt` after a Stop or idle/watchdog cancel (the host bridge treats
  cancel as turn-scoped and keeps its sessionId). Cancel is now turn-scoped: it
  still interrupts the in-flight generation immediately (the backend request is
  aborted via the same cancellation token) and drains that turn's in-flight
  state, but the session entry and its in-memory conversation history survive —
  a follow-up prompt on the same sessionId succeeds with full context. A
  `session/cancel` with no active turn (or for an unknown id) is a successful
  no-op.
- **Schema-aware coercion of stringified tool arguments**: local models frequently
  double-encode a structured parameter as a JSON *string* (e.g. `add_variables:
  "[{\"name\":\"IsActive\",\"type\":\"bool\"}]"`), which the host bridge's typed field
  reader then type-fails, silently skipping the operation. The shim now parses such a
  string and dispatches the real value — but ONLY when the tool's inputSchema declares
  that property as exactly one non-string JSON type (`array`/`object`/`boolean`/
  `number`/`integer`). String-typed, union-typed (`type: [..]`), `oneOf`/`anyOf`, and
  schema-less properties are never touched, and a string that fails to parse (or parses
  to a non-matching type) dispatches verbatim — the host plugin owns
  validation/rejection. Each coercion is logged (tool + field names, never values).
  Known limitation: union-typed (`oneOf`/`anyOf`) properties are deliberately never
  coerced.

### Added
- **Structured `error.data` on unknown-session prompt errors**: the JSON-RPC error
  for a `session/prompt` against an unknown sessionId keeps its exact code
  (`-32000`) and message text (`ACP framing error: unknown session: <id>`) and now
  additionally carries `error.data = {"reason": "unknown_session", "sessionId":
  "<id>"}`, so the host bridge can distinguish this case from other `-32000`
  errors without parsing the message string.

<!-- ──────────────────────────────────────────────────────────────────────
     RELEASE-SPLIT NOTE: the section below targets the NEXT MINOR (v0.5.0).
     The Fixed/Added entries ABOVE ship as v0.4.1. Split at release time.
     ────────────────────────────────────────────────────────────────────── -->

### Added (unreleased — target v0.5.0: session persistence)
- **Session persistence + `loadSession: true`**: the shim now persists each
  session's durable conversation state (history, per-session model, tool tier,
  learned tool ceiling, pruned-turn count) to one JSON envelope per session and
  advertises `agentCapabilities.loadSession: true` on `initialize`, so the host
  bridge can resume a prior conversation across a shim restart via ACP
  `session/load`.
  - **Storage**: `<cwd>/Saved/NwiroIntegrationKit/shim-sessions/<encoded-session-id>.json`,
    where `cwd` is the absolute project directory the host supplies on
    `session/new` / `session/load`. Writes happen at turn end and on
    model/tool-tier config changes, via an atomic same-directory temp-file +
    rename (fsync best-effort); a write failure logs and never fails the turn.
    Session ids are percent-encoded onto an `[A-Za-z0-9_-]` allowlist so a
    hostile id cannot escape the storage directory.
  - **`session/load` semantics**: restores state and returns an **empty object**
    result — nothing is replayed (no `session/update` frames; the host
    suppresses replayed chunks). The same sessionId is immediately live for
    `session/prompt` with a fresh cancel token; MCP reconnects per normal turn
    flow. ANY anomaly (unknown id, corrupt file, `schema_version` mismatch,
    envelope/requested id mismatch, invalid cwd, persistence disabled) answers
    JSON-RPC **`-32002` "session not found: \<id\>"**, which the host treats as
    resource-not-found and silently falls back to `session/new`.
  - **Kill switch**: `NWIRO_SHIM_PERSIST` (default ON; `0`/`false`/`off`
    disables). Disabled ⇒ `loadSession: false` is advertised, nothing is
    written, and `session/load` answers `-32002`.
  - **`NWIRO_SHIM_STATE_DIR`** overrides the storage ROOT (it replaces
    `<cwd>/Saved/NwiroIntegrationKit`; must be absolute). **Privacy warning:**
    persisted history can contain project file contents and tool results —
    pointing the override at a shared or synced directory moves that data
    outside the project. The default location keeps it inside the project.
  - **Eviction**: per storage dir, the newest ~50 session files are kept and
    files older than ~30 days are deleted (after successful writes and at first
    storage use per process); stale `*.tmp` leftovers are cleaned. Eviction
    errors log and never block.
  - The envelope is a versioned contract (`schema_version: 1`) — see the new
    AGENTS.md invariant. The flag-gated connector path (non-default) does not
    participate: its `session/load` answers `-32002`.

## [0.3.0] — 2026-06-16

Prompt-path resilience hardening: three layered timeout guards (inactivity, pre-stream,
and the existing wall-clock) plus exponential retry, so a stalled, wedged, or flaky
local backend always fails fast with a diagnosable `errorKind` instead of hanging the
editor — all model- and backend-agnostic.

### Added
- **SEC-DOS-1 per-token inactivity timeout** (`NWIRO_LOCAL_LLM_INACTIVITY_TIMEOUT_SECS`,
  default 120 s, `0` disables). Aborts the turn with `errorKind=stream_inactivity_timeout`
  if the backend emits no token for the configured window — a silent-stall guard that
  complements the wall-clock `MAX_TURN_DURATION_SECS` (runaway emission) and the
  `MAX_RESPONSE_BYTES` ceiling. Resets on every received token.
- **Per-attempt pre-stream timeout** (`NWIRO_LOCAL_LLM_PROMPT_PRESTREAM_TIMEOUT_SECS`,
  default 30 s, `0` disables). Bounds each prompt attempt's pre-stream phase — the
  request send PLUS the admission-gate body reads (a non-2xx error body, the LM Studio
  `200 + application/json` "model unloaded" envelope) — so a backend that accepts the
  connection then never sends response headers (or sends headers then stalls the body)
  fails fast with `errorKind=timeout` instead of hanging. (`connect_timeout` bounds only
  the TCP connect, not the wait for response headers.) Clamped above the connect timeout
  so a slow connect still surfaces as `unreachable`. The retry count auto-reduces so
  total pre-stream time (`attempts × cap`) stays under nwiro's ~300 s first-token
  watchdog — a raised `CONNECT_TIMEOUT_SECS` trades retries for one longer attempt
  rather than blowing the watchdog. Default worst case `3×30 s + backoffs ≈ 91 s`. Never
  covers the streamed SSE body (the SEC-DOS-1 inactivity guard owns that).
- **Exponential prompt-path retry backoff** plus a third attempt (`MAX_PROMPT_ATTEMPTS`
  2 → 3, now safe because each attempt's pre-stream phase is time-bounded). Transient
  classes (`rate_limited` / `timeout` / `server_error`) back off `base · 2^retry`
  (clamped to 2 s); `rate_limited` still honors a `Retry-After` hint.
- **Warmup wait surfaced at start.** The effective warmup timeout
  (`NWIRO_LOCAL_LLM_WARMUP_TIMEOUT_SECS`, default 300 s) is now logged when warmup
  *begins* — the model load the editor spinner blocks on — instead of only on failure.
  The other `NWIRO_LOCAL_LLM_*` knobs remain env-configured and unlogged.

### Notes
- **P0-C scope clarification:** the v0.2.7 prompt-path error-taxonomy remap (typed
  `errorKind` + the generic bridge degrader replacing the flat `-32000`) is COMPLETE.
  The follow-on P1 hardening it named — exponential-backoff tuning and a per-attempt
  pre-stream timeout cap — LANDED in this release (see Added). Both were always P1
  hardening, NOT residual P0-C work.

## [0.2.7] — 2026-06-16

General, model- and backend-agnostic robustness for the LM Studio and llama-server
(llama.cpp) local backends. Every degrade is keyed off behaviour (finish reason,
content emptiness, argument validity), never model names, so new reasoning models and
flaky backends are covered without per-model patches.

### Added
- **Context-overflow tool budgeting.** On a backend context-overflow (HTTP 400),
  parse the measured prompt-token count (`n_keep` for LM Studio, `n_prompt_tokens`
  for llama-server) and tail-trim the tool array to fit, with bounded retries that
  recompute from each fresh overflow. Falls back to a conservative chars/3 estimate
  when neither field is present. The learned per-session tool ceiling is committed
  only after a successful retry, so a still-overflowing retry cannot poison the cache.
- **`reasoning_budget_exhausted` degrade.** A `length`/`max_tokens` finish with empty
  content and no tool call now surfaces a helpful message + clean refusal instead of
  an empty turn, and repairs history so the empty assistant turn cannot poison the
  next one.
- **`NWIRO_LOCAL_LLM_MAX_TOKENS`** generation cap (default 16384, `0` disables) as a
  runaway-reasoning guardrail.
- **Runaway/unbounded-stream protection:** a hard response-size ceiling
  (`NWIRO_LOCAL_LLM_MAX_RESPONSE_BYTES`, default 8 MiB → `errorKind=response_too_large`)
  plus a per-turn wall-clock deadline (`NWIRO_LOCAL_LLM_MAX_TURN_DURATION_SECS`,
  default 1800 s → `errorKind=turn_timeout`).
- CI `verify-version` gate: a release tag must match `Cargo.toml`'s version.

### Changed
- Rewrote the tool-invocation mandate (the sole runtime lever — the request sets no
  `tool_choice`) to add the non-action path: answer greetings/questions directly
  without a tool, decide quickly, and never echo the tool list. Stops weak reasoners
  from exhausting their budget or firing a random tool on plain chat.

### Fixed
- `server_error` degrade: a backend HTTP 5xx now surfaces a sanitized refusal instead
  of leaking the raw error string into the UE5 chat as a `-32000`.
- `execute_tool` rejects malformed or non-object tool-call arguments as a clean tool
  failure instead of silently dispatching `{}` (which previously ran the wrong
  side-effecting action).

### Security
- Bounded the runaway/unbounded stream (size ceiling + wall-clock deadline +
  `max_tokens`), closing the DoS-by-repetition-loop path that could OOM the editor.

## [0.2.6]

- Fixed the LM Studio / llama.cpp tool-capability probe: those backends reject the
  object-form `tool_choice` with HTTP 400, which fail-opened the probe to Emulated and
  mis-tiered every native LM Studio model. The probe now retries without `tool_choice`.

## [0.2.5]

- Fixed the shim tool-result envelope being nested one level too deep
  (`rawOutput.result.content`), which produced an empty result in the UI and a green
  badge on errored tools.

## [0.2.1]

- Bounded the warmup model-load request with `NWIRO_LOCAL_LLM_WARMUP_TIMEOUT_SECS`
  (default 300 s, `0` = unbounded); a timed-out warmup now fails fast with
  `errorKind=timeout` instead of hanging the UE5 spinner.

## [0.2.0]

- Model-robustness track complete: the 14-model matrix runs **0 BLACK**
  (review-validated). Wave 0 (spec-conformance fixes) and Wave 1 (the connector seam
  behind `LOCAL_LLM_USE_CONNECTOR`) landed. The shim's critical path handed off to
  nwiro (the `isError:true` keystone).

## [0.1.x]

Pre-0.2.0 history (v0.1.0 – v0.1.39): tool-tier probing, the emulated tool-call
parser, schema-bleed detection, real-time streaming with bounded mpsc, the
cancellation fast-path, circuit breakers, history pruning, and the MCP correlation
map. The full per-version record lives in the git history.

<details>
<summary><strong>Previously resolved limitations (v0.1.18 – v0.1.37)</strong> — once listed as open in the README; moved here so the README stays lean.</summary>

- Real-time `session/update` streaming (v0.1.18).
- Mid-prompt cancellation via `session/cancel` (v0.1.18, streaming path).
- Emulated-tier tool execution via Qwen XML / inline JSON / Markdown parsers in
  `src/bridge/emulated_parser.rs` (v0.1.17, extended v0.1.19).
- Backend error envelope unwrap inside SSE streams (v0.1.21) — surfaces
  `n_keep >= n_ctx` and similar mid-stream errors cleanly.
- Lazy tool-capability re-probe so a model that warms up `None` can recover its
  tier without a session restart (v0.1.33).
- Schema-bleed reaching the UI as prose: detected at buffer-flush, garbage
  suppressed, one clean `stopReason: "refusal"` emitted instead (v0.1.33, Finding J).
- The tool result handed back to the model is now the tool's **text**, not the raw
  MCP envelope, and a typed `errorKind` advisory is published on `result._meta` +
  `WarmupResult` (v0.1.34).
- **Model-agnostic** tool invocation: the GLM-only name-based predicates were
  deleted; behaviour now keys solely on the MCP `isError` signal and the probed
  tool tier (v0.1.35).
- Opt-in tool-I/O observability (`NWIRO_LOCAL_LLM_LOG_TOOL_IO`) — see a tool's args
  and response in the trace (v0.1.36).
- **Mid-tool `session/cancel` is now responsive** (v0.1.37, Finding C): a cancel
  arriving while the shim awaits an MCP round-trip used to wait the full 30 s
  timeout and surface a spurious tool failure. The MCP-await is now cancel-aware on
  both the legacy and connector paths (a cancel sentinel maps to
  `stopReason: cancelled`); cancel now lands in well under 500 ms with no bogus
  tool-failure frame.

</details>

[0.3.0]: https://github.com/leartesstudios/nwiro-acp/releases
[0.2.7]: https://github.com/leartesstudios/nwiro-acp/releases
[0.2.6]: https://github.com/leartesstudios/nwiro-acp/releases
[0.2.5]: https://github.com/leartesstudios/nwiro-acp/releases
[0.2.1]: https://github.com/leartesstudios/nwiro-acp/releases
[0.2.0]: https://github.com/leartesstudios/nwiro-acp/releases
