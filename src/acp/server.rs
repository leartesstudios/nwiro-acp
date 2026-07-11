use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::json;
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::sync::oneshot;
use uuid::Uuid;

use crate::{
    acp::{
        frame,
        messages::{
            InitializeParams, SessionCancelParams, SessionLoadParams, SessionNewParams,
            SessionPromptParams, SessionUpdateNotification, SetConfigOptionParams, ToolTier,
            WarmupParams,
        },
    },
    bridge,
    error::{Result, ShimError},
    openai::{self, messages::ChatMessage},
    ApiKey,
};

/// Normalize an Ollama-style model id for tier-match comparison: strip a
/// trailing digest (`@sha256:..`) then a default `:latest` tag, so that
/// `qwen3:14b`, `qwen3:14b:latest`, and `qwen3:14b@sha256:..` all compare equal.
///
/// M-1 root cause: a byte-exact id mismatch in `session/set_config_option`
/// silently collapsed the session to `ToolTier::None`, which strips the
/// outbound `tools` array — so the model saw only tool NAMES in prose and
/// improvised Python (`spawn_actor(...)` as a code block, the "describer trap")
/// instead of emitting a native tool_call. Distinct models (`qwen3:14b` vs
/// `qwen3:4b`, or `qwen3:14b` vs `qwen3`) still compare unequal.
fn normalize_model_id(id: &str) -> &str {
    // Strip a trailing Ollama content digest (`@sha256:...`) if present, then a
    // default `:latest` tag, so `qwen3:14b`, `qwen3:14b:latest`, and
    // `qwen3:14b@sha256:..` all compare equal. ONLY a `sha256:` digest is
    // stripped — an arbitrary `@` suffix (e.g. a registry-qualified id, or
    // `foo@rev1` vs `foo@rev2`) is left intact so distinct revisions can't
    // collapse to the same tier key (review finding).
    let id = match id.rsplit_once('@') {
        Some((base, suffix)) if suffix.starts_with("sha256:") => base,
        _ => id,
    };
    id.strip_suffix(":latest").unwrap_or(id)
}

/// Resolve the session tool tier for a `set_config_option(model)` against the
/// warmup record. Shared by BOTH set_config handlers (the connector path and
/// the primary `handle_set_config_option`) so they can never diverge — the M-1
/// describer trap was one handler getting the tolerant match while the other
/// kept a byte-exact `==`. Tolerates Ollama `:latest`/`@digest` noise so a
/// cosmetic id difference doesn't collapse a warmed tool-capable model to None
/// (which strips the `tools` array). A genuine mismatch, or no warmup, falls to
/// None and is warn-logged.
fn resolve_set_config_tier(
    warmed: &Option<(String, ToolTier)>,
    requested_model: &str,
) -> ToolTier {
    match warmed {
        Some((warmed_model, tier))
            if normalize_model_id(warmed_model) == normalize_model_id(requested_model) =>
        {
            *tier
        }
        Some((warmed_model, _)) => {
            tracing::warn!(
                requested = %requested_model,
                warmed = %warmed_model,
                "set_config model id does not match the warmed model (even after \
                 tag/digest normalization) — tier falls to None and tools are \
                 stripped (M-1 describer trap). Warm up the exact id being prompted."
            );
            ToolTier::None
        }
        None => ToolTier::None,
    }
}

/// Extract a JSON-RPC id as the u64 the shim uses to key `pending_requests`.
/// The shim allocates numeric ids (>= 1_000_000) for its OWN outbound `mcp/*`
/// requests; a conformant peer echoes them back as JSON numbers. Some host
/// bridges stringify ids, so also accept a numeric string (`"1000042"`) — a
/// string echo must still correlate to its pending request instead of being
/// dropped as an orphan and timing the tool round-trip out at 30s.
fn json_rpc_id_as_u64(v: &serde_json::Value) -> Option<u64> {
    v.as_u64()
        .or_else(|| v.as_str().and_then(|s| s.parse::<u64>().ok()))
}

/// Build the `session/set_mode` response. We advertise NO session modes, so the
/// only honest outcomes are: ACK the degenerate default mode (empty or `"auto"`
/// — we are always effectively in `"auto"`, and the host bridge sends
/// `modeId:"auto"` after every `session/new` and ignores the result), and
/// REJECT any other requested mode with `-32602` rather than falsely ACKing a
/// switch that never happened. Keeps the bridge's log quiet while staying
/// truthful to a stricter client that might request a real mode (e.g. `plan`).
fn build_set_mode_response(req_id: serde_json::Value, mode_id: &str) -> serde_json::Value {
    if mode_id.is_empty() || mode_id == "auto" {
        json!({ "jsonrpc": "2.0", "id": req_id, "result": {} })
    } else {
        json!({
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": -32602,
                "message": format!(
                    "unsupported session mode '{mode_id}': this agent advertises no \
                     session modes (only the default 'auto' is accepted)"
                ),
            }
        })
    }
}

#[cfg(test)]
mod router_helpers_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn json_rpc_id_accepts_number_and_numeric_string_else_none() {
        // Conformant peer echoes the shim's numeric id.
        assert_eq!(json_rpc_id_as_u64(&json!(1_000_042u64)), Some(1_000_042));
        // A bridge that stringifies the id must STILL correlate (the C2 fix) —
        // otherwise the mcp/* reply orphans and the tool round-trip times out.
        assert_eq!(json_rpc_id_as_u64(&json!("1000042")), Some(1_000_042));
        // Non-numeric / null / negative ids have no pending entry to match.
        assert_eq!(json_rpc_id_as_u64(&json!("abc")), None);
        assert_eq!(json_rpc_id_as_u64(&json!(null)), None);
        assert_eq!(json_rpc_id_as_u64(&json!(-5)), None);
    }

    #[test]
    fn set_mode_acks_auto_default_but_rejects_unknown_modes() {
        // "auto" and the empty/default mode are the only honest ACKs (we have no
        // real modes); the nwiro bridge only ever sends "auto" and ignores the result.
        for m in ["auto", ""] {
            let r = build_set_mode_response(json!(7), m);
            assert_eq!(r["result"], json!({}), "mode {m:?} should ACK {{}}");
            assert!(r.get("error").is_none());
        }
        // A real mode we don't offer must NOT be falsely ACKed as success.
        let r = build_set_mode_response(json!(7), "plan");
        assert!(r.get("result").is_none(), "unknown mode must not ACK success");
        assert_eq!(r["error"]["code"], json!(-32602));
        assert_eq!(r["id"], json!(7));
    }
}

#[cfg(test)]
mod model_id_norm_tests {
    use super::*;

    #[test]
    fn normalizes_tag_and_digest_noise_but_keeps_distinct_models_distinct() {
        // The same model with default-tag / digest noise must compare EQUAL so the
        // session keeps the warmed tier instead of collapsing to None (M-1).
        assert_eq!(normalize_model_id("qwen3:14b"), normalize_model_id("qwen3:14b:latest"));
        assert_eq!(normalize_model_id("qwen3:14b"), normalize_model_id("qwen3:14b@sha256:deadbeef"));
        assert_eq!(normalize_model_id("qwen3:latest"), "qwen3");
        // Genuinely different models must NOT be conflated.
        assert_ne!(normalize_model_id("qwen3:14b"), normalize_model_id("qwen3:4b"));
        assert_ne!(normalize_model_id("qwen3:14b"), normalize_model_id("qwen3"));
        // Only a `sha256:` digest is stripped — an arbitrary `@` is kept, so
        // distinct revisions don't collapse to one tier key (review finding).
        assert_eq!(normalize_model_id("qwen3:14b@sha256:abc"), "qwen3:14b");
        assert_ne!(normalize_model_id("foo@rev1"), normalize_model_id("foo@rev2"));
        assert_eq!(normalize_model_id("foo@rev1"), "foo@rev1");
    }

    #[test]
    fn set_config_tier_resolution_tolerates_tag_noise_but_refuses_distinct_models() {
        // BOTH set_config handlers delegate to resolve_set_config_tier, so this
        // single test guards the connector path AND handle_set_config_option (the
        // primary path that was still byte-exact — the incomplete-fix bug).
        let warmed = Some(("qwen3:14b".to_string(), ToolTier::Native));
        // cosmetic :latest / @digest differences keep the warmed (Native) tier
        assert_eq!(resolve_set_config_tier(&warmed, "qwen3:14b"), ToolTier::Native);
        assert_eq!(resolve_set_config_tier(&warmed, "qwen3:14b:latest"), ToolTier::Native);
        assert_eq!(
            resolve_set_config_tier(&warmed, "qwen3:14b@sha256:deadbeef"),
            ToolTier::Native
        );
        // a genuinely different model must still collapse to None (filter intact)
        assert_eq!(resolve_set_config_tier(&warmed, "qwen3:4b"), ToolTier::None);
        // no warmup record → None
        assert_eq!(resolve_set_config_tier(&None, "qwen3:14b"), ToolTier::None);
        // a non-Native warmed tier is carried through unchanged on a tolerant match
        let warmed_emu = Some(("llama3.1:8b".to_string(), ToolTier::Emulated));
        assert_eq!(
            resolve_set_config_tier(&warmed_emu, "llama3.1:8b:latest"),
            ToolTier::Emulated
        );
    }
}

/// Default ceiling on how long a single shim→bridge MCP round-trip may
/// block before the shim synthesises a -32000 timeout error and unwinds.
/// 30s comfortably covers a real tools/call dispatch inside the UE5
/// bridge; below this the user-visible wait would be confusing.
///
/// v0.1.16 makes this overridable via `NWIRO_LOCAL_LLM_MCP_TIMEOUT_SECS`
/// (matching the existing NWIRO_LOCAL_LLM_* pattern in main.rs) so ops
/// can tune without a recompile. Per-session config remains a future
/// option if individual sessions need different ceilings.
const PHASE3_MCP_TIMEOUT_SECS_DEFAULT: u64 = 30;

/// v0.1.37 (Finding C): the distinguished marker a cancel-aware MCP-await returns
/// when the session `CancellationToken` trips mid-round-trip. `bridge/tools.rs`
/// recognises it (`is_cancel_sentinel`) and maps it to `ShimError::Cancelled`
/// (→ `stopReason: cancelled`) — NOT the generic missing-`result` path, which
/// would become an in-band `isError:true` tool failure (a spurious
/// `tool_call_failed` frame). The text is deliberately DISTINCT from the generic
/// "mcp round-trip cancelled" (sender-dropped) message so only a real token
/// cancel takes the cancelled path.
pub(crate) const MCP_CANCELLED_SENTINEL: &str = "mcp round-trip cancelled (token)";

pub(crate) fn mcp_cancelled_marker() -> serde_json::Value {
    serde_json::json!({ "error": { "code": -32800, "message": MCP_CANCELLED_SENTINEL } })
}

/// Resolve the MCP timeout at call time. Env var read is sub-microsecond
/// and the call frequency is bounded by `mcp/connect` + `mcp/message`
/// per tool round-trip, so caching via OnceLock would optimise the
/// wrong axis. Returns the default on parse failure or absence.
fn phase3_mcp_timeout_secs() -> u64 {
    std::env::var("NWIRO_LOCAL_LLM_MCP_TIMEOUT_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(PHASE3_MCP_TIMEOUT_SECS_DEFAULT)
}

/// An in-flight shim→bridge JSON-RPC request (`mcp/connect`,
/// `mcp/message`) awaiting a response from the bridge.
///
/// Lifecycle:
///   1. `write_mcp_real` (in `handle_session_prompt`) constructs the
///      entry, inserts it into `Server.pending_requests` keyed by a
///      freshly-allocated shim id, and writes the request frame to
///      stdout.
///   2. The dispatcher loop in `Server::run` receives the bridge's
///      response, looks up the entry by id, removes it, and calls
///      `sender.send(response_value)`.
///   3. If `handle_session_cancel` fires before step 2, the entry's
///      sender is dropped (via `retain` removal). The awaiting
///      `Receiver` in `write_mcp_real` then surfaces `RecvError`,
///      which the closure translates to a `-32000` cancelled error
///      for the tool handler.
struct PendingRequest {
    /// ACP session that originated this request. The cancel-drain path
    /// filters on this field to remove only the affected session's
    /// in-flight requests (O(in-flight), not O(total)).
    session_id: String,
    /// JSON-RPC method name. Diagnostic only (logged on routing); the
    /// dispatcher routes on id presence, never on method.
    method: String,
    /// Channel half that delivers the response back to `write_mcp_real`.
    /// Consumed by `Sender::send` on dispatch, or dropped on cancel.
    sender: oneshot::Sender<serde_json::Value>,
}

/// Production `McpTransport` for the connector path (W1-09). `prepare` allocates
/// a shim id, registers a oneshot in `pending_requests`, and returns the stamped
/// `mcp/*` frame + a response future — but does NOT write the frame. The
/// connector emits it as `ConnectorEvent::OutboundFrame` so the dispatcher (the
/// single writer) writes it IN ORDER with session/update; this is the
/// two-writer-reorder fix (mcp/* used to be written directly here, racing the
/// drainer-buffered session/update stream). The frame-router in `run()` still
/// routes the inbound response to the oneshot by id — the same correlation
/// machinery the legacy path uses. No `output` sink field: this type correlates
/// only; the dispatcher writes.
struct AcpMcpTransport {
    pending_requests: Arc<Mutex<HashMap<u64, PendingRequest>>>,
    next_shim_id: Arc<AtomicU64>,
}

impl crate::connector::mcp::McpTransport for AcpMcpTransport {
    fn prepare(
        &self,
        req: serde_json::Value,
        cancel: tokio_util::sync::CancellationToken,
    ) -> (
        serde_json::Value,
        crate::connector::runtime::BoxFuture<'static, serde_json::Value>,
    ) {
        // Allocate a shim id, stamp it, and register the pending request
        // SYNCHRONOUSLY (so the correlation entry exists before the caller routes
        // the frame through the dispatcher's writer, mirroring the legacy
        // register-before-write order). The frame is NOT written here — the
        // connector emits it as ConnectorEvent::OutboundFrame so the dispatcher
        // writes it IN ORDER with session/update (no two-writer reorder). The
        // returned future awaits the response with the MCP timeout; the
        // frame-router routes the reply to the oneshot by id.
        let id = self.next_shim_id.fetch_add(1, Ordering::Relaxed);
        let method = req
            .get("method")
            .and_then(|m| m.as_str())
            .unwrap_or("unknown")
            .to_string();
        let mut frame = req;
        if let Some(obj) = frame.as_object_mut() {
            obj.insert("id".to_string(), serde_json::json!(id));
        }
        let (tx, rx) = oneshot::channel::<serde_json::Value>();
        let registered = match self.pending_requests.lock() {
            Ok(mut map) => {
                map.insert(
                    id,
                    PendingRequest {
                        session_id: "connector".to_string(),
                        method,
                        sender: tx,
                    },
                );
                true
            }
            Err(_) => false,
        };
        let pending_requests = Arc::clone(&self.pending_requests);
        let future: crate::connector::runtime::BoxFuture<'static, serde_json::Value> =
            Box::pin(async move {
                if !registered {
                    return serde_json::json!({
                        "error": { "code": -32000, "message": "pending_requests lock poisoned" }
                    });
                }
                let timeout = std::time::Duration::from_secs(phase3_mcp_timeout_secs());
                tokio::select! {
                    biased;
                    // v0.1.37 (Finding C): a mid-round-trip session/cancel trips
                    // the token — abort immediately instead of waiting the full
                    // MCP timeout; drain our pending entry so a late reply
                    // doesn't route to a dropped Sender.
                    _ = cancel.cancelled() => {
                        if let Ok(mut map) = pending_requests.lock() {
                            map.remove(&id);
                        }
                        mcp_cancelled_marker()
                    }
                    res = tokio::time::timeout(timeout, rx) => match res {
                        Ok(Ok(v)) => v,
                        Ok(Err(_)) => serde_json::json!({
                            "error": { "code": -32000, "message": "mcp round-trip cancelled" }
                        }),
                        Err(_) => {
                            if let Ok(mut map) = pending_requests.lock() {
                                map.remove(&id);
                            }
                            serde_json::json!({
                                "error": { "code": -32000, "message": "mcp round-trip timeout" }
                            })
                        }
                    }
                }
            });
        (frame, future)
    }
}

pub struct Server {
    client: openai::Client,
    sessions: HashMap<String, bridge::SessionState>,
    /// Lines from the dedicated stdin thread (see main.rs). We don't read
    /// `tokio::io::stdin()` directly because it deadlocks against UE's
    /// FInteractiveProcess pipes on Windows.
    stdin_rx: UnboundedReceiver<String>,
    /// Most recent warmup outcome: the model name that was warmed and the
    /// tier that came back from `Client::probe_tool_capability`. Lives on
    /// `Server` (not on `SessionState` or `Client`) because warmup runs
    /// OUTSIDE any session context.
    ///
    /// The model name is part of the tuple specifically to detect the
    /// **mid-session model switch** case: `session/set_config_option`
    /// changes `state.current_model` without re-running warmup. Without
    /// the name pair, a session warmed against `qwen2.5:14b` (Native)
    /// could switch to `gemma2:2b` and silently retain `Native`, letting
    /// tool calls flow to a model that can't execute them — exactly the
    /// failure mode this whole feature exists to prevent.
    ///
    /// `None` means no warmup has ever run. Sessions created in that
    /// state default to `ToolTier::None` and refuse tools.
    last_warmup: Option<(String, ToolTier)>,
    /// Phase 3 correlation map for in-flight shim→bridge JSON-RPC
    /// requests (`mcp/connect`, `mcp/message`). Keyed by the
    /// shim-allocated request id from `next_shim_id`. `Arc`-shared so
    /// the dispatcher loop AND the per-session `write_mcp_real`
    /// closures can both reach it without struct-level borrow conflicts.
    ///
    /// `std::sync::Mutex` (not `tokio::sync::Mutex`) because every
    /// critical section is a single HashMap op with no `.await`
    /// inside — sync locking is correct and avoids dragging async
    /// poison into the cancel path (which is itself a sync fn).
    pending_requests: Arc<Mutex<HashMap<u64, PendingRequest>>>,
    /// Monotonic counter for shim-allocated request ids. Starts at
    /// 1_000_000 to stay well above the bridge's small-integer id
    /// space (the bridge issues `id: 1, 2, 3, ...` for its inbound
    /// requests). `AtomicU64` keeps the counter cheap to clone via
    /// `Arc` without lock contention.
    next_shim_id: Arc<AtomicU64>,
    /// v0.1.18 STREAM-002: side-map of session_id → CancellationToken,
    /// populated by `handle_session_new` and re-armed (fresh token) by
    /// `handle_session_cancel`. The frame-router task uses this map
    /// to fire `token.cancel()` IMMEDIATELY on receiving a
    /// `session/cancel` frame, bypassing the dispatcher's serialized
    /// processing.
    ///
    /// Why this is needed: the dispatcher awaits
    /// `handle_session_prompt.await` inline. While the prompt is
    /// streaming, `session/cancel` arrives, gets forwarded to
    /// `bridge_tx` by the frame-router, but sits in `bridge_rx`
    /// because `bridge_rx.recv().await` doesn't fire until the
    /// current handler returns. By the time `handle_session_cancel`
    /// runs, the stream has already completed naturally. The
    /// fast-path via this map shortcuts that wait — proven by the
    /// streaming-cancel smoke test.
    ///
    /// `CancellationToken::cancel()` is idempotent (sets an atomic
    /// flag), so the double-cancel from the fast-path AND the
    /// later dispatcher-path `handle_session_cancel` is harmless.
    /// The dispatcher path still runs to re-arm the session with a
    /// fresh token and drain `pending_requests` (cancel is
    /// turn-scoped — the session itself survives).
    cancel_tokens: Arc<Mutex<HashMap<String, tokio_util::sync::CancellationToken>>>,
    /// Sink for every outbound ACP frame. Production is `frame::StdoutSink`,
    /// which forwards verbatim to `frame::write_frame` (byte-identical to the
    /// pre-seam direct call). Tests inject a `frame::CaptureSink` to record
    /// frames in memory for golden-transcript assertions. `Arc<dyn>` so the
    /// dispatcher, the per-prompt session/update drainer task, and the `mcp/*`
    /// closure can each hold a cheap clone.
    output: Arc<dyn frame::OutputSink>,
}

impl Server {
    pub fn new(client: openai::Client, stdin_rx: UnboundedReceiver<String>) -> Self {
        Self {
            client,
            sessions: HashMap::new(),
            stdin_rx,
            last_warmup: None,
            pending_requests: Arc::new(Mutex::new(HashMap::new())),
            next_shim_id: Arc::new(AtomicU64::new(1_000_000)),
            cancel_tokens: Arc::new(Mutex::new(HashMap::new())),
            output: Arc::new(frame::StdoutSink),
        }
    }

    /// Test-only constructor that injects a custom output sink (e.g.
    /// `frame::CaptureSink`) so golden-transcript tests can capture the shim's
    /// ACP output in memory instead of writing to process stdout. Behaviour is
    /// otherwise identical to [`Server::new`].
    #[cfg(test)]
    pub(crate) fn new_with_output(
        client: openai::Client,
        stdin_rx: UnboundedReceiver<String>,
        output: Arc<dyn frame::OutputSink>,
    ) -> Self {
        Self {
            client,
            sessions: HashMap::new(),
            stdin_rx,
            last_warmup: None,
            pending_requests: Arc::new(Mutex::new(HashMap::new())),
            next_shim_id: Arc::new(AtomicU64::new(1_000_000)),
            cancel_tokens: Arc::new(Mutex::new(HashMap::new())),
            output,
        }
    }

    pub async fn run(mut self) -> Result<()> {
        // Stdout writes go through frame::write_frame which uses synchronous
        // std::io::stdout() (see frame.rs comment). No tokio::io::Stdout
        // handle is held here.

        // ── Frame router ──────────────────────────────────────────
        //
        // A dedicated task owns `stdin_rx` and reads every inbound
        // frame. It routes by frame shape:
        //
        //   * frame with NO `method` field  →  JSON-RPC response to
        //     one of our outbound mcp/* requests. Look up `id` in
        //     `pending_requests`; if present, send the frame on the
        //     stored oneshot::Sender. Map *presence* is the routing
        //     predicate — the 1_000_000+ allocator range is just
        //     policy.
        //
        //   * frame WITH a `method` field   →  bridge→shim request.
        //     Forward on `bridge_tx` for the dispatcher loop below.
        //
        // Why this is structural and not just a "branch in the
        // dispatcher loop": the dispatcher awaits `handle_session_prompt`
        // inline, and that handler itself awaits the 30s receiver in
        // `write_mcp_real`. If stdin reading lived in the same task,
        // the response to mcp/connect couldn't be drained while the
        // dispatcher was blocked waiting for it — classic deadlock.
        // Running the router as a separate tokio task lets stdin
        // continue to drain in parallel with prompt processing.
        let pending_requests = Arc::clone(&self.pending_requests);
        let cancel_tokens = Arc::clone(&self.cancel_tokens);
        // Move stdin_rx out of `self` so the spawned task can own it.
        // After this, `self.stdin_rx` is unreachable (partial-move),
        // which is fine because nothing else in `run` touches it.
        let mut stdin_rx = std::mem::replace(
            &mut self.stdin_rx,
            tokio::sync::mpsc::unbounded_channel::<String>().1,
        );
        let (bridge_tx, mut bridge_rx) =
            tokio::sync::mpsc::unbounded_channel::<serde_json::Value>();

        tokio::spawn(async move {
            loop {
                let msg = match frame::read_frame(&mut stdin_rx).await {
                    Ok(Some(m)) => m,
                    Ok(None) => {
                        tracing::info!("frame-router: stdin EOF, exiting");
                        break;
                    }
                    Err(e) => {
                        tracing::error!(error = %e, "frame-router: parse error; continuing");
                        continue;
                    }
                };
                if msg.get("method").is_none() {
                    // Correlate by the numeric id the shim allocated for its
                    // outbound mcp/* request. Accept a JSON number OR a numeric
                    // string so a bridge that stringifies the echoed id still
                    // matches its pending request (C2) instead of orphaning.
                    if let Some(id) = msg.get("id").and_then(json_rpc_id_as_u64) {
                        let entry = match pending_requests.lock() {
                            Ok(mut map) => map.remove(&id),
                            Err(_) => {
                                tracing::error!(
                                    "pending_requests lock poisoned — cannot route response"
                                );
                                None
                            }
                        };
                        if let Some(pending) = entry {
                            tracing::debug!(
                                id = id,
                                method = %pending.method,
                                "frame-router: response → pending"
                            );
                            // Best-effort send: receiver may already be
                            // dropped if timeout fired or cancel-drain
                            // pulled the entry first. Late frames are
                            // silently discarded.
                            let _ = pending.sender.send(msg);
                            continue;
                        }
                    }
                    tracing::warn!(
                        msg = %msg,
                        "frame-router: orphan response (no method, no matching pending request)"
                    );
                    continue;
                }
                // v0.1.18 STREAM-002 fast-path: if this is a
                // `session/cancel` frame, fire the matching
                // CancellationToken IMMEDIATELY without waiting for
                // the dispatcher to drain its serialized backlog.
                // The dispatcher will still process the frame via
                // bridge_tx below — that's where the token re-arm
                // and pending_requests drain happen.
                // CancellationToken::cancel() is idempotent so the
                // double-fire is safe.
                if msg.get("method").and_then(|m| m.as_str()) == Some("session/cancel") {
                    if let Some(sid) = msg
                        .pointer("/params/sessionId")
                        .and_then(|v| v.as_str())
                    {
                        let token_opt = match cancel_tokens.lock() {
                            Ok(map) => map.get(sid).cloned(),
                            Err(_) => {
                                tracing::error!(
                                    "cancel_tokens lock poisoned — fast-path skipped"
                                );
                                None
                            }
                        };
                        if let Some(token) = token_opt {
                            token.cancel();
                            tracing::debug!(
                                session_id = sid,
                                "frame-router: fast-path cancel fired"
                            );
                        }
                    }
                }
                if bridge_tx.send(msg).is_err() {
                    tracing::info!("frame-router: dispatcher dropped, exiting");
                    break;
                }
            }
        });

        // Clone the output sink for the dispatcher loop. Cloning the `Arc`
        // (rather than borrowing `self.output`) keeps the loop body free of a
        // `self` borrow, so the `&mut self` handler calls below stay
        // conflict-free.
        let output = Arc::clone(&self.output);

        // W1-09: route session ops through `LocalOpenAiConnector` when
        // `LOCAL_LLM_USE_CONNECTOR=1`. Both paths live; default-off keeps the
        // legacy path authoritative. The connector is built lazily on the first
        // session/new (so it picks up any `initialize` client reconfig), SHARING
        // the cancel-token map (frame-router fast-path + `connector.cancel()`
        // trip the same token) and the `pending_requests` correlation map (via
        // `AcpMcpTransport`). initialize / warmup / set_mode still go to the
        // legacy handlers below.
        let use_connector = std::env::var("LOCAL_LLM_USE_CONNECTOR").as_deref() == Ok("1");
        let pending_arc = Arc::clone(&self.pending_requests);
        let next_id_arc = Arc::clone(&self.next_shim_id);
        let cancel_arc = Arc::clone(&self.cancel_tokens);
        let mut connector: Option<crate::connector::local_openai::LocalOpenAiConnector> = None;

        // ── Dispatcher ────────────────────────────────────────────
        //
        // Consumes only bridge→shim *requests* from the router. The
        // router has already filtered out response frames. Each
        // iteration may block arbitrarily long (handle_session_prompt
        // can await an mcp/* round-trip) without affecting the
        // router's stdin-draining throughput.
        loop {
            let msg = match bridge_rx.recv().await {
                Some(m) => m,
                None => {
                    tracing::info!("frame-router channel closed — dispatcher shutting down");
                    break;
                }
            };

            let method = match msg.get("method").and_then(|v| v.as_str()) {
                Some(m) => m.to_string(),
                None => {
                    // Router only forwards method-bearing frames, so
                    // this is unreachable. Defensive log + continue.
                    tracing::warn!(msg = %msg, "dispatcher: received frame without method (router invariant violated)");
                    continue;
                }
            };

            // JSON-RPC: presence of "id" field distinguishes request from notification.
            let id = msg.get("id").cloned();
            let params = msg.get("params").cloned().unwrap_or(json!({}));

            // ── Connector path (W1-09, flag-gated) ────────────────────
            // Owns session/new, set_config, prompt, cancel when the flag is on;
            // returns `true` to skip the legacy match. Write failures are logged
            // (never `?`-propagated) so a failing prompt can't kill the server.
            if use_connector {
                use crate::connector::event::ConnectorEvent;
                use crate::connector::runtime::AgentRuntimeConnector;
                use crate::connector::session::{
                    ConnectorConfig, ConnectorPrompt, ConnectorSessionId, StartSessionParams,
                };
                use futures_util::StreamExt;

                let handled = match method.as_str() {
                    "session/new" => {
                        if connector.is_none() {
                            let transport: Arc<dyn crate::connector::mcp::McpTransport> =
                                Arc::new(AcpMcpTransport {
                                    pending_requests: Arc::clone(&pending_arc),
                                    next_shim_id: Arc::clone(&next_id_arc),
                                });
                            connector = Some(
                                crate::connector::local_openai::LocalOpenAiConnector::with_cancel_tokens(
                                    self.client.clone(),
                                    transport,
                                    Arc::clone(&cancel_arc),
                                ),
                            );
                        }
                        let conn = connector.as_ref().unwrap();
                        let sys = serde_json::from_value::<SessionNewParams>(params.clone())
                            .ok()
                            .and_then(|p| p.meta)
                            .and_then(|m| m.system_prompt)
                            .and_then(|sp| sp.append)
                            .filter(|s| !s.trim().is_empty());
                        let started = conn
                            .start_session(StartSessionParams {
                                session_id: None,
                                client_capabilities: Default::default(),
                                system_prompt: sys,
                            })
                            .await;
                        if let Some(req_id) = id.clone() {
                            let response = match started {
                                Ok(sid) => json!({"jsonrpc":"2.0","id":req_id,"result":{"sessionId": sid.as_str()}}),
                                Err(e) => json!({"jsonrpc":"2.0","id":req_id,"error":{"code": e.to_acp_jsonrpc_code(), "message": e.to_string()}}),
                            };
                            if let Err(e) = output.write_frame(response).await {
                                tracing::error!(error = %e, "connector session/new response write failed");
                            }
                        }
                        true
                    }
                    "session/set_config_option" => {
                        if let (Some(conn), Ok(p)) = (
                            connector.as_ref(),
                            serde_json::from_value::<SetConfigOptionParams>(params.clone()),
                        ) {
                            if p.config_id == "model" {
                                // Replicate legacy handle_set_config_option: grant
                                // the warmed tier iff the new model matches the
                                // warmed model, else None (fail-safe). Without this
                                // the connector session stays ToolTier::None and
                                // the shared bridge strips `tools` from every
                                // request (a warmed Native model could never call a
                                // tool). The goldens mask this via the forced-tier
                                // env override; production has no such override.
                                // M-1: shared tolerant tier resolution — both
                                // set_config handlers MUST use resolve_set_config_tier
                                // so a `:latest`/digest id difference can't collapse a
                                // warmed model to None and strip its tools.
                                let tier = resolve_set_config_tier(&self.last_warmup, &p.value);
                                let _ = conn
                                    .set_config(
                                        &ConnectorSessionId::new(p.session_id),
                                        ConnectorConfig {
                                            model: Some(p.value),
                                            tool_tier: Some(tier),
                                        },
                                    )
                                    .await;
                            }
                        }
                        if let Some(req_id) = id.clone() {
                            let _ = output
                                .write_frame(json!({"jsonrpc":"2.0","id":req_id,"result":{}}))
                                .await;
                        }
                        true
                    }
                    "session/prompt" => {
                        if let (Some(conn), Some(req_id)) = (connector.as_ref(), id.clone()) {
                            match serde_json::from_value::<SessionPromptParams>(params.clone()) {
                                Ok(p) => {
                                    let sid = p.session_id.clone();
                                    let (text, images) = p.content_parts();
                                    let cprompt = ConnectorPrompt {
                                        text,
                                        images,
                                        tools: p.tools.map(serde_json::Value::Array),
                                    };
                                    let (_handle, mut stream) =
                                        conn.prompt(ConnectorSessionId::new(sid.clone()), cprompt);
                                    // If the spawned bridge task panics, tokio
                                    // swallows it (panic=unwind keeps the shim
                                    // alive) and the stream ends with NO terminal
                                    // event. `responded` lets the post-loop fallback
                                    // answer req_id with -32000 instead of hanging.
                                    let mut responded = false;
                                    while let Some(event) = stream.next().await {
                                        match &event {
                                            ConnectorEvent::TurnFinished { stop_reason, error_kind, .. } => {
                                                responded = true;
                                                let result = crate::acp::messages::PromptResponseResult {
                                                    stop_reason: stop_reason.to_acp_str().to_string(),
                                                    meta: error_kind.clone().map(|ek| crate::acp::messages::PromptResponseMeta {
                                                        error_kind: Some(ek),
                                                    }),
                                                };
                                                let _ = output.write_frame(json!({
                                                    "jsonrpc":"2.0","id":req_id.clone(),
                                                    "result": result
                                                })).await;
                                            }
                                            ConnectorEvent::Failed { error, .. } => {
                                                responded = true;
                                                let resp = if error.is_cancellation() {
                                                    json!({"jsonrpc":"2.0","id":req_id.clone(),"result":{"stopReason":"cancelled"}})
                                                } else {
                                                    json!({"jsonrpc":"2.0","id":req_id.clone(),"error":{"code": error.to_acp_jsonrpc_code(), "message": error.to_string()}})
                                                };
                                                let _ = output.write_frame(resp).await;
                                            }
                                            ConnectorEvent::OutboundFrame { frame, .. } => {
                                                // Raw outbound frame (mcp/*): write
                                                // it verbatim, IN ORDER with the
                                                // session/update frames below. The
                                                // dispatcher is the single writer,
                                                // so mcp/* can't reorder ahead of the
                                                // session/update that preceded it.
                                                let _ = output.write_frame(frame.clone()).await;
                                            }
                                            other => {
                                                if let Some(notif) =
                                                    crate::acp::translator::event_to_notification(other, &sid)
                                                {
                                                    let frame = json!({
                                                        "jsonrpc":"2.0","method":"session/update",
                                                        "params":{"sessionId": sid, "update": notif.update}
                                                    });
                                                    let _ = output.write_frame(frame).await;
                                                }
                                            }
                                        }
                                    }
                                    if !responded {
                                        // Stream ended with no terminal event — the
                                        // prompt task panicked. Legacy surfaces an
                                        // Err -> -32000; match that so req_id is
                                        // always answered (never a silent hang).
                                        let _ = output
                                            .write_frame(json!({
                                                "jsonrpc":"2.0","id":req_id.clone(),
                                                "error":{"code":-32000,"message":"prompt task ended without completion"}
                                            }))
                                            .await;
                                    }
                                }
                                Err(e) => {
                                    let _ = output.write_frame(json!({
                                        "jsonrpc":"2.0","id":req_id,
                                        "error":{"code":-32602,"message": format!("invalid session/prompt params: {e}")}
                                    })).await;
                                }
                            }
                        } else if connector.is_none() {
                            // A prompt arrived before any session/new — the legacy
                            // path answers an unknown session with an error; match
                            // that rather than silently skipping the response,
                            // which would hang the client forever. (The goldens
                            // never hit this: they always session/new first.)
                            if let Some(req_id) = id.clone() {
                                let _ = output
                                    .write_frame(json!({
                                        "jsonrpc":"2.0","id":req_id,
                                        "error":{"code":-32602,"message":"session/prompt received before session/new (no active session)"}
                                    }))
                                    .await;
                            }
                        }
                        true
                    }
                    "session/cancel" => {
                        if let (Some(conn), Ok(p)) = (
                            connector.as_ref(),
                            serde_json::from_value::<SessionCancelParams>(params.clone()),
                        ) {
                            let sid = ConnectorSessionId::new(p.session_id);
                            // Trip the token (interrupt the turn), THEN tear the
                            // session down — removing it from the connector's
                            // sessions map AND the shared cancel_tokens map.
                            // NOTE: legacy handle_session_cancel is now
                            // TURN-scoped (v0.4.0 — the session survives with a
                            // re-armed token); this flag-gated non-default path
                            // keeps the old teardown because a safe re-arm here
                            // needs its own pass (per-turn state holds the
                            // session mutex for the whole in-flight turn).
                            // Without the teardown the cancelled token lingers
                            // (a re-prompt of the same id would clone an
                            // already-cancelled token and instantly cancel,
                            // corrupting history) and both maps grow
                            // unbounded. The MCP-await unblock that legacy's
                            // pending_requests drain would provide is Finding C
                            // (open on BOTH paths — the dispatcher is blocked, so
                            // neither path's drain runs mid-turn); not replicated.
                            let _ = conn.cancel(&sid).await;
                            let _ = conn.close_session(sid).await;
                        }
                        true
                    }
                    // Session persistence is a legacy-path feature: the
                    // connector path answers session/load with -32002 (the
                    // host classifies resource_not_found and silently falls
                    // back to session/new). Deliberately NOT extended — the
                    // connector keeps its own sessions map, so a legacy-side
                    // restore would be invisible to it.
                    "session/load" => {
                        if let Some(req_id) = id.clone() {
                            let sid = params
                                .get("sessionId")
                                .and_then(|v| v.as_str())
                                .unwrap_or_default();
                            let _ = output
                                .write_frame(json!({
                                    "jsonrpc":"2.0","id":req_id,
                                    "error":{"code":-32002,"message":format!("session not found: {sid}")}
                                }))
                                .await;
                        }
                        true
                    }
                    // initialize / warmup / set_mode -> legacy handlers below.
                    _ => false,
                };
                if handled {
                    continue;
                }
            }

            tracing::debug!(method = %method, has_id = id.is_some(), "dispatching");

            match method.as_str() {
                "initialize" => {
                    let result = self.handle_initialize(params);
                    if let Some(req_id) = id {
                        let response = json!({
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "result": result,
                        });
                        output.write_frame(response).await?;
                    }
                }

                "session/new" => {
                    let result = self.handle_session_new(params)?;
                    if let Some(req_id) = id {
                        let response = json!({
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "result": result,
                        });
                        output.write_frame(response).await?;
                    }
                }

                // Session persistence (targets v0.5.0): restore a persisted
                // session across a shim restart. Contract: replay NOTHING (the
                // result is an empty object; the host suppresses replayed
                // chunks), and EVERY anomaly answers -32002 so the host
                // classifies resource_not_found and silently falls back to
                // session/new. Trivially inside the host's 30s load budget —
                // it's one file read.
                "session/load" => {
                    let outcome = self.handle_session_load(params);
                    if let Some(req_id) = id {
                        let response = match outcome {
                            Ok(result) => json!({
                                "jsonrpc": "2.0",
                                "id": req_id,
                                "result": result,
                            }),
                            Err(message) => json!({
                                "jsonrpc": "2.0",
                                "id": req_id,
                                "error": { "code": -32002, "message": message },
                            }),
                        };
                        output.write_frame(response).await?;
                    }
                }

                "session/set_config_option" => {
                    let result = self.handle_set_config_option(params)?;
                    if let Some(req_id) = id {
                        let response = json!({
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "result": result,
                        });
                        output.write_frame(response).await?;
                    }
                }

                // `session/warmup` pre-loads the configured model into the
                // backend's working memory without generating any tokens. The
                // bridge sends this on Save (auto) and via the "Load Model"
                // button (manual). For Ollama this performs the empty-prompt
                // + keep_alive trick; for LM Studio / cloud it falls back to
                // a `/models` round-trip that confirms reachability without
                // forcing a load. Always responds with a `WarmupResult`
                // envelope so the frontend can categorise the outcome.
                "session/warmup" => {
                    let result = self.handle_warmup(params).await;
                    if let Some(req_id) = id {
                        let response = json!({
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "result": result,
                        });
                        output.write_frame(response).await?;
                    }
                }

                // `session/set_mode` is sent by the UE5 bridge after every
                // `session/new` with `modeId:"auto"` (and it ignores the
                // result). Local LLMs have no mode concept (cloud adapters use
                // modes for policy/safety profiles like Claude's
                // `permissive`/`strict`). We advertise NO session modes, so we
                // ACK the degenerate default/"auto" mode with `{}` (truthful —
                // we are always effectively in "auto") to keep the bridge's log
                // quiet, but reject any OTHER requested mode with `-32602`
                // rather than falsely ACKing a switch that never happened. This
                // refines the 2026-05-09 always-`{}` design to stay honest to a
                // stricter client. See `build_set_mode_response`.
                "session/set_mode" => {
                    if let Some(req_id) = id {
                        let mode_id = params
                            .get("modeId")
                            .and_then(|m| m.as_str())
                            .unwrap_or("");
                        let response = build_set_mode_response(req_id, mode_id);
                        output.write_frame(response).await?;
                    }
                }

                "session/prompt" => {
                    // Requests only (must have id). Notifications without id are protocol errors.
                    if let Some(req_id) = id {
                        // A failing prompt must NOT kill the whole connector.
                        // The success / cancelled / genuine-error responses are
                        // written INSIDE handle_session_prompt; an Err returned
                        // here means it failed *before* writing one (bad params,
                        // unknown session, or — the case that motivated this — a
                        // session/update drainer-task panic surfacing as
                        // AcpFraming). Previously the `?` propagated that Err out
                        // of `run()`, exiting the process and taking every other
                        // live session down with it. Instead: emit a -32000 for
                        // this request id and keep serving.
                        if let Err(e) = self.handle_session_prompt(req_id.clone(), params).await {
                            tracing::error!(
                                error = %e,
                                "session/prompt failed; emitting -32000 and continuing"
                            );
                            // Additive: unknown-session failures also carry
                            // structured `error.data` so the client can
                            // distinguish them from other -32000 errors
                            // without parsing the message text (which stays
                            // byte-identical, as does the code).
                            let error_body = match &e {
                                ShimError::UnknownSession(sid) => json!({
                                    "code": -32000,
                                    "message": e.to_string(),
                                    "data": {
                                        "reason": "unknown_session",
                                        "sessionId": sid
                                    }
                                }),
                                _ => json!({ "code": -32000, "message": e.to_string() }),
                            };
                            let err = json!({
                                "jsonrpc": "2.0",
                                "id": req_id,
                                "error": error_body
                            });
                            // Best-effort: a write failure here (parent gone)
                            // must not abort the dispatcher loop either.
                            if let Err(we) = output.write_frame(err).await {
                                tracing::error!(
                                    error = %we,
                                    "failed to write session/prompt error response"
                                );
                            }
                        }
                    } else {
                        tracing::warn!("session/prompt received as notification (no id) — ignoring");
                    }
                }

                "session/cancel" => {
                    // Notification: no id, no response expected.
                    self.handle_session_cancel(params)?;
                }

                other => {
                    tracing::warn!(method = %other, "unknown method");
                    if let Some(req_id) = id {
                        let error = json!({
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "error": {
                                "code": -32601,
                                "message": "Method not found",
                                "data": { "method": other }
                            }
                        });
                        output.write_frame(error).await?;
                    }
                }
            }
        }

        Ok(())
    }

    // ── Handlers ─────────────────────────────────────────────────────────

    fn handle_initialize(&mut self, params: serde_json::Value) -> serde_json::Value {
        let init: InitializeParams = serde_json::from_value(params).unwrap_or_default();

        // Extract localLlm config when P1-006 injects it. Until then we keep
        // the env-var-sourced client constructed in main.
        //
        // v0.1.32: localLlm config moved under `params._meta.localLlm` (the
        // ACP extensibility convention). The legacy `params.context.localLlm`
        // form is still honoured for one release so the UE5 bridge can roll
        // the change without a flag-day. `_meta` wins when both are present;
        // the legacy path logs a deprecation warning.
        let meta_llm = init.meta.and_then(|m| m.local_llm);
        let ctx_llm = init.context.and_then(|c| c.local_llm);
        let llm = match (meta_llm, ctx_llm) {
            (Some(m), Some(_)) => {
                tracing::warn!(
                    "localLlm config supplied via BOTH `params._meta.localLlm` and legacy \
                     `params.context.localLlm`; using `_meta.localLlm`. Drop the \
                     `context.localLlm` form — it will be removed in a future release."
                );
                Some(m)
            }
            (Some(m), None) => Some(m),
            (None, Some(c)) => {
                tracing::warn!(
                    "localLlm config received via legacy `params.context.localLlm`; migrate to \
                     `params._meta.localLlm` (the ACP extensibility convention). The legacy \
                     path is accepted for now but will be removed in a future release."
                );
                Some(c)
            }
            (None, None) => None,
        };
        if let Some(llm) = llm {
            if llm.base_url.is_some() || llm.model.is_some() {
                let base_url = llm
                    .base_url
                    .unwrap_or_else(|| self.client.base_url().to_string());
                let model = llm
                    .model
                    .unwrap_or_else(|| self.client.model().to_string());
                let api_key = ApiKey::from_env();
                self.client = openai::Client::new(base_url, model, api_key);
                tracing::info!("client reconfigured from ACP initialize localLlm config");
            }
        }

        // Spec-correct ACP `initialize` result: `agentCapabilities` /
        // `agentInfo` / `protocolVersion` / `authMethods` (the bridge already
        // reads these; the old non-spec `serverCapabilities`/`serverInfo` keys it
        // never read). `promptCapabilities.image` reflects whether the configured
        // model is vision-capable so the host can gate image emission. Never
        // advertise terminal — blockCommandExecution is always-on per PLAN.md.
        let supports_vision = crate::vision::model_supports_vision(self.client.model());
        json!({
            "protocolVersion": 1,
            "agentInfo": {
                "name": "local-llm-acp",
                "version": env!("CARGO_PKG_VERSION")
            },
            "agentCapabilities": {
                // Session persistence (targets v0.5.0): the host only ever
                // attempts a `session/load` resume when this is true. Gated on
                // the NWIRO_SHIM_PERSIST kill switch (+ a sane
                // NWIRO_SHIM_STATE_DIR override) so a disabled/broken storage
                // config degrades to the host's plain session/new flow.
                "loadSession": crate::persist::persistence_available(),
                "promptCapabilities": {
                    "image": supports_vision,
                    "audio": false,
                    "embeddedContext": false
                },
                "session": {
                    "prompt": true,
                    "setConfigOption": true,
                    "cancel": true
                }
            },
            "authMethods": []
        })
    }

    fn handle_session_new(&mut self, params: serde_json::Value) -> Result<serde_json::Value> {
        let p: SessionNewParams = serde_json::from_value(params)
            .map_err(|e| ShimError::AcpFraming(format!("invalid session/new params: {e}")))?;

        // Session persistence (targets v0.5.0): resolve the storage dir from
        // the host-supplied cwd (or the NWIRO_SHIM_STATE_DIR override) ONCE at
        // creation. `None` — kill switch off, absent/relative/nonexistent cwd —
        // permanently disables persistence for this session (no writes; a
        // restart cannot resume it). Validation happens here, not per-write.
        let persist_handle = crate::persist::resolve_storage_dir(p.cwd.as_deref()).map(|dir| {
            // First-use-per-process init: create the dir, clean stale *.tmp,
            // run an eviction pass. Best-effort — never fails session/new.
            crate::persist::init_storage_dir(&dir);
            crate::persist::PersistHandle {
                dir,
                created_at: crate::persist::now_unix(),
            }
        });

        // Extract the bridge-supplied system prompt from `_meta.systemPrompt.append`.
        // NwiroIKBridge::DoCreateSession sets this for Claude and localllm; codex-acp
        // strips `_meta` so codex sessions get no system message via this path (the
        // bridge falls back to a per-turn prepend in DoSendPrompt for codex). Empty
        // or whitespace-only content is treated as absent so we don't push a useless
        // message that would still consume context budget.
        let system_prompt_text = p
            .meta
            .and_then(|m| m.system_prompt)
            .and_then(|sp| sp.append)
            .filter(|s| !s.trim().is_empty());

        let session_id = Uuid::new_v4().to_string();
        // Seed history with the system message when supplied. Lives at
        // history[0] for the life of the session; every subsequent turn's
        // OpenAI request naturally leads with it.
        let mut history: Vec<ChatMessage> = Vec::new();
        if let Some(text) = system_prompt_text {
            tracing::debug!(session_id = %session_id, chars = text.len(), "seeding session history with system prompt from _meta.systemPrompt.append");
            history.push(ChatMessage::system(text));
        }
        let state = bridge::SessionState {
            session_id: session_id.clone(),
            current_model: String::new(), // set by session/set_config_option before first prompt
            history,
            cancel_token: tokio_util::sync::CancellationToken::new(),
            // Start with `None` deliberately. The tier is resolved by
            // `handle_set_config_option` once the bridge tells us which
            // model this session is actually using — at that point we
            // compare the model name against `self.last_warmup` and
            // either grant the warmed tier or refuse (fail-safe).
            // Inheriting from warmup at session creation would be unsafe
            // because the session might switch models mid-flight.
            tool_tier: ToolTier::None,
            // v0.1.22 Fix D + G1: telemetry state for token-budget
            // warning + history pruning. Both start clean — the warn
            // hasn't fired and we haven't pruned anything yet.
            token_budget_warned: false,
            pruned_turn_count: 0,
            // No context overflow learned yet this session — set on the
            // first overflow-recover (see bridge::handle_session_prompt).
            learned_tool_ceiling: None,
            persist: persist_handle,
        };
        // Register the session's cancel_token in the side-map so the
        // frame-router's fast-path can fire it without waiting on the
        // dispatcher. Done BEFORE inserting into self.sessions so any
        // immediate cancel arriving on the next stdin read finds the
        // token (the bridge can't legally send session/cancel before
        // it sees the session/new response, but defense in depth).
        if let Ok(mut map) = self.cancel_tokens.lock() {
            map.insert(session_id.clone(), state.cancel_token.clone());
        } else {
            tracing::error!(
                session_id = %session_id,
                "cancel_tokens lock poisoned — session/cancel fast-path will be unavailable for this session"
            );
        }
        self.sessions.insert(session_id.clone(), state);

        tracing::info!(session_id = %session_id, "session created");
        Ok(json!({ "sessionId": session_id }))
    }

    /// `session/load` (targets v0.5.0): rebuild a persisted session from its
    /// on-disk envelope under the SAME sessionId, with a fresh (untripped)
    /// cancel token and no replay — the empty-object result IS the whole
    /// restore signal. `Err(message)` maps to JSON-RPC `-32002` at the
    /// dispatcher; the wire message is always the bland
    /// `"session not found: <id>"` (the detailed reason goes to the trace log
    /// only) because the host's fallback path doesn't read it and the storage
    /// internals shouldn't leak.
    fn handle_session_load(
        &mut self,
        params: serde_json::Value,
    ) -> std::result::Result<serde_json::Value, String> {
        // Fields are optional at parse time so malformed params degrade to the
        // -32002 anomaly path below rather than a generic parse error.
        let p: SessionLoadParams = serde_json::from_value(params).unwrap_or_default();
        let requested = p.session_id.unwrap_or_default();
        let not_found = |reason: &str| -> String {
            tracing::warn!(
                session_id = %requested_for_log(&requested),
                reason,
                "session/load failed — answering -32002 (host falls back to session/new)"
            );
            format!("session not found: {requested}")
        };
        fn requested_for_log(id: &str) -> &str {
            if id.is_empty() { "(missing)" } else { id }
        }

        if requested.is_empty() {
            return Err(not_found("missing/empty sessionId"));
        }
        if !crate::persist::persistence_available() {
            return Err(not_found("persistence disabled (NWIRO_SHIM_PERSIST / state-dir config)"));
        }
        let Some(dir) = crate::persist::resolve_storage_dir(p.cwd.as_deref()) else {
            return Err(not_found("invalid or missing cwd — no storage dir"));
        };
        // First-use-per-process init (stale-*.tmp cleanup + eviction pass).
        // Best-effort; a load-only process still gets its storage hygiene.
        crate::persist::init_storage_dir(&dir);
        let envelope = crate::persist::load(&dir, &requested).map_err(|reason| not_found(&reason))?;

        // Rebuild live state: persisted conversation state verbatim, fresh
        // CancellationToken, non-durable flags re-defaulted. MCP state is not
        // restored — it reconnects per normal turn flow.
        let state = crate::persist::state_from_envelope(envelope, dir);

        // Register the fresh cancel token in the frame-router's fast-path map,
        // exactly like session/new does (and BEFORE the sessions insert, for
        // the same defense-in-depth reason).
        if let Ok(mut map) = self.cancel_tokens.lock() {
            map.insert(requested.clone(), state.cancel_token.clone());
        } else {
            tracing::error!(
                session_id = %requested,
                "cancel_tokens lock poisoned — session/cancel fast-path will be \
                 unavailable for this restored session"
            );
        }
        self.sessions.insert(requested.clone(), state);

        tracing::info!(session_id = %requested, "session restored from disk");
        // REPLAY NOTHING: the host ignores load-result content and
        // suppresses/forbids replayed chunks. An empty object is the contract.
        Ok(json!({}))
    }

    fn handle_set_config_option(&mut self, params: serde_json::Value) -> Result<serde_json::Value> {
        let p: SetConfigOptionParams = serde_json::from_value(params)
            .map_err(|e| ShimError::AcpFraming(format!("invalid set_config_option params: {e}")))?;

        if p.config_id == "model" {
            // Snapshot the warmup state BEFORE taking the mutable session
            // borrow so we don't fight the borrow checker. Cheap clone:
            // one Option, one String, one Copy enum.
            let warmed = self.last_warmup.clone();
            if let Some(state) = self.sessions.get_mut(&p.session_id) {
                tracing::debug!(session_id = %p.session_id, model = %p.value, "model updated");
                state.current_model = p.value;
                // Refresh tier against the warmed model via the SHARED resolver
                // (M-1): a tolerant `:latest`/digest match grants the warmed tier,
                // a genuine mismatch (or no warmup) falls to None + warns. Same fn
                // the connector handler uses — they must not diverge (that was the
                // incomplete-fix bug: this primary path stayed byte-exact).
                state.tool_tier = resolve_set_config_tier(&warmed, &state.current_model);
                // Session persistence: model + tool tier are durable session
                // state — write through so a restart BEFORE the next turn
                // still resumes with the right model/tier. Best-effort by
                // contract: a failure logs and never fails this request.
                crate::persist::save_session_state(state);
            }
        }

        Ok(json!({}))
    }

    async fn handle_session_prompt(
        &mut self,
        req_id: serde_json::Value,
        params: serde_json::Value,
    ) -> Result<()> {
        // Stdout writes happen via frame::write_frame which manages its own
        // synchronous std::io::stdout() lock per call (see frame.rs).

        // Clone the output sink up front (Arc clone — holds no `self` borrow)
        // so the drainer task, the `mcp/*` closure, and the final response
        // write all route through it. Production: `StdoutSink` (identical to
        // the former direct `frame::write_frame`). Tests: the injected
        // `CaptureSink`.
        let output = Arc::clone(&self.output);

        let req: SessionPromptParams = serde_json::from_value(params)
            .map_err(|e| ShimError::AcpFraming(format!("invalid session/prompt params: {e}")))?;

        let session_id = req.session_id.clone();

        // ---- Lazy capability probe (probe-None self-heal) ----
        // If this session would strip all tools because its tier is None AND
        // no `session/warmup` has run this process (last_warmup is None), run
        // one warmup+probe now against the (by-now-warm) model before the
        // prompt is processed. This self-heals the failures where the bridge
        // restarted and never re-issued session/warmup, or the first probe
        // raced a cold model load — both leave a capable model misclassified
        // as None for the whole process with no recovery path. Reusing
        // `warmup()` (not a bare probe) preserves the family force-Emulated
        // downgrade and the probe observability. One-shot: once last_warmup is
        // set (a real Native/Emulated tier OR a genuine None verdict), this
        // never runs again — a truly tool-less model still caches None cheaply.
        // Two gates keep this precise (and inert for existing flows):
        //   - only when the prompt actually carries tools (else tier is moot);
        //   - only when no FORCE_TOOL_TIER override is set (an operator-forced
        //     tier wins downstream regardless, so there is no misclassification
        //     left to repair — and we must not second-guess an explicit force).
        let prompt_has_tools = req
            .tools
            .as_ref()
            .map(|t| !t.is_empty())
            .unwrap_or(false);
        let force_tier_set = std::env::var("NWIRO_LOCAL_LLM_FORCE_TOOL_TIER").is_ok();
        if prompt_has_tools
            && !force_tier_set
            && self.last_warmup.is_none()
            && self
                .sessions
                .get(&session_id)
                .map(|s| s.tool_tier == ToolTier::None)
                .unwrap_or(false)
        {
            let current = self
                .sessions
                .get(&session_id)
                .map(|s| s.current_model.clone())
                .unwrap_or_default();
            let model = if current.is_empty() {
                self.client.model().to_string()
            } else {
                current
            };
            tracing::warn!(model = %model, "lazy probe: session tier is None and no warmup ran this process — running a one-shot warmup+probe against the warm model before stripping tools");
            let result = self.client.warmup(&model, "15m").await;
            // We adopt `result.tool_tier` even when `result.status == "failed"`.
            // This is deliberate: v0.1.35 makes the probe fail OPEN, so an
            // inconclusive / transient warmup carries `Emulated` (not `None`),
            // and adopting it yields a graceful Emulated attempt WITH any
            // diagnostic in the trace — strictly better than gating on status
            // and silently stripping tools by forcing `None`. Genuine
            // backend-down warmups still carry `None`, so adopting it is the
            // correct no-op there. (Follow-up: a richer path could surface the
            // warmup error to the user directly instead of relying on the trace.)
            self.last_warmup = Some((model.clone(), result.tool_tier));
            if let Some(state) = self.sessions.get_mut(&session_id) {
                if state.current_model.is_empty() || state.current_model == model {
                    state.tool_tier = result.tool_tier;
                }
            }
            tracing::info!(model = %model, tier = ?result.tool_tier, "lazy probe complete — tier adopted for this session");
        }

        let state = self
            .sessions
            .get_mut(&session_id)
            // Dedicated variant (not AcpFraming) so the dispatcher's error
            // response can carry structured `error.data.reason =
            // "unknown_session"`; the wire message text is unchanged.
            .ok_or_else(|| ShimError::UnknownSession(session_id.clone()))?;
        // v0.1.37 (Finding C): clone the session cancel token for the cancel-aware
        // MCP-await in `write_mcp_real` below. This is an immutable read through
        // `state` that completes before the `&mut state` reborrow at
        // `handle_session_prompt`, so it does not conflict with that borrow.
        let mcp_cancel_token = state.cancel_token.clone();

        // v0.1.18 STREAM-001: real-time session/update emission via a
        // bounded mpsc channel + per-prompt drainer task.
        //
        // The bridge handler's `write_update` is sync (chat_completion_stream
        // calls it from a sync FnMut on_chunk). To emit each notification
        // in real-time without making the entire callback chain async, we
        // hand the closure an mpsc::Sender that try_sends per call; a
        // dedicated drainer task receives + calls frame::write_frame in
        // parallel with the running stream.
        //
        // Ordering invariant: all session/update frames for this prompt
        // MUST precede the prompt response on stdout. We guarantee this
        // via drain-by-close: dropping the closure (which moved-in the
        // Sender) closes the channel → drainer's recv() returns None →
        // drainer task exits → we await its JoinHandle BEFORE writing
        // the response frame. There is no race because mpsc::Receiver
        // processes items sequentially, and JoinHandle resolves only
        // after the task's last frame::write_frame has been submitted.
        //
        // Capacity 64 with try_send: empirically the steady-state depth
        // is ~0.1 frames at 25 tok/s × 2ms-per-write — 640x headroom.
        // try_send drops + warn on a full channel rather than blocking
        // the sync on_chunk callback (blocking it would stall the
        // tokio worker running the HTTP read). Drops are observable in
        // the trace log if they ever happen.
        let (update_tx, update_rx) =
            tokio::sync::mpsc::channel::<serde_json::Value>(64);
        // mcp/* frames are routed through this SAME ordered channel (see
        // write_mcp_real below) so they cannot reorder relative to the
        // session/update frames the bridge emitted before them. The drainer is
        // then the single writer for the whole turn — eliminating the two-writer
        // race that made the tool goldens flaky under load.
        let mcp_update_tx = update_tx.clone();
        let sid_clone = session_id.clone();

        let write_update = move |notif: SessionUpdateNotification| {
            let frame = json!({
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": &sid_clone,
                    "update": notif.update
                }
            });
            if let Err(e) = update_tx.try_send(frame) {
                tracing::warn!(
                    error = ?e,
                    "session/update channel full or closed — dropping frame"
                );
            }
        };

        // Spawn the drainer task. It owns `update_rx` and pumps the
        // channel until close-and-empty (drain-by-close). Each
        // frame::write_frame internally uses spawn_blocking + the
        // process-global stdout lock, so concurrent writes from this
        // task and the v0.1.15 frame-router task never byte-interleave.
        let drainer_output = Arc::clone(&output);
        let update_drainer = tokio::spawn(async move {
            let mut rx = update_rx;
            while let Some(frame) = rx.recv().await {
                if let Err(e) = drainer_output.write_frame(frame).await {
                    // Don't abort — log and keep draining so subsequent
                    // updates aren't silently dropped. A stdout-write
                    // failure here is unusual (would mean parent process
                    // died); the prompt handler will surface a write
                    // failure separately when it tries to send the
                    // response frame.
                    tracing::error!(
                        error = ?e,
                        "session/update drainer: write_frame failed; continuing"
                    );
                }
            }
        });

        // Clone client to avoid simultaneous &mut self.sessions + &self.client borrow.
        let client = self.client.clone();

        // Phase 3 transport: real `write_mcp_request` closure that
        // routes outbound `mcp/connect` / `mcp/message` frames through
        // the pending-requests correlation map on `Server`. Replaces
        // the v0.1.14 stub that returned -32601 unconditionally.
        //
        // Capture order matters: clone the Arcs BEFORE the closure
        // body so each closure invocation cheaply re-clones them for
        // the async future. The `move` on the outer closure moves the
        // Arc handles into the closure once; inside, we `Arc::clone`
        // for each call, keeping the closure `Fn` (not `FnOnce`).
        let pending_requests_arc = Arc::clone(&self.pending_requests);
        let next_shim_id_arc = Arc::clone(&self.next_shim_id);
        let session_id_for_mcp = session_id.clone();
        let cancel_for_mcp = mcp_cancel_token.clone();

        let write_mcp_real = move |req: serde_json::Value| {
            let pending_requests = Arc::clone(&pending_requests_arc);
            let next_shim_id = Arc::clone(&next_shim_id_arc);
            let session_id = session_id_for_mcp.clone();
            let update_tx = mcp_update_tx.clone();
            let cancel = cancel_for_mcp.clone();
            async move {
                // 1. Allocate a fresh shim→bridge request id and stamp
                // it onto the outbound frame, overriding whatever
                // placeholder `bridge/tools.rs::execute_tool` baked in.
                // (The placeholders stay vestigial for readability.)
                let id = next_shim_id.fetch_add(1, Ordering::Relaxed);
                let method = req
                    .get("method")
                    .and_then(|m| m.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "unknown".to_string());
                let mut req = req;
                if let Some(obj) = req.as_object_mut() {
                    obj.insert("id".to_string(), serde_json::json!(id));
                }

                // 2. Register a oneshot::Sender in the correlation map
                // BEFORE writing to stdout. If the bridge somehow
                // replies before we await `rx` below, the dispatcher
                // already has a Sender to push the value into — the
                // oneshot channel holds the value buffered until we
                // pick it up.
                let (tx, rx) = oneshot::channel::<serde_json::Value>();
                {
                    let mut map = match pending_requests.lock() {
                        Ok(m) => m,
                        Err(_) => {
                            return serde_json::json!({
                                "error": {
                                    "code": -32000,
                                    "message": "pending_requests lock poisoned"
                                }
                            });
                        }
                    };
                    map.insert(
                        id,
                        PendingRequest {
                            session_id: session_id.clone(),
                            method,
                            sender: tx,
                        },
                    );
                }

                // 3. Enqueue the request frame on the SAME ordered drainer
                // channel that carries session/update (not a direct stdout
                // write), so this mcp/* frame lands in the bridge's call order
                // relative to the session/update frames emitted before it. The
                // drainer is then the single writer for the turn — no two-writer
                // reorder (the race that made the tool goldens flaky under load).
                // Async send (not the sync try_send write_update uses): an mcp
                // frame must never be dropped, and we are already async here. On
                // send failure the drainer is gone (channel closed); drain our
                // own entry rather than leak it, and surface an error.
                // BOUND the enqueue by the MCP ceiling: routing mcp/* through the
                // bounded(64) channel couples its progress to the drainer, so a
                // wedged drainer (stdout back-pressured by a client not reading)
                // would otherwise block the SOLE dispatcher task FOREVER on channel
                // capacity — no new prompts, no cancels. Transient slowness waits
                // and succeeds; a persistent stall fails the round-trip (draining
                // the just-registered pending entry) so the dispatcher makes
                // progress instead of deadlocking. (Does not cure a fully-stalled
                // stdout — the response write would then block too, a pre-existing
                // condition — but removes the unbounded mcp-enqueue block the
                // channel routing would add.)
                let enqueue_ceiling = std::time::Duration::from_secs(phase3_mcp_timeout_secs());
                match tokio::time::timeout(enqueue_ceiling, update_tx.send(req)).await {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => {
                        if let Ok(mut map) = pending_requests.lock() {
                            map.remove(&id);
                        }
                        return serde_json::json!({
                            "error": {
                                "code": -32000,
                                "message": format!("update channel closed: {e}")
                            }
                        });
                    }
                    Err(_) => {
                        if let Ok(mut map) = pending_requests.lock() {
                            map.remove(&id);
                        }
                        return serde_json::json!({
                            "error": {
                                "code": -32000,
                                "message": "mcp enqueue timeout (drainer stalled)"
                            }
                        });
                    }
                }

                // 4. Await response with a 30s ceiling. Three terminal
                // cases:
                //   Ok(Ok(v))         — dispatcher delivered the response
                //   Ok(Err(RecvError))— Sender dropped (cancel-drain fired)
                //   Err(Elapsed)      — bridge never responded within budget
                //
                // The Elapsed case must also drain our own HashMap entry —
                // dispatcher would otherwise route a late response to a
                // dropped Sender. That send fails silently (no panic),
                // but leaks the entry until process exit.
                let timeout = std::time::Duration::from_secs(phase3_mcp_timeout_secs());
                tokio::select! {
                    biased;
                    // v0.1.37 (Finding C): a mid-round-trip session/cancel trips
                    // the token — abort immediately instead of waiting the full
                    // MCP timeout; drain our own entry so a late reply doesn't
                    // route to a dropped Sender.
                    _ = cancel.cancelled() => {
                        if let Ok(mut map) = pending_requests.lock() {
                            map.remove(&id);
                        }
                        mcp_cancelled_marker()
                    }
                    res = tokio::time::timeout(timeout, rx) => match res {
                        Ok(Ok(v)) => v,
                        Ok(Err(_recv_err)) => serde_json::json!({
                            "error": {
                                "code": -32000,
                                "message": "mcp round-trip cancelled"
                            }
                        }),
                        Err(_elapsed) => {
                            if let Ok(mut map) = pending_requests.lock() {
                                map.remove(&id);
                            }
                            serde_json::json!({
                                "error": {
                                    "code": -32000,
                                    "message": "mcp round-trip timeout"
                                }
                            })
                        }
                    }
                }
            }
        };
        let prompt_result = bridge::handle_session_prompt(req, state, &client, write_update, write_mcp_real).await;

        // Session persistence: the turn is over and `state.history` is final
        // for this turn — every return path of bridge::handle_session_prompt
        // (clean finish, degrade, cancel) appends/repairs history before
        // returning. Write the envelope through now (best-effort by contract:
        // a failure logs and never fails the turn). Placed BEFORE the drainer
        // join so the drainer-panic early return below cannot skip it; the
        // `state` borrow ended with the call above, so this re-borrow is clean.
        if let Some(state) = self.sessions.get(&session_id) {
            crate::persist::save_session_state(state);
        }

        // bridge::handle_session_prompt consumed `write_update` by
        // value; on return, the closure (and its captured update_tx
        // Sender) is dropped. That closes the mpsc channel.
        //
        // Now wait for the drainer to flush any residual updates and
        // exit cleanly. This is the ORDERING BARRIER: after this
        // await, every session/update frame for this prompt has been
        // submitted via frame::write_frame, so the response frame we
        // write next is guaranteed to land last on stdout.
        //
        // On cancel: chat_completion_stream returns Err(Cancelled),
        // the closure still drops the Sender, and the drainer drains
        // any partial-update frames already queued before exiting.
        // That's the "α drain-before-ack" design choice — users see partial reasoning
        // up to the moment of cancel, then the -32800 error.
        if let Err(e) = update_drainer.await {
            return Err(ShimError::AcpFraming(format!(
                "session/update drainer panicked: {e}"
            )));
        }

        // Send the prompt result (or error) as the JSON-RPC response.
        // v0.1.24 G2: map the bridge's finish_reason to an ACP
        // `stopReason` field on success. Per ACP prompt-turn spec
        // (https://agentclientprotocol.com/protocol/prompt-turn), the
        // RESPONSE — not a session/update event — carries the turn-
        // completion signal.
        //
        // Round-3 (post-review): Cancelled is a *turn completion*, not
        // a transport error — ACP requires `result.stopReason="cancelled"`
        // here, not a `-32800` JSON-RPC error. The pre-v0.1.24 behavior
        // of emitting -32800 was non-compliant with ACP spec.
        // Infrastructure errors (AcpFraming/OpenAiHttp/Config) stay as
        // -32000 because those represent failures, not completions.
        let response = match prompt_result {
            Ok((finish_reason, error_kind)) => {
                let stop_reason = crate::acp::messages::map_finish_reason_to_acp_stop_reason(
                    &finish_reason,
                );
                let result = crate::acp::messages::PromptResponseResult {
                    stop_reason: stop_reason.to_string(),
                    meta: error_kind.map(|ek| crate::acp::messages::PromptResponseMeta {
                        error_kind: Some(ek),
                    }),
                };
                json!({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": result
                })
            }
            Err(ShimError::Cancelled) => json!({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": { "stopReason": "cancelled" }
            }),
            Err(e) => json!({
                "jsonrpc": "2.0",
                "id": req_id,
                "error": { "code": -32000, "message": e.to_string() }
            }),
        };
        output.write_frame(response).await
    }

    /// Dispatch a warmup request to the OpenAI client, with optional
    /// per-call overrides for model / base_url. Reconfigures the client
    /// in-place if either override is supplied — this mirrors the path
    /// `handle_initialize` takes when the bridge first sends config and
    /// keeps the live `Client` instance authoritative for subsequent
    /// chat requests. (Without this, a Save-time warmup against a newly
    /// chosen model would warm the right model but leave the client
    /// pointed at the previous one for the next chat.)
    async fn handle_warmup(&mut self, params: serde_json::Value) -> serde_json::Value {
        let p: WarmupParams = serde_json::from_value(params).unwrap_or_default();
        let keep_alive = p.keep_alive.as_deref().unwrap_or("15m");

        // Reconfigure the client if the caller passed either field.
        // Skipping reconfiguration when both are absent avoids the cost
        // of rebuilding the reqwest::Client when the bridge just wants
        // to re-warm the already-active config (e.g. "Load Model"
        // button after Ollama auto-unloaded).
        if p.base_url.is_some() || p.model.is_some() {
            let new_base_url = p
                .base_url
                .clone()
                .unwrap_or_else(|| self.client.base_url().to_string());
            let new_model = p
                .model
                .clone()
                .unwrap_or_else(|| self.client.model().to_string());
            let api_key = ApiKey::from_env();
            self.client = openai::Client::new(new_base_url, new_model, api_key);
            tracing::info!("warmup: client reconfigured for warm-up call");
        }

        let model = p
            .model
            .as_deref()
            .unwrap_or(self.client.model())
            .to_string();
        let result = self.client.warmup(&model, keep_alive).await;
        // Capture (model name, tier) so the next set_config_option can
        // decide whether to grant the warmed tier or fall back to None.
        // We store even on failure (tier=None) so subsequent prompts
        // against the failed-warmup model are refused fast and don't
        // accidentally inherit a stale Native from a previous warmup.
        self.last_warmup = Some((model.clone(), result.tool_tier));
        serde_json::to_value(result).unwrap_or_else(|_| json!({"status":"failed","elapsedMs":0,"errorKind":"unknown","message":"failed to serialise WarmupResult"}))
    }

    /// TURN-scoped cancel (v0.4.0). `session/cancel` interrupts the in-flight
    /// turn and clears only that turn's state — the session entry and its
    /// in-memory conversation history SURVIVE, so a follow-up `session/prompt`
    /// with the same sessionId keeps working. The host bridge treats cancel as
    /// turn-scoped (it keeps its sessionId after a Stop/idle cancel); the
    /// previous whole-session `sessions.remove` wedged every subsequent prompt
    /// with an "unknown session" -32000. A cancel with no active turn (or for
    /// an id we never knew) is a successful no-op.
    fn handle_session_cancel(&mut self, params: serde_json::Value) -> Result<()> {
        let p: SessionCancelParams = serde_json::from_value(params)
            .map_err(|e| ShimError::AcpFraming(format!("invalid session/cancel params: {e}")))?;

        let Some(state) = self.sessions.get_mut(&p.session_id) else {
            // Unknown session: successful no-op. Defensively drop any stray
            // fast-path token entry so the map can't leak for ids that have
            // no session state.
            if let Ok(mut map) = self.cancel_tokens.lock() {
                map.remove(&p.session_id);
            }
            tracing::debug!(
                session_id = %p.session_id,
                "session/cancel for unknown session — no-op"
            );
            return Ok(());
        };

        // Trip the in-flight turn's token. The frame-router fast-path has
        // usually fired this already (mid-turn cancels can only reach here
        // after the turn ends, because the dispatcher is serialized);
        // cancel() is idempotent, and this call covers the paths the
        // fast-path can miss (poisoned lock, missing map entry). This is
        // also what actually stops the backend HTTP generation —
        // `chat_completion_stream` and the MCP awaits select on this token.
        state.cancel_token.cancel();

        // Re-arm: swap in a FRESH token for the session's next turn, in BOTH
        // the session state and the frame-router's fast-path map. Without
        // this the next prompt would clone the already-tripped token and be
        // cancelled at birth.
        let fresh = tokio_util::sync::CancellationToken::new();
        state.cancel_token = fresh.clone();
        if let Ok(mut map) = self.cancel_tokens.lock() {
            map.insert(p.session_id.clone(), fresh);
        } else {
            tracing::error!(
                session_id = %p.session_id,
                "cancel_tokens lock poisoned — session/cancel fast-path will be \
                 unavailable for this session's next turn"
            );
        }

        // Drain in-flight shim→bridge requests owned by this
        // session. Dropping each `PendingRequest` drops its
        // `oneshot::Sender`, which makes the awaiting `Receiver`
        // in `write_mcp_real` surface `RecvError` — the closure
        // translates that to a `-32000` cancelled error rather
        // than blocking until the 30s timeout fires.
        //
        // `retain` filters on `session_id` so cancellation of one
        // session never disturbs another session's in-flight
        // requests. O(in-flight), not O(total-sessions).
        let drained = match self.pending_requests.lock() {
            Ok(mut map) => {
                let before = map.len();
                map.retain(|_id, pending| pending.session_id != p.session_id);
                before - map.len()
            }
            Err(_) => {
                tracing::error!(
                    session_id = %p.session_id,
                    "pending_requests lock poisoned — cannot drain on cancel"
                );
                0
            }
        };

        tracing::info!(
            session_id = %p.session_id,
            drained_requests = drained,
            "turn cancelled — session retained"
        );

        Ok(())
    }
}
