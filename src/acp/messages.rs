use serde::{Deserialize, Serialize};

// ── Inbound: bridge → shim ────────────────────────────────────────────────

/// Filesystem capabilities advertised by the bridge client.
#[derive(Debug, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct FsCapabilities {
    // TODO: gate fs read access on this flag when shim grows file-system tool support.
    #[allow(dead_code)]
    #[serde(default)]
    pub read_text_file: bool,
    // TODO: gate fs write access on this flag when shim grows file-system tool support.
    #[allow(dead_code)]
    #[serde(default)]
    pub write_text_file: bool,
}

/// Safety capabilities. May not be present in current bridge (pre-P1-006).
/// The shim treats `blockCommandExecution` as always-on regardless.
#[derive(Debug, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct SafetyCapabilities {
    // TODO: enforce per-session when shim grows safety-policy routing.
    // Currently the shim treats block_command_execution as always-on
    // regardless (no command execution path exists in the shim today).
    #[allow(dead_code)]
    #[serde(default)]
    pub block_command_execution: bool,
}

/// Client capabilities sent by the bridge in `initialize`.
#[derive(Debug, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ClientCapabilities {
    // TODO: gate file-system tool availability per session when shim
    // grows fs tool support.
    #[allow(dead_code)]
    #[serde(default)]
    pub fs: Option<FsCapabilities>,
    /// The bridge currently sends `terminal: false` directly. The shim
    /// advertises no terminal capability regardless.
    // TODO: route terminal capability when shim grows terminal support.
    #[allow(dead_code)]
    #[serde(default)]
    pub terminal: Option<bool>,
    // NOTE: parsed but not routed — shim treats block_command_execution
    // as always-on regardless (no command execution path exists today).
    #[allow(dead_code)]
    #[serde(default)]
    pub safety: Option<SafetyCapabilities>,
}

/// Local LLM endpoint config injected by the bridge (requires P1-006).
/// Fields are optional so the handler can fall back to env vars until P1-006 lands.
#[derive(Debug, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct LocalLlmContext {
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
}

/// `params.context` object in `initialize`.
///
/// DEPRECATED as the home for `localLlm` config — see `InitializeMeta`.
/// Still parsed for back-compat (v0.1.32); `handle_initialize` emits a
/// deprecation warning when config arrives via this path.
#[derive(Debug, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct InitializeContext {
    #[serde(default)]
    pub local_llm: Option<LocalLlmContext>,
}

/// `params._meta` block on `initialize`.
///
/// ACP's extensibility convention (<https://agentclientprotocol.com/protocol/extensibility>)
/// places vendor extensions under `_meta` rather than ad-hoc `context` or
/// top-level fields. v0.1.32 migrates `localLlm` config here. The legacy
/// `params.context.localLlm` form is still accepted (with a deprecation
/// warning) for one release so the UE5 bridge can roll the change without a
/// flag-day. `_meta.localLlm` takes precedence when both are present.
#[derive(Debug, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct InitializeMeta {
    #[serde(default)]
    pub local_llm: Option<LocalLlmContext>,
}

/// Params for the `initialize` request (bridge → shim).
#[derive(Debug, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct InitializeParams {
    // TODO: gate on protocol_version when shim supports multi-version
    // ACP negotiation. Currently any value is accepted.
    #[allow(dead_code)]
    #[serde(default)]
    pub protocol_version: Option<u32>,
    // TODO: route per-session feature flags when capability negotiation
    // becomes a real shim concern. Currently parsed-but-ignored.
    #[allow(dead_code)]
    #[serde(default)]
    pub client_capabilities: ClientCapabilities,
    /// DEPRECATED home for `localLlm` config — prefer `_meta.localLlm`.
    /// Still parsed for back-compat (v0.1.32); see `InitializeMeta`.
    #[serde(default)]
    pub context: Option<InitializeContext>,
    /// ACP-extensibility home for `localLlm` config (v0.1.32+). Preferred
    /// over `context.localLlm` when both are present.
    #[serde(default, rename = "_meta")]
    pub meta: Option<InitializeMeta>,
}

/// `_meta.systemPrompt` directive on `session/new`. Mirrors the Claude Code
/// host-metadata convention so the same `NwiroIKBridge::DoCreateSession`
/// payload reaches Claude, localllm, and any future ACP host without per-
/// adapter divergence.
#[derive(Debug, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct SystemPromptDirective {
    /// Content the bridge wants to seed the session's system message with.
    /// Treated as the full system message text by the shim — Claude Code's
    /// `append` semantics are functionally equivalent here because the shim
    /// does not maintain its own system-prompt baseline.
    #[serde(default)]
    pub append: Option<String>,
}

/// `_meta` block on `session/new`. Currently carries only `systemPrompt`;
/// kept as a struct so future host-metadata fields can be added without
/// breaking the wire format.
#[derive(Debug, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct SessionMeta {
    #[serde(default)]
    pub system_prompt: Option<SystemPromptDirective>,
}

/// Params for `session/new` (bridge → shim).
#[derive(Debug, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct SessionNewParams {
    // TODO: forward to MCP servers when shim grows session-cwd routing.
    #[allow(dead_code)]
    #[serde(default)]
    pub cwd: Option<String>,
    // NOTE: parsed for ACP spec compliance but intentionally ignored.
    // The current architecture routes ALL MCP traffic via the bridge
    // (`bridge::tools::execute_tool` → JSON-RPC `mcp/message`), not via
    // per-session MCP server URLs. If a future bridge protocol moves
    // MCP server registration to session-scope, this field becomes
    // load-bearing — until then, accepting + ignoring is correct.
    // Per the v0.1.23 "what's left" review pass: the field's
    // ambiguity was flagged; this comment is the decision.
    #[allow(dead_code)]
    #[serde(default)]
    pub mcp_servers: Option<Vec<serde_json::Value>>,
    /// `_meta.systemPrompt.append` carries the bridge-built system prompt
    /// when present. Hosts that don't supply it (e.g. codex-acp, which
    /// strips `_meta` from `session/new`) leave the session without a
    /// baked-in system message — the bridge should fall back to per-turn
    /// prepending for those, as it already does for Codex.
    #[serde(default, rename = "_meta")]
    pub meta: Option<SessionMeta>,
}

/// Params for `session/set_config_option` (bridge → shim).
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SetConfigOptionParams {
    pub session_id: String,
    pub config_id: String,
    pub value: String,
}

/// Params for `session/warmup` (bridge → shim). All fields optional so the
/// bridge can ask the shim to warm "whatever is currently configured" without
/// having to reach into the shim's state.
#[derive(Debug, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct WarmupParams {
    /// Override the model the shim is currently configured with. Used when
    /// the frontend wants to warm a not-yet-active model — typically passed
    /// alongside the same model the user just saved in settings.
    #[serde(default)]
    pub model: Option<String>,
    /// Override the base_url. Same rationale as `model`.
    #[serde(default)]
    pub base_url: Option<String>,
    /// Keepalive duration to request from Ollama (e.g. "15m", "-1", "0").
    /// Defaults to 15m on the shim side if absent — see
    /// `openai::Client::warmup` for the rationale.
    #[serde(default)]
    pub keep_alive: Option<String>,
}

/// Classification of a model's tool-calling capability, derived from a
/// one-shot probe at warmup time. Stored on `SessionState` so the bridge
/// can refuse tool calls cleanly when the selected model can't service
/// them — instead of letting the LLM produce useless prose.
///
/// Wire format is `snake_case` so the frontend reads `"native" / "emulated"
/// / "none"`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolTier {
    /// Model emitted `tool_calls[0].function.name` matching the probe tool.
    /// Safe to forward MCP tool round-trips.
    Native,
    /// Model emitted the tool name in `content` (wrong field but recognises
    /// the schema). v0.1.13 refuses with the same message as `None`; a
    /// later sprint may layer `<<TOOL>>` prompt emulation here.
    Emulated,
    /// Model ignored the tools field entirely — pure prose response, or
    /// any probe failure (network, non-200, non-JSON, timeout). Default.
    #[default]
    None,
}

/// Result envelope for `session/warmup` (shim → bridge → frontend).
/// Shape is intentionally narrow so the same envelope can carry results
/// from heterogeneous backends (Ollama returns a real load result; LM
/// Studio + cloud adapters return a no-op success).
#[derive(Debug, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct WarmupResult {
    /// `loaded` (Ollama actually loaded weights), `noop` (backend doesn't
    /// support pre-warming — LM Studio, cloud), or `failed` (warmup hit
    /// an error; see `error_kind`/`message`).
    pub status: String,
    /// Wall-clock duration of the warmup HTTP call, in milliseconds.
    /// Frontend uses this to update its "estimated load time" metric.
    pub elapsed_ms: u64,
    /// Tool-call capability tier, set by `Client::probe_tool_capability`
    /// on the success path of `warmup`. Defaults to `None` on any failure
    /// path so the bridge always sees a populated field.
    #[serde(default)]
    pub tool_tier: ToolTier,
    /// Approximate model size in bytes if the backend can report it.
    /// Used by the frontend to compute estimated load time on cold first
    /// warmup before any historical sample exists.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_size_bytes: Option<u64>,
    /// Categorised failure mode when `status == "failed"`. One of
    /// `not_found`, `oom`, `unreachable`, `auth`, `model_unloaded`,
    /// `broken_chat_template`, `timeout` (v0.2.1 — the load request
    /// exceeded `NWIRO_LOCAL_LLM_WARMUP_TIMEOUT_SECS`), `unknown`.
    /// Consumers treat unrecognized values as `unknown` — the set may
    /// grow. Lets the frontend show a specific actionable message
    /// instead of a generic timeout (per design decision 5).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_kind: Option<String>,
    /// Human-readable failure detail. Safe to display to the end-user.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    /// Empirical per-model upper bound on the tool count this backend
    /// + model can handle without schema-bleed (§8 item 2; sourced from
    /// ModelFamily::recommended_tool_ceiling). None = no known ceiling
    /// (well-behaved families or unrecognized name); the consumer
    /// (Nwiro) falls back to its tier-default. The shim publishes the
    /// hardware ceiling; the app is free to apply additional safety
    /// margin on top (acceptable daylight per closed_open_questions[3]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recommended_tool_ceiling: Option<u32>,
}

/// Result envelope for `session/prompt` (shim → bridge). Advisory
/// metadata sits under `_meta` per spec §131 to keep top-level
/// `result.*` keys aligned with the closed ACP key set (`stopReason`
/// only). Consumers MUST tolerate `_meta` being absent.
#[derive(Debug, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct PromptResponseResult {
    pub stop_reason: String,
    // advisory hint in _meta per spec §131; consumers must tolerate absent field
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<PromptResponseMeta>,
}

/// Advisory metadata block carried under `result._meta` on the
/// `session/prompt` response. Currently carries only `errorKind`;
/// kept as a struct so future advisory fields can be added without
/// changing the wire envelope.
#[derive(Debug, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct PromptResponseMeta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_kind: Option<PromptErrorKind>,
}

/// Typed advisory error-kind. Hand-rolled Serialize keeps wire format
/// as a plain string (e.g. "schema_bleed"); adding a variant forces
/// compile errors at every match site downstream.
///
/// `PromptErrorKind::Unknown(String)` is a forward-compat catch-all: since v0.3.0
/// (P0-C) the bridge's generic degrader constructs it for any backend error kind it
/// cannot map to a more specific variant, and a future wire receiver round-tripping
/// an unknown string back through the type still compiles.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub enum PromptErrorKind {
    SchemaBleed,
    ContextOverflow,
    Unknown(String),
}

impl serde::Serialize for PromptErrorKind {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::SchemaBleed => s.serialize_str("schema_bleed"),
            Self::ContextOverflow => s.serialize_str("context_overflow"),
            Self::Unknown(other) => s.serialize_str(other),
        }
    }
}

impl<'de> serde::Deserialize<'de> for PromptErrorKind {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        // Connector roundtrip (event.rs) requires this for the
        // ConnectorEvent::TurnFinished serde derive. Wire is a plain
        // string — known sentinels map to typed variants; everything
        // else folds into `Unknown(String)` so a future receiver can
        // round-trip unknown values without losing data.
        let s = String::deserialize(d)?;
        Ok(match s.as_str() {
            "schema_bleed" => Self::SchemaBleed,
            "context_overflow" => Self::ContextOverflow,
            _ => Self::Unknown(s),
        })
    }
}

/// Map a finish_reason sentinel to a PromptErrorKind. `schema_bleed` and
/// `context_overflow` surface an advisory hint; other finish_reasons return
/// None so the `_meta` envelope is omitted entirely (wire stays byte-identical
/// to baseline for those paths).
pub fn finish_reason_to_prompt_error_kind(finish_reason: &str) -> Option<PromptErrorKind> {
    match finish_reason {
        "schema_bleed" => Some(PromptErrorKind::SchemaBleed),
        "context_overflow" => Some(PromptErrorKind::ContextOverflow),
        // A backend HTTP 5xx surfaced as an advisory errorKind so the UE5
        // plugin can show a sanitized message instead of the raw error string.
        "server_error" => Some(PromptErrorKind::Unknown("server_error".to_string())),
        // A reasoning model that exhausted its budget without answering.
        "reasoning_budget_exhausted" => {
            Some(PromptErrorKind::Unknown("reasoning_budget_exhausted".to_string()))
        }
        // Gap-5 hardening: a bare provider `finish_reason:"error"` — a failed
        // generation signalled with NO top-level `error` object. The in-band
        // `error`-object guard in client.rs only catches the variant WITH a
        // top-level error object; without this arm the bare sentinel returns
        // `Ok(("error", None))`, a SILENT clean finish (the exact mask Gap-5
        // exists to close). Classify it as a generic backend error.
        "error" => Some(PromptErrorKind::Unknown("server_error".to_string())),
        _ => None,
    }
}

/// Extract the leading `[kind]` tag from a tagged `ShimError::OpenAiHttp`
/// message (e.g. `"[rate_limited] HTTP 429: ..."` → `"rate_limited"`). Returns
/// `None` when the message does not begin with a `[...]` tag. Plain
/// prefix-slice — no regex — so a stray `]` later in the body is ignored (P0-C).
pub fn extract_error_kind(msg: &str) -> Option<&str> {
    let rest = msg.strip_prefix('[')?;
    let end = rest.find(']')?;
    Some(&rest[..end])
}

/// Operator-worded, sanitized one-line message for a degraded prompt-path
/// failure (P0-C). Deliberately NOT phrased as a model "refusal" — these are
/// backend / transport / configuration faults the operator can act on, so the
/// text names the cause and the fix. The machine-readable discriminator travels
/// separately in `result._meta.errorKind`; this is only the human line shown in
/// the UE5 chat. `context_overflow` has its own bespoke arm and never reaches
/// here; the `_` fallback covers `unknown` and any unrecognized kind.
pub fn kind_to_user_message(kind: &str) -> &'static str {
    match kind {
        "auth" => {
            "Backend authentication failed — check the API key and the endpoint configuration."
        }
        "not_found" => {
            "The configured model was not found on the backend — check the model name and that it \
             is loaded."
        }
        "rate_limited" => {
            "The backend rate-limited this request and an automatic retry did not clear it. Wait a \
             moment and try again."
        }
        "timeout" => {
            "The backend timed out handling this request. Try again, or check the backend's load."
        }
        "server_error" => {
            "The local model backend returned an internal error while handling this request. Try \
             again, or rephrase the request."
        }
        "model_unloaded" => {
            "The model is not loaded on the backend. Load the model and try again."
        }
        "oom" => {
            "The backend ran out of memory for this request. Free VRAM/RAM, reduce the context, or \
             use a smaller model."
        }
        "tls_cert" => {
            "The backend's TLS certificate is not trusted (the shim trusts bundled roots, not the \
             OS store — a corporate TLS-intercepting proxy fails here). Use a direct endpoint."
        }
        "unreachable" => {
            "Cannot reach the local model backend — check that it is running and that the \
             configured base URL is correct."
        }
        "turn_timeout" => {
            "This turn exceeded the wall-clock limit and was aborted (possible runaway). Try a \
             shorter request, or raise the turn-duration limit."
        }
        "response_too_large" => {
            "The response exceeded the size limit and was aborted (possible repetition loop). \
             Raise the response-size limit if this was a legitimately long answer."
        }
        "stream_inactivity_timeout" => {
            "The backend stopped sending tokens (a stall) and the request was aborted. Check the \
             backend's load or connectivity, or raise NWIRO_LOCAL_LLM_INACTIVITY_TIMEOUT_SECS for a \
             very slow model."
        }
        _ => {
            "The local model backend returned an unexpected error while handling this request. Try \
             again."
        }
    }
}

/// A single content block in a `session/prompt` payload.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptBlock {
    // `kind` (ACP "type": text|image|resource|resource_link) is retained for
    // wire fidelity; routing is by field presence (`data` ⇒ image) in
    // `content_parts()`, so `kind` itself stays read-only.
    #[allow(dead_code)]
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Base64-encoded image data (ACP `image` content block). Rendered into an
    /// OpenAI `image_url` data-URL by `content_parts()` for vision-capable models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
    /// MIME type for `data` (e.g. `image/png`, `image/jpeg`). Defaults to
    /// `image/png` in `content_parts()` when the bridge omits it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

/// Params for `session/prompt` (bridge → shim).
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionPromptParams {
    pub session_id: String,
    pub prompt: Vec<PromptBlock>,
    /// Optional tool list forwarded from MCP; may be absent.
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
}

/// One image extracted from an ACP prompt block, ready to render into an OpenAI
/// `image_url` content part.
#[derive(Debug, Clone)]
pub struct ImageInput {
    pub mime: String,
    /// Base64-encoded image payload (no `data:` prefix).
    pub data: String,
}

impl SessionPromptParams {
    /// Concatenate all `text` prompt blocks into a single string for the
    /// `role: user` content slot. (Image blocks contribute nothing here — use
    /// `content_parts()` to get text + images together.)
    pub fn text_content(&self) -> String {
        self.prompt
            .iter()
            .filter_map(|b| b.text.as_ref().cloned())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Split the prompt into the joined text and the list of image inputs
    /// (blocks carrying base64 `data`). The prompt-build sites use this to
    /// construct a multimodal user message for vision-capable models, or to
    /// append an omission note for text-only models.
    pub fn content_parts(&self) -> (String, Vec<ImageInput>) {
        let images = self
            .prompt
            .iter()
            .filter_map(|b| {
                let data = b.data.as_ref()?;
                if data.is_empty() {
                    return None;
                }
                Some(ImageInput {
                    mime: b
                        .mime_type
                        .clone()
                        .unwrap_or_else(|| "image/png".to_string()),
                    data: data.clone(),
                })
            })
            .collect();
        (self.text_content(), images)
    }
}

/// Naming alias used by the bridge module — keeps both implementer
/// conventions valid while we settle on one. New code should prefer
/// `SessionPromptParams` (matches the ACP spec's "params" naming).
pub type SessionPromptRequest = SessionPromptParams;

/// Params for `session/cancel` notification (bridge → shim, no `id` field).
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionCancelParams {
    pub session_id: String,
}

// ── Outbound: shim → bridge ───────────────────────────────────────────────

/// A single `session/update` notification payload.
/// The `update` field carries whatever the OpenAI stream chunk contains.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionUpdateNotification {
    pub session_id: String,
    pub update: serde_json::Value,
}

impl SessionUpdateNotification {
    /// Convenience constructor for streaming a token of the assistant's
    /// **answer** to the bridge as an ACP `agent_message_chunk`.
    ///
    /// Field shape (`sessionUpdate` discriminator + `content.{type,text}`)
    /// matches the ACP spec at <https://agentclientprotocol.com/protocol/sessionUpdate>
    /// — the same wire format used by claude-agent-acp and codex-acp. Earlier
    /// versions of this shim emitted `{type, delta}` (OpenAI-style),
    /// which the UE5 bridge silently dropped because its dispatcher
    /// keys on `sessionUpdate`,
    /// not `type`. That mismatch is why localllm chats produced empty
    /// bubbles even when the model streamed normal content.
    pub fn content_delta(session_id: String, delta: String) -> Self {
        Self {
            session_id,
            update: serde_json::json!({
                "sessionUpdate": "agent_message_chunk",
                "content": { "type": "text", "text": delta }
            }),
        }
    }

    /// Convenience constructor for streaming a token of the model's
    /// **chain-of-thought / reasoning** to the bridge as an ACP
    /// `agent_thought_chunk`. The bridge's existing handler turns this
    /// into a "thinking…" UI indicator (bouncing dots) and disarms its
    /// first_token_timer the same way `agent_message_chunk` would —
    /// long reasoning phases (Qwen3-27B can reason for minutes) no longer
    /// trip the 300s waiting_for_first_token timeout.
    pub fn thought_delta(session_id: String, delta: String) -> Self {
        Self {
            session_id,
            update: serde_json::json!({
                "sessionUpdate": "agent_thought_chunk",
                "content": { "type": "text", "text": delta }
            }),
        }
    }

    /// v0.1.26: signal that a tool call has been emitted by the LLM
    /// and is about to be executed. The host bridge's dispatcher
    /// routes this to `tool_start` UI display so users see
    /// "tools used" indicators. Field shape: the host bridge reads
    /// `toolCallId`, `status`, `title`, and `rawInput.arguments`
    /// (parsed JSON object, not
    /// stringified). Symmetric for Native and Emulated tier — bridge
    /// dispatcher doesn't care about provenance.
    ///
    /// `arguments` is a parsed `serde_json::Value` (not a raw String)
    /// because the bridge reads it via `GetObjectField("rawInput")`
    /// → `GetObjectField("arguments")`. Callers that hold the
    /// stringified OpenAI tool_call.arguments field should parse via
    /// `serde_json::from_str` (falling back to `Value::String` on
    /// parse failure) before invoking this constructor.
    pub fn tool_call_pending(
        session_id: String,
        tool_call_id: String,
        title: String,
        arguments: serde_json::Value,
    ) -> Self {
        Self {
            session_id,
            update: serde_json::json!({
                "sessionUpdate": "tool_call",
                "toolCallId": tool_call_id,
                "status": "pending",
                "title": title,
                "rawInput": { "arguments": arguments },
            }),
        }
    }

    /// v0.1.26: signal that a tool call's MCP round-trip completed
    /// successfully. Bridge consumes `rawOutput` to display the
    /// returned content (object `{content:[...]}` / array `[{type,
    /// text}]` / plain string — all handled by the dispatcher).
    /// `result_value` should be the same value pushed to history as
    /// the `tool` message content — usually the MCP response envelope
    /// `{content: [...], isError: false}` from `bridge::tools::execute_tool`.
    pub fn tool_call_completed(
        session_id: String,
        tool_call_id: String,
        result_value: serde_json::Value,
    ) -> Self {
        Self {
            session_id,
            update: serde_json::json!({
                "sessionUpdate": "tool_call_update",
                "toolCallId": tool_call_id,
                "status": "completed",
                "rawOutput": result_value,
            }),
        }
    }

    /// v0.1.26: signal that a tool call's MCP round-trip failed
    /// (transport error, F1's in-band `isError: true` envelope, or
    /// circuit-breaker / orphan-call cleanup on F2 abort). Bridge
    /// reads `rawOutput` for the failure content — passing the same
    /// `{content: [{type:"text", text: <err-msg>}], isError: true}`
    /// envelope that v0.1.23 F1 already produces aligns this with the
    /// rest of the error pipeline. No top-level `error` field is sent
    /// because the host bridge dispatcher doesn't read one.
    pub fn tool_call_failed(
        session_id: String,
        tool_call_id: String,
        error_envelope: serde_json::Value,
    ) -> Self {
        Self {
            session_id,
            update: serde_json::json!({
                "sessionUpdate": "tool_call_update",
                "toolCallId": tool_call_id,
                "status": "failed",
                "rawOutput": error_envelope,
            }),
        }
    }
}

/// v0.1.24 G2: map an OpenAI-style `finish_reason` to an ACP-canonical
/// `stopReason` value (per the ACP prompt-turn spec at
/// <https://agentclientprotocol.com/protocol/prompt-turn>). The mapping
/// is conservative — unknown values pass through as `"end_turn"`
/// rather than fabricating a new ACP variant, because ACP clients
/// (including NwiroIKBridge) validate stopReason against a closed
/// enum.
///
/// ACP defines 5 stopReason values: `end_turn`, `max_tokens`,
/// `max_turn_requests`, `refusal`, `cancelled`.
///
/// Inputs accepted by this mapper:
/// - OpenAI finish_reason values: `stop`, `length`, `tool_calls`,
///   `content_filter`, `refusal`
/// - Synthetic shim-side sentinels passed through directly:
///   `circuit_breaker` (F2 abort), `max_turn_requests` (tool-round
///   ceiling), `cancelled` (user-initiated session/cancel)
///
/// This is the round-2 review correction: round-1 invented a
/// `sessionUpdate: "end_of_turn"` notification, but ACP defines turn
/// completion on the `session/prompt` RESPONSE via `stopReason`.
/// Round-3 extends to cover `cancelled` and `max_turn_requests`
/// (per round-2 review — HIGH severity on Cancelled going
/// through `-32800` error path instead of `stopReason: "cancelled"`).
pub fn map_finish_reason_to_acp_stop_reason(finish_reason: &str) -> &'static str {
    match finish_reason {
        "stop" | "tool_calls" => "end_turn",
        "length" => "max_tokens",
        "content_filter" | "refusal" => "refusal",
        // `circuit_breaker` (F2 repeated-failure abort), `schema_bleed`
        // (real-request schema-bleed guard), `context_overflow` (the prompt
        // plus attached tools no longer fit the model's loaded context window)
        // `server_error` (a backend HTTP 5xx on the prompt round) and
        // `reasoning_budget_exhausted` (a thinking model spent its whole
        // generation budget on chain-of-thought and produced no answer) are
        // shim-side content refusals — a clean degrade, not a transport failure
        // that would leak a raw backend error string to the UI as a -32000.
        "circuit_breaker" | "schema_bleed" | "context_overflow" | "server_error"
        | "reasoning_budget_exhausted"
        // P0-C: every classified prompt-path backend/transport/config failure
        // degrades to a clean refusal carrying an advisory errorKind instead of a
        // flat -32000. These mirror client.rs::classify_http_error_kind plus the
        // P0-E shim-side abort guards (turn_timeout, response_too_large).
        | "auth" | "not_found" | "rate_limited" | "timeout" | "oom" | "model_unloaded"
        | "tls_cert" | "unreachable" | "unknown" | "turn_timeout"
        | "response_too_large" | "stream_inactivity_timeout"
        // Gap-5: a bare provider `finish_reason:"error"` (failed generation with
        // no top-level error object) is a backend failure, never a clean finish.
        | "error" => "refusal",
        // Round-3 round-trip: shim-side sentinels that already name
        // ACP values directly. Pass through verbatim. Decoupled from
        // the OpenAI → ACP mapping above because the shim emits these
        // for protocol-defined conditions (cancellation, turn-request
        // budget exhaustion) that OpenAI's finish_reason vocabulary
        // doesn't cover.
        "max_turn_requests" => "max_turn_requests",
        "cancelled" => "cancelled",
        _ => "end_turn",
    }
}

// v0.1.24 C1: typed MCP structs (McpConnectParams, McpConnectResult,
// McpMessageParams) were removed. The shim → bridge MCP transport
// uses `serde_json::Value` end-to-end via `bridge::tools::execute_tool`
// (see `src/bridge/tools.rs::execute_tool` which constructs the
// `mcp/message` JSON-RPC envelope inline). The typed structs were
// speculative pre-implementation sketches that never gained callers.
// They were flagged as STALE in the v0.1.23 "what's left"
// review pass.

#[cfg(test)]
mod tests {
    use super::*;

    // --- Phase 2: image content parts ---

    #[test]
    fn content_parts_splits_text_and_images_with_camelcase_mime() {
        let p: SessionPromptParams = serde_json::from_value(serde_json::json!({
            "sessionId": "s1",
            "prompt": [
                {"type": "text", "text": "describe"},
                {"type": "image", "data": "BASE64", "mimeType": "image/jpeg"}
            ]
        }))
        .unwrap();
        let (text, images) = p.content_parts();
        assert_eq!(text, "describe");
        assert_eq!(images.len(), 1);
        // Regression: the ACP wire sends `mimeType` (camelCase); PromptBlock must
        // deserialise it into `mime_type`. (Pre-fix this was None → "image/png".)
        assert_eq!(images[0].mime, "image/jpeg");
        assert_eq!(images[0].data, "BASE64");
    }

    #[test]
    fn content_parts_text_only_yields_no_images() {
        let p: SessionPromptParams = serde_json::from_value(serde_json::json!({
            "sessionId": "s1", "prompt": [{"type": "text", "text": "hi"}]
        }))
        .unwrap();
        let (text, images) = p.content_parts();
        assert_eq!(text, "hi");
        assert!(images.is_empty());
    }

    #[test]
    fn image_block_mime_defaults_to_png_when_omitted() {
        let p: SessionPromptParams = serde_json::from_value(serde_json::json!({
            "sessionId": "s1", "prompt": [{"type": "image", "data": "X"}]
        }))
        .unwrap();
        let (_t, images) = p.content_parts();
        assert_eq!(images[0].mime, "image/png");
    }

    // v0.1.24 G2 round-2 (post-review) — verify the OpenAI →
    // ACP stopReason mapping. The bridge consumes stopReason via
    // ACP's prompt-response channel, so the mapping must produce
    // valid ACP enum values.

    #[test]
    fn stop_reason_maps_natural_stop_to_end_turn() {
        assert_eq!(map_finish_reason_to_acp_stop_reason("stop"), "end_turn");
    }

    #[test]
    fn stop_reason_maps_tool_calls_to_end_turn() {
        // A tool-call completion is still a natural turn end — the
        // tool exchange wrapped up cleanly. ACP doesn't have a
        // separate "tool_use" stop reason.
        assert_eq!(
            map_finish_reason_to_acp_stop_reason("tool_calls"),
            "end_turn"
        );
    }

    #[test]
    fn stop_reason_maps_length_to_max_tokens() {
        // OpenAI "length" = hit max_tokens budget. ACP equivalent.
        assert_eq!(
            map_finish_reason_to_acp_stop_reason("length"),
            "max_tokens"
        );
    }

    #[test]
    fn stop_reason_maps_content_filter_to_refusal() {
        assert_eq!(
            map_finish_reason_to_acp_stop_reason("content_filter"),
            "refusal"
        );
    }

    #[test]
    fn stop_reason_maps_circuit_breaker_to_refusal() {
        // F2 circuit-breaker abort surfaces as refusal so ACP clients
        // don't try to interpret it as a "model finished naturally"
        // signal.
        assert_eq!(
            map_finish_reason_to_acp_stop_reason("circuit_breaker"),
            "refusal"
        );
    }

    #[test]
    fn stop_reason_maps_shim_abort_guards_to_refusal() {
        // The v0.3.0 / P0-E shim-side abort guards each surface as a CLEAN refusal — a
        // deliberate, safety-bearing choice, NOT end_turn (which would look like a
        // natural finish). `stream_inactivity_timeout` in particular is TERMINAL by
        // design: it fires mid-stream after bytes may already have been emitted, so the
        // turn cannot be retried without tearing output. Pinning it here stops a future
        // change from silently reclassifying it as retriable or end_turn (the risk a
        // v0.3.0 risk-assessment review flagged). To allow longer HEALTHY think-pauses
        // on slow reasoning models, raise NWIRO_LOCAL_LLM_INACTIVITY_TIMEOUT_SECS — do
        // NOT change this mapping.
        for kind in [
            "stream_inactivity_timeout",
            "turn_timeout",
            "response_too_large",
        ] {
            assert_eq!(
                map_finish_reason_to_acp_stop_reason(kind),
                "refusal",
                "{kind} must map to a clean refusal"
            );
        }
    }

    #[test]
    fn bare_finish_reason_error_maps_to_refusal_with_error_kind() {
        // Gap-5 hardening (audit MAJOR-A): a bare provider `finish_reason:"error"`
        // — a failed generation signalled with NO top-level `error` object — must
        // NOT slip through as a clean `end_turn`. The in-band error-OBJECT guard in
        // client.rs only covers the variant WITH a top-level error object; this
        // pins the bare-sentinel half. Before the fix, "error" hit the
        // `_ => "end_turn"` fallback AND finish_reason_to_prompt_error_kind("error")
        // returned None, so the turn ended Ok(("error", None)) — a silent clean
        // finish (the exact mask Gap-5 exists to close).
        assert_eq!(
            map_finish_reason_to_acp_stop_reason("error"),
            "refusal",
            "bare finish_reason:\"error\" must degrade to a refusal, never end_turn"
        );
        assert!(
            finish_reason_to_prompt_error_kind("error").is_some(),
            "bare finish_reason:\"error\" must carry an advisory errorKind, not None"
        );
    }

    #[test]
    fn stop_reason_unknown_falls_through_to_end_turn() {
        // Conservative fallback: an unknown OpenAI finish_reason value
        // becomes end_turn rather than fabricating a new ACP variant.
        // ACP clients reject unrecognised stopReason values.
        assert_eq!(
            map_finish_reason_to_acp_stop_reason("brand_new_openai_value"),
            "end_turn"
        );
        assert_eq!(map_finish_reason_to_acp_stop_reason(""), "end_turn");
    }

    #[test]
    fn stop_reason_max_turn_requests_passes_through() {
        // Round-3 review correction: the tool-round ceiling path in
        // bridge/mod.rs now emits "max_turn_requests" directly (it's
        // already an ACP value, not an OpenAI finish_reason). Mapper
        // must pass it through verbatim, not fall through to end_turn.
        assert_eq!(
            map_finish_reason_to_acp_stop_reason("max_turn_requests"),
            "max_turn_requests"
        );
    }

    #[test]
    fn stop_reason_cancelled_passes_through() {
        // Round-3 review correction: the Cancelled path in acp/server.rs
        // sends "cancelled" directly as the stopReason (ACP requires
        // this for user-initiated session/cancel — it's a turn-completion
        // reason, not a transport error). Mapper passes through verbatim.
        assert_eq!(
            map_finish_reason_to_acp_stop_reason("cancelled"),
            "cancelled"
        );
    }

    // -------- v0.1.26: tool_call / tool_call_update wire-shape pins --------
    //
    // The host bridge dispatcher reads specific field names for
    // `tool_call` and `tool_call_update`. These tests pin the wire
    // shape so any future change is immediately caught — a sessionUpdate
    // value the bridge has no case for is silently dropped.

    #[test]
    fn tool_call_pending_wire_shape() {
        let n = SessionUpdateNotification::tool_call_pending(
            "session-1".to_string(),
            "call_abc".to_string(),
            "find_blueprints".to_string(),
            serde_json::json!({"searchTerm": "test"}),
        );
        let u = &n.update;
        assert_eq!(u.get("sessionUpdate").and_then(|v| v.as_str()), Some("tool_call"));
        assert_eq!(u.get("toolCallId").and_then(|v| v.as_str()), Some("call_abc"));
        assert_eq!(u.get("status").and_then(|v| v.as_str()), Some("pending"));
        assert_eq!(u.get("title").and_then(|v| v.as_str()), Some("find_blueprints"));
        // rawInput.arguments must be an OBJECT (the bridge calls
        // GetObjectField), not a stringified JSON.
        let raw_input = u.get("rawInput").expect("rawInput missing");
        assert!(raw_input.is_object(), "rawInput must be a JSON object");
        let args = raw_input.get("arguments").expect("rawInput.arguments missing");
        assert!(args.is_object(), "rawInput.arguments must be a JSON object");
        assert_eq!(
            args.get("searchTerm").and_then(|v| v.as_str()),
            Some("test")
        );
    }

    #[test]
    fn tool_call_completed_wire_shape() {
        let result = serde_json::json!({
            "content": [{"type": "text", "text": "done"}],
            "isError": false,
        });
        let n = SessionUpdateNotification::tool_call_completed(
            "session-1".to_string(),
            "call_abc".to_string(),
            result.clone(),
        );
        let u = &n.update;
        assert_eq!(
            u.get("sessionUpdate").and_then(|v| v.as_str()),
            Some("tool_call_update"),
            "discriminator MUST be tool_call_update (not tool_call) for status changes"
        );
        assert_eq!(u.get("toolCallId").and_then(|v| v.as_str()), Some("call_abc"));
        assert_eq!(u.get("status").and_then(|v| v.as_str()), Some("completed"));
        // rawOutput carries the MCP result envelope verbatim — bridge
        // handles {content:[...]} | [...] | string shapes.
        let raw_output = u.get("rawOutput").expect("rawOutput missing");
        assert_eq!(raw_output, &result);
    }

    #[test]
    fn tool_call_failed_wire_shape() {
        let err_envelope = serde_json::json!({
            "content": [{"type": "text", "text": "Tool execution failed: timeout"}],
            "isError": true,
        });
        let n = SessionUpdateNotification::tool_call_failed(
            "session-1".to_string(),
            "call_xyz".to_string(),
            err_envelope.clone(),
        );
        let u = &n.update;
        assert_eq!(
            u.get("sessionUpdate").and_then(|v| v.as_str()),
            Some("tool_call_update")
        );
        assert_eq!(u.get("toolCallId").and_then(|v| v.as_str()), Some("call_xyz"));
        assert_eq!(u.get("status").and_then(|v| v.as_str()), Some("failed"));
        assert_eq!(u.get("rawOutput").expect("rawOutput missing"), &err_envelope);
        // No top-level `error` field — the host bridge dispatcher
        // doesn't read one.
        assert!(
            u.get("error").is_none(),
            "must NOT include a top-level `error` field; failure text rides in rawOutput"
        );
    }

    #[test]
    fn content_delta_and_thought_delta_unchanged() {
        // Regression guard: C1's struct deletions and G2's additions
        // must not have disturbed the existing constructors.
        let c = SessionUpdateNotification::content_delta(
            "s".to_string(),
            "hello".to_string(),
        );
        assert_eq!(
            c.update.get("sessionUpdate").and_then(|v| v.as_str()),
            Some("agent_message_chunk")
        );
        let t = SessionUpdateNotification::thought_delta(
            "s".to_string(),
            "thinking".to_string(),
        );
        assert_eq!(
            t.update.get("sessionUpdate").and_then(|v| v.as_str()),
            Some("agent_thought_chunk")
        );
    }

    // -------- v0.1.32 W0-A: ACP enum-string audit --------
    //
    // Audit conclusion: the shim ALREADY emits ACP-canonical tool-call
    // status strings and sessionUpdate discriminators (verified by the
    // per-constructor wire-shape tests above). There is NO `kind` field
    // (read|edit|delete|…) in the current tool-call notifications — they
    // carry `title`, not `kind` — so there is nothing to audit there.
    // This test consolidates the canonical status-string set as a single
    // regression guard so the audit conclusion can't silently regress.

    #[test]
    fn acp_tool_call_status_strings_are_canonical() {
        // ACP tool-call status vocabulary: pending | in_progress |
        // completed | failed. The shim emits pending (start), completed
        // (success), failed (error). `in_progress` has no emitter yet (no
        // streaming-args path exists); it is intentionally NOT emitted.
        let pending = SessionUpdateNotification::tool_call_pending(
            "s".into(),
            "c".into(),
            "t".into(),
            serde_json::json!({}),
        );
        assert_eq!(
            pending.update.get("status").and_then(|v| v.as_str()),
            Some("pending")
        );

        let completed = SessionUpdateNotification::tool_call_completed(
            "s".into(),
            "c".into(),
            serde_json::json!({}),
        );
        assert_eq!(
            completed.update.get("status").and_then(|v| v.as_str()),
            Some("completed")
        );

        let failed = SessionUpdateNotification::tool_call_failed(
            "s".into(),
            "c".into(),
            serde_json::json!({}),
        );
        assert_eq!(
            failed.update.get("status").and_then(|v| v.as_str()),
            Some("failed")
        );
    }

    // -------- v0.1.32 W0-B: localLlm config _meta namespacing --------
    //
    // localLlm config migrated from `params.context.localLlm` to
    // `params._meta.localLlm` (ACP extensibility convention). Both forms
    // must deserialize; the server prefers `_meta` and warns on the legacy
    // path (precedence logic lives in `acp::server::handle_initialize`).
    // These tests pin the wire contract both sides must agree on.

    #[test]
    fn initialize_parses_meta_local_llm() {
        let params = serde_json::json!({
            "protocolVersion": 1,
            "_meta": { "localLlm": { "baseUrl": "http://localhost:1234/v1", "model": "qwen3-27b" } }
        });
        let init: InitializeParams = serde_json::from_value(params).unwrap();
        let llm = init
            .meta
            .and_then(|m| m.local_llm)
            .expect("_meta.localLlm should parse");
        assert_eq!(llm.base_url.as_deref(), Some("http://localhost:1234/v1"));
        assert_eq!(llm.model.as_deref(), Some("qwen3-27b"));
    }

    #[test]
    fn initialize_parses_legacy_context_local_llm() {
        // Back-compat: the pre-v0.1.32 form must still deserialize so the
        // server can honour it (with a deprecation warning).
        let params = serde_json::json!({
            "context": { "localLlm": { "baseUrl": "http://localhost:11434/v1", "model": "gemma3" } }
        });
        let init: InitializeParams = serde_json::from_value(params).unwrap();
        let llm = init
            .context
            .and_then(|c| c.local_llm)
            .expect("context.localLlm should parse");
        assert_eq!(llm.base_url.as_deref(), Some("http://localhost:11434/v1"));
        assert_eq!(llm.model.as_deref(), Some("gemma3"));
    }

    #[test]
    fn initialize_meta_and_context_both_survive_parse() {
        // When both are present, both deserialize so the server's
        // precedence logic (handle_initialize) has both to choose from;
        // it picks `_meta`. This pins that neither is dropped at parse time.
        let params = serde_json::json!({
            "_meta": { "localLlm": { "model": "from-meta" } },
            "context": { "localLlm": { "model": "from-context" } }
        });
        let init: InitializeParams = serde_json::from_value(params).unwrap();
        assert_eq!(
            init.meta
                .and_then(|m| m.local_llm)
                .and_then(|l| l.model)
                .as_deref(),
            Some("from-meta")
        );
        assert_eq!(
            init.context
                .and_then(|c| c.local_llm)
                .and_then(|l| l.model)
                .as_deref(),
            Some("from-context")
        );
    }

    #[test]
    fn initialize_empty_params_yield_no_config() {
        // Neither form present — both None, server keeps the env-var client.
        let init: InitializeParams =
            serde_json::from_value(serde_json::json!({})).unwrap();
        assert!(init.meta.is_none());
        assert!(init.context.is_none());
    }
}
