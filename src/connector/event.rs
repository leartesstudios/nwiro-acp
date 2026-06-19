//! Normalized connector event vocabulary (Wave 1 W1-02).
//!
//! The runtime emits `ConnectorEvent`s; the ACP server translates them to
//! `session/update` notifications and the prompt response. Phase A: defined,
//! not yet emitted/translated (W1-08 emits, W1-09 translates).
#![allow(dead_code)]

use super::error::ConnectorError;

/// Routing envelope on every `ConnectorEvent`. Multi-agent orchestrators
/// (Wave 6) populate these; Wave 1 leaves them all `None`. `session_id` is
/// intentionally NOT here — the dispatcher passes it to `prompt()` and already
/// knows it; duplicating risks drift.
///
/// All four routing fields are kept per the product requirement that Wave 1 be
/// wire-ready for multi-agent without a later format break. They serialize to
/// nothing when `None`.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct EventEnvelope {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_run_id: Option<String>,
}

/// ACP-canonical tool-call kind. Exact strings matter — the IDE renders on them.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ToolKind {
    Read,
    Edit,
    Delete,
    Move,
    Search,
    Execute,
    Think,
    Fetch,
    Other,
}

/// Diagnostic severity for `ConnectorEvent::Diagnostic`.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticLevel {
    Info,
    Warn,
    Error,
}

/// A single plan entry (Wave 4 emission; maps to ACP `plan`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PlanEntry {
    pub content: String,
    pub status: String,
    pub priority: String,
}

/// A permission option offered to the client (Wave 3 emission).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PermissionOption {
    pub option_id: String,
    pub name: String,
    /// allow_once | allow_always | reject_once | reject_always
    pub kind: String,
}

/// Filesystem operation for `ConnectorEvent::FsRequest` (Wave 3 emission).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum FsOp {
    ReadTextFile,
    WriteTextFile { contents: String },
}

/// ACP-canonical stop reasons for the `session/prompt` response.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConnectorStopReason {
    EndTurn,
    MaxTokens,
    MaxTurnRequests,
    Refusal,
    Cancelled,
}

impl ConnectorStopReason {
    /// Map to the ACP-canonical stop-reason string (the 5 spec values).
    pub fn to_acp_str(self) -> &'static str {
        match self {
            ConnectorStopReason::EndTurn => "end_turn",
            ConnectorStopReason::MaxTokens => "max_tokens",
            ConnectorStopReason::MaxTurnRequests => "max_turn_requests",
            ConnectorStopReason::Refusal => "refusal",
            ConnectorStopReason::Cancelled => "cancelled",
        }
    }

    /// Map a backend/sentinel finish_reason to a stop reason. Mirrors the
    /// existing `acp::messages::map_finish_reason_to_acp_stop_reason` policy so
    /// the connector path produces identical stop reasons to the legacy path.
    pub fn from_finish_reason(finish_reason: &str) -> Self {
        match finish_reason {
            "length" => ConnectorStopReason::MaxTokens,
            "content_filter" | "refusal" | "circuit_breaker" | "schema_bleed"
            | "context_overflow" | "server_error" | "reasoning_budget_exhausted"
            // P0-C: classified prompt-path backend/transport/config failures +
            // the P0-E abort guards degrade to a refusal (mirrors
            // acp::messages::map_finish_reason_to_acp_stop_reason).
            | "auth" | "not_found" | "rate_limited" | "timeout" | "oom" | "model_unloaded"
            | "tls_cert" | "unreachable" | "unknown" | "turn_timeout"
            | "response_too_large" | "stream_inactivity_timeout"
            // Gap-5: bare provider `finish_reason:"error"` -> refusal (mirror).
            | "error" => ConnectorStopReason::Refusal,
            "max_turn_requests" => ConnectorStopReason::MaxTurnRequests,
            "cancelled" => ConnectorStopReason::Cancelled,
            // "stop" | "tool_calls" | unknown -> end_turn (conservative).
            _ => ConnectorStopReason::EndTurn,
        }
    }
}

/// The normalized events any `AgentRuntimeConnector` emits. Wave 1's
/// `LocalOpenAiConnector` emits variants 1-8, 11, and 14; 9, 10, 12, 13 are
/// defined for later waves so the enum never has to break to add them.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
// Tag is "event" (not "kind") because ToolCallStarted has a `kind` field.
// This serde form is INTERNAL (test/debug only) — the ACP wire shape is
// produced by the W1-09 translator, not by serializing this enum.
#[serde(tag = "event", rename_all = "snake_case")]
pub enum ConnectorEvent {
    /// 1. Streaming prose token -> ACP `agent_message_chunk`.
    AgentMessageDelta { env: EventEnvelope, text: String },
    /// 2. Streaming reasoning token -> ACP `agent_thought_chunk`.
    AgentThoughtDelta { env: EventEnvelope, text: String },
    /// 3. Tool call about to execute -> ACP `tool_call` (status=pending).
    ToolCallStarted {
        env: EventEnvelope,
        call_id: String,
        name: String,
        kind: ToolKind,
        arguments: serde_json::Value,
    },
    /// 4. Tool call completed -> ACP `tool_call_update` (status=completed).
    ToolCallCompleted {
        env: EventEnvelope,
        call_id: String,
        output: serde_json::Value,
    },
    /// 5. Tool call failed -> ACP `tool_call_update` (status=failed).
    ToolCallFailed {
        env: EventEnvelope,
        call_id: String,
        error_envelope: serde_json::Value,
    },
    /// 6. Usage update (Wave 4 emission).
    UsageUpdated {
        env: EventEnvelope,
        prompt_tokens: u64,
        completion_tokens: u64,
        cost_usd: Option<f64>,
    },
    /// 7. Turn finished -> written to the prompt RESPONSE (not a session/update).
    /// Cancellation goes through here, not `Failed`.
    TurnFinished {
        env: EventEnvelope,
        stop_reason: ConnectorStopReason,
        /// Optional advisory hint surfaced as `result._meta.errorKind`
        /// on the prompt response. None for ordinary turn ends.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error_kind: Option<crate::acp::messages::PromptErrorKind>,
    },
    /// 8. Fatal runtime error -> JSON-RPC error response on the prompt request.
    /// NOTE: `ConnectorError` must derive Serialize/Deserialize/Clone/Debug
    /// (it does) — this embedded variant is why the whole enum requires it.
    Failed { env: EventEnvelope, error: ConnectorError },
    /// 9. Permission request (Wave 3 emission).
    PermissionRequested {
        env: EventEnvelope,
        request_id: String,
        tool_name: String,
        arguments: serde_json::Value,
        options: Vec<PermissionOption>,
    },
    /// 10. Plan / TodoWrite update (Wave 4 emission) -> ACP `plan`.
    PlanUpdate {
        env: EventEnvelope,
        entries: Vec<PlanEntry>,
    },
    /// 11. Generic diagnostic — logs, not user UI (schema-bleed, history
    /// pruning, circuit breaker; inspectable in tests).
    Diagnostic {
        env: EventEnvelope,
        level: DiagnosticLevel,
        code: String,
        message: String,
        fields: serde_json::Value,
    },
    /// 12. Filesystem read/write request (Wave 3 emission via FsProvider).
    FsRequest {
        env: EventEnvelope,
        request_id: String,
        op: FsOp,
        /// MUST be absolute (`connector::path::reject_relative_path`).
        path: String,
    },
    /// 13. Outbound JSON-RPC request the connector needs the ACP client to
    /// perform (fs/*, request_permission, ...). The ACP server performs it and
    /// returns the result via `AgentRuntimeConnector::deliver_response`. This is
    /// the "stream-up, handle-down" replacement for passing a sender into
    /// `prompt()`. Wave 1: defined + logged; emitted in Wave 3.
    ClientRequest {
        env: EventEnvelope,
        request_id: String,
        method: String,
        params: serde_json::Value,
    },
    /// 14. A pre-stamped outbound JSON-RPC frame the connector needs WRITTEN in
    /// order with the session/update stream (Wave 1: `mcp/*` requests). Unlike
    /// `ClientRequest` (Wave 3, correlated via `deliver_response`), the response
    /// here is correlated out-of-band by the `McpTransport` — the prompt task
    /// awaits the future that `McpTransport::prepare` returned. This variant
    /// exists ONLY so the FRAME is emitted by the same single writer (the ACP
    /// dispatcher) that writes session/update, eliminating the two-writer
    /// reorder. The dispatcher writes `frame` verbatim; the translator maps it
    /// to `None` (it is not a session/update).
    OutboundFrame {
        env: EventEnvelope,
        frame: serde_json::Value,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_all_none_serializes_to_empty_object() {
        let env = EventEnvelope::default();
        assert_eq!(serde_json::to_value(&env).unwrap(), serde_json::json!({}));
    }

    #[test]
    fn stop_reason_acp_strings() {
        assert_eq!(ConnectorStopReason::EndTurn.to_acp_str(), "end_turn");
        assert_eq!(ConnectorStopReason::MaxTokens.to_acp_str(), "max_tokens");
        assert_eq!(
            ConnectorStopReason::MaxTurnRequests.to_acp_str(),
            "max_turn_requests"
        );
        assert_eq!(ConnectorStopReason::Refusal.to_acp_str(), "refusal");
        assert_eq!(ConnectorStopReason::Cancelled.to_acp_str(), "cancelled");
    }

    #[test]
    fn finish_reason_mapping_matches_legacy_policy() {
        assert_eq!(
            ConnectorStopReason::from_finish_reason("stop"),
            ConnectorStopReason::EndTurn
        );
        assert_eq!(
            ConnectorStopReason::from_finish_reason("tool_calls"),
            ConnectorStopReason::EndTurn
        );
        assert_eq!(
            ConnectorStopReason::from_finish_reason("length"),
            ConnectorStopReason::MaxTokens
        );
        assert_eq!(
            ConnectorStopReason::from_finish_reason("circuit_breaker"),
            ConnectorStopReason::Refusal
        );
        assert_eq!(
            // Parity with the legacy mapper (messages.rs): the real-request
            // schema-bleed guard surfaces as a refusal on BOTH paths.
            ConnectorStopReason::from_finish_reason("schema_bleed"),
            ConnectorStopReason::Refusal
        );
        assert_eq!(
            ConnectorStopReason::from_finish_reason("cancelled"),
            ConnectorStopReason::Cancelled
        );
        assert_eq!(
            ConnectorStopReason::from_finish_reason("brand_new"),
            ConnectorStopReason::EndTurn
        );
    }

    #[test]
    fn all_fourteen_variants_construct_and_serde_roundtrip() {
        let env = EventEnvelope::default();
        let events = vec![
            ConnectorEvent::AgentMessageDelta { env: env.clone(), text: "hi".into() },
            ConnectorEvent::AgentThoughtDelta { env: env.clone(), text: "think".into() },
            ConnectorEvent::ToolCallStarted {
                env: env.clone(),
                call_id: "c1".into(),
                name: "t".into(),
                kind: ToolKind::Search,
                arguments: serde_json::json!({}),
            },
            ConnectorEvent::ToolCallCompleted { env: env.clone(), call_id: "c1".into(), output: serde_json::json!({}) },
            ConnectorEvent::ToolCallFailed { env: env.clone(), call_id: "c1".into(), error_envelope: serde_json::json!({}) },
            ConnectorEvent::UsageUpdated { env: env.clone(), prompt_tokens: 1, completion_tokens: 2, cost_usd: None },
            ConnectorEvent::TurnFinished {
                env: env.clone(),
                stop_reason: ConnectorStopReason::EndTurn,
                error_kind: None,
            },
            ConnectorEvent::Failed { env: env.clone(), error: ConnectorError::Cancelled },
            ConnectorEvent::PermissionRequested { env: env.clone(), request_id: "r".into(), tool_name: "t".into(), arguments: serde_json::json!({}), options: vec![] },
            ConnectorEvent::PlanUpdate { env: env.clone(), entries: vec![] },
            ConnectorEvent::Diagnostic { env: env.clone(), level: DiagnosticLevel::Info, code: "c".into(), message: "m".into(), fields: serde_json::json!({}) },
            ConnectorEvent::FsRequest { env: env.clone(), request_id: "r".into(), op: FsOp::ReadTextFile, path: "/abs".into() },
            ConnectorEvent::ClientRequest { env: env.clone(), request_id: "r".into(), method: "fs/read_text_file".into(), params: serde_json::json!({}) },
            ConnectorEvent::OutboundFrame { env, frame: serde_json::json!({"jsonrpc":"2.0","id":1_000_000,"method":"mcp/connect"}) },
        ];
        assert_eq!(events.len(), 14);
        for e in &events {
            let s = serde_json::to_string(e).expect("serialize");
            let _back: ConnectorEvent = serde_json::from_str(&s).expect("deserialize");
        }
    }
}
