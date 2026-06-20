//! `LocalOpenAiConnector` — the first `AgentRuntimeConnector` (Wave 1 W1-08).
//!
//! Reuse-adapter strategy: `prompt()` REUSES the existing
//! `bridge::handle_session_prompt` (the proven ReAct loop / SSE / emulated
//! parser / circuit breaker), passing it adapter closures that translate its
//! `SessionUpdateNotification` output into normalized `ConnectorEvent`s and route
//! its `mcp/*` requests through an `McpTransport`. No 829-line duplication; the
//! runtime is behaviour-identical by construction. The W1-09 translator reverses
//! the notification->event mapping, and the 10 golden transcripts verify the
//! round-trip is byte-for-byte-normalized identical under `LOCAL_LLM_USE_CONNECTOR=1`.
//!
//! Wired at `acp/server.rs` behind `LOCAL_LLM_USE_CONNECTOR=1` (default-off): when
//! set, the dispatcher drives `session/prompt` through this connector instead of
//! the inline legacy path. Both paths are golden-gated in CI.
#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use futures_util::stream::{BoxStream, StreamExt};
use tokio_util::sync::CancellationToken;

use crate::acp::messages::{SessionUpdateNotification, ToolTier};
use crate::bridge::{self, SessionState};
use crate::openai;

use super::capabilities::{ConnectorCapabilities, LocalLlmDiagnostics};
use super::error::ConnectorError;
use super::event::{ConnectorEvent, ConnectorStopReason, DiagnosticLevel, EventEnvelope, ToolKind};
use super::mcp::McpTransport;
use super::runtime::{AgentRuntimeConnector, BoxFuture};
use super::session::{ConnectorConfig, ConnectorPrompt, ConnectorSessionId, StartSessionParams};
use super::submission::{Op, SubmissionHandle};

pub struct LocalOpenAiConnector {
    client: openai::Client,
    mcp_transport: Arc<dyn McpTransport>,
    /// Per-session state behind a `tokio::sync::Mutex` so `prompt()` can hold it
    /// across the multi-second runtime call.
    sessions: Arc<Mutex<HashMap<String, Arc<tokio::sync::Mutex<SessionState>>>>>,
    /// Cancel tokens, SEPARATE from `sessions` so `cancel()` never blocks on the
    /// per-session state lock (preserves the v0.1.18 fast-path).
    cancel_tokens: Arc<Mutex<HashMap<String, CancellationToken>>>,
}

impl LocalOpenAiConnector {
    pub fn new(client: openai::Client, mcp_transport: Arc<dyn McpTransport>) -> Self {
        Self::with_cancel_tokens(
            client,
            mcp_transport,
            Arc::new(Mutex::new(HashMap::new())),
        )
    }

    /// Construct sharing an EXISTING cancel-token map (the ACP server's), so the
    /// frame-router fast-path cancel and `connector.cancel()` trip the SAME
    /// token. Used by the W1-09 wiring; `cancel()` never locks per-session state.
    pub fn with_cancel_tokens(
        client: openai::Client,
        mcp_transport: Arc<dyn McpTransport>,
        cancel_tokens: Arc<Mutex<HashMap<String, CancellationToken>>>,
    ) -> Self {
        Self {
            client,
            mcp_transport,
            sessions: Arc::new(Mutex::new(HashMap::new())),
            cancel_tokens,
        }
    }

    fn session(&self, id: &ConnectorSessionId) -> Option<Arc<tokio::sync::Mutex<SessionState>>> {
        self.sessions.lock().ok()?.get(id.as_str()).cloned()
    }
}

/// Translate the bridge's ACP-shaped `SessionUpdateNotification` into the
/// normalized `ConnectorEvent`. The W1-09 translator is the inverse — the
/// round-trip must reproduce the original notification (the goldens verify).
pub(crate) fn notification_to_event(notif: SessionUpdateNotification) -> ConnectorEvent {
    let env = EventEnvelope::default();
    let u = &notif.update;
    match u.get("sessionUpdate").and_then(|v| v.as_str()).unwrap_or("") {
        "agent_message_chunk" => ConnectorEvent::AgentMessageDelta {
            env,
            text: u
                .pointer("/content/text")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
        },
        "agent_thought_chunk" => ConnectorEvent::AgentThoughtDelta {
            env,
            text: u
                .pointer("/content/text")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
        },
        "tool_call" => ConnectorEvent::ToolCallStarted {
            env,
            call_id: u.get("toolCallId").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            name: u.get("title").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            kind: ToolKind::Other,
            arguments: u.pointer("/rawInput/arguments").cloned().unwrap_or(serde_json::json!({})),
        },
        "tool_call_update" => {
            let call_id = u.get("toolCallId").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let raw = u.get("rawOutput").cloned().unwrap_or(serde_json::Value::Null);
            if u.get("status").and_then(|v| v.as_str()) == Some("failed") {
                ConnectorEvent::ToolCallFailed { env, call_id, error_envelope: raw }
            } else {
                ConnectorEvent::ToolCallCompleted { env, call_id, output: raw }
            }
        }
        other => ConnectorEvent::Diagnostic {
            env,
            level: DiagnosticLevel::Warn,
            code: "untranslated_update".into(),
            message: format!("unmapped sessionUpdate: {other}"),
            fields: u.clone(),
        },
    }
}

impl AgentRuntimeConnector for LocalOpenAiConnector {
    fn start_session(
        &self,
        params: StartSessionParams,
    ) -> BoxFuture<'_, Result<ConnectorSessionId, ConnectorError>> {
        Box::pin(async move {
            let id = params
                .session_id
                .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
            let cancel_token = CancellationToken::new();
            let mut history = Vec::new();
            if let Some(sys) = params.system_prompt.filter(|s| !s.trim().is_empty()) {
                history.push(crate::openai::messages::ChatMessage::system(sys));
            }
            let state = SessionState {
                session_id: id.clone(),
                current_model: String::new(),
                history,
                cancel_token: cancel_token.clone(),
                tool_tier: ToolTier::None,
                token_budget_warned: false,
                pruned_turn_count: 0,
                learned_tool_ceiling: None,
            };
            self.cancel_tokens
                .lock()
                .map_err(|_| ConnectorError::Internal("cancel_tokens poisoned".into()))?
                .insert(id.clone(), cancel_token);
            self.sessions
                .lock()
                .map_err(|_| ConnectorError::Internal("sessions poisoned".into()))?
                .insert(id.clone(), Arc::new(tokio::sync::Mutex::new(state)));
            Ok(ConnectorSessionId::new(id))
        })
    }

    fn set_config(
        &self,
        session: &ConnectorSessionId,
        config: ConnectorConfig,
    ) -> BoxFuture<'_, Result<(), ConnectorError>> {
        let state = self.session(session);
        Box::pin(async move {
            let state = state.ok_or_else(|| {
                ConnectorError::InvalidParams("unknown session".into())
            })?;
            let mut guard = state.lock().await;
            if let Some(model) = config.model {
                guard.current_model = model;
            }
            // Mirror legacy handle_set_config_option: the dispatcher resolves the
            // tier against the warmed model and passes it here, so a warmed
            // Native model keeps its tool capability. Without this the session
            // stays ToolTier::None and the shared bridge strips `tools`.
            if let Some(tier) = config.tool_tier {
                guard.tool_tier = tier;
            }
            Ok(())
        })
    }

    fn prompt(
        &self,
        session: ConnectorSessionId,
        prompt: ConnectorPrompt,
    ) -> (SubmissionHandle, BoxStream<'static, ConnectorEvent>) {
        let (events_tx, events_rx) = tokio::sync::mpsc::unbounded_channel::<ConnectorEvent>();
        let (op_tx, mut op_rx) = tokio::sync::mpsc::unbounded_channel::<Op>();

        let state = self.session(&session);
        let cancel_token = self
            .cancel_tokens
            .lock()
            .ok()
            .and_then(|m| m.get(session.as_str()).cloned());
        let client = self.client.clone();
        let mcp_transport = Arc::clone(&self.mcp_transport);

        // Op forwarder: SubmissionHandle::submit(Op::Cancel) -> trip the token.
        if let Some(token) = cancel_token.clone() {
            tokio::spawn(async move {
                while let Some(op) = op_rx.recv().await {
                    // Exhaustive within-crate (no `_` arm): when Wave 6 adds an
                    // `Op` variant this becomes a compile error here, forcing the
                    // new op to be handled rather than silently ignored.
                    match op {
                        Op::Cancel => token.cancel(),
                    }
                }
            });
        }

        tokio::spawn(async move {
            let Some(state) = state else {
                let _ = events_tx.send(ConnectorEvent::Failed {
                    env: EventEnvelope::default(),
                    error: ConnectorError::InvalidParams("unknown session".into()),
                });
                return;
            };
            let mut guard = state.lock().await;

            // Build the ACP-shaped request the reused bridge runtime expects.
            // Rebuild the ACP prompt array: the text block plus any image blocks
            // (Phase 2). `handle_session_prompt` re-derives text + images via
            // `content_parts()` and gates on model vision support.
            let mut prompt_blocks =
                vec![serde_json::json!({ "type": "text", "text": prompt.text })];
            for img in &prompt.images {
                prompt_blocks.push(serde_json::json!({
                    "type": "image",
                    "data": img.data,
                    "mimeType": img.mime,
                }));
            }
            let mut params = serde_json::json!({
                "sessionId": guard.session_id,
                "prompt": prompt_blocks,
            });
            if let Some(tools) = prompt.tools {
                params["tools"] = tools;
            }
            let req: crate::acp::messages::SessionPromptParams =
                match serde_json::from_value(params) {
                    Ok(r) => r,
                    Err(e) => {
                        let _ = events_tx.send(ConnectorEvent::Failed {
                            env: EventEnvelope::default(),
                            error: ConnectorError::InvalidParams(e.to_string()),
                        });
                        return;
                    }
                };

            // Adapter: bridge SessionUpdateNotification -> ConnectorEvent.
            let ev_tx = events_tx.clone();
            let write_update = move |notif: SessionUpdateNotification| {
                let _ = ev_tx.send(notification_to_event(notif));
            };
            // Adapter: bridge mcp/* request -> McpTransport::prepare (allocate id
            // + register pending, NO write) -> emit the stamped frame as a
            // ConnectorEvent::OutboundFrame so the ACP dispatcher writes it IN
            // ORDER with session/update -> await the response future. Routing
            // mcp/* through the SAME event stream (single writer) is what stops
            // mcp/* frames racing ahead of the session/update that preceded them.
            let ev_tx_mcp = events_tx.clone();
            let cancel_for_mcp = cancel_token.clone();
            let write_mcp = move |req: serde_json::Value| {
                let t = Arc::clone(&mcp_transport);
                let ev = ev_tx_mcp.clone();
                // v0.1.37 (Finding C): pass the session cancel token so the
                // MCP-await aborts on cancel instead of waiting the full timeout.
                // Default = a never-cancelled token (a live prompt always has a
                // session token; the fallback just preserves pre-fix behaviour).
                let cancel = cancel_for_mcp.clone().unwrap_or_default();
                async move {
                    let (frame, resp) = t.prepare(req, cancel);
                    let _ = ev.send(ConnectorEvent::OutboundFrame {
                        env: EventEnvelope::default(),
                        frame,
                    });
                    resp.await
                }
            };

            let result =
                bridge::handle_session_prompt(req, &mut guard, &client, write_update, write_mcp)
                    .await;

            let env = EventEnvelope::default();
            let terminal = match result {
                Ok((finish_reason, error_kind)) => ConnectorEvent::TurnFinished {
                    env,
                    stop_reason: ConnectorStopReason::from_finish_reason(&finish_reason),
                    error_kind,
                },
                Err(crate::error::ShimError::Cancelled) => ConnectorEvent::TurnFinished {
                    env,
                    stop_reason: ConnectorStopReason::Cancelled,
                    error_kind: None,
                },
                Err(e) => ConnectorEvent::Failed { env, error: ConnectorError::from(e) },
            };
            let _ = events_tx.send(terminal);
        });

        let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(events_rx).boxed();
        (SubmissionHandle::new(op_tx), stream)
    }

    fn deliver_response(
        &self,
        _session: &ConnectorSessionId,
        _request_id: String,
        _result: Result<serde_json::Value, ConnectorError>,
    ) {
        // Wave 3: routes the outbound fs/permission reply back into the prompt
        // task. No ClientRequest is emitted in Wave 1, so this is unreachable.
    }

    fn cancel(
        &self,
        session: &ConnectorSessionId,
    ) -> BoxFuture<'_, Result<(), ConnectorError>> {
        // Trip the token WITHOUT locking the per-session state (fast-path).
        let token = self
            .cancel_tokens
            .lock()
            .ok()
            .and_then(|m| m.get(session.as_str()).cloned());
        Box::pin(async move {
            if let Some(token) = token {
                token.cancel();
            }
            Ok(())
        })
    }

    fn close_session(
        &self,
        session: ConnectorSessionId,
    ) -> BoxFuture<'_, Result<(), ConnectorError>> {
        if let Ok(mut m) = self.sessions.lock() {
            m.remove(session.as_str());
        }
        if let Ok(mut m) = self.cancel_tokens.lock() {
            m.remove(session.as_str());
        }
        Box::pin(async { Ok(()) })
    }

    fn capabilities(&self, _session: Option<&ConnectorSessionId>) -> ConnectorCapabilities {
        ConnectorCapabilities {
            supports_cancellation: true,
            supports_parallel_tools: false,
            // Phase 2: reflects whether the configured model is vision-capable
            // (registry + NWIRO_LOCAL_LLM_FORCE_VISION override). The actual
            // image gate lives in `handle_session_prompt`; this surfaces the
            // same signal for capability reporting.
            supports_image_input: crate::vision::model_supports_vision(self.client.model()),
            supports_usage_accounting: false,
            max_context_tokens: None,
        }
    }

    fn as_local_llm(&self, session: Option<&ConnectorSessionId>) -> Option<LocalLlmDiagnostics> {
        let session = session?;
        let state = self.session(session)?;
        // try_lock: as_local_llm is a cheap synchronous query; if a prompt holds
        // the state lock, report None rather than block.
        let guard = state.try_lock().ok()?;
        Some(LocalLlmDiagnostics {
            tool_tier: guard.tool_tier,
            schema_bleed_detected: false,
            // Informational; the raw configured model name. Family derivation is
            // Wave-3 polish — LocalLlmDiagnostics is unused in Wave 1.
            model_family: guard.current_model.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn connector() -> LocalOpenAiConnector {
        struct NoopMcp;
        impl McpTransport for NoopMcp {
            fn prepare(
                &self,
                req: serde_json::Value,
                _cancel: tokio_util::sync::CancellationToken,
            ) -> (serde_json::Value, BoxFuture<'static, serde_json::Value>) {
                (req, Box::pin(async { serde_json::json!({}) }))
            }
        }
        let client = openai::Client::new(
            "http://127.0.0.1:1/v1".to_string(),
            "test-model".to_string(),
            None,
        );
        LocalOpenAiConnector::new(client, Arc::new(NoopMcp))
    }

    #[tokio::test]
    async fn lifecycle_start_config_cancel_close() {
        let c: Box<dyn AgentRuntimeConnector> = Box::new(connector());
        let sid = c.start_session(StartSessionParams::default()).await.unwrap();
        c.set_config(&sid, ConnectorConfig { model: Some("qwen".into()), ..Default::default() })
            .await
            .unwrap();
        // as_local_llm reflects the configured model family + default tier.
        let diag = c.as_local_llm(Some(&sid)).expect("local diagnostics");
        assert_eq!(diag.tool_tier, ToolTier::None);
        // cancel is a no-op-safe fast path.
        c.cancel(&sid).await.unwrap();
        c.close_session(sid).await.unwrap();
    }

    #[tokio::test]
    async fn set_config_applies_tool_tier() {
        // Fix #4 guard: the dispatcher resolves the warmed tier and passes it
        // via ConnectorConfig.tool_tier; set_config must apply it to the session
        // so the shared bridge does not strip `tools` from a warmed Native model.
        let c = connector();
        let sid = c.start_session(StartSessionParams::default()).await.unwrap();
        c.set_config(
            &sid,
            ConnectorConfig {
                model: Some("qwen".into()),
                tool_tier: Some(ToolTier::Native),
            },
        )
        .await
        .unwrap();
        let diag = c.as_local_llm(Some(&sid)).expect("diagnostics");
        assert_eq!(
            diag.tool_tier,
            ToolTier::Native,
            "tool_tier from ConnectorConfig must be applied to the session"
        );
    }

    #[tokio::test]
    async fn capabilities_advertise_cancellation() {
        let c = connector();
        assert!(c.capabilities(None).supports_cancellation);
    }

    #[test]
    fn notification_translation_covers_the_emitted_set() {
        let n = SessionUpdateNotification::content_delta("s".into(), "hi".into());
        assert!(matches!(
            notification_to_event(n),
            ConnectorEvent::AgentMessageDelta { .. }
        ));
        let n = SessionUpdateNotification::tool_call_pending(
            "s".into(),
            "c1".into(),
            "find".into(),
            serde_json::json!({"x": 1}),
        );
        match notification_to_event(n) {
            ConnectorEvent::ToolCallStarted { call_id, name, .. } => {
                assert_eq!(call_id, "c1");
                assert_eq!(name, "find");
            }
            other => panic!("expected ToolCallStarted, got {other:?}"),
        }
        let n = SessionUpdateNotification::tool_call_failed(
            "s".into(),
            "c1".into(),
            serde_json::json!({"isError": true}),
        );
        assert!(matches!(
            notification_to_event(n),
            ConnectorEvent::ToolCallFailed { .. }
        ));
    }
}
