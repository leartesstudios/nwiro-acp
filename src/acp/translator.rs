//! `ConnectorEvent` -> ACP `SessionUpdateNotification` (Wave 1 W1-09).
//!
//! The forward translator: the ACP server drives a connector's event stream and
//! turns each streaming event into a `session/update` notification (terminal
//! events — `TurnFinished` / `Failed` — go to the prompt RESPONSE instead, so
//! they return `None` here).
//!
//! This is the exact inverse of `connector::local_openai::notification_to_event`.
//! The round-trip `notification -> event -> notification` MUST be identity so the
//! 10 golden transcripts pass byte-for-byte-normalized under
//! `LOCAL_LLM_USE_CONNECTOR=1`. The round-trip is proven below.
#![allow(dead_code)]

use crate::acp::messages::SessionUpdateNotification;
use crate::connector::event::ConnectorEvent;

/// Translate a streaming `ConnectorEvent` into a `session/update` notification.
/// Returns `None` for events that are NOT session/update frames:
/// - `TurnFinished` / `Failed` -> the prompt RESPONSE (`stopReason` / `-32000`)
/// - `UsageUpdated` / `Diagnostic` / `PermissionRequested` / `PlanUpdate` /
///   `FsRequest` / `ClientRequest` -> no Wave-1 wire frame (later waves).
pub fn event_to_notification(
    event: &ConnectorEvent,
    session_id: &str,
) -> Option<SessionUpdateNotification> {
    let sid = session_id.to_string();
    match event {
        ConnectorEvent::AgentMessageDelta { text, .. } => {
            Some(SessionUpdateNotification::content_delta(sid, text.clone()))
        }
        ConnectorEvent::AgentThoughtDelta { text, .. } => {
            Some(SessionUpdateNotification::thought_delta(sid, text.clone()))
        }
        ConnectorEvent::ToolCallStarted {
            call_id,
            name,
            arguments,
            ..
        } => Some(SessionUpdateNotification::tool_call_pending(
            sid,
            call_id.clone(),
            name.clone(),
            arguments.clone(),
        )),
        ConnectorEvent::ToolCallCompleted { call_id, output, .. } => Some(
            SessionUpdateNotification::tool_call_completed(sid, call_id.clone(), output.clone()),
        ),
        ConnectorEvent::ToolCallFailed {
            call_id,
            error_envelope,
            ..
        } => Some(SessionUpdateNotification::tool_call_failed(
            sid,
            call_id.clone(),
            error_envelope.clone(),
        )),
        ConnectorEvent::TurnFinished { .. }
        | ConnectorEvent::Failed { .. }
        | ConnectorEvent::UsageUpdated { .. }
        | ConnectorEvent::Diagnostic { .. }
        | ConnectorEvent::PermissionRequested { .. }
        | ConnectorEvent::PlanUpdate { .. }
        | ConnectorEvent::FsRequest { .. }
        | ConnectorEvent::ClientRequest { .. }
        // Raw outbound frame (mcp/*) — written verbatim by the dispatcher, not a
        // session/update; the dispatcher handles it before reaching the translator.
        | ConnectorEvent::OutboundFrame { .. } => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connector::local_openai::notification_to_event;

    /// The load-bearing W1-09 invariant: for every notification the bridge can
    /// emit, `notification -> event -> notification` reproduces the EXACT
    /// `.update` payload. If this holds for all emitted types, the connector
    /// path produces identical session/update frames to the legacy path.
    fn assert_roundtrip(original: SessionUpdateNotification) {
        let want = original.update.clone();
        let sid = original.session_id.clone();
        let event = notification_to_event(original);
        let back = event_to_notification(&event, &sid).expect("must translate back");
        assert_eq!(back.update, want, "round-trip changed the update payload");
        assert_eq!(back.session_id, sid);
    }

    #[test]
    fn roundtrip_content_delta() {
        assert_roundtrip(SessionUpdateNotification::content_delta(
            "s".into(),
            "hello world".into(),
        ));
    }

    #[test]
    fn roundtrip_thought_delta() {
        assert_roundtrip(SessionUpdateNotification::thought_delta(
            "s".into(),
            "let me think".into(),
        ));
    }

    #[test]
    fn roundtrip_tool_call_pending() {
        assert_roundtrip(SessionUpdateNotification::tool_call_pending(
            "s".into(),
            "call_1".into(),
            "find_blueprints".into(),
            serde_json::json!({"searchTerm": "door"}),
        ));
    }

    #[test]
    fn roundtrip_tool_call_completed() {
        assert_roundtrip(SessionUpdateNotification::tool_call_completed(
            "s".into(),
            "call_1".into(),
            serde_json::json!({"content": [{"type": "text", "text": "3 results"}], "isError": false}),
        ));
    }

    #[test]
    fn roundtrip_tool_call_failed() {
        assert_roundtrip(SessionUpdateNotification::tool_call_failed(
            "s".into(),
            "call_1".into(),
            serde_json::json!({"content": [{"type": "text", "text": "boom"}], "isError": true}),
        ));
    }

    #[test]
    fn terminal_events_produce_no_notification() {
        let env = Default::default();
        assert!(event_to_notification(
            &ConnectorEvent::TurnFinished {
                env,
                stop_reason: crate::connector::event::ConnectorStopReason::EndTurn,
                error_kind: None,
            },
            "s"
        )
        .is_none());
    }
}
