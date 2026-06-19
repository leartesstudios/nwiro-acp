//! The `AgentRuntimeConnector` trait (Wave 1 W1-05) — the single contract
//! between the ACP server and any agent runtime.
//!
//! Object-safe (held as `Box<dyn AgentRuntimeConnector>`); async via hand-rolled
//! `BoxFuture` / `BoxStream` (no `async-trait` dep). Phase A: defined, with the
//! `LocalOpenAiConnector` impl arriving in W1-08.
#![allow(dead_code)]

use futures_util::stream::BoxStream;

use super::capabilities::{ConnectorCapabilities, LocalLlmDiagnostics};
use super::error::ConnectorError;
use super::event::ConnectorEvent;
use super::session::{ConnectorConfig, ConnectorPrompt, ConnectorSessionId, StartSessionParams};
use super::submission::SubmissionHandle;

/// Boxed, `Send`, lifetime-parameterized future — the return type of the
/// trait's async methods.
pub type BoxFuture<'a, T> =
    std::pin::Pin<Box<dyn std::future::Future<Output = T> + Send + 'a>>;

pub trait AgentRuntimeConnector: Send + Sync {
    /// Create a session; returns its handle.
    fn start_session(
        &self,
        params: StartSessionParams,
    ) -> BoxFuture<'_, Result<ConnectorSessionId, ConnectorError>>;

    /// Update per-session config (model; future: mode). Idempotent.
    fn set_config(
        &self,
        session: &ConnectorSessionId,
        config: ConnectorConfig,
    ) -> BoxFuture<'_, Result<(), ConnectorError>>;

    /// Drive one prompt to completion. Returns a `'static` event stream (so it
    /// can be driven from a spawned task / future actor) PLUS a handle to
    /// interject ops. The stream MUST terminate with exactly one `TurnFinished`
    /// OR one `Failed` event, then `None`.
    ///
    /// The `'static` lifetime and owned `session` are the v2.1 catastrophic-fix:
    /// a `BoxStream<'_, ..>` tied to `&self` cannot be spawned, which the Wave 2
    /// actor model requires.
    fn prompt(
        &self,
        session: ConnectorSessionId,
        prompt: ConnectorPrompt,
    ) -> (SubmissionHandle, BoxStream<'static, ConnectorEvent>);

    /// Deliver the result of an outbound `ConnectorEvent::ClientRequest`
    /// (fs/*, permission) back into the connector. Wave 1: defined; exercised
    /// in Wave 3.
    fn deliver_response(
        &self,
        session: &ConnectorSessionId,
        request_id: String,
        result: Result<serde_json::Value, ConnectorError>,
    );

    /// Cancel the in-flight prompt for a session. Idempotent, cheap, and MUST
    /// NOT block on per-session state locks (it trips a shared
    /// `CancellationToken` — preserves the v0.1.18 fast-path).
    fn cancel(
        &self,
        session: &ConnectorSessionId,
    ) -> BoxFuture<'_, Result<(), ConnectorError>>;

    /// Tear down a session.
    fn close_session(
        &self,
        session: ConnectorSessionId,
    ) -> BoxFuture<'_, Result<(), ConnectorError>>;

    /// Protocol-level capabilities (meaningful for any connector).
    fn capabilities(&self, session: Option<&ConnectorSessionId>) -> ConnectorCapabilities;

    /// Local-LLM-specific diagnostics, if this is a local-LLM connector. Default
    /// `None` keeps the trait generic for Claude/Codex/Antigravity connectors.
    fn as_local_llm(&self, _session: Option<&ConnectorSessionId>) -> Option<LocalLlmDiagnostics> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;

    /// Minimal connector proving (a) object-safety (`Box<dyn ..>`), (b) that the
    /// `'static` stream can be returned from a spawned-friendly source, and
    /// (c) the default `as_local_llm` returns `None`.
    struct StubConnector;

    impl AgentRuntimeConnector for StubConnector {
        fn start_session(
            &self,
            _params: StartSessionParams,
        ) -> BoxFuture<'_, Result<ConnectorSessionId, ConnectorError>> {
            Box::pin(async { Ok(ConnectorSessionId::new("stub")) })
        }
        fn set_config(
            &self,
            _session: &ConnectorSessionId,
            _config: ConnectorConfig,
        ) -> BoxFuture<'_, Result<(), ConnectorError>> {
            Box::pin(async { Ok(()) })
        }
        fn prompt(
            &self,
            _session: ConnectorSessionId,
            _prompt: ConnectorPrompt,
        ) -> (SubmissionHandle, BoxStream<'static, ConnectorEvent>) {
            let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
            let ev = ConnectorEvent::TurnFinished {
                env: Default::default(),
                stop_reason: super::super::event::ConnectorStopReason::EndTurn,
                error_kind: None,
            };
            let stream = futures_util::stream::once(async move { ev }).boxed();
            (SubmissionHandle::new(tx), stream)
        }
        fn deliver_response(
            &self,
            _session: &ConnectorSessionId,
            _request_id: String,
            _result: Result<serde_json::Value, ConnectorError>,
        ) {
        }
        fn cancel(
            &self,
            _session: &ConnectorSessionId,
        ) -> BoxFuture<'_, Result<(), ConnectorError>> {
            Box::pin(async { Ok(()) })
        }
        fn close_session(
            &self,
            _session: ConnectorSessionId,
        ) -> BoxFuture<'_, Result<(), ConnectorError>> {
            Box::pin(async { Ok(()) })
        }
        fn capabilities(&self, _session: Option<&ConnectorSessionId>) -> ConnectorCapabilities {
            ConnectorCapabilities {
                supports_cancellation: true,
                ..Default::default()
            }
        }
    }

    #[tokio::test]
    async fn trait_is_object_safe_and_stream_is_static() {
        let c: Box<dyn AgentRuntimeConnector> = Box::new(StubConnector);
        assert!(c.capabilities(None).supports_cancellation);
        assert!(c.as_local_llm(None).is_none());

        let (_handle, mut stream) = c.prompt(
            ConnectorSessionId::new("s"),
            ConnectorPrompt::default(),
        );
        // The 'static stream can be moved into a spawned task.
        let first = tokio::spawn(async move { stream.next().await })
            .await
            .unwrap();
        assert!(matches!(first, Some(ConnectorEvent::TurnFinished { .. })));
    }
}
