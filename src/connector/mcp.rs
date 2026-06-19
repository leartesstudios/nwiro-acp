//! MCP round-trip transport (Wave 1 W1-06).
//!
//! The explicit replacement for the anonymous `write_mcp_real` closure: a
//! `Fn(Value) -> Fut` generic cannot be boxed into the connector, so the seam
//! needs a named trait. Production (`AcpMcpTransport`, W1-08) wraps the existing
//! `pending_requests` correlation map; tests script responses.
#![allow(dead_code)]

use super::runtime::BoxFuture;
use tokio_util::sync::CancellationToken;

pub trait McpTransport: Send + Sync {
    /// Prepare a shim->bridge `mcp/*` round-trip WITHOUT writing it: allocate a
    /// correlation id, stamp it onto the frame, register the pending request,
    /// and return `(the stamped frame to be WRITTEN BY THE CALLER, a 'static
    /// future resolving to the response)`. The caller writes the frame through
    /// its own ordered writer — the connector emits it as
    /// `ConnectorEvent::OutboundFrame` so the ACP dispatcher writes it in order
    /// with session/update. THIS decoupling of write-from-correlate is what keeps
    /// `mcp/*` frames from racing the session/update stream (the two-writer
    /// reorder that made tool goldens flaky under load). Impls clone any `&self`
    /// state into the returned future (the `'static` bound).
    ///
    /// v0.1.37 (Finding C): `cancel` aborts the response-await early when the
    /// session is cancelled mid-round-trip, so a `session/cancel` does not wait
    /// the full MCP timeout. Impls select on `cancel.cancelled()` and, on cancel,
    /// drain their pending entry + return `mcp_cancelled_marker()`.
    fn prepare(
        &self,
        req: serde_json::Value,
        cancel: CancellationToken,
    ) -> (serde_json::Value, BoxFuture<'static, serde_json::Value>);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// Scripted transport: returns a fixed result regardless of the request.
    struct StubMcpTransport {
        result: serde_json::Value,
    }

    impl McpTransport for StubMcpTransport {
        fn prepare(
            &self,
            req: serde_json::Value,
            _cancel: CancellationToken,
        ) -> (serde_json::Value, BoxFuture<'static, serde_json::Value>) {
            // Clone into the future so it borrows nothing from &self -> 'static.
            let result = self.result.clone();
            (req, Box::pin(async move { result }))
        }
    }

    #[tokio::test]
    async fn arc_dyn_transport_returns_scripted_value() {
        let t: Arc<dyn McpTransport> = Arc::new(StubMcpTransport {
            result: serde_json::json!({"result": {"connectionId": "c"}}),
        });
        let (_frame, resp) = t.prepare(serde_json::json!({"method": "mcp/connect"}), CancellationToken::new());
        let resp = resp.await;
        assert_eq!(resp.pointer("/result/connectionId").unwrap(), "c");
    }
}
