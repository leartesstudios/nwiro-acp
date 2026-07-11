//! Connector error taxonomy (Wave 1 W1-03).
//!
//! Replaces the flat 5-variant `ShimError` at the connector boundary with a
//! richer, ACP-mappable taxonomy. Phase A: defined, not yet wired.
//!
//! REQUIRED derives: `Serialize/Deserialize/Clone/Debug`. `ConnectorError` is
//! embedded in `ConnectorEvent::Failed`, and that enum derives the same — the
//! bound propagates structurally, so every variant payload MUST be
//! `Serialize + Deserialize + Clone`. We therefore use only `String` /
//! `serde_json::Value` payloads — NEVER `std::io::Error` or `anyhow::Error`
//! (neither is `Serialize`/`Clone`). `From<ShimError>` stringifies any
//! non-serializable source into a `String` payload.
#![allow(dead_code)]

use crate::error::ShimError;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
// Externally tagged (the default): internal tagging can't represent tuple
// variants like `Transport(String)`. This serde form is internal (test/debug);
// the ACP error response is built by the translator from `to_acp_jsonrpc_code`
// + `Display`, not by serializing this enum.
#[serde(rename_all = "snake_case")]
pub enum ConnectorError {
    /// HTTP/transport failure reaching the model backend.
    Transport(String),
    /// Backend returned a non-success status / rejected the request.
    ProviderRejected(String),
    /// SSE stream malformed or an error envelope arrived mid-stream.
    MalformedStream(String),
    /// The conversation/prompt exceeded the model's context window.
    ContextOverflow,
    /// The configured model is not available on the backend.
    ModelUnavailable(String),
    /// The shim->bridge MCP transport failed (round-trip error/timeout).
    ToolTransport(String),
    /// The tool ran but returned an in-band error envelope (`isError: true`).
    ToolInBandError(serde_json::Value),
    /// A guarded operation was denied (no client capability / sandbox).
    PermissionDenied,
    /// User-initiated cancellation. NOT a JSON-RPC error — maps to
    /// `stopReason: "cancelled"` on the prompt response.
    Cancelled,
    /// Malformed request parameters.
    InvalidParams(String),
    /// Internal runtime bug / unexpected state.
    Internal(String),
}

impl ConnectorError {
    /// JSON-RPC error code for ACP error responses. `Cancelled` is intentionally
    /// absent — callers must route it to `stopReason: "cancelled"`, not an error.
    pub fn to_acp_jsonrpc_code(&self) -> i64 {
        match self {
            ConnectorError::InvalidParams(_) => -32602,
            // Everything else is a server-side runtime failure.
            _ => -32000,
        }
    }

    /// True when this should surface as a turn-completion `stopReason`, not a
    /// JSON-RPC error.
    pub fn is_cancellation(&self) -> bool {
        matches!(self, ConnectorError::Cancelled)
    }
}

impl std::fmt::Display for ConnectorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConnectorError::Transport(m) => write!(f, "transport error: {m}"),
            ConnectorError::ProviderRejected(m) => write!(f, "provider rejected: {m}"),
            ConnectorError::MalformedStream(m) => write!(f, "malformed stream: {m}"),
            ConnectorError::ContextOverflow => write!(f, "context overflow"),
            ConnectorError::ModelUnavailable(m) => write!(f, "model unavailable: {m}"),
            ConnectorError::ToolTransport(m) => write!(f, "tool transport error: {m}"),
            ConnectorError::ToolInBandError(v) => write!(f, "tool returned an error: {v}"),
            ConnectorError::PermissionDenied => write!(f, "permission denied"),
            ConnectorError::Cancelled => write!(f, "cancelled"),
            ConnectorError::InvalidParams(m) => write!(f, "invalid params: {m}"),
            ConnectorError::Internal(m) => write!(f, "internal error: {m}"),
        }
    }
}

impl std::error::Error for ConnectorError {}

/// Bridge the existing flat `ShimError` into the connector taxonomy so code
/// inside `bridge::*` keeps compiling while the connector emits the richer form.
impl From<ShimError> for ConnectorError {
    fn from(e: ShimError) -> Self {
        match e {
            ShimError::AcpFraming(m) => ConnectorError::Internal(m),
            // Same taxonomy the connector's own session lookup uses
            // (`InvalidParams("unknown session")` in local_openai.rs).
            ShimError::UnknownSession(sid) => {
                ConnectorError::InvalidParams(format!("unknown session: {sid}"))
            }
            ShimError::OpenAiHttp(m) => ConnectorError::Transport(m),
            ShimError::Config(m) => ConnectorError::InvalidParams(m),
            ShimError::McpRoundtrip(m) => ConnectorError::ToolTransport(m),
            ShimError::Cancelled => ConnectorError::Cancelled,
        }
    }
}

impl From<super::path::RelativePathError> for ConnectorError {
    fn from(e: super::path::RelativePathError) -> Self {
        ConnectorError::InvalidParams(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_params_maps_to_minus_32602() {
        assert_eq!(
            ConnectorError::InvalidParams("x".into()).to_acp_jsonrpc_code(),
            -32602
        );
    }

    #[test]
    fn transport_maps_to_minus_32000() {
        assert_eq!(ConnectorError::Transport("x".into()).to_acp_jsonrpc_code(), -32000);
    }

    #[test]
    fn cancelled_is_a_cancellation_not_an_error() {
        assert!(ConnectorError::Cancelled.is_cancellation());
        assert!(!ConnectorError::Transport("x".into()).is_cancellation());
    }

    #[test]
    fn from_shim_error_preserves_category() {
        assert!(matches!(
            ConnectorError::from(ShimError::Cancelled),
            ConnectorError::Cancelled
        ));
        assert!(matches!(
            ConnectorError::from(ShimError::OpenAiHttp("boom".into())),
            ConnectorError::Transport(_)
        ));
        assert!(matches!(
            ConnectorError::from(ShimError::McpRoundtrip("boom".into())),
            ConnectorError::ToolTransport(_)
        ));
    }

    #[test]
    fn serde_roundtrip_holds_the_derive_chain() {
        // The whole point of the required derives: ConnectorError must serialize
        // so ConnectorEvent::Failed can too.
        let e = ConnectorError::ToolInBandError(serde_json::json!({"isError": true}));
        let s = serde_json::to_string(&e).unwrap();
        let back: ConnectorError = serde_json::from_str(&s).unwrap();
        assert!(matches!(back, ConnectorError::ToolInBandError(_)));
    }
}
