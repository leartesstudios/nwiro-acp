use thiserror::Error;

#[derive(Debug, Error)]
pub enum ShimError {
    #[error("ACP framing error: {0}")]
    AcpFraming(String),
    /// `session/prompt` referenced a sessionId this process has no state for.
    /// Split out of `AcpFraming` (the wire message text stays byte-identical —
    /// the `#[error]` string preserves the "ACP framing error:" prefix) so the
    /// dispatcher can attach structured `error.data`
    /// (`reason: "unknown_session"` + the offending id) without string-sniffing
    /// the message. Carries the sessionId the client sent.
    #[error("ACP framing error: unknown session: {0}")]
    UnknownSession(String),
    #[error("OpenAI HTTP error: {0}")]
    OpenAiHttp(String),
    #[error("Configuration error: {0}")]
    Config(String),
    #[error("MCP round-trip error: {0}")]
    McpRoundtrip(String),
    #[error("Cancelled by client")]
    Cancelled,
}

pub type Result<T> = std::result::Result<T, ShimError>;
