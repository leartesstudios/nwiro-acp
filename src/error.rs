use thiserror::Error;

#[derive(Debug, Error)]
pub enum ShimError {
    #[error("ACP framing error: {0}")]
    AcpFraming(String),
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
