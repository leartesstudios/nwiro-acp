//! Connector session value types (Wave 1 W1-02).
//!
//! Phase A: defined but not yet wired (W1-09 connects the ACP server to the
//! connector). `#![allow(dead_code)]` keeps the build clean until then; each
//! type is exercised by unit tests so the shapes can't rot.
#![allow(dead_code)]

/// Opaque handle to a connector session — the ACP session id, newtyped so the
/// connector API can't be passed a bare string by accident.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConnectorSessionId(String);

impl ConnectorSessionId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ConnectorSessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Per-session configuration update (currently the model; future: mode).
/// Maps from ACP `session/set_config_option`.
#[derive(Debug, Clone, Default)]
pub struct ConnectorConfig {
    pub model: Option<String>,
    /// Tool-capability tier resolved from `session/warmup`. The ACP dispatcher
    /// computes this (matching the new model against the warmed model, exactly
    /// like the legacy `handle_set_config_option`) and passes it so the
    /// connector session's tier mirrors the legacy path. Without it the session
    /// stays `ToolTier::None` and the shared bridge strips `tools` from every
    /// request — a warmed Native model could never emit a tool call.
    pub tool_tier: Option<crate::acp::messages::ToolTier>,
}

/// A user prompt handed to the connector. `text` is the concatenation of the
/// ACP `prompt[]` text content blocks; `tools` is the optional tool list the
/// ACP client forwarded.
#[derive(Debug, Clone, Default)]
pub struct ConnectorPrompt {
    pub text: String,
    /// Image inputs extracted from the ACP prompt blocks (Phase 2 image input).
    /// Empty for text-only prompts; rendered to OpenAI `image_url` parts
    /// downstream when the target model is vision-capable.
    pub images: Vec<crate::acp::messages::ImageInput>,
    pub tools: Option<serde_json::Value>,
}

/// Client filesystem capabilities advertised in ACP `initialize`. Drives the
/// per-session `FsProvider` selection (Wave 3) — see `connector::fs`.
#[derive(Debug, Clone, Copy, Default)]
pub struct ClientFsCapabilities {
    pub read_text_file: bool,
    pub write_text_file: bool,
}

/// Parameters for [`AgentRuntimeConnector::start_session`]. Carries the
/// `initialize`-time capabilities the per-session `FsProvider` needs, so the
/// provider is chosen AFTER capabilities arrive (not at connector construction).
#[derive(Debug, Clone, Default)]
pub struct StartSessionParams {
    pub session_id: Option<String>,
    pub client_capabilities: ClientFsCapabilities,
    pub system_prompt: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_id_roundtrips() {
        let id = ConnectorSessionId::new("abc-123");
        assert_eq!(id.as_str(), "abc-123");
        assert_eq!(id.to_string(), "abc-123");
        assert_eq!(id, ConnectorSessionId::new("abc-123"));
    }

    #[test]
    fn config_and_prompt_defaults() {
        assert!(ConnectorConfig::default().model.is_none());
        let p = ConnectorPrompt::default();
        assert!(p.text.is_empty());
        assert!(p.tools.is_none());
    }

    #[test]
    fn start_params_defaults_conservative() {
        let p = StartSessionParams::default();
        assert!(!p.client_capabilities.read_text_file);
        assert!(!p.client_capabilities.write_text_file);
    }
}
