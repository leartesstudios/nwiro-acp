//! Connector capability reporting (Wave 1 W1-04).
//!
//! Split into protocol-level `ConnectorCapabilities` (meaningful for ANY
//! connector) and local-LLM-specific `LocalLlmDiagnostics` (exposed only via
//! `AgentRuntimeConnector::as_local_llm`, default `None`) so the core trait
//! stays IDE-agnostic for future Claude/Codex/Antigravity connectors.
#![allow(dead_code)]

use crate::acp::messages::ToolTier;

/// Protocol-level capabilities — the questions an ACP host can ask of ANY
/// runtime, with no local-LLM concepts leaking in.
#[derive(Debug, Clone, Copy, Default)]
pub struct ConnectorCapabilities {
    pub supports_cancellation: bool,
    pub supports_parallel_tools: bool,
    pub supports_image_input: bool,
    pub supports_usage_accounting: bool,
    pub max_context_tokens: Option<u32>,
}

/// Local-LLM-specific signals. Quarantined here (behind `as_local_llm`) so a
/// Claude/Codex connector never has to defensively default `tool_tier` /
/// `schema_bleed_detected`.
#[derive(Debug, Clone)]
pub struct LocalLlmDiagnostics {
    pub tool_tier: ToolTier,
    pub schema_bleed_detected: bool,
    pub model_family: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_capabilities_are_conservative() {
        let c = ConnectorCapabilities::default();
        assert!(!c.supports_cancellation);
        assert!(!c.supports_parallel_tools);
        assert!(c.max_context_tokens.is_none());
    }

    #[test]
    fn local_diagnostics_carry_tier() {
        let d = LocalLlmDiagnostics {
            tool_tier: ToolTier::Native,
            schema_bleed_detected: false,
            model_family: "qwen".into(),
        };
        assert_eq!(d.tool_tier, ToolTier::Native);
    }
}
