//! Connector module — the agent-runtime boundary for the ACP shim.
//!
//! v0.1.32 seeds this module with `path` (W0-C absolute-path validation).
//! Wave 1 fills it out with the `AgentRuntimeConnector` trait, the
//! `ConnectorEvent` enum, the `ClientSender` / `McpTransport` / `FsProvider`
//! traits, and the `LocalOpenAiConnector` implementation.
//!
//! Why seed it now: `path::reject_relative_path` is a pure, dependency-free
//! validator that several Wave-3 call sites (`session/new` cwd, `fs/*`) will
//! share. Landing it in v0.1.32 keeps the eventual connector PR focused on the
//! runtime seam rather than on utilities.

pub mod path;

// Wave 1 W1-02..W1-07 — the AgentRuntimeConnector seam. Phase A: these compile
// alongside the existing bridge path and are wired in at W1-09.
pub mod capabilities;
pub mod error;
pub mod event;
pub mod fs;
pub mod local_openai;
pub mod mcp;
pub mod runtime;
pub mod sender;
pub mod session;
pub mod submission;
