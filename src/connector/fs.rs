//! Filesystem access abstraction (Wave 1 W1-07 — trait + selection only).
//!
//! The connector calls `FsProvider` instead of `std::fs::*` directly — the
//! invariant that makes the Wave 3 host/MCP split + sandbox/permission models
//! possible. Provider is chosen PER SESSION in `start_session()` AFTER the ACP
//! `initialize` capabilities arrive (NOT at connector construction, where caps
//! are still unknown). Bodies are `todo!("Wave 3")`; nothing routes here in
//! Wave 1, so the panics are loud guards.
#![allow(dead_code)]

use super::error::ConnectorError;
use super::runtime::BoxFuture;
use super::session::StartSessionParams;

pub trait FsProvider: Send + Sync {
    fn read_text_file(&self, path: &str) -> BoxFuture<'_, Result<String, ConnectorError>>;
    fn write_text_file(
        &self,
        path: &str,
        contents: &str,
    ) -> BoxFuture<'_, Result<(), ConnectorError>>;
    fn capabilities(&self) -> FsCapabilities;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FsCapabilities {
    pub can_read: bool,
    pub can_write: bool,
}

/// Claude-style host pass-through (PRIMARY). Wave 3: emits `fs/*` JSON-RPC
/// outbound to the ACP client via `ClientSender::send_request`.
pub struct HostFsProvider;

impl FsProvider for HostFsProvider {
    fn read_text_file(&self, _path: &str) -> BoxFuture<'_, Result<String, ConnectorError>> {
        Box::pin(async { todo!("Wave 3: emit fs/read_text_file outbound") })
    }
    fn write_text_file(
        &self,
        _path: &str,
        _contents: &str,
    ) -> BoxFuture<'_, Result<(), ConnectorError>> {
        Box::pin(async { todo!("Wave 3: emit fs/write_text_file outbound") })
    }
    fn capabilities(&self) -> FsCapabilities {
        FsCapabilities { can_read: true, can_write: true }
    }
}

/// MCP-tool fallback (SECONDARY). Wave 3: routes to MCP file tools, resolved
/// lazily on first call.
pub struct McpFsProvider;

impl FsProvider for McpFsProvider {
    fn read_text_file(&self, _path: &str) -> BoxFuture<'_, Result<String, ConnectorError>> {
        Box::pin(async { todo!("Wave 3: route to MCP _fs_read tool") })
    }
    fn write_text_file(
        &self,
        _path: &str,
        _contents: &str,
    ) -> BoxFuture<'_, Result<(), ConnectorError>> {
        Box::pin(async { todo!("Wave 3: route to MCP _fs_write tool") })
    }
    fn capabilities(&self) -> FsCapabilities {
        FsCapabilities { can_read: true, can_write: true }
    }
}

/// No filesystem access — every call is denied. Used when neither host fs caps
/// nor MCP file tools are available.
pub struct NoOpFsProvider;

impl FsProvider for NoOpFsProvider {
    fn read_text_file(&self, _path: &str) -> BoxFuture<'_, Result<String, ConnectorError>> {
        Box::pin(async { Err(ConnectorError::PermissionDenied) })
    }
    fn write_text_file(
        &self,
        _path: &str,
        _contents: &str,
    ) -> BoxFuture<'_, Result<(), ConnectorError>> {
        Box::pin(async { Err(ConnectorError::PermissionDenied) })
    }
    fn capabilities(&self) -> FsCapabilities {
        FsCapabilities::default()
    }
}

/// Per-session provider selection (W1-07). Host pass-through wins when the
/// client advertised fs capabilities in `initialize`; otherwise fall back to
/// MCP file tools (probed lazily in Wave 3). This runs inside `start_session()`
/// so it sees the negotiated capabilities, never the (None) construction-time
/// ones.
pub fn select_fs_provider(params: &StartSessionParams) -> Box<dyn FsProvider> {
    let fs = &params.client_capabilities;
    if fs.read_text_file || fs.write_text_file {
        Box::new(HostFsProvider)
    } else {
        Box::new(McpFsProvider)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connector::session::ClientFsCapabilities;

    #[test]
    fn host_provider_chosen_when_client_advertises_fs() {
        let params = StartSessionParams {
            client_capabilities: ClientFsCapabilities {
                read_text_file: true,
                write_text_file: false,
            },
            ..Default::default()
        };
        // HostFsProvider advertises read+write.
        assert_eq!(
            select_fs_provider(&params).capabilities(),
            FsCapabilities { can_read: true, can_write: true }
        );
    }

    #[test]
    fn mcp_fallback_when_no_fs_caps() {
        let params = StartSessionParams::default();
        // McpFsProvider also advertises read+write (resolved lazily in Wave 3).
        assert_eq!(
            select_fs_provider(&params).capabilities(),
            FsCapabilities { can_read: true, can_write: true }
        );
    }

    #[tokio::test]
    async fn noop_provider_denies() {
        let p = NoOpFsProvider;
        assert!(matches!(
            p.read_text_file("/abs").await,
            Err(ConnectorError::PermissionDenied)
        ));
        assert_eq!(p.capabilities(), FsCapabilities::default());
    }

    // NOTE: Host/Mcp read/write bodies are `todo!("Wave 3")` and are NOT called
    // here — only `capabilities()` and selection are exercised in Wave 1.
}
