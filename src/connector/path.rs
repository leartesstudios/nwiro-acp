//! Absolute-path validation (W0-C).
//!
//! ACP requires every filesystem path — `fs/read_text_file`,
//! `fs/write_text_file`, `session/new.cwd`, `tool_call.locations[].path` — to
//! be ABSOLUTE (<https://agentclientprotocol.com/protocol/file-system>). This
//! module is the single validation point the connector will call before
//! forwarding any path to the host (Claude-style pass-through) or to MCP file
//! tools.
//!
//! Status: DEFINED, NOT YET WIRED. v0.1.32 ships the validator; Wave 3 wires
//! the call sites (`session/new` cwd + `FsProvider` read/write). Until a caller
//! exists, the items are `#[allow(dead_code)]` to keep `cargo build` clean —
//! the unit tests below exercise them so the logic can't silently rot.

/// Error returned by [`reject_relative_path`] when a path is not absolute.
///
/// Self-contained on purpose: the connector module's richer `ConnectorError`
/// taxonomy arrives in Wave 1, at which point a `From<RelativePathError>` impl
/// will wrap this into `ConnectorError::InvalidParams`. Keeping it standalone
/// now means v0.1.32 carries no forward-dependency on unwritten Wave-1 types.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelativePathError {
    /// The offending (relative) path, preserved for the error message and for
    /// the eventual `ConnectorError::InvalidParams` payload.
    pub path: String,
}

impl std::fmt::Display for RelativePathError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "path must be absolute, got relative path: {}", self.path)
    }
}

impl std::error::Error for RelativePathError {}

/// Return `true` if `p` is an absolute path in any of the forms an ACP client
/// might legitimately send, independent of the platform the shim is compiled
/// for. Accepts:
///   - POSIX-absolute:        `/etc/hosts`
///   - Windows drive-absolute: `C:\Users\x`, `C:/Users/x`
///   - UNC / leading-sep:      `\\server\share`, `//server/share`
/// Rejects drive-RELATIVE (`C:foo`, relative to the CWD on drive C) and
/// dot-relative (`./`, `../`, `foo`) paths.
///
/// We do NOT use `std::path::Path::is_absolute()` because it is
/// platform-conditional: on Windows it rejects `/abs`, on Unix it rejects
/// `C:\abs`. The shim cross-compiles to 6 targets and must accept both forms
/// on every one of them.
#[allow(dead_code)]
fn is_absolute_acp_path(p: &str) -> bool {
    let bytes = p.as_bytes();
    // Leading separator: POSIX absolute (`/…`) or UNC (`\\…` / `//…`).
    if matches!(bytes.first(), Some(b'/') | Some(b'\\')) {
        return true;
    }
    // Windows drive-absolute: `X:\…` or `X:/…`. A separator AFTER the colon is
    // required — `C:foo` (no separator) is drive-relative, not absolute.
    if bytes.len() >= 3
        && bytes[0].is_ascii_alphabetic()
        && bytes[1] == b':'
        && matches!(bytes[2], b'/' | b'\\')
    {
        return true;
    }
    false
}

/// Reject a non-absolute path. Returns `Ok(())` for absolute paths (see
/// [`is_absolute_acp_path`] for the accepted forms) and
/// `Err(RelativePathError)` otherwise.
///
/// DEFINED, NOT YET WIRED: Wave 3 calls this from the `session/new` cwd handler
/// and the `FsProvider` read/write paths.
#[allow(dead_code)]
pub fn reject_relative_path(path: &str) -> Result<(), RelativePathError> {
    if is_absolute_acp_path(path) {
        Ok(())
    } else {
        Err(RelativePathError {
            path: path.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_posix_absolute() {
        assert!(reject_relative_path("/abs").is_ok());
        assert!(reject_relative_path("/etc/hosts").is_ok());
    }

    #[test]
    fn accepts_windows_drive_absolute() {
        // `"C:\\abs"` in source is the path `C:\abs`.
        assert!(reject_relative_path("C:\\abs").is_ok());
        assert!(reject_relative_path("C:/abs").is_ok());
        assert!(reject_relative_path("d:\\Users\\x").is_ok());
    }

    #[test]
    fn accepts_unc_and_leading_separator() {
        // `"\\\\server\\share"` in source is `\\server\share`.
        assert!(reject_relative_path("\\\\server\\share").is_ok());
        assert!(reject_relative_path("//server/share").is_ok());
    }

    #[test]
    fn rejects_dot_relative_and_bare() {
        assert!(reject_relative_path("foo/bar").is_err());
        assert!(reject_relative_path("./foo").is_err());
        assert!(reject_relative_path("../foo").is_err());
        assert!(reject_relative_path("foo").is_err());
        assert!(reject_relative_path("").is_err());
    }

    #[test]
    fn rejects_drive_relative() {
        // `C:foo` is drive-RELATIVE (relative to the CWD on drive C), NOT
        // absolute — ACP requires absolute, so this must be rejected.
        assert!(reject_relative_path("C:foo").is_err());
    }

    #[test]
    fn error_carries_offending_path() {
        let err = reject_relative_path("rel/path").unwrap_err();
        assert_eq!(err.path, "rel/path");
        assert!(err.to_string().contains("rel/path"));
    }
}
