//! Session persistence (targets v0.5.0): one JSON envelope per session so the
//! host bridge can resume a prior conversation across a shim restart via ACP
//! `session/load`.
//!
//! Storage model
//! -------------
//! One file per session at
//! `<cwd>/Saved/NwiroIntegrationKit/shim-sessions/<encoded-session-id>.json`,
//! where `cwd` is the absolute UE project directory the host supplies on
//! `session/new` (and again on `session/load`). The default location keeps the
//! data INSIDE the project the conversation is about.
//!
//! `NWIRO_SHIM_STATE_DIR` overrides the ROOT (it replaces
//! `<cwd>/Saved/NwiroIntegrationKit`; the `shim-sessions` leaf is always
//! appended). **Privacy warning:** persisted history can contain project file
//! contents and tool results — pointing the override at a shared or synced
//! directory moves that data outside the project. The default location exists
//! precisely to avoid that.
//!
//! `NWIRO_SHIM_PERSIST` is the kill switch (default ON; `0`/`false`/`off`
//! disables). Disabled means: `initialize` advertises `loadSession: false`,
//! nothing is ever written, and `session/load` answers `-32002` (the host
//! classifies that as resource-not-found and silently falls back to
//! `session/new`).
//!
//! Durability contract
//! -------------------
//! - The envelope is a VERSIONED contract (`schema_version`). Breaking its
//!   shape requires a version bump; older files must fail `session/load` with
//!   `-32002` — never a crash or a partial load. (AGENTS.md invariant.)
//! - Writes are atomic: same-directory temp file + rename, fsync best-effort.
//!   A write failure logs and never fails the turn.
//! - Only durable conversation state is persisted. `cancel_token`,
//!   `token_budget_warned`, and MCP state are deliberately NOT in the envelope
//!   (MCP reconnects per normal turn flow; non-durable flags re-default).
//! - Session ids are untrusted input: filenames are built from an allowlisted
//!   percent-encoding so a hostile id (`../../evil`) cannot escape the storage
//!   directory in either direction (write or read).
//!
//! Eviction: after successful writes and at first storage-dir use per process,
//! keep the newest [`MAX_SESSION_FILES`] session files (by modified time — the
//! file is rewritten on every update, so mtime IS `updated_at`) and delete
//! files older than [`MAX_SESSION_AGE`]. Leftover `*.tmp` files are dead by
//! construction (all persistence ops run serialized on the dispatcher task)
//! and are removed. Eviction errors log and never block.

use std::collections::HashSet;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::acp::messages::ToolTier;
use crate::bridge::SessionState;
use crate::openai::messages::ChatMessage;

/// Version of the on-disk envelope shape. Bump on ANY breaking change to
/// [`SessionEnvelope`]; `load` rejects files with a different version so the
/// host falls back to `session/new` instead of resuming from a misparse.
pub const SCHEMA_VERSION: u32 = 1;

/// Count cap per storage dir: keep the newest ~50 session files.
const MAX_SESSION_FILES: usize = 50;

/// Age cap: session files older than ~30 days are deleted.
const MAX_SESSION_AGE: Duration = Duration::from_secs(30 * 24 * 60 * 60);

/// Per-session persistence anchor, stored on `SessionState`. `None` means the
/// session is not persisted (kill switch off, no/invalid cwd on `session/new`,
/// or the connector path, which does not participate in persistence).
pub struct PersistHandle {
    /// Resolved storage directory (…/shim-sessions), validated at creation.
    pub dir: PathBuf,
    /// Unix seconds when the session was first created — preserved verbatim
    /// across every rewrite of the envelope (and across `session/load`).
    pub created_at: u64,
}

/// The on-disk session envelope — a VERSIONED contract (see module docs).
/// Unknown extra fields are tolerated on read (additive evolution is
/// non-breaking); missing/mistyped fields fail deserialization → `-32002`.
#[derive(Debug, Serialize, Deserialize)]
pub struct SessionEnvelope {
    pub schema_version: u32,
    pub session_id: String,
    pub current_model: String,
    pub tool_tier: ToolTier,
    pub history: Vec<ChatMessage>,
    pub learned_tool_ceiling: Option<usize>,
    pub pruned_turn_count: usize,
    /// Unix seconds (see [`now_unix`]).
    pub created_at: u64,
    pub updated_at: u64,
}

/// Current time as unix seconds. `0` on a pre-epoch clock (never fails).
pub fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Kill-switch parse: only an explicit `0` / `false` / `off` (case-insensitive,
/// trimmed) disables persistence. Absent or anything else = enabled.
fn flag_disables(value: Option<&str>) -> bool {
    matches!(
        value.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("0") | Some("false") | Some("off")
    )
}

/// Kill switch: `NWIRO_SHIM_PERSIST`, default ON.
pub fn persistence_enabled() -> bool {
    !flag_disables(std::env::var("NWIRO_SHIM_PERSIST").ok().as_deref())
}

/// What `initialize` advertises as `agentCapabilities.loadSession`. True only
/// when the kill switch is on AND, if the `NWIRO_SHIM_STATE_DIR` override is
/// set, it passes the same absolute-path validation the storage resolver
/// applies — a misconfigured override must not advertise a capability every
/// `session/load` would then fail.
pub fn persistence_available() -> bool {
    if !persistence_enabled() {
        return false;
    }
    match std::env::var("NWIRO_SHIM_STATE_DIR") {
        Ok(root) if !root.trim().is_empty() => Path::new(root.trim()).is_absolute(),
        _ => true,
    }
}

/// Resolve the storage directory for a host-supplied `cwd` (from `session/new`
/// or `session/load`). `None` disables persistence for that session: kill
/// switch off, relative/absent/nonexistent cwd, or a relative state-dir
/// override. Pure logic lives in [`resolve_storage_dir_with`] for testability.
pub fn resolve_storage_dir(cwd: Option<&str>) -> Option<PathBuf> {
    resolve_storage_dir_with(
        cwd,
        std::env::var("NWIRO_SHIM_STATE_DIR").ok().as_deref(),
        persistence_enabled(),
    )
}

fn resolve_storage_dir_with(
    cwd: Option<&str>,
    override_root: Option<&str>,
    enabled: bool,
) -> Option<PathBuf> {
    if !enabled {
        return None;
    }
    if let Some(root) = override_root.map(str::trim).filter(|s| !s.is_empty()) {
        let root = Path::new(root);
        if !root.is_absolute() {
            tracing::warn!(
                root = %root.display(),
                "NWIRO_SHIM_STATE_DIR is not an absolute path — session persistence disabled"
            );
            return None;
        }
        return Some(root.join("shim-sessions"));
    }
    let cwd = cwd.map(str::trim).filter(|s| !s.is_empty())?;
    let cwd_path = Path::new(cwd);
    if !cwd_path.is_absolute() || !cwd_path.is_dir() {
        tracing::warn!(
            cwd = %cwd_path.display(),
            "session cwd is not an absolute existing directory — session persistence \
             disabled for this session"
        );
        return None;
    }
    Some(
        cwd_path
            .join("Saved")
            .join("NwiroIntegrationKit")
            .join("shim-sessions"),
    )
}

/// Encode an untrusted session id into a safe single-component filename stem:
/// `[A-Za-z0-9_-]` pass through; every other byte becomes `%XX` (uppercase
/// hex). Injective (`%` itself is encoded), so distinct ids never collide, and
/// the output can contain no path separators or dots — a hostile id like
/// `../../evil` cannot escape the storage dir. Empty ids are rejected.
pub fn encode_session_id(id: &str) -> Option<String> {
    if id.is_empty() {
        return None;
    }
    let mut out = String::with_capacity(id.len());
    for b in id.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'_' | b'-' => out.push(b as char),
            other => out.push_str(&format!("%{other:02X}")),
        }
    }
    Some(out)
}

/// Full path of the envelope file for `session_id` inside `dir`.
/// `None` for an unencodable (empty) id.
pub fn session_file(dir: &Path, session_id: &str) -> Option<PathBuf> {
    Some(dir.join(format!("{}.json", encode_session_id(session_id)?)))
}

/// Once-per-process-per-dir storage init: create the directory and run one
/// eviction pass (which also removes stale `*.tmp` leftovers from a previous
/// crashed process). Best-effort — failures log and never block.
pub fn init_storage_dir(dir: &Path) {
    static INITIALIZED: OnceLock<Mutex<HashSet<PathBuf>>> = OnceLock::new();
    let set = INITIALIZED.get_or_init(|| Mutex::new(HashSet::new()));
    match set.lock() {
        Ok(mut guard) => {
            if !guard.insert(dir.to_path_buf()) {
                return; // already initialized this process
            }
        }
        Err(_) => {
            tracing::warn!("persist init set poisoned — skipping first-use eviction");
            return;
        }
    }
    if let Err(e) = std::fs::create_dir_all(dir) {
        tracing::warn!(dir = %dir.display(), error = %e, "could not create session storage dir");
        return;
    }
    evict(dir);
}

/// Atomic envelope write: serialize to a same-directory `*.json.tmp`, fsync
/// best-effort, then rename over the target. The rename is what makes a
/// concurrent crash leave either the OLD file or the NEW file — never a
/// truncated hybrid. Errors are returned; the caller logs and continues (a
/// persistence failure must never fail the turn).
pub fn save(dir: &Path, envelope: &SessionEnvelope) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let file = session_file(dir, &envelope.session_id).ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "empty session id")
    })?;
    let tmp = file.with_extension("json.tmp");
    let bytes = serde_json::to_vec(envelope)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(&bytes)?;
        // fsync is best-effort by design: a lost-on-power-cut envelope is an
        // acceptable outcome (the host falls back to session/new); a failed
        // sync must not fail the write.
        let _ = f.sync_all();
    }
    // std::fs::rename replaces an existing destination on both Unix and
    // Windows (MOVEFILE_REPLACE_EXISTING), so updates are in-place-atomic.
    std::fs::rename(&tmp, &file)?;
    Ok(())
}

/// Read + validate the envelope for `session_id` from `dir`. Every anomaly is
/// an `Err(reason)` the caller maps to `-32002` (the reason string is for the
/// shim log only, never the wire). Reads ONLY the `*.json` target — never a
/// `*.tmp` (an in-progress or abandoned write is invisible to load).
pub fn load(dir: &Path, session_id: &str) -> Result<SessionEnvelope, String> {
    let file =
        session_file(dir, session_id).ok_or_else(|| "invalid (empty) session id".to_string())?;
    let bytes =
        std::fs::read(&file).map_err(|e| format!("read {} failed: {e}", file.display()))?;
    let envelope: SessionEnvelope = serde_json::from_slice(&bytes)
        .map_err(|e| format!("corrupt or incompatible envelope: {e}"))?;
    if envelope.schema_version != SCHEMA_VERSION {
        return Err(format!(
            "unsupported schema_version {} (this shim speaks {SCHEMA_VERSION})",
            envelope.schema_version
        ));
    }
    if envelope.session_id != session_id {
        return Err(format!(
            "envelope session_id {:?} does not match requested id",
            envelope.session_id
        ));
    }
    Ok(envelope)
}

/// Build the durable envelope from live session state. `None` when the session
/// has no persistence anchor. `updated_at` is stamped now; `created_at` is
/// carried from the handle so it survives every rewrite.
pub fn envelope_from_state(state: &SessionState) -> Option<SessionEnvelope> {
    let handle = state.persist.as_ref()?;
    Some(SessionEnvelope {
        schema_version: SCHEMA_VERSION,
        session_id: state.session_id.clone(),
        current_model: state.current_model.clone(),
        tool_tier: state.tool_tier,
        history: state.history.clone(),
        learned_tool_ceiling: state.learned_tool_ceiling,
        pruned_turn_count: state.pruned_turn_count,
        created_at: handle.created_at,
        updated_at: now_unix(),
    })
}

/// Rebuild a live `SessionState` from a loaded envelope: fresh (untripped)
/// `CancellationToken`, non-durable flags re-default (`token_budget_warned`),
/// and a persistence handle pointing back at `dir` so subsequent turns keep
/// writing through. MCP state is not restored — it reconnects per normal turn
/// flow.
pub fn state_from_envelope(envelope: SessionEnvelope, dir: PathBuf) -> SessionState {
    SessionState {
        session_id: envelope.session_id,
        current_model: envelope.current_model,
        history: envelope.history,
        cancel_token: tokio_util::sync::CancellationToken::new(),
        tool_tier: envelope.tool_tier,
        token_budget_warned: false,
        pruned_turn_count: envelope.pruned_turn_count,
        learned_tool_ceiling: envelope.learned_tool_ceiling,
        persist: Some(PersistHandle {
            dir,
            created_at: envelope.created_at,
        }),
    }
}

/// Best-effort write-through for a session (turn end + config changes). A
/// no-op for sessions without a persistence handle. A write failure logs and
/// never propagates; a successful write runs an eviction pass.
pub fn save_session_state(state: &SessionState) {
    let Some(handle) = state.persist.as_ref() else {
        return;
    };
    let Some(envelope) = envelope_from_state(state) else {
        return;
    };
    match save(&handle.dir, &envelope) {
        Ok(()) => evict(&handle.dir),
        Err(e) => tracing::warn!(
            session_id = %state.session_id,
            dir = %handle.dir.display(),
            error = %e,
            "session persistence write failed — the turn is unaffected"
        ),
    }
}

/// Eviction pass with the production caps. See [`evict_with`].
pub fn evict(dir: &Path) {
    evict_with(dir, MAX_SESSION_FILES, MAX_SESSION_AGE);
}

/// Keep the newest `keep` session files (by mtime — the file is rewritten on
/// every update, so mtime tracks `updated_at`), delete files older than
/// `max_age`, and remove leftover `*.tmp` files (dead by construction: all
/// persistence ops are serialized on the dispatcher task, so no write is in
/// flight while eviction runs). Every error logs and is skipped — eviction
/// never blocks or fails the caller.
fn evict_with(dir: &Path, keep: usize, max_age: Duration) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return, // dir missing — nothing to evict
    };
    let now = SystemTime::now();
    let mut sessions: Vec<(PathBuf, SystemTime)> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };
        if name.ends_with(".tmp") {
            if let Err(e) = std::fs::remove_file(&path) {
                tracing::warn!(file = %path.display(), error = %e, "could not remove stale .tmp");
            }
            continue;
        }
        if !name.ends_with(".json") {
            continue; // foreign file — leave it alone
        }
        let meta = match entry.metadata() {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!(file = %path.display(), error = %e, "unreadable entry — skipped");
                continue;
            }
        };
        if !meta.is_file() {
            tracing::warn!(
                file = %path.display(),
                "non-file garbage in session storage dir — skipped"
            );
            continue;
        }
        let mtime = meta.modified().unwrap_or(now);
        sessions.push((path, mtime));
    }

    let delete = |path: &Path, why: &str| {
        if let Err(e) = std::fs::remove_file(path) {
            tracing::warn!(file = %path.display(), error = %e, why, "eviction delete failed — skipped");
        } else {
            tracing::debug!(file = %path.display(), why, "evicted session file");
        }
    };

    // Age cap first: anything older than max_age goes regardless of count.
    sessions.retain(|(path, mtime)| {
        let age = now.duration_since(*mtime).unwrap_or(Duration::ZERO);
        if age > max_age {
            delete(path, "age cap");
            false
        } else {
            true
        }
    });

    // Count cap: newest `keep` survive.
    if sessions.len() > keep {
        sessions.sort_by(|a, b| b.1.cmp(&a.1)); // newest first
        for (path, _) in sessions.iter().skip(keep) {
            delete(path, "count cap");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::messages::{ToolCall, ToolCallFunction};

    /// Fresh unique temp dir for one test. Not auto-cleaned on panic — fine
    /// for a test scratch area under the OS temp root.
    fn tmpdir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir()
            .join("nwiro-shim-persist-tests")
            .join(format!("{tag}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("create test tmp dir");
        dir
    }

    fn envelope(session_id: &str) -> SessionEnvelope {
        SessionEnvelope {
            schema_version: SCHEMA_VERSION,
            session_id: session_id.to_string(),
            current_model: "test-model".to_string(),
            tool_tier: ToolTier::None,
            history: vec![ChatMessage::user("hi")],
            learned_tool_ceiling: None,
            pruned_turn_count: 0,
            created_at: 100,
            updated_at: 200,
        }
    }

    // ── kill switch parse ─────────────────────────────────────────────────

    #[test]
    fn kill_switch_default_on_and_only_explicit_values_disable() {
        assert!(!flag_disables(None), "absent = enabled");
        assert!(!flag_disables(Some("1")));
        assert!(!flag_disables(Some("yes")));
        for off in ["0", "false", "off", " OFF ", "False"] {
            assert!(flag_disables(Some(off)), "{off:?} must disable");
        }
    }

    // ── storage-dir resolution ────────────────────────────────────────────

    #[test]
    fn resolve_requires_absolute_existing_cwd() {
        let dir = tmpdir("resolve");
        // Happy path: absolute existing cwd → <cwd>/Saved/NwiroIntegrationKit/shim-sessions.
        let resolved =
            resolve_storage_dir_with(Some(dir.to_str().unwrap()), None, true).expect("resolves");
        assert_eq!(
            resolved,
            dir.join("Saved").join("NwiroIntegrationKit").join("shim-sessions")
        );
        // Relative cwd → None.
        assert!(resolve_storage_dir_with(Some("relative/path"), None, true).is_none());
        // Nonexistent cwd → None.
        let ghost = dir.join("does-not-exist");
        assert!(resolve_storage_dir_with(Some(ghost.to_str().unwrap()), None, true).is_none());
        // Absent / empty cwd → None.
        assert!(resolve_storage_dir_with(None, None, true).is_none());
        assert!(resolve_storage_dir_with(Some("  "), None, true).is_none());
        // Kill switch off → None even with a valid cwd.
        assert!(resolve_storage_dir_with(Some(dir.to_str().unwrap()), None, false).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resolve_state_dir_override_replaces_the_root() {
        let dir = tmpdir("override");
        let root = dir.to_str().unwrap();
        // Absolute override replaces <cwd>/Saved/NwiroIntegrationKit entirely
        // (cwd not even required).
        let resolved = resolve_storage_dir_with(None, Some(root), true).expect("resolves");
        assert_eq!(resolved, dir.join("shim-sessions"));
        // Relative override is rejected (path-validation failure → disabled).
        assert!(resolve_storage_dir_with(None, Some("rel/state"), true).is_none());
        // Empty override falls back to the cwd rule.
        assert!(resolve_storage_dir_with(None, Some("  "), true).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── session-id encoding (hostile-input safety) ───────────────────────

    #[test]
    fn encode_passes_allowlist_and_escapes_everything_else() {
        assert_eq!(encode_session_id("abc-DEF_09").as_deref(), Some("abc-DEF_09"));
        // '%' itself is escaped → the encoding is injective.
        assert_eq!(encode_session_id("%").as_deref(), Some("%25"));
        // No dots or separators survive.
        let enc = encode_session_id("../../evil").unwrap();
        assert!(!enc.contains('.') && !enc.contains('/') && !enc.contains('\\'), "{enc}");
        assert_eq!(enc, "%2E%2E%2F%2E%2E%2Fevil");
        // Empty is rejected.
        assert!(encode_session_id("").is_none());
    }

    #[test]
    fn hostile_session_id_cannot_write_outside_the_storage_dir() {
        let dir = tmpdir("hostile-write");
        let store = dir.join("store");
        let env = envelope("../../evil");
        save(&store, &env).expect("save succeeds inside the store");
        // Exactly one file, INSIDE the store, single path component.
        let entries: Vec<_> = std::fs::read_dir(&store)
            .unwrap()
            .flatten()
            .map(|e| e.path())
            .collect();
        assert_eq!(entries.len(), 1, "one envelope file expected: {entries:?}");
        assert_eq!(entries[0].parent(), Some(store.as_path()));
        // Nothing escaped upward.
        assert!(!dir.join("evil.json").exists());
        assert!(!dir.parent().unwrap().join("evil.json").exists());
        // And it round-trips under the same hostile id.
        assert!(load(&store, "../../evil").is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn hostile_session_id_cannot_read_outside_the_storage_dir() {
        let dir = tmpdir("hostile-read");
        let store = dir.join("store");
        std::fs::create_dir_all(&store).unwrap();
        // Plant a VALID envelope OUTSIDE the store at the exact path a naive
        // `<dir>/../evil.json` join would hit.
        let outside = envelope("../evil");
        let planted = dir.join("evil.json");
        std::fs::write(&planted, serde_json::to_vec(&outside).unwrap()).unwrap();
        // The load must NOT find it: the encoded filename has no separators.
        assert!(load(&store, "../evil").is_err(), "must not read outside the store");
        assert!(planted.exists(), "the planted file must be untouched");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── round-trip: save → load → SessionState ───────────────────────────

    #[test]
    fn round_trip_save_load_rebuilds_session_state() {
        let dir = tmpdir("roundtrip");
        let state = SessionState {
            session_id: "sess-1".to_string(),
            current_model: "qwen3:14b".to_string(),
            history: vec![
                ChatMessage::system("you are helpful"),
                ChatMessage::user("spawn a light"),
                ChatMessage::assistant(
                    Some("calling".to_string()),
                    Some(vec![ToolCall {
                        id: "call_1".to_string(),
                        r#type: "function".to_string(),
                        function: ToolCallFunction {
                            name: "spawn_actor".to_string(),
                            arguments: "{\"kind\":\"PointLight\"}".to_string(),
                        },
                    }]),
                ),
                ChatMessage::tool(
                    "call_1",
                    serde_json::json!({"content":[{"type":"text","text":"ok"}],"isError":false}),
                ),
                ChatMessage::assistant(Some("done".to_string()), None),
            ],
            cancel_token: tokio_util::sync::CancellationToken::new(),
            tool_tier: ToolTier::Native,
            token_budget_warned: true, // non-durable — must NOT survive
            pruned_turn_count: 3,
            learned_tool_ceiling: Some(7),
            persist: Some(PersistHandle {
                dir: dir.clone(),
                created_at: 111,
            }),
        };

        let envelope = envelope_from_state(&state).expect("state has a persist handle");
        assert_eq!(envelope.schema_version, SCHEMA_VERSION);
        assert_eq!(envelope.created_at, 111, "created_at carried from the handle");
        save(&dir, &envelope).expect("save");

        let loaded = load(&dir, "sess-1").expect("load");
        let restored = state_from_envelope(loaded, dir.clone());
        assert_eq!(restored.session_id, "sess-1");
        assert_eq!(restored.current_model, "qwen3:14b");
        assert_eq!(restored.tool_tier, ToolTier::Native);
        assert_eq!(restored.pruned_turn_count, 3);
        assert_eq!(restored.learned_tool_ceiling, Some(7));
        // Non-durable state re-defaults.
        assert!(!restored.token_budget_warned);
        assert!(!restored.cancel_token.is_cancelled(), "fresh token");
        // History is byte-identical on the OpenAI wire (serde round-trip).
        assert_eq!(
            serde_json::to_value(&restored.history).unwrap(),
            serde_json::to_value(&state.history).unwrap()
        );
        // The handle points back at the same dir with the original created_at.
        let handle = restored.persist.as_ref().expect("restored handle");
        assert_eq!(handle.dir, dir);
        assert_eq!(handle.created_at, 111);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── -32002 anomaly matrix (storage layer half) ────────────────────────

    #[test]
    fn load_rejects_missing_corrupt_wrong_version_and_mismatched_id() {
        let dir = tmpdir("anomalies");
        std::fs::create_dir_all(&dir).unwrap();
        // Missing file.
        assert!(load(&dir, "ghost").is_err());
        // Corrupt JSON.
        std::fs::write(session_file(&dir, "corrupt").unwrap(), b"{ not json").unwrap();
        assert!(load(&dir, "corrupt").is_err());
        // Wrong schema_version.
        let mut v2 = serde_json::to_value(envelope("versioned")).unwrap();
        v2["schema_version"] = serde_json::json!(SCHEMA_VERSION + 1);
        std::fs::write(
            session_file(&dir, "versioned").unwrap(),
            serde_json::to_vec(&v2).unwrap(),
        )
        .unwrap();
        let err = load(&dir, "versioned").unwrap_err();
        assert!(err.contains("schema_version"), "{err}");
        // Envelope id != requested id (file renamed / copied across sessions).
        save(&dir, &envelope("real-id")).unwrap();
        std::fs::rename(
            session_file(&dir, "real-id").unwrap(),
            session_file(&dir, "stolen-id").unwrap(),
        )
        .unwrap();
        let err = load(&dir, "stolen-id").unwrap_err();
        assert!(err.contains("does not match"), "{err}");
        // Empty id.
        assert!(load(&dir, "").is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── atomicity: tmp files are invisible to load and get cleaned ────────

    #[test]
    fn leftover_tmp_is_never_loaded_and_is_cleaned_by_eviction() {
        let dir = tmpdir("tmpfiles");
        std::fs::create_dir_all(&dir).unwrap();
        // A VALID envelope that only ever made it to the tmp name (crashed
        // mid-write): load must not see it.
        let tmp = session_file(&dir, "half-written").unwrap().with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec(&envelope("half-written")).unwrap()).unwrap();
        assert!(load(&dir, "half-written").is_err(), "a load never reads a .tmp");
        // Eviction removes the stale tmp and leaves real session files alone.
        save(&dir, &envelope("alive")).unwrap();
        evict(&dir);
        assert!(!tmp.exists(), "stale .tmp must be cleaned");
        assert!(session_file(&dir, "alive").unwrap().exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_leaves_no_tmp_behind() {
        let dir = tmpdir("no-tmp");
        save(&dir, &envelope("s")).unwrap();
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "tmp must be renamed away: {leftovers:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── eviction caps ─────────────────────────────────────────────────────

    #[test]
    fn eviction_count_cap_keeps_the_newest() {
        let dir = tmpdir("count-cap");
        for i in 0..4 {
            save(&dir, &envelope(&format!("s{i}"))).unwrap();
            // Distinct mtimes so "newest" is well-defined.
            std::thread::sleep(Duration::from_millis(30));
        }
        evict_with(&dir, 2, Duration::from_secs(3600));
        let survivors: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .filter_map(|e| e.file_name().to_str().map(String::from))
            .collect();
        assert_eq!(survivors.len(), 2, "keep=2 of 4: {survivors:?}");
        assert!(survivors.contains(&"s2.json".to_string()), "{survivors:?}");
        assert!(survivors.contains(&"s3.json".to_string()), "{survivors:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn eviction_age_cap_deletes_old_files() {
        let dir = tmpdir("age-cap");
        save(&dir, &envelope("old")).unwrap();
        std::thread::sleep(Duration::from_millis(30));
        // Everything older than 1ms is "old" by now.
        evict_with(&dir, 100, Duration::from_millis(1));
        assert!(
            !session_file(&dir, "old").unwrap().exists(),
            "age cap must delete files older than max_age"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn eviction_skips_non_evictable_garbage_and_still_processes_the_rest() {
        let dir = tmpdir("garbage");
        std::fs::create_dir_all(&dir).unwrap();
        // A DIRECTORY named like a session file: remove_file can't delete it —
        // it must be logged + skipped, never a panic or an abort of the pass.
        let junk = dir.join("junk.json");
        std::fs::create_dir_all(&junk).unwrap();
        // A foreign non-json file: left alone.
        std::fs::write(dir.join("README.txt"), b"keep me").unwrap();
        save(&dir, &envelope("old")).unwrap();
        std::thread::sleep(Duration::from_millis(30));
        evict_with(&dir, 100, Duration::from_millis(1));
        assert!(junk.exists(), "garbage dir must be skipped, not deleted");
        assert!(dir.join("README.txt").exists(), "foreign files are untouched");
        assert!(
            !session_file(&dir, "old").unwrap().exists(),
            "the real old session file must still be evicted despite the garbage"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── best-effort save via SessionState ────────────────────────────────

    #[test]
    fn save_session_state_is_a_noop_without_a_handle_and_writes_with_one() {
        let dir = tmpdir("write-through");
        let mut state = state_from_envelope(envelope("s1"), dir.clone());
        state.persist = None;
        save_session_state(&state); // must not create anything
        assert!(!session_file(&dir, "s1").unwrap().exists());
        state.persist = Some(PersistHandle { dir: dir.clone(), created_at: 5 });
        save_session_state(&state);
        let on_disk = load(&dir, "s1").expect("written through");
        assert_eq!(on_disk.created_at, 5);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
