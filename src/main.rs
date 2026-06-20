mod acp;
mod bridge;
mod connector;
mod error;
mod model_family;
mod openai;
mod vision;

// Re-export at crate root so submodules can `use crate::ShimError` /
// `use crate::Result` without going through the longer error:: path.
// The bridge and openai modules rely on this short path; if you delete
// these re-exports, recompile carefully — many call sites will break.
pub use error::{Result, ShimError};

use anyhow::Context;
use std::sync::OnceLock;

/// Diagnostic file path for tracing the bridge ↔ shim handshake.
/// Off by default in release builds. To enable: set the env var
/// NWIRO_LOCAL_LLM_DEBUG_LOG to a writable file path before spawning
/// the shim. Each call appends one timestamped line; best-effort —
/// silent on IO error so a misconfigured path can't crash the shim.
static DEBUG_LOG_PATH: OnceLock<Option<String>> = OnceLock::new();

pub fn dbg_log(msg: &str) {
    let path_opt = DEBUG_LOG_PATH.get_or_init(|| std::env::var("NWIRO_LOCAL_LLM_DEBUG_LOG").ok());
    let Some(path) = path_opt else { return };
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = writeln!(f, "[{now_ms}] {msg}");
    }
}

/// Newtype wrapper for the API key. Manual Debug impl prevents accidental logging.
#[derive(Clone)]
pub struct ApiKey(String);

impl std::fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ApiKey([REDACTED])")
    }
}

impl ApiKey {
    /// Read from env var. Returns None when the local endpoint has no auth (e.g. Ollama).
    pub fn from_env() -> Option<Self> {
        std::env::var("NWIRO_LOCAL_LLM_API_KEY_localllm").ok().map(Self)
    }

    /// Only call from openai::client to build the Authorization header.
    /// Named `as_str` (not `as_bearer`) to match the convention `openai::client`
    /// adopted; the call site format!s `"Bearer {}"` itself.
    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }

    /// Test-only constructor for the SEC-KEY-1 sentinel-scan golden. Production
    /// code builds the key ONLY via `from_env`; this keeps that invariant while
    /// letting a test inject a known sentinel without racing the process-global
    /// env var.
    #[cfg(test)]
    pub(crate) fn for_test(raw: &str) -> Self {
        Self(raw.to_string())
    }
}

// Use the default multi_thread tokio runtime. Earlier we tried current_thread
// (single user, simpler), but `tokio::io::stdin()` on Windows hangs against
// the anonymous pipes that UE5's FInteractiveProcess provides — `ReadFile`
// goes via spawn_blocking and the current_thread driver can't make progress
// on the read. The reference ACP shims (claude-agent-acp, codex-acp) both
// use multi_thread; matching them eliminates the initialize_timeout symptom
// observed against the Nwiro Integration Kit bridge.
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dbg_log("main: enter");

    // Install a panic hook before doing ANYTHING else. If the shim panics
    // during startup (e.g. tracing init fails, channel allocation OOMs),
    // the bridge sees "process started, never responded, then exited" with
    // no other signal. The hook writes the panic to the debug log so we
    // can see what crashed.
    std::panic::set_hook(Box::new(|info| {
        dbg_log(&format!("PANIC: {info}"));
        eprintln!("local-llm-acp panicked: {info}");
    }));
    dbg_log("main: panic hook installed");

    // Tracing setup — CRITICAL: must NOT write to stderr.
    //
    // FInteractiveProcess on Windows (UE5's child-process spawner) merges
    // stderr into the stdout pipe delegate. Any stderr write from this shim
    // arrives as the FIRST bytes the bridge sees on stdout, ahead of the
    // ACP `initialize` response. The bridge's ProcessLine then chokes on
    // ANSI-coded tracing junk and the RPC matcher breaks.
    //
    // Two options:
    //   1. Write tracing to a FILE if NWIRO_LOCAL_LLM_TRACING_FILE is set.
    //   2. Silently discard tracing otherwise.
    //
    // Stdout STAYS JSON-RPC-only, no exceptions. The dbg_log diagnostics in
    // this binary already cover startup-failure debugging (panic hook +
    // file-based handshake trace) without needing tracing on stderr.
    if let Ok(path) = std::env::var("NWIRO_LOCAL_LLM_TRACING_FILE") {
        if let Ok(file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
            tracing_subscriber::fmt()
                .with_env_filter(
                    tracing_subscriber::EnvFilter::try_from_default_env()
                        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
                )
                .with_ansi(false)
                .with_writer(std::sync::Mutex::new(file))
                .init();
        }
        // If file open fails, silently fall through — tracing is best-effort.
    }
    // No `else` branch — when no tracing file is configured, tracing is
    // simply uninitialized and tracing!() macros are no-ops.
    dbg_log("main: tracing init complete (no stderr writer)");

    let api_key = ApiKey::from_env();

    // Fallback env vars for testability before P1-006 (C++ DoInitialize injection) lands.
    // Once P1-006 is merged, the initialize handler overwrites these from the ACP params.
    let base_url = std::env::var("NWIRO_LOCAL_LLM_BASE_URL")
        .unwrap_or_else(|_| "http://localhost:11434/v1".to_string());
    let model = std::env::var("NWIRO_LOCAL_LLM_MODEL")
        .unwrap_or_else(|_| "llama3".to_string());

    tracing::info!(base_url = %base_url, model = %model, "local-llm-acp starting");

    let client = openai::Client::new(base_url, model, api_key);

    // Spawn a dedicated OS thread for stdin reads.
    //
    // Why not `tokio::io::stdin()` directly: on Windows, tokio's async stdin
    // calls `ReadFile` via `spawn_blocking`. UE5's `FInteractiveProcess` gives
    // its child an anonymous overlapped pipe as stdin. Tokio's async wrapper
    // around blocking ReadFile against that pipe deadlocks unreliably — the
    // initialize_timeout symptom we hit on the Nwiro Integration Kit bridge.
    //
    // The fix: do plain blocking `std::io::stdin().lock().read_line()` on a
    // dedicated thread, forward each line through a tokio mpsc channel. The
    // async server reads from the channel like any other tokio stream. This
    // is the canonical Windows-friendly stdio pattern.
    let (stdin_tx, stdin_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    std::thread::spawn(move || {
        dbg_log("stdin thread: enter");
        use std::io::BufRead;
        let stdin = std::io::stdin();
        let mut handle = stdin.lock();
        let mut buf = String::new();
        let mut iter = 0usize;
        loop {
            buf.clear();
            dbg_log(&format!("stdin thread: about to read_line (iter={iter})"));
            match handle.read_line(&mut buf) {
                Ok(0) => {
                    dbg_log("stdin thread: EOF (read_line returned 0)");
                    break;
                }
                Ok(n) => {
                    dbg_log(&format!("stdin thread: read {n} bytes; first 80: {:?}", buf.get(..80.min(buf.len())).unwrap_or("")));
                    if stdin_tx.send(buf.clone()).is_err() {
                        dbg_log("stdin thread: server dropped receiver");
                        break;
                    }
                }
                Err(e) => {
                    dbg_log(&format!("stdin thread: read error: {e}"));
                    eprintln!("stdin thread read error: {e}");
                    break;
                }
            }
            iter += 1;
        }
        dbg_log("stdin thread: exit");
    });
    dbg_log("main: stdin thread spawned");

    let server = acp::Server::new(client, stdin_rx);
    dbg_log("main: server constructed; entering run loop");

    let result = server.run().await;
    dbg_log(&format!("main: server.run returned: {:?}", result.is_ok()));
    result.context("ACP server exited with error")?;

    Ok(())
}
