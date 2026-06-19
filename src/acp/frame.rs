use std::io::Write as _;
use tokio::sync::mpsc::UnboundedReceiver;

use crate::error::{Result, ShimError};

/// Read one JSON-RPC frame from the stdin channel. Returns None when the
/// sender is dropped (i.e. stdin reader thread saw EOF or errored).
///
/// We feed lines through a tokio mpsc channel populated by a dedicated
/// blocking thread (spawned in main.rs). This sidesteps `tokio::io::stdin()`
/// which on Windows reads via `spawn_blocking` against `ReadFile` and
/// hangs unreliably when the parent uses anonymous overlapped pipes — the
/// pattern UE5's `FInteractiveProcess` produces. The blocking-thread +
/// channel pattern is the canonical Windows-friendly stdio shape and is
/// what every robust Rust CLI tool that targets pipe-driven IPC uses.
///
/// Empty lines are skipped (some ACP clients emit keep-alive blank lines).
pub async fn read_frame(rx: &mut UnboundedReceiver<String>) -> Result<Option<serde_json::Value>> {
    loop {
        crate::dbg_log("read_frame: awaiting rx.recv()");
        let line = match rx.recv().await {
            Some(l) => {
                crate::dbg_log(&format!("read_frame: got line ({} bytes)", l.len()));
                l
            }
            None => {
                crate::dbg_log("read_frame: rx.recv() returned None (EOF)");
                return Ok(None);
            }
        };

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue; // skip keep-alive blank lines
        }

        let value: serde_json::Value = serde_json::from_str(trimmed)
            .map_err(|e| ShimError::AcpFraming(format!("JSON parse: {e}")))?;

        return Ok(Some(value));
    }
}

/// Write one JSON-RPC frame to stdout and flush immediately.
///
/// Uses synchronous `std::io::stdout()` rather than `tokio::io::stdout()`.
/// Symmetric reason to read_frame's stdin channel: on Windows, tokio's
/// async stdout dispatches writes via `spawn_blocking + WriteFile` against
/// the overlapped pipe handle UE5's `FInteractiveProcess` provides, and
/// the write future can stall — initialize_timeout symptom even when the
/// stdin path was already fixed. Synchronous Win32 `WriteFile` against the
/// same pipe handle works deterministically. With the multi_thread tokio
/// runtime, this blocking call doesn't stall other tasks (other worker
/// threads keep making progress on the SSE stream, MCP round-trips, etc).
///
/// Each frame is small (~200 bytes) so the write is microseconds. Flushing
/// inside the lock guarantees the bridge sees the response before this
/// function returns.
pub async fn write_frame(value: &serde_json::Value) -> Result<()> {
    crate::dbg_log("write_frame: serializing");
    let mut line =
        serde_json::to_string(value).map_err(|e| ShimError::AcpFraming(format!("JSON serialize: {e}")))?;
    line.push('\n');
    let bytes_len = line.len();
    crate::dbg_log(&format!("write_frame: spawn_blocking write+flush ({bytes_len} bytes)"));

    let bytes = line.into_bytes();
    let join_result = tokio::task::spawn_blocking(move || -> std::io::Result<()> {
        let stdout = std::io::stdout();
        let mut handle = stdout.lock();
        crate::dbg_log("write_frame[blocking]: stdout locked, writing");
        handle.write_all(&bytes)?;
        crate::dbg_log("write_frame[blocking]: write_all done, flushing");
        handle.flush()?;
        crate::dbg_log("write_frame[blocking]: flush done");
        Ok(())
    })
    .await;
    crate::dbg_log("write_frame: spawn_blocking returned");

    join_result
        .map_err(|e| ShimError::AcpFraming(format!("stdout join: {e}")))?
        .map_err(|e| ShimError::AcpFraming(format!("stdout write: {e}")))?;

    crate::dbg_log("write_frame: success");
    Ok(())
}

// ── Output sink abstraction (Wave 1 W1-01) ────────────────────────────────
//
// All outbound ACP frames are written through an `OutputSink`. Production uses
// `StdoutSink`, which forwards VERBATIM to `write_frame` above — so production
// output stays byte-for-byte identical to the pre-seam shim; the seam adds only
// an indirection. Its sole purpose is to let the golden-transcript tests capture
// frames in memory (`CaptureSink`) instead of process stdout, which is otherwise
// impossible because `write_frame` targets `std::io::stdout()` directly (and for
// good Windows-pipe reasons — see its doc comment).

/// Boxed, `Send`, `'static` future — the return type of [`OutputSink::write_frame`].
/// `'static` (borrowing nothing from `&self`) is what lets the spawned
/// session/update drainer task own its sink across `.await` points.
pub type BoxFuture<'a, T> =
    std::pin::Pin<Box<dyn std::future::Future<Output = T> + Send + 'a>>;

/// Where outbound ACP frames go. Object-safe so `Server` holds
/// `Arc<dyn OutputSink>` and the drainer task / per-prompt `mcp/*` closure can
/// each keep a cheap `Arc` clone.
pub trait OutputSink: Send + Sync {
    /// Write one JSON-RPC frame. Takes an owned `Value` so the returned future
    /// borrows nothing from `&self` and is therefore `'static`.
    fn write_frame(&self, value: serde_json::Value) -> BoxFuture<'static, Result<()>>;
}

/// Production sink: forwards to the unchanged synchronous-stdout [`write_frame`].
/// Zero behaviour change — that is the entire point of the seam.
pub struct StdoutSink;

impl OutputSink for StdoutSink {
    fn write_frame(&self, value: serde_json::Value) -> BoxFuture<'static, Result<()>> {
        Box::pin(async move { write_frame(&value).await })
    }
}

/// Test sink: captures every frame in emission order for golden snapshots.
/// `Arc<Mutex<Vec<_>>>` so the Server's drainer task, `mcp/*` closure, and
/// dispatcher (all holding `Arc` clones of the same sink) append to one buffer
/// the test reads back after `run()` returns.
#[cfg(test)]
#[derive(Clone, Default)]
pub struct CaptureSink {
    frames: std::sync::Arc<std::sync::Mutex<Vec<serde_json::Value>>>,
}

#[cfg(test)]
impl CaptureSink {
    pub fn new() -> Self {
        Self::default()
    }

    /// Snapshot of all frames captured so far, in emission order.
    pub fn frames(&self) -> Vec<serde_json::Value> {
        self.frames
            .lock()
            .expect("CaptureSink mutex poisoned")
            .clone()
    }
}

#[cfg(test)]
impl OutputSink for CaptureSink {
    fn write_frame(&self, value: serde_json::Value) -> BoxFuture<'static, Result<()>> {
        let frames = std::sync::Arc::clone(&self.frames);
        Box::pin(async move {
            frames
                .lock()
                .expect("CaptureSink mutex poisoned")
                .push(value);
            Ok(())
        })
    }
}

/// Test sink that panics on the Nth `session/update` write to deterministically
/// crash the per-prompt drainer task — the regression guard for "a drainer panic
/// must not kill the whole connector" (Finding A). Non-`session/update` frames
/// and later updates delegate to an inner [`CaptureSink`], so the dispatcher
/// responses, the recovered `-32000`, and a follow-up prompt are all captured.
#[cfg(test)]
#[derive(Clone)]
pub struct PanicOnNthUpdateSink {
    inner: CaptureSink,
    seen_updates: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    panic_at: usize,
}

#[cfg(test)]
impl PanicOnNthUpdateSink {
    /// Panic on the `panic_at`-th (1-indexed) `session/update` write; all other
    /// writes (and later updates) delegate to `inner`.
    pub fn new(inner: CaptureSink, panic_at: usize) -> Self {
        Self {
            inner,
            seen_updates: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            panic_at,
        }
    }
}

#[cfg(test)]
impl OutputSink for PanicOnNthUpdateSink {
    fn write_frame(&self, value: serde_json::Value) -> BoxFuture<'static, Result<()>> {
        let is_update =
            value.get("method").and_then(|m| m.as_str()) == Some("session/update");
        if is_update {
            let n = self
                .seen_updates
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
                + 1;
            if n == self.panic_at {
                return Box::pin(async move {
                    panic!("PanicOnNthUpdateSink: forced drainer panic on session/update #{n}")
                });
            }
        }
        self.inner.write_frame(value)
    }
}
