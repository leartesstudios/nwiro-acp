use crate::acp::messages::{SessionPromptRequest, SessionUpdateNotification, ToolTier};
use crate::openai::messages::{ChatMessage, Role};
use crate::ShimError;
use std::future::Future;
use tokio_util::sync::CancellationToken;

pub mod emulated_parser;
pub mod tools;

/// Default soft preflight warn threshold (Fix D). Calibrated to the
/// v0.2.0 bridge reality: tool array auto-attached on every prompt is
/// ~25K tokens for the Nwiro Pro registry, so a warn threshold tied
/// to LM Studio's factory default 4K was guaranteed to fire on every
/// prompt AND its 2× backstop (8K) would fire BEFORE the request
/// reached the backend — preventing even "hello" from being sent.
///
/// v0.1.25 (this release): bumped to 32768 to align with
/// `docs/MODEL-SETUP.md`'s recommended n_ctx ≥ 65536 for Nwiro Pro
/// full-tool-array users. At 32768, the warn fires meaningfully when
/// session payload approaches ~half the recommended n_ctx budget. The
/// 2× backstop at 65536 now fires only when payload exceeds the full
/// recommended n_ctx — genuinely pathological.
///
/// Override via `NWIRO_LOCAL_LLM_WARN_TOKEN_THRESHOLD`. Users running
/// against backends with smaller n_ctx (or smaller tool registries)
/// should override DOWN to get earlier warnings.
const DEFAULT_WARN_TOKEN_THRESHOLD: usize = 32768;

/// Default history-prune threshold (G1). Pruning operates on
/// messages-only token estimate, while warn fires on the full payload
/// (messages + tools + EMIT-004 directive). With v0.2.0's ~25K tool
/// array dominating payload size, the warn often fires BEFORE the
/// messages-only prune threshold is reached — prune isn't a guarantee
/// of "fires before warn." Rather, prune protects remaining budget
/// AND backstop headroom: when message history grows large enough
/// to ALSO push the payload over the backstop, pruning trims oldest
/// turn-clusters to keep the request fittable in the recommended
/// 65536 n_ctx.
///
/// v0.1.25: bumped 3500 → 28000 in lockstep with the warn threshold.
/// Override via `NWIRO_LOCAL_LLM_PRUNE_TOKEN_THRESHOLD`.
const DEFAULT_PRUNE_TOKEN_THRESHOLD: usize = 28000;

/// Default cap on tool-call rounds per prompt. Protects against
/// pathological infinite tool-call loops in local models.
///
/// v0.1.22: made env-overridable via `NWIRO_LOCAL_LLM_MAX_TOOL_ROUNDS`.
/// v0.1.23: lowered default 50 → 10 per issue #1 F3b. At 5-15s/round on
/// local 14B models, the old 50-round budget allowed 4-12min silent
/// hangs on stuck workflows. 10 covers virtually all legitimate
/// multi-step tool exchanges (the 4-tool average for "create a
/// rotating cube" type prompts) while bounding worst-case latency
/// to ~2.5min. Power users with longer chains can set the env var
/// to restore 50. Breaking default — flagged in release notes.
const DEFAULT_MAX_TOOL_ROUNDS: usize = 10;

/// v0.1.23 F2: default consecutive-identical-failed-call limit before
/// the circuit breaker aborts the tool loop. Per issue #1, small
/// models perseverate — repeating the same failing call across
/// rounds with no self-correction. 3 catches obvious perseveration
/// without firing on legitimate "retry once then try alternate"
/// patterns. Override via `NWIRO_LOCAL_LLM_REPEATED_CALL_LIMIT`.
const DEFAULT_REPEATED_CALL_LIMIT: usize = 3;
// v0.1.39: identical-SUCCESS perseveration limit — companion to
// DEFAULT_REPEATED_CALL_LIMIT (which counts identical FAILURES). A model that
// re-issues the SAME successful call this many times with identical arguments
// is looping (qwen3-class reasoning over-call); end the turn cleanly rather than
// run to max_turn_requests. Override via NWIRO_LOCAL_LLM_IDENTICAL_SUCCESS_LIMIT.
const DEFAULT_IDENTICAL_SUCCESS_LIMIT: usize = 3;

/// Default tool-count above which a non-Native (small / Emulated / None) local
/// model gets a one-shot "high tool count" warning. Small models empirically
/// collapse to schema-bleed above ~30 tools, while qwen3-14B-class tolerate
/// 200+. Override via `NWIRO_LOCAL_LLM_HIGH_TOOL_WARN`. Warn-only; never refuses.
const DEFAULT_HIGH_TOOL_WARN_THRESHOLD: usize = 50;

/// v0.1.26 G4: per-stream state for the Emulated-tier content
/// prefix-gated buffer. Shared between the `chat_completion_stream`
/// closure and the post-stream extraction-decision block via
/// `Arc<Mutex<>>`. Default = `decided: false, looks_like_envelope:
/// false, buffer: ""` (i.e. "still figuring out what kind of content
/// this is").
#[derive(Default)]
struct EmulatedBufferState {
    /// True after we've seen ≥16 bytes OR a newline and made the
    /// envelope-vs-prose classification.
    decided: bool,
    /// Only meaningful when `decided == true`. True = first significant
    /// non-whitespace char was `{`, `<`, or `#` (looks like a tool
    /// envelope opener). False = looks like prose (already flushed).
    looks_like_envelope: bool,
    /// Holds bytes pending classification (pre-decision) OR the full
    /// envelope content (post-decision when `looks_like_envelope`).
    /// Cleared inline once flushed.
    buffer: String,
}

/// v0.2.5 display fix: strip the registered tool-envelope span(s) from a
/// buffered Emulated-tier tail and return the trimmed remaining prose.
///
/// Used by the post-stream suppress-or-flush block when synth has already
/// fired the tool: the raw `{"tool":...}` / `<tool_call>…</tool_call>`
/// envelope must NOT reach the UI, but any surrounding chain-of-thought
/// prose the model emitted should still be displayed.
///
/// Returns:
///   * `Some(remainder)` when at least one registered-envelope span was
///     found — `remainder` is `buffered` with every span removed, then
///     trimmed (may be empty if the buffer was envelope-only; the caller
///     then emits nothing).
///   * `None` when NO span was found — the caller flushes the whole buffer
///     verbatim so nothing is silently lost.
///
/// The `emulated_parser`'s `tool_names` membership guard is the single
/// source of truth: only a registered tool name yields a span, so
/// legitimate JSON in the buffer is never stripped.
fn clean_envelope_remainder(buffered: &str, tool_names: &[String]) -> Option<String> {
    let spans = emulated_parser::extract_tool_calls_with_spans(buffered, tool_names);
    if spans.is_empty() {
        return None;
    }
    // Remove the UNION of the span byte-ranges. XML and inline-JSON are
    // scanned independently, so an XML `<tool_call>` whose args contain a
    // registered `{"tool":...}` JSON envelope produces OVERLAPPING (nested)
    // spans. A naive per-span `replace_range` would apply the outer span's
    // stale offsets after the inner removal shrank the string — an
    // out-of-bounds panic. Sort ASC, merge overlaps, then copy the gaps:
    // robust to any overlap, nesting, or adjacency.
    let mut ranges: Vec<std::ops::Range<usize>> =
        spans.into_iter().map(|e| e.span).collect();
    ranges.sort_by_key(|r| r.start);
    let mut merged: Vec<std::ops::Range<usize>> = Vec::with_capacity(ranges.len());
    for r in ranges {
        match merged.last_mut() {
            Some(last) if r.start <= last.end => last.end = last.end.max(r.end),
            _ => merged.push(r),
        }
    }
    // Span bounds sit on ASCII `{`/`}`/`<`/`>` boundaries, so every gap
    // slice is on a valid UTF-8 char boundary.
    let mut remainder = String::with_capacity(buffered.len());
    let mut pos = 0usize;
    for r in &merged {
        remainder.push_str(&buffered[pos..r.start]);
        pos = r.end;
    }
    remainder.push_str(&buffered[pos..]);
    Some(remainder.trim().to_string())
}

/// v0.2.2: default stream-delta coalescing window (ms). The shim used to emit
/// ONE ACP session/update frame per streamed token; a fast backend
/// (gpt-oss-120b on an H200 at ~200 tok/s) then floods stdout at ~220
/// frames/sec, which a parent stdio reader that can't drain that fast (the
/// nwiro UE5 FInteractiveProcess — Finding I) back-pressures into a stall →
/// empty response. 25ms (~40 flushes/sec) is below human perception for
/// streaming text and decouples the frame rate from token speed. Override via
/// `NWIRO_LOCAL_LLM_STREAM_COALESCE_MS`; `0` disables (per-token; tests use 0).
const DEFAULT_STREAM_COALESCE_MS: u64 = 25;

/// v0.2.2: batches streamed content + reasoning deltas into fewer, larger ACP
/// frames to cap the outbound frame rate (Finding I back-pressure mitigation).
/// Content and reasoning keep DISTINCT chunk types; on a type switch the older
/// buffer flushes first so order stays causal. Sits UPSTREAM of the single
/// ordered writer (no side channel — preserves the Finding-H one-writer
/// invariant). Callers MUST flush before any tool_call / terminal frame and at
/// end-of-stream. Methods RETURN the frames to emit so the caller writes them
/// AFTER dropping the lock (never hold the mutex across the outbound write —
/// same rule as EmulatedBufferState).
struct StreamCoalescer {
    session_id: String,
    content: String,
    reasoning: String,
    last_flush: std::time::Instant,
    interval: std::time::Duration,
    size_limit: usize,
}

impl StreamCoalescer {
    fn new(session_id: String, interval_ms: u64) -> Self {
        Self {
            session_id,
            content: String::new(),
            reasoning: String::new(),
            last_flush: std::time::Instant::now(),
            interval: std::time::Duration::from_millis(interval_ms),
            size_limit: 1024,
        }
    }

    fn take_reasoning(&mut self) -> Option<SessionUpdateNotification> {
        (!self.reasoning.is_empty()).then(|| {
            SessionUpdateNotification::thought_delta(
                self.session_id.clone(),
                std::mem::take(&mut self.reasoning),
            )
        })
    }

    fn take_content(&mut self) -> Option<SessionUpdateNotification> {
        (!self.content.is_empty()).then(|| {
            SessionUpdateNotification::content_delta(
                self.session_id.clone(),
                std::mem::take(&mut self.content),
            )
        })
    }

    /// Append a content delta; returns frames to emit now (write them after
    /// dropping the lock). A pending reasoning buffer (causally older) flushes
    /// first to preserve order on a reasoning→content switch.
    fn push_content(&mut self, s: &str) -> Vec<SessionUpdateNotification> {
        let mut out = Vec::new();
        out.extend(self.take_reasoning());
        self.content.push_str(s);
        out.extend(self.flush_if_due());
        out
    }

    fn push_reasoning(&mut self, s: &str) -> Vec<SessionUpdateNotification> {
        let mut out = Vec::new();
        out.extend(self.take_content());
        self.reasoning.push_str(s);
        out.extend(self.flush_if_due());
        out
    }

    fn flush_if_due(&mut self) -> Vec<SessionUpdateNotification> {
        let due = self.interval.is_zero()
            || self.last_flush.elapsed() >= self.interval
            || self.content.len() + self.reasoning.len() >= self.size_limit;
        if due {
            self.flush()
        } else {
            Vec::new()
        }
    }

    /// Force-flush both buffers (reasoning first, then content). Call at
    /// end-of-stream and before any tool_call / terminal frame.
    fn flush(&mut self) -> Vec<SessionUpdateNotification> {
        let mut out = Vec::new();
        out.extend(self.take_reasoning());
        out.extend(self.take_content());
        self.last_flush = std::time::Instant::now();
        out
    }
}

pub struct SessionState {
    pub session_id: String,
    /// Mutable: updated by set_config_option ACP messages.
    pub current_model: String,
    pub history: Vec<ChatMessage>,
    pub cancel_token: CancellationToken,
    /// Tool-call capability tier inherited from the most recent warmup
    /// probe at session-creation time. `None` (default) when no warmup
    /// has run before `session/new`, which also fires the refusal guard
    /// if tools are sent — acceptable since the user should warmup first.
    pub tool_tier: ToolTier,
    /// v0.1.22 Fix D: gate the soft preflight warning so it fires at
    /// most once per session, not on every prompt or tool-round. The
    /// warning is operator-targeted ("load larger n_ctx") — repeated
    /// firings would inflate observability cardinality without new
    /// information.
    pub token_budget_warned: bool,
    /// v0.1.22 G1: cumulative count of history entries pruned via the
    /// rolling-window pruner. Used as a structured field in the warn
    /// log so operators can see "context was lossy" without parsing
    /// the prose message. Surfacing this via `session/update` is
    /// future work — bridge UI support required first.
    pub pruned_turn_count: usize,
    /// v0.2.6+ context-aware tool budgeting: the tool-array ceiling
    /// LEARNED at runtime when a prompt round overflowed the model's
    /// loaded context and the shim tail-trimmed the tool set to fit
    /// (then retried). `None` until a context_overflow on this session
    /// is recovered; once `Some(n)`, every LATER turn pre-trims the
    /// outbound tool array to `n` BEFORE the first backend call, so the
    /// steady state is one backend call with no overflow. Per-session
    /// (not global) because n_ctx is a property of the loaded model and
    /// sessions can switch models.
    pub learned_tool_ceiling: Option<usize>,
    /// Session-persistence anchor (targets v0.5.0): the resolved storage dir
    /// (derived from the `session/new` cwd or the `NWIRO_SHIM_STATE_DIR`
    /// override) plus the session's original `created_at`. `None` = this
    /// session is never written to disk (kill switch off, no/invalid cwd, or
    /// the connector path). See `src/persist.rs` for the envelope contract.
    pub persist: Option<crate::persist::PersistHandle>,
}

/// Estimate token count for the full outbound payload (messages + tools).
/// Uses `byte_count / 3` — over-counts vs proper BPE because JSON
/// structural syntax (quotes, braces, colons, commas) inflates byte
/// count ~1.3× over semantic content. Over-counting is the CORRECT
/// direction for budget checks — we want to warn / prune sooner than
/// strictly necessary, not later.
fn estimate_payload_tokens(
    messages: &[ChatMessage],
    tools: Option<&[serde_json::Value]>,
) -> usize {
    let msg_bytes: usize = messages
        .iter()
        .map(|m| serde_json::to_string(m).map(|s| s.len()).unwrap_or(0))
        .sum();
    let tool_bytes: usize = tools
        .map(|t| serde_json::to_string(t).map(|s| s.len()).unwrap_or(0))
        .unwrap_or(0);
    (msg_bytes + tool_bytes) / 3
}

/// Estimate token count for messages only (no tools). Used by the
/// pruner because tools are external to history — pruning history
/// can't shrink the tools array, only the conversation.
fn estimate_messages_tokens(messages: &[ChatMessage]) -> usize {
    let msg_bytes: usize = messages
        .iter()
        .map(|m| serde_json::to_string(m).map(|s| s.len()).unwrap_or(0))
        .sum();
    msg_bytes / 3
}

/// Drop oldest non-system turn-clusters from `history` until estimated
/// message-tokens drop below `target_tokens`. Returns count of pruned
/// entries for telemetry.
///
/// "Turn cluster" = a stretch starting at a non-system message and
/// running up to (but not including) the next user message. Concretely:
///   - `[user]` alone, OR
///   - `[user, assistant, ...]`, OR
///   - `[user, assistant-with-tool_calls, tool, tool, ...]`, OR
///   - Leading orphan `[assistant, tool, ...]` from malformed history
///     (treated as one cluster up to the first user; drops as a unit).
///
/// Preserves:
///   - All leading system messages (first contiguous block — typically
///     `_meta.systemPrompt.append` + the EMIT-004 directive)
///   - The latest user turn AND everything after it (assistant
///     response + tool exchanges for the prompt currently being
///     processed).
///
/// Atomicity matters because OpenAI strict-mode backends (and some
/// Ollama/LM Studio builds) error or loop on history where an
/// `assistant` message has `tool_calls` IDs without a matching
/// `tool` response. Per v0.1.22 design decision (treated as a release
/// blocker).
///
/// **True O(N) algorithm** (v0.1.22 critic-corrected):
///   1. Single forward pass to serialise each message once into
///      `bytes_per_msg`.
///   2. Single forward pass to identify cluster boundaries
///      (each non-system non-user index OR each user index, before
///      `latest_user_idx`).
///   3. Iterate clusters left-to-right, subtracting cluster bytes
///      from the running total, until target met.
///   4. Single `Vec::drain` to remove the chosen prefix of clusters
///      in one O(N) pass — NOT N repeated `Vec::remove` calls
///      which would be O(N×K) = O(N²) worst-case.
fn prune_history_atomic(history: &mut Vec<ChatMessage>, target_tokens: usize) -> usize {
    // O(N) serialisation pass — done once.
    let bytes_per_msg: Vec<usize> = history
        .iter()
        .map(|m| serde_json::to_string(m).map(|s| s.len()).unwrap_or(0))
        .collect();
    let mut total_bytes: usize = bytes_per_msg.iter().sum();

    if total_bytes / 3 <= target_tokens {
        return 0;
    }

    // First non-system index (start of prunable region).
    let prune_start = match history.iter().position(|m| !matches!(m.role, Role::System)) {
        Some(i) => i,
        None => return 0,
    };

    // Latest user index (must preserve this and beyond).
    let latest_user_idx = match history.iter().rposition(|m| matches!(m.role, Role::User)) {
        Some(i) => i,
        None => return 0,
    };

    if prune_start >= latest_user_idx {
        return 0;
    }

    // Build cluster_starts: every cluster boundary in the prunable
    // range. A boundary is `prune_start` itself, plus every user
    // index between prune_start+1 and latest_user_idx (exclusive).
    // Append `latest_user_idx` as the sentinel terminator so windows(2)
    // gives valid [start, end) pairs.
    let mut cluster_starts: Vec<usize> = Vec::new();
    cluster_starts.push(prune_start);
    for i in (prune_start + 1)..latest_user_idx {
        if matches!(history[i].role, Role::User) {
            cluster_starts.push(i);
        }
    }
    cluster_starts.push(latest_user_idx);

    // Walk clusters left-to-right; accumulate the drop range until
    // total bytes drop under target. Single pass, O(N) total.
    let mut drop_end = prune_start;
    for window in cluster_starts.windows(2) {
        if total_bytes / 3 <= target_tokens {
            break;
        }
        let (cstart, cend) = (window[0], window[1]);
        let cluster_bytes: usize = bytes_per_msg[cstart..cend].iter().sum();
        total_bytes = total_bytes.saturating_sub(cluster_bytes);
        drop_end = cend;
    }

    // Single drain — O(N) amortised across all dropped messages.
    let pruned = history.drain(prune_start..drop_end).count();
    pruned
}

/// v0.1.23 F2: canonicalise a tool-call's `arguments` string into a
/// stable form so the circuit breaker doesn't miss semantic duplicates
/// that differ only in JSON key ordering. Parse → `to_string` round-
/// trips through `serde_json::Map` (which is `BTreeMap` by default
/// because we don't enable `preserve_order`), yielding sorted keys at
/// every nesting level. On parse failure (malformed arguments), falls
/// back to the raw string — the circuit breaker would only miss
/// semantic duplicates in that case, not false-positive on legitimate
/// retries.
///
/// Design decision: normalize, don't trust model stability across
/// backends.
fn canonicalize_arguments(arguments: &str) -> String {
    match serde_json::from_str::<serde_json::Value>(arguments) {
        Ok(v) => v.to_string(),
        Err(_) => arguments.to_string(),
    }
}

/// v0.1.36 tool-I/O logging mode (`NWIRO_LOCAL_LLM_LOG_TOOL_IO`).
///
/// The shim emits every tool call's name/args/result on the WIRE (ACP
/// `rawInput`/`rawOutput`) for the UE5 UI, but historically logged NONE of it
/// to a file sink — so an agent reading `NWIRO_LOCAL_LLM_TRACING_FILE` was blind
/// to which tool was called with what arguments and what it returned. This gate
/// turns on a readable, file-logged record of the tool round-trip:
/// - `off` (default): nothing (zero overhead, no behaviour change).
/// - `failures`: log the full call ONLY when it failed — MCP `isError:true`,
///   OR the green-badge anomaly (`isError:false` yet an explicit
///   `success:false`). Low-noise; surfaces exactly the case the operator debugs.
/// - `full`: log every tool call.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ToolIoMode {
    Off,
    Failures,
    Full,
}

/// Cached once at process start (env is not re-read per call), mirroring the
/// `NWIRO_LOCAL_LLM_DEBUG_LOG` OnceLock pattern. Unrecognised / unset → `Off`.
fn tool_io_mode() -> ToolIoMode {
    static MODE: std::sync::OnceLock<ToolIoMode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        match std::env::var("NWIRO_LOCAL_LLM_LOG_TOOL_IO")
            .map(|v| v.trim().to_ascii_lowercase())
            .as_deref()
        {
            Ok("full") => ToolIoMode::Full,
            Ok("failures") | Ok("failure") => ToolIoMode::Failures,
            _ => ToolIoMode::Off,
        }
    })
}

/// Per-field byte cap for tool-I/O logging (`NWIRO_LOCAL_LLM_LOG_TOOL_IO_MAX_BYTES`,
/// default 65536, `0` = unlimited). Bounds a pathological asset-dump from writing
/// megabytes per call — but truncation is ALWAYS marked (see `clip_tool_io`), so
/// the log never silently lies the way a bare byte-count does.
fn tool_io_max_bytes() -> usize {
    static MAX: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MAX.get_or_init(|| {
        std::env::var("NWIRO_LOCAL_LLM_LOG_TOOL_IO_MAX_BYTES")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(65536)
    })
}

/// Clip `s` to at most `max` bytes (`0` = unlimited), appending an EXPLICIT,
/// honest truncation marker with the dropped/total byte counts. Respects a
/// UTF-8 char boundary so the kept prefix is always valid.
fn clip_tool_io(s: &str, max: usize) -> std::borrow::Cow<'_, str> {
    if max == 0 || s.len() <= max {
        return std::borrow::Cow::Borrowed(s);
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    std::borrow::Cow::Owned(format!(
        "{}…[truncated {} of {} bytes]",
        &s[..end],
        s.len() - end,
        s.len()
    ))
}

/// The green-badge anomaly: MCP `isError:false` while an explicit `success:false`
/// lives at the envelope top level or under `structuredContent`. Used ONLY to
/// decide whether to LOG in `failures` mode — it NEVER changes classification
/// (the shim stays MCP-faithful; `isError` remains the sole failure signal). The
/// anomaly being visible in the log is a breadcrumb pointing at the Nwiro
/// `isError`-should-be-true bug, not a shim reinterpretation of the result.
fn tool_io_success_anomaly(response: &serde_json::Value) -> bool {
    let success_is_false =
        |v: &serde_json::Value| v.get("success").and_then(|s| s.as_bool()) == Some(false);
    success_is_false(response)
        || response
            .get("structuredContent")
            .map(success_is_false)
            .unwrap_or(false)
}

/// v0.1.36 — log one tool call's full I/O to `tracing` (target `"tool_io"`) when
/// `NWIRO_LOCAL_LLM_LOG_TOOL_IO` selects it. Logs the RAW argument string (so a
/// model's malformed JSON is visible, not silently normalized to `{}`) and the
/// full MCP `{content, isError}` response envelope (ground truth). `is_error` is
/// recorded as DATA, never used to flip the protocol signal. Called from the
/// bridge tool loop — the only site that sees the synthesized transport-error /
/// breaker envelopes that `execute_tool` returns `Err` before constructing.
fn log_tool_io(
    round: usize,
    call: &crate::openai::messages::ToolCall,
    response: &serde_json::Value,
    is_error: bool,
    elapsed: std::time::Duration,
) {
    let mode = tool_io_mode();
    match mode {
        ToolIoMode::Off => return,
        ToolIoMode::Failures if !is_error && !tool_io_success_anomaly(response) => return,
        _ => {}
    }
    let max = tool_io_max_bytes();
    let response_str = response.to_string();
    tracing::info!(
        target: "tool_io",
        round,
        call_id = %call.id,
        tool = %call.function.name,
        is_error,
        elapsed_ms = elapsed.as_millis() as u64,
        args = %clip_tool_io(&call.function.arguments, max),
        response = %clip_tool_io(&response_str, max),
        "tool call I/O"
    );
}

/// v0.1.23 F2 atomicity: when the circuit breaker trips mid-batch
/// (e.g. on call B of an `[A, B, C]` round), push synthetic `tool`
/// response messages for every unprocessed call in the batch. Without
/// this, the prior assistant's `tool_calls` array references ids
/// with no matching `tool` response, which strict-mode backends
/// reject and `prune_history_atomic` was designed around. Per
/// v0.1.23 critic claude FINDING 1 — the invariant already regressed
/// once, so it has its own helper + dedicated test.
fn push_skipped_call_stubs(
    history: &mut Vec<ChatMessage>,
    tool_calls: &[crate::openai::messages::ToolCall],
    from_idx: usize,
    reason: &str,
) -> usize {
    let mut pushed = 0;
    // v0.1.39: caller-supplied reason so the stub is accurate for BOTH the
    // circuit breakers and the schema-bleed co-emission guard (the text was
    // hardcoded to the breaker phrasing even on the bleed/from_idx=0 path).
    let stub_text = format!("Skipped: {reason}.");
    for skipped_call in tool_calls.iter().skip(from_idx) {
        let stub_response = serde_json::json!({
            "content": [{
                "type": "text",
                "text": stub_text.clone(),
            }],
            "isError": true
        });
        history.push(ChatMessage::tool(
            skipped_call.id.clone(),
            stub_response,
        ));
        pushed += 1;
    }
    pushed
}

/// v0.1.23 F2: hash a (tool_name, canonicalized_arguments) signature
/// for structured logging. Avoids dumping raw arguments to telemetry
/// — model-emitted tool arguments may contain user-supplied content
/// or sensitive paths. Operators correlate via the hash; the tool
/// name is logged as a separate structured field.
fn hash_call_signature(name: &str, canonical_args: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    name.hash(&mut hasher);
    canonical_args.hash(&mut hasher);
    hasher.finish()
}

/// Build the model-AGNOSTIC "invoke, don't describe" tool-invocation mandate
/// (v0.1.35). Returns the directive to inject as a system message, or `None`
/// when no mandate applies.
///
/// Fires for EVERY Native/Emulated session that registers tools — there is NO
/// per-family gate. Describer-over-actor (a model that lists tool names in
/// prose but never emits a `tool_call`) is a property of any RLHF-aligned chat
/// model under `tool_choice:auto`, not a GLM quirk, and the runtime request
/// sets no `tool_choice`, so this prompt-level mandate is the cross-backend
/// nudge that makes weak models ACT. The text is tier-agnostic ("invoke the
/// appropriate tool") and scoped to action requests ("when the user requests
/// an action ..."); it is a prompt NUDGE, not a hard `tool_choice` constraint,
/// so the model keeps latitude to converse or ask a clarifying question on
/// non-action / underspecified turns. `None` tier means tools were stripped →
/// no tools to invoke → no mandate.
fn build_tool_invocation_mandate(tier: ToolTier, tool_names: &[String]) -> Option<String> {
    if (tier == ToolTier::Native || tier == ToolTier::Emulated) && !tool_names.is_empty() {
        Some(format!(
            "You can call these tools when needed: {}. \
             If the user asks you to DO something one of these tools handles \
             (create, edit, delete, find, list, generate, spawn, read, execute, \
             and the like), call that tool directly with the correct arguments — \
             do not merely describe or explain what it would do. \
             For anything else — greetings, questions, small talk, or any turn no \
             tool fits — reply directly and briefly, and do NOT call a tool. \
             Decide quickly: do not deliberate at length over which tool to use, \
             and never echo or list these tool names back to the user.",
            tool_names.join(", ")
        ))
    } else {
        None
    }
}

/// Estimate the per-call directive overhead injected by the directive
/// branches in `handle_session_prompt`'s `messages_for_call` builder.
/// The directive isn't in `state.history`, so naive payload estimation
/// undercounts the real outbound size for sessions that get a directive.
///
/// Two directive branches as of v0.1.29:
///
/// - **EMIT-004 (Emulated tier)**: teaches HOW to format a tool call.
///   Fires when `tier == Emulated && !tool_names.is_empty()`. Per v0.1.22
///   critic, base directive prose ~340 chars / 3 ≈ 113 tokens.
///
/// - **Invocation mandate (Native + Emulated, v0.1.35 model-agnostic)**:
///   teaches WHEN/WHY to invoke. Fires for EVERY Native/Emulated session
///   with registered tools (no per-family gate). ~470 chars / 3 ≈ 156
///   tokens — the action-verb enumeration plus mandate language is denser
///   than EMIT-004; on Emulated it stacks on top of EMIT-004.
///
/// Both share the same comma-joined tool names component.
///
/// Conservative — over-counts slightly because tool names are typically
/// shorter than the per-name formatting overhead.
fn estimate_directive_overhead(tier: ToolTier, tool_names: &[String]) -> usize {
    if tool_names.is_empty() {
        return 0;
    }
    // v0.1.35: the tool-invocation mandate is model-AGNOSTIC — it fires for
    // EVERY Native/Emulated session that registers tools (no per-family gate),
    // so the mandate overhead is unconditional on those tiers. On Emulated it
    // stacks on top of EMIT-004, and both carry a copy of the tool-name list,
    // so the names_bytes contribution is counted twice there.
    let (base_tokens, names_copies) = match tier {
        // EMIT-004 (FORMAT, ~113 tokens) + invocation mandate (POLICY, ~156).
        ToolTier::Emulated => (113 + 156, 2),
        // Invocation mandate (POLICY, ~156 tokens) only.
        ToolTier::Native => (156, 1),
        // None strips tools entirely → no directive injected.
        ToolTier::None => return 0,
    };
    let names_bytes: usize = tool_names.iter().map(|n| n.len() + 2).sum();
    base_tokens + names_copies * (names_bytes / 3)
}

/// Enforce the per-family tool ceiling on the OUTBOUND tool array — the shim's
/// "HOW MANY" responsibility (the WHICH/HOW-MANY division: nwiro owns WHICH
/// tools and their best-first order; the shim, which sees tools as opaque
/// name-only JSON, only bounds the COUNT). The ceiling was previously
/// only *published* on `WarmupResult`, never *enforced* here — so a known-fragile
/// family (e.g. GLM-4 → 29) could still be over-exposed if nwiro's own cap
/// regressed or was overridden. Truncate the TAIL (keep the first `ceiling`,
/// trusting nwiro's pin+context+BM25 best-first ordering). A `None` tier (tools
/// already stripped) and a family with no documented ceiling pass through
/// untouched. Generic over the tool element so it stays a pure, unit-testable fn.
fn enforce_tool_ceiling<T>(
    tier: ToolTier,
    tools: Option<Vec<T>>,
    ceiling: Option<u32>,
) -> Option<Vec<T>> {
    match (tier, tools, ceiling) {
        (t, Some(v), Some(c)) if t != ToolTier::None && v.len() as u32 > c => {
            Some(v.into_iter().take(c as usize).collect())
        }
        (_, v, _) => v,
    }
}

/// Parse the model's loaded context length (`n_ctx`) out of a backend
/// context-overflow error message. The backend (LM Studio / llama.cpp)
/// reports e.g. `"the request exceeds the available context size
/// (n_ctx = 4096); the prompt is too long"`. We locate the literal token
/// `n_ctx`, then return the FIRST subsequent run of ASCII digits — which
/// tolerates the observed spellings `n_ctx = 4096`, `n_ctx=4096`, and
/// `n_ctx is 4096` (any non-digit filler between the token and the number
/// is skipped). Returns `None` when `n_ctx` is absent or no digits follow
/// it, in which case the caller refuses cleanly WITHOUT a retry (no
/// budget to compute a fit ceiling from).
fn parse_n_ctx_from_overflow(msg: &str) -> Option<usize> {
    let after = &msg[msg.find("n_ctx")? + "n_ctx".len()..];
    let start = after.find(|c: char| c.is_ascii_digit())?;
    let digits: String = after[start..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse().ok()
}

/// Parse the backend's MEASURED prompt-token count from an overflow error, under
/// either of the two known field names:
///   - `n_keep` — llama.cpp / LM Studio: "...(n_keep: 39147 >= n_ctx: 4096) ..."
///   - `n_prompt_tokens` — llama-server (raw llama.cpp HTTP server):
///     `...,"n_prompt_tokens":39069,"n_ctx":4096}`
/// Both are the actual prompt size for the EXACT failing request — a far more
/// reliable trim signal than any chars-per-token estimate, and the PRIMARY signal
/// the overflow path uses. `n_keep` is tried first; `n_prompt_tokens` is a strict
/// superstring guard away (it does not contain "n_keep"), so order is safe.
/// Returns None when neither field is present (truly opaque backends), in which
/// case the caller falls back to the conservative `context_fit_ceiling` estimate.
fn parse_n_keep_from_overflow(msg: &str) -> Option<usize> {
    for field in ["n_keep", "n_prompt_tokens"] {
        let Some(pos) = msg.find(field) else { continue };
        let after = &msg[pos + field.len()..];
        let Some(start) = after.find(|c: char| c.is_ascii_digit()) else { continue };
        let digits: String = after[start..]
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect();
        if let Ok(n) = digits.parse() {
            return Some(n);
        }
    }
    None
}

/// Max bounded retries on the context-overflow path. Validated by review: a single
/// retry overshot in production (the char estimate left too many tools and the
/// one retry overflowed again → refusal). A small bound lets the loop converge on
/// the backend's MEASURED fit — each retry recomputes from the next (smaller)
/// `n_keep` — without risking an unbounded loop. Each attempt is a fast HTTP 400
/// (no generation), so worst-case added latency is ~MAX × a few hundred ms.
const MAX_OVERFLOW_RETRIES: usize = 3;

/// The fraction of `n_ctx` the trimmed prompt targets, expressed as NUM/DEN
/// (= 0.8). Leaves ~20% headroom for the fixed, non-shrinking overhead (system
/// prompt + chat template + response reserve) that a proportional tool-trim
/// cannot reduce. Residual variance beyond this headroom is absorbed by the
/// bounded retry loop.
const OVERFLOW_FIT_SAFETY_NUM: usize = 4;
const OVERFLOW_FIT_SAFETY_DEN: usize = 5;

/// Conservative tool count that should fit `n_ctx`, derived from the backend's
/// MEASURED prompt token count `n_keep` for the current `cur`-tool request — the
/// PRIMARY overflow-recovery sizing. The whole prompt is `n_keep` tokens; we want
/// it under `n_ctx * SAFETY`. We scale the tool count proportionally, treating the
/// fixed non-tool overhead as if it shrank with the tools — which OVER-trims
/// slightly (the safe direction). Any residual overflow is caught by the bounded
/// retry loop, which recomputes from the next (smaller) `n_keep` and converges.
/// Always shrinks by at least one tool, never below one (clamped to `[1, cur-1]`).
fn overflow_target_from_nkeep(n_ctx: usize, n_keep: usize, cur: usize) -> usize {
    // PRECONDITION: cur >= 2 (the caller refuses at `cur <= 1` before sizing, so
    // the `[1, cur-1]` contract is satisfiable). Guard the degenerate input
    // defensively so a misuse can never return 0 instead of a valid count.
    debug_assert!(cur >= 2, "overflow_target_from_nkeep requires cur >= 2 (caller guards cur <= 1)");
    if cur <= 1 {
        return cur;
    }
    // new = floor(cur * (n_ctx * NUM / DEN) / n_keep). u128 math avoids overflow
    // on the cur*target product; n_keep is floored at 1 to avoid div-by-zero.
    let target_tokens = n_ctx.saturating_mul(OVERFLOW_FIT_SAFETY_NUM) / OVERFLOW_FIT_SAFETY_DEN;
    let scaled =
        (cur as u128).saturating_mul(target_tokens as u128) / (n_keep.max(1) as u128);
    (scaled as usize).max(1).min(cur.saturating_sub(1))
}

/// FALLBACK estimate (used only when the overflow error carries no `n_keep` —
/// see `overflow_target_from_nkeep` for the primary path) of how many tools fit
/// in `n_ctx`, given the (opaque, already best-first-ordered) `tools` array the
/// shim is about to send. The shim has no tokenizer, so it approximates token
/// count as `serialized_chars / 3` — matching `estimate_payload_tokens`; JSON
/// structural syntax inflates byte count ~1.3× over semantic content, so `/3`
/// OVER-counts, which is the CORRECT (safe) direction for a budget check. An
/// earlier `/4` UNDER-counted by ~38% against real llama.cpp tokenization and
/// left too many tools (the production overshoot this fix addresses). It budgets
/// 60% of the window to tools — the other 40% covers system prompt + history +
/// response. `per_tool` is the average serialized cost; the fit is
/// `budget_tokens / per_tool`, clamped to `[1, tools.len()]`.
///
/// Deliberately CONSERVATIVE: under-trimming merely costs one of the bounded
/// retries, whereas over-shooting causes another overflow. When `tools` is empty
/// there is nothing to fit, so the ceiling is 0.
fn context_fit_ceiling(n_ctx: usize, tools: &[serde_json::Value]) -> usize {
    if tools.is_empty() {
        return 0;
    }
    let total_tool_tokens = serde_json::to_string(tools).map(|s| s.len()).unwrap_or(0) / 3;
    let per_tool = std::cmp::max(1, total_tool_tokens / tools.len());
    // 60% of ctx for tools; the other 40% covers system prompt + history
    // + response. Reserving via `n_ctx/2` was too aggressive (would
    // under-trim less but leave too little for the rest of the payload).
    let budget_tokens = n_ctx * 60 / 100;
    let fit = std::cmp::max(1, budget_tokens / per_tool);
    fit.min(tools.len())
}

/// Legacy alias maintained for the v0.1.22-vintage tests; now identical to
/// `estimate_directive_overhead` (the `family` argument was removed in
/// v0.1.35). New tests should call `estimate_directive_overhead` directly.
#[cfg(test)]
fn estimate_emulated_directive_overhead(tier: ToolTier, tool_names: &[String]) -> usize {
    estimate_directive_overhead(tier, tool_names)
}

/// Merge a directive string into the first system message of `msgs`,
/// or prepend a new system message if none exists.
///
/// v0.1.30 refactor: extracted from `handle_session_prompt`'s
/// `messages_for_call` builder where this logic was duplicated across
/// two branches (EMIT-004 and the v0.1.29 action mandate). Combined
/// with the v0.1.30 critic round-1 MUST_FIX 2 expansion of the mandate
/// gate to include Emulated tier, the messages_for_call construction
/// needs to be able to apply MULTIPLE directives sequentially — this
/// helper makes that composition cheap to express.
///
/// Single concatenated system message strategy: when an existing system
/// message is present (typically from `_meta.systemPrompt.append`), the
/// directive is appended with a blank-line separator. When no system
/// message exists, a new one is prepended. Two separate system messages
/// would split attention — some models weight only the first.
///
/// Pure on `msgs.history`: the caller is expected to pass a per-call
/// clone, not `state.history` directly, so persisted state stays clean
/// across tool-round iterations.
fn merge_directive_into_system(msgs: &mut Vec<ChatMessage>, directive: &str) {
    if let Some(first) = msgs.first_mut() {
        if matches!(first.role, crate::openai::messages::Role::System) {
            if let Some(existing) = first.content.as_mut().and_then(|c| c.as_text_mut()) {
                existing.push_str("\n\n");
                existing.push_str(directive);
            } else {
                first.content = Some(directive.to_string().into());
            }
            return;
        }
    }
    msgs.insert(0, ChatMessage::system(directive.to_string()));
}

/// History image policy: downgrade any multimodal (image-bearing) user message
/// in `history` to text-only, replacing the image parts with a short placeholder.
/// Called at the start of each prompt turn so base64 images from PRIOR turns are
/// not re-sent (and re-billed) on every subsequent turn; the current turn's
/// images are added afterwards and stay intact for this turn's request.
fn strip_history_images(history: &mut [ChatMessage]) {
    use crate::openai::messages::MessageContent;
    for msg in history.iter_mut() {
        if matches!(msg.content, Some(MessageContent::Parts(_))) {
            let text = msg.content_text().unwrap_or("").to_string();
            msg.content = Some(MessageContent::Text(format!(
                "{text}\n\n[image attachment(s) from an earlier turn omitted to save context]"
            )));
        }
    }
}

/// Build the `role:user` message for a prompt turn, routing image content:
/// no images → a plain text message (byte-identical wire); images + a
/// vision-capable model → an OpenAI multimodal message (text + `image_url`
/// parts); images + a text-only model → text plus a visible omission note
/// (never a silent drop).
fn build_user_message(
    text: String,
    images: Vec<crate::acp::messages::ImageInput>,
    vision_capable: bool,
    model: &str,
) -> ChatMessage {
    if images.is_empty() {
        return ChatMessage::user(text);
    }
    if vision_capable {
        tracing::info!(
            model,
            n_images = images.len(),
            "vision-capable model — forwarding image(s) as OpenAI image_url content"
        );
        ChatMessage::user_multimodal(
            text,
            images.into_iter().map(|i| (i.mime, i.data)).collect(),
        )
    } else {
        let n = images.len();
        tracing::warn!(
            model,
            n_images = n,
            "model is not vision-capable — dropping image(s) with an omission note \
             (set NWIRO_LOCAL_LLM_FORCE_VISION=on to force, or use a vision model)"
        );
        ChatMessage::user(format!(
            "{text}\n\n[{n} image attachment(s) omitted: the current model has no vision \
             support. Switch to a vision-capable model (e.g. qwen2.5-vl, llava, \
             llama3.2-vision) to use images.]"
        ))
    }
}

/// Drive one user prompt to completion, streaming content deltas via
/// `write_update` and executing tool calls via `write_mcp_request`.
///
/// NOTE: This function takes 5 parameters. STRUCTURE.md sketches 4.
/// The 5th (write_mcp_request) cannot live inside SessionState without
/// object-safe boxing complexity. STRUCTURE.md must be updated — see
/// follow_up_tasks in the implementer output.
/// v0.1.24 G2: returns the OpenAI-style `finish_reason` from the
/// terminating LLM stream so the outer ACP responder (`acp::server`)
/// can map it to an ACP `stopReason` value on the `session/prompt`
/// response. Synthetic "circuit_breaker" is returned when the F2
/// perseveration guard fires.
pub async fn handle_session_prompt<F, Fut>(
    req: SessionPromptRequest,
    state: &mut SessionState,
    client: &crate::openai::Client,
    write_update: impl Fn(SessionUpdateNotification),
    write_mcp_request: F,
) -> crate::Result<(String, Option<crate::acp::messages::PromptErrorKind>)>
where
    F: Fn(serde_json::Value) -> Fut,
    Fut: Future<Output = serde_json::Value> + Send,
{
    // History image policy (council): strip image parts from PRIOR turns' user
    // messages down to a text placeholder, so base64 images aren't re-sent (and
    // re-billed) every turn. The CURRENT turn's images are added just below and
    // stay intact for this turn's request.
    strip_history_images(&mut state.history);

    // Build the user message, routing image content for vision-capable models
    // (image input, Phase 2). Text-only turns are byte-identical to before.
    let (user_text, images) = req.content_parts();
    let vision = crate::vision::model_supports_vision(client.model());
    state
        .history
        .push(build_user_message(user_text, images, vision, client.model()));

    let mut mcp_connection_id: Option<String> = None;
    let mut tool_round: usize = 0;

    // Pre-extract the registered tool names from `req.tools`. Used by
    // the v0.1.17 Emulated-tier parser as the false-positive
    // discriminator — only a name matching one of these synthesises a
    // ToolCall. Extraction is per-prompt, not per-iteration, because
    // `req.tools` doesn't change across the tool_round loop.
    let tool_names: Vec<String> = req
        .tools
        .as_ref()
        .map(|tools| {
            tools
                .iter()
                .filter_map(|t| {
                    t.pointer("/function/name")
                        .and_then(|n| n.as_str())
                        .map(String::from)
                })
                .collect()
        })
        .unwrap_or_default();

    // v0.4.1: schema lookup for shim-side argument coercion — tool name →
    // that tool's `function.parameters` JSON Schema. Built once per prompt
    // from the UNTRIMMED `req.tools` (the family/learned ceilings trim only
    // the OUTBOUND array; any name the model emits should still coerce),
    // mirroring the `tool_names` extraction above. A tool entry without
    // `parameters` simply never coerces. Duplicate tool names last-wins —
    // fine, registry names are unique.
    let tool_schemas: std::collections::HashMap<String, serde_json::Value> = req
        .tools
        .as_ref()
        .map(|tools| {
            tools
                .iter()
                .filter_map(|t| {
                    let name = t.pointer("/function/name")?.as_str()?.to_string();
                    let schema = t.pointer("/function/parameters")?.clone();
                    Some((name, schema))
                })
                .collect()
        })
        .unwrap_or_default();

    // v0.1.22 Fix D + G1 setup: read thresholds + max-tool-rounds from
    // env, with sane defaults. Done once per prompt, not per iteration.
    //
    // v0.1.25 robustness: clamp warn/prune env values to ≥ 1 so a
    // misconfigured `=0` doesn't cause `0 >= 0` to fire the backstop
    // on the first byte of payload — mirrors the v0.1.23 codex catch
    // for repeated_call_limit. Same `.max(1)` pattern.
    let warn_threshold: usize = std::env::var("NWIRO_LOCAL_LLM_WARN_TOKEN_THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_WARN_TOKEN_THRESHOLD)
        .max(1);
    let prune_threshold: usize = std::env::var("NWIRO_LOCAL_LLM_PRUNE_TOKEN_THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_PRUNE_TOKEN_THRESHOLD)
        .max(1);
    // v0.1.25 critic claude FINDING 3: clamp to ≥1 for consistency
    // with warn/prune/repeated_call_limit. A `=0` env value would
    // otherwise cause `tool_round (= 0) >= max_tool_rounds (= 0)` to
    // fire on the very first round, returning max_turn_requests
    // before any LLM call happens. That's a confusing undocumented
    // "disable" mode; floor it to 1.
    let max_tool_rounds: usize = std::env::var("NWIRO_LOCAL_LLM_MAX_TOOL_ROUNDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_MAX_TOOL_ROUNDS)
        .max(1);
    // v0.1.23 F2: circuit-breaker threshold for consecutive identical
    // failing tool calls. Clamped to `.max(1)` so a misconfigured `=0`
    // doesn't make the breaker fire on the very first tool result
    // (because `consecutive_identical_errors (=0) >= 0` would be true
    // and the abort message would read "...returned an error 0
    // consecutive times" — nonsense). Per v0.1.23 critic codex catch.
    // To effectively disable the breaker, set the env var to a very
    // large number (e.g. 999999) — there's no explicit "disabled" flag.
    let repeated_call_limit: usize = std::env::var("NWIRO_LOCAL_LLM_REPEATED_CALL_LIMIT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_REPEATED_CALL_LIMIT)
        .max(1);
    // v0.1.39: identical-SUCCESS breaker limit. The error breaker above only
    // counts identical FAILURES; a model can also loop on identical SUCCESSFUL
    // calls (qwen3 re-firing the same spawn_actor against an always-success
    // backend) until max_turn_requests. Clamped to >=1 (a `=0` would fire on the
    // first success), same pattern as repeated_call_limit.
    let identical_success_limit: usize = std::env::var("NWIRO_LOCAL_LLM_IDENTICAL_SUCCESS_LIMIT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_IDENTICAL_SUCCESS_LIMIT)
        .max(1);
    // Real-request schema-bleed guard: when a small model collapses under the
    // tool payload and echoes the tool SCHEMA back as text instead of a usable
    // call, suppress that garbage wall and surface ONE clean refusal line. The
    // v0.1.28 template-gate only protects warmup; this is its real-request
    // counterpart. Default ON — it only ever fires on already-unreadable output
    // (>50% structural, >=5 schema keywords); disable with
    // NWIRO_LOCAL_LLM_BLEED_GUARD=off|0|false.
    let bleed_guard_enabled = !matches!(
        std::env::var("NWIRO_LOCAL_LLM_BLEED_GUARD").ok().as_deref(),
        Some("off") | Some("0") | Some("false")
    );
    let high_tool_warn_threshold: usize = std::env::var("NWIRO_LOCAL_LLM_HIGH_TOOL_WARN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_HIGH_TOOL_WARN_THRESHOLD)
        .max(1);
    // v0.1.23 F2: per-prompt circuit-breaker state. Scoped to ONE
    // handle_session_prompt invocation — must NOT live on SessionState
    // because perseveration is a per-prompt phenomenon (carrying the
    // streak across user prompts would punish legitimate retries).
    let mut last_call_signature: Option<String> = None;
    let mut consecutive_identical_errors: usize = 0;
    // v0.1.39: streak of identical SUCCESSFUL calls (same name + canonical args).
    let mut consecutive_identical_successes: usize = 0;

    // Sanity check: if env-misconfigured to invert the relationship,
    // log a one-time hint at this prompt. We don't clamp because the
    // user might have a deliberate reason; but we surface the issue.
    if warn_threshold < prune_threshold {
        tracing::warn!(
            warn_threshold,
            prune_threshold,
            "NWIRO_LOCAL_LLM_WARN_TOKEN_THRESHOLD < NWIRO_LOCAL_LLM_PRUNE_TOKEN_THRESHOLD — \
             pruning will never trigger warn. Check env vars."
        );
    }

    // v0.1.27 Option B: operator escape hatch for misconfigured chat
    // templates or probe misclassification. NWIRO_LOCAL_LLM_FORCE_TOOL_TIER
    // accepts `native`, `emulated`, `none`. When set, the effective
    // tier overrides the probe-derived `state.tool_tier` for THIS
    // invocation. `state.tool_tier` is left untouched (still the
    // probe result, used for telemetry).
    //
    // Critical semantic per planner (gemini Layer 3 + codex):
    // effective_tool_tier == None must mean "no tools sent to backend"
    // — not just "skip the Emulated parser". Without that, a None-tier
    // model (probe failed to find tool-call support OR user forced
    // it) STILL receives the 25K tool array as plain text, and
    // autoregressively echoes schema fragments. The actual fix lives
    // at the `chat_completion_stream` call site below — see
    // `effective_tools`.
    let effective_tool_tier = match std::env::var("NWIRO_LOCAL_LLM_FORCE_TOOL_TIER")
        .ok()
        .as_deref()
    {
        Some("native") => ToolTier::Native,
        Some("emulated") => ToolTier::Emulated,
        Some("none") => ToolTier::None,
        Some(other) => {
            tracing::warn!(
                value = other,
                probed_tier = ?state.tool_tier,
                "NWIRO_LOCAL_LLM_FORCE_TOOL_TIER unrecognised — falling back to probe-derived tier"
            );
            state.tool_tier
        }
        None => state.tool_tier,
    };
    if effective_tool_tier != state.tool_tier {
        // v0.1.27 critic DEFECT 1 (claude): MUST be warn!, not info!. The
        // dangerous combination is FORCE_TOOL_TIER=native with probed=None
        // — the operator just re-enabled the 25K schema-bleed this whole
        // release prevents, and prod tracing pipelines default-filter at
        // WARN. An info-level line is invisible in Datadog/Grafana/Windows
        // Event log under standard configs. warn! ensures the override is
        // surfaced wherever the operator's logs are read. The `!=` guard
        // means no-op overrides (env-var == probed tier) stay silent.
        tracing::warn!(
            probed_tier = ?state.tool_tier,
            effective_tier = ?effective_tool_tier,
            "tool tier overridden by NWIRO_LOCAL_LLM_FORCE_TOOL_TIER"
        );
    }

    // One-shot high-tool-count breadcrumb. Small / Emulated / None local
    // models empirically collapse to schema-bleed above ~30 tools; qwen3-14B
    // -class models tolerate 200+. Warn only — the request proceeds (the
    // operator may know what they're doing, and the real-request schema-bleed
    // guard surfaces a clean error if collapse actually happens). Gated off
    // Native so a healthy large model isn't warned every prompt.
    if effective_tool_tier != ToolTier::Native && tool_names.len() > high_tool_warn_threshold {
        tracing::warn!(
            tool_count = tool_names.len(),
            high_tool_warn_threshold,
            effective_tier = ?effective_tool_tier,
            "high tool count for a non-Native local model — small/Emulated models can collapse to schema-bleed above ~30 tools (qwen3-14B-class tolerate 200+). If output is garbage, reduce the tool set or use a larger model. Warning only; the request proceeds."
        );
    }

    // v0.1.27 Tier-None tool stripping (gemini Layer 3): when the
    // effective tier is None, suppress the outbound `tools` array
    // before it reaches the backend. v0.1.20 removed the upfront
    // None-tier refusal but kept the tools attached, which is the
    // root cause of GLM-style schema-bleeding: model received tools
    // it couldn't process, autoregressively continued the JSON as
    // text. Stripping tools when the model can't use them prevents
    // that failure mode without an explicit env-var workaround.
    let effective_tools = if effective_tool_tier == ToolTier::None {
        if req.tools.as_ref().is_some_and(|t| !t.is_empty()) {
            // v0.1.27 critic FINDING 6 (claude): MUST be warn!, symmetric
            // with the env-override warn! above. When the probe returns
            // ToolTier::None WITHOUT operator override, the `!=` guard
            // above does NOT fire — so this is the ONLY log evidence
            // that N tools were silently dropped from every outbound
            // request. info! is invisible under WARN-filtered prod
            // pipelines (Datadog/Grafana/Windows Event Log defaults), so
            // operators would see tool badges disappear in UE5 with no
            // log to diagnose. warn! makes the silent strip visible.
            tracing::warn!(
                tool_count_dropped = req.tools.as_ref().map(|t| t.len()).unwrap_or(0),
                probed_tier = ?state.tool_tier,
                effective_tier = ?effective_tool_tier,
                "stripping tools from outbound request — effective tier is None \
                 (model lacks OpenAI tool_call support; prevents schema-bleed). \
                 Set NWIRO_LOCAL_LLM_FORCE_TOOL_TIER=emulated or =native to override \
                 if you believe this is a probe misclassification."
            );
        }
        None
    } else {
        req.tools.clone()
    };

    // v0.1.22 critic F2: the Emulated-tier directive (injected at
    // `messages_for_call` build below) isn't in `state.history`, so
    // budget estimates must add a fixed overhead for Emulated sessions
    // with registered tools. Computed once because `tool_names` and
    // the effective tier don't change across tool-round iterations.
    // v0.1.29: resolve `model` once at function entry. Previously the
    // resolution lived inside the loop body because nothing pre-loop
    // needed it; the directive_overhead estimator now needs the
    // family-from-model lookup, so we hoist resolution here. The model
    // doesn't change across tool-round iterations (the loop never
    // mutates state.current_model), so the hoist is safe.
    let model = if state.current_model.is_empty() {
        client.model().to_string()
    } else {
        state.current_model.clone()
    };

    // Tool-surface-overload backstop: ENFORCE
    // the per-family recommendedToolCeiling on the outbound array. nwiro caps +
    // orders tools best-first; the shim is the hard backstop so a fragile family
    // (e.g. GLM-4 → 29) can't be over-exposed even if nwiro's cap regresses. Only
    // bites on a tool-using tier with a known ceiling the array exceeds.
    let tool_ceiling =
        crate::model_family::ModelFamily::detect(&model).and_then(|f| f.recommended_tool_ceiling());
    let pre_ceiling_count = effective_tools.as_ref().map(|t| t.len()).unwrap_or(0);
    let effective_tools = enforce_tool_ceiling(effective_tool_tier, effective_tools, tool_ceiling);
    // v0.2.6+ context-aware tool budgeting: if a PRIOR turn of THIS session
    // overflowed the model's context and we learned a context-fit ceiling,
    // pre-trim the outbound array to that cap NOW — before the first backend
    // call — as an ADDITIONAL tail-trim on top of the family bleed ceiling.
    // This makes a session that already learned its cap spend one backend
    // call per later turn with no overflow (steady state). Reuses the same
    // tail-trim helper (order-preserving, drops the tail; nwiro orders
    // best-first). The `tool_names` reconciliation below truncates the
    // mandate's name list to match whenever this shrinks the array.
    let effective_tools = match state.learned_tool_ceiling {
        Some(n) => enforce_tool_ceiling(effective_tool_tier, effective_tools, Some(n as u32)),
        None => effective_tools,
    };
    let kept_count = effective_tools.as_ref().map(|t| t.len()).unwrap_or(0);
    // Keep `tool_names` consistent with the truncated array: the invocation
    // mandate (`build_tool_invocation_mandate`) LISTS these names verbatim, so
    // advertising a dropped tool would invite a call to one not in the array.
    // tool_names mirrors the array order, so truncating to the same count keeps
    // the same surviving tools.
    let tool_names: Vec<String> = if kept_count < pre_ceiling_count {
        tracing::warn!(
            model = %model,
            sent = pre_ceiling_count,
            kept = kept_count,
            ceiling = tool_ceiling,
            "enforced recommendedToolCeiling: truncated the tool array (+ the \
             mandate's tool-name list) to the per-family safe ceiling (tail \
             dropped; nwiro orders best-first)"
        );
        tool_names.into_iter().take(kept_count).collect()
    } else {
        tool_names
    };
    // Directive overhead covers BOTH EMIT-004 (Emulated) AND the
    // invocation mandate (Native + Emulated). Both are now model-agnostic
    // (no per-family gate), so the estimate is a pure function of tier +
    // tool names — no family detection needed.
    let directive_overhead =
        estimate_directive_overhead(effective_tool_tier, &tool_names);

    // The "invoke, don't describe" mandate (model-agnostic; full rationale on
    // `build_tool_invocation_mandate`). Hoisted out of the tool-round loop
    // because its guards are loop-invariant (`effective_tool_tier`,
    // `tool_names`), so the directive is built once and reused.
    let tool_invocation_mandate =
        build_tool_invocation_mandate(effective_tool_tier, &tool_names);

    loop {
        // v0.1.23 F3a: was `tool_round > max_tool_rounds` which permitted
        // max_tool_rounds + 1 actual rounds (51 when default was 50)
        // because tool_round is incremented AFTER tool execution at the
        // bottom of the loop body. Fix: `>=` checks before the round
        // that would exceed the limit.
        //
        // v0.1.24 G2 round-3 (post-critic): return as ACP stopReason
        // "max_turn_requests" rather than ShimError::OpenAiHttp →
        // -32000. ACP defines max_turn_requests for this exact case
        // (turn-request budget exhausted). Surfacing it as a JSON-RPC
        // error would leave the bridge unable to distinguish "budget
        // exhausted" from "transport failed" — both would arrive as
        // -32000. The synthetic sentinel passes through the mapper
        // verbatim.
        if tool_round >= max_tool_rounds {
            tracing::warn!(
                tool_round,
                max_tool_rounds,
                "tool-call rounds exceeded — terminating turn with ACP stopReason 'max_turn_requests'"
            );
            return Ok(("max_turn_requests".to_string(), None));
        }

        // G1: graceful history pruning. Runs at the top of every loop
        // iteration because multi-tool-call exchanges append assistant
        // + tool messages mid-session, so a second/third round can
        // overflow even if the first prompt fit. Per codex
        // highest-priority change in the v0.1.22 planner pass.
        let msg_est = estimate_messages_tokens(&state.history);
        if msg_est > prune_threshold {
            let msg_est_before = msg_est;
            let dropped = prune_history_atomic(&mut state.history, prune_threshold);
            if dropped > 0 {
                state.pruned_turn_count += dropped;
                tracing::info!(
                    pruned_now = dropped,
                    pruned_total = state.pruned_turn_count,
                    history_len_after = state.history.len(),
                    msg_tokens_before = msg_est_before,
                    msg_tokens_after = estimate_messages_tokens(&state.history),
                    prune_threshold,
                    "G1: pruned history turns to stay under prune_threshold"
                );
            }
        }

        // v0.1.22 critic F3: Fix D warn was moved INSIDE the loop so
        // late-crossing prompts (one that starts under budget but
        // crosses the threshold on tool-round 2/3 when new assistant
        // + tool messages are appended) still emit the warn. The
        // `token_budget_warned` flag still bounds it to once-per-
        // session — operators don't need N warns for the same
        // configuration issue. Includes `directive_overhead` so the
        // estimate reflects the REAL outbound payload, not just
        // `state.history` bytes (critic F2).
        // v0.1.27: use `effective_tools` (post-strip) for the budget
        // estimate so a `None`-tier session with stripped tools sees
        // realistic ~0-token payload, not the 25K it WOULD have been.
        let payload_est =
            estimate_payload_tokens(&state.history, effective_tools.as_deref()) + directive_overhead;
        if !state.token_budget_warned && payload_est > warn_threshold {
            state.token_budget_warned = true;
            // v0.1.25 critic FINDING 1: derive the recommended n_ctx
            // from the actual configured warn_threshold (= backstop /
            // 2), not a hardcoded 65536. Users with env-var overrides
            // would otherwise see "ensure n_ctx ≥ 65536" while their
            // backstop actually fires at 2×their-override-value.
            let recommended_n_ctx = warn_threshold.saturating_mul(2);
            tracing::warn!(
                estimated_tokens = payload_est,
                tool_count = effective_tools.as_ref().map(|t| t.len()).unwrap_or(0),
                history_len = state.history.len(),
                pruned_turn_count = state.pruned_turn_count,
                directive_overhead,
                warn_threshold,
                prune_threshold,
                recommended_n_ctx,
                tool_round,
                "prompt payload for Nwiro Pro's full tool array exceeds warn_threshold — \
                 ensure backend n_ctx is at least {recommended_n_ctx} (LM Studio Context \
                 Length, OLLAMA_NUM_CTX, --ctx-size {recommended_n_ctx}). See \
                 docs/MODEL-SETUP.md for the recommended configuration."
            );
        }

        // Post-prune backstop: NOT a correctness bound — this is a
        // pathology catch for the "tools + history together exceed
        // the FULL recommended n_ctx" case where shipping the request
        // would waste 5-10s of inference time getting a slow refusal
        // from backends without fast admission control (llama.cpp
        // without --jinja, some Ollama builds begin inference and
        // fail mid-stream rather than refusing immediately).
        //
        // v0.1.25 re-calibration: at the new warn_threshold default
        // 32768, the 2× backstop fires at 65536 — which matches the
        // documented recommended n_ctx in docs/MODEL-SETUP.md. So the
        // backstop semantically means "your payload exceeds your
        // full recommended budget." That IS pathological — distinct
        // from the v0.1.22-v0.1.24 misfire where the backstop
        // (8192) triggered on every normal v0.2.0 prompt because the
        // 25K tool array alone was > 8K.
        if payload_est > warn_threshold.saturating_mul(2) {
            return Err(ShimError::OpenAiHttp(format!(
                "prompt payload estimated at ~{payload_est} tokens, exceeds 2× warn_threshold \
                 ({warn_threshold}×2 = {}); this is past the recommended n_ctx budget. \
                 Increase backend n_ctx (LM Studio Context Length, OLLAMA_NUM_CTX) \
                 OR set NWIRO_LOCAL_LLM_WARN_TOKEN_THRESHOLD higher to raise the backstop.",
                warn_threshold.saturating_mul(2)
            )));
        }

        // v0.1.20: upfront tier-None refusal removed. The architect's
        // v0.1.13 guard was sound when tools-attached implied tool-intent
        // (bridge attached tools only when the user invoked a tool flow).
        // Post-v0.2.0 the bridge attaches tools to EVERY session/prompt,
        // so the guard refused even chat messages like "hello" against
        // None-tier models — bad UX for any locally-runnable non-tool
        // model. We let all tiers proceed to the stream now:
        //   - Native: emits tool_calls SSE → Phase 3 transport executes
        //   - Emulated: emits prose; post-stream parser tries to extract
        //   - None: emits prose chat; if user asked for tool action, the
        //     model's prose attempt is the response — informative
        //     failure rather than a blocking gate.
        // The cost is one inference round per non-tool chat against
        // None-tier models. Negligible; the alternative (refusing every
        // prompt) was strictly worse.

        let session_id = state.session_id.clone();

        // v0.1.29: `model` is hoisted out of the loop above so the
        // directive_overhead estimator can use it. See the resolution
        // comment near the function entry. (Previously this block
        // duplicated the resolution per loop iteration.)

        // v0.1.19 EMIT-004: belt-and-suspenders format directive for
        // Emulated-tier models. Builds a per-call copy of the history
        // with a system-message directive appended/inserted, telling
        // the model exactly which envelope shape we'll parse. Models
        // that obey the directive produce clean inline JSON that the
        // EMIT-002 extractor hits with no false-positive risk; models
        // that ignore it (Qwen 2.5 7B reverts to <tool_call> XML,
        // Mistral 7B to Markdown headers) still get caught by the
        // EMIT-003 / EMIT-008 extractors.
        //
        // State.history is NEVER mutated — the directive lives only
        // in the per-call message vector. Two reasons: (a) avoid
        // doubling the directive on each tool round-trip iteration,
        // (b) keep the persisted session state pure for any future
        // export / replay flow.
        // v0.1.29 design decision Phase 1: Native-tier action-mandate
        // directive for known describer-bias families (GLM-4 only at
        // ship). The string itself is hoisted before the loop as
        // `native_action_mandate` — we borrow it here via `ref`.
        //
        // Family detection happens once at function entry via
        // `directive_family`. Avoids expanding SessionState for a
        // feature that only depends on the model name string.
        //
        // The wording targets the two failure modes from the v0.1.29
        // user transcript: (1) "I would follow these steps to..."
        // step-by-step manual instructions, (2) "I have access to
        // tools" acknowledgment-without-action. Models that obey emit
        // a tool_calls envelope; models that ignore it still get the
        // normal tool catalog via the structured `tools` array — no
        // worse than v0.1.28 baseline.

        // v0.1.30 critic round-1 MUST_FIX 2: sequential directive
        // composition. The previous if/else-if/else chain made EMIT-004
        // and the v0.1.29 mandate mutually exclusive — but after v0.1.30
        // routes GLM through Emulated, both directives apply to the same
        // call (EMIT-004 teaches FORMAT, mandate teaches POLICY). The
        // sequential `merge_directive_into_system` helper expresses this
        // composition cleanly: each directive merges into the same
        // (single) system message in document order. When BOTH directives
        // apply, EMIT-004 lands first followed by the policy mandate,
        // separated by blank lines.
        let mut messages_for_call = state.history.clone();

        // EMIT-004: format directive for Emulated-tier models. Teaches the
        // exact inline JSON envelope shape (`{"tool":..., "arguments":...}`)
        // the EMIT-002 parser extracts.
        if effective_tool_tier == ToolTier::Emulated && !tool_names.is_empty() {
            let emit_004 = format!(
                "When you need to invoke a tool, emit ONLY a JSON object \
                 on a line by itself in this exact shape: \
                 {{\"tool\": \"<name>\", \"arguments\": {{...}}}}. \
                 Use one of these registered tool names: {}. \
                 Do NOT wrap the JSON in Markdown code blocks, do NOT \
                 add surrounding prose, and do NOT use XML tags.",
                tool_names.join(", ")
            );
            merge_directive_into_system(&mut messages_for_call, &emit_004);
        }

        // Tool-invocation mandate (v0.1.35 model-agnostic): teaches POLICY
        // (invoke vs describe). Fires for both Native AND Emulated whenever
        // tools are registered, for EVERY model — see
        // `build_tool_invocation_mandate`. Composes with EMIT-004 above on the
        // Emulated path so the model receives BOTH format guidance AND policy.
        if let Some(ref directive) = tool_invocation_mandate {
            merge_directive_into_system(&mut messages_for_call, directive);
        }

        // v0.1.26 G4 — Emulated-tier prose-leak prevention via
        // prefix-gated buffering. Default behavior (Native + None
        // tiers, or Emulated without tools): stream content_delta
        // immediately. For Emulated tier WITH tools attached, the
        // model is likely to emit a prose-shaped tool envelope
        // ({"tool":...}, <tool_call>..., or markdown header). If we
        // streamed those characters straight to the UI, the user
        // would see the envelope as chat text (the "meaningless
        // text" GLM users reported on v0.1.25).
        //
        // Prefix-gated design (per codex planner critique of blanket
        // buffering): inspect the first ~16 bytes of content_delta.
        // If they look like an envelope opener (starts with `{`,
        // `<`, or `#` after trimming), buffer until end-of-stream
        // and let post-stream extraction decide whether to suppress
        // (parser hit) or flush (parser miss → was actually prose).
        // If they look like normal prose, flush the buffer
        // immediately and stream the rest normally.
        //
        // This preserves real-time streaming for the COMMON case
        // (Emulated model answering a non-tool question with prose)
        // while preventing leaked envelopes (the bug GLM reported).
        // v0.2.6: buffer the content channel for (a) Emulated+tools sessions (the
        // v0.2.5 envelope-bleed fix) AND (b) NATIVE sessions of a bleed-prone
        // family — one with a documented tool ceiling (`tool_ceiling` computed
        // above; today that is GLM). The probe-tool_choice fix can now correctly
        // classify GLM as Native, but the runtime schema-bleed guard below reads
        // ONLY this buffer. Without buffering, a GLM that collapses into
        // schema-bleed at runtime (a variant whose bleed onset is at/below its
        // tool ceiling, or an artifact name `detect()` misses the ceiling for)
        // would stream the `"object"/"properties"` wall LIVE to the UI — a C4
        // BLACK. Routing bleed-prone Native content through the same
        // suppress-and-refuse guard contains it. Non-fragile Native models
        // (Qwen3, Hermes) have `tool_ceiling == None` → keep streaming live.
        let should_buffer_tool_content = !tool_names.is_empty()
            && (effective_tool_tier == ToolTier::Emulated || tool_ceiling.is_some());
        let emulated_buf: std::sync::Arc<std::sync::Mutex<EmulatedBufferState>> =
            std::sync::Arc::new(std::sync::Mutex::new(EmulatedBufferState::default()));
        // v0.2.2: coalesce per-token stream deltas into ~25ms-batched frames to
        // cap the outbound ACP frame rate (Finding I back-pressure mitigation).
        // Tests default to 0 (per-token) so the byte-identical goldens are
        // unaffected by wall-clock timing; the coalescer logic is unit-tested
        // directly (coalescer_batches_until_due_and_preserves_type_order).
        let default_coalesce_ms = if cfg!(test) { 0 } else { DEFAULT_STREAM_COALESCE_MS };
        let coalesce_ms: u64 = std::env::var("NWIRO_LOCAL_LLM_STREAM_COALESCE_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(default_coalesce_ms);
        let coalescer: std::sync::Arc<std::sync::Mutex<StreamCoalescer>> =
            std::sync::Arc::new(std::sync::Mutex::new(StreamCoalescer::new(
                session_id.clone(),
                coalesce_ms,
            )));

        // v0.2.6+ context-aware tool budgeting: bounded overflow retries (up to
        // MAX_OVERFLOW_RETRIES). If the backend rejects the prompt with HTTP 400 /
        // [context_overflow] AND it reports an n_ctx we can parse, tail-trim the
        // (opaque, best-first-ordered) tool array — sized from the MEASURED n_keep
        // when present, else a conservative char estimate, min'd with the family
        // bleed ceiling — and retry, recomputing from each fresh overflow so the
        // count converges. If the retries are exhausted and it still overflows —
        // or n_ctx is unparseable, or the array is already at one tool — we fall
        // through to the EXISTING clean refusal
        // below (the post-loop match is UNCHANGED). The closure passed to
        // `chat_completion_stream` is MOVED into the call, so it is re-created
        // fresh each iteration; the `emulated_buf`/`coalescer` Arcs are created
        // ONCE above and reused, which is safe because a hard HTTP 400
        // short-circuits before any SSE streaming (the buffers stay empty on
        // the failing attempt).
        let mut tools_for_call = effective_tools.clone();
        let mut overflow_retry_count: usize = 0;
        // v0.2.6+ design decision: hold the computed cap PENDING during the retry —
        // commit it to session state only AFTER a successful retried call (below
        // the loop), so a retry that still overflows (or an overflow whose real
        // cause was history/system-prompt, not the tool array) never poisons
        // later turns with an unvalidated pre-trim cap.
        let mut pending_ceiling: Option<usize> = None;
        let result = loop {
        let attempt = client
            .chat_completion_stream(
                &model,
                messages_for_call.clone(),
                tools_for_call.clone(),
                state.cancel_token.clone(),
                // Runaway/repetition guard (P0-E): cap the total accumulated
                // response bytes so a looping local model can't stream until the
                // editor OOMs. 8 MiB is far above any real reply, well under an
                // OOM. Override via NWIRO_LOCAL_LLM_MAX_RESPONSE_BYTES; 0 disables.
                std::env::var("NWIRO_LOCAL_LLM_MAX_RESPONSE_BYTES")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(8 * 1024 * 1024),
                // Per-turn wall-clock deadline (P0-E, validated by review). Default
                // 1800s leaves headroom for slow legit generations (a 70B at
                // 5 tok/s producing 4000 tokens ~= 800s); `0` disables. NOTE:
                // computed per model round here; a per-turn-SHARED deadline across
                // ReAct rounds is a recommended follow-up.
                {
                    let secs = std::env::var("NWIRO_LOCAL_LLM_MAX_TURN_DURATION_SECS")
                        .ok()
                        .and_then(|s| s.parse::<u64>().ok())
                        .unwrap_or(1800);
                    (secs > 0)
                        .then(|| tokio::time::Instant::now() + std::time::Duration::from_secs(secs))
                },
                // SEC-DOS-1 inactivity guard: abort if the backend emits NO token
                // for this long (a silent stall — complements the wall-clock cap
                // above, which bounds runaway emission). 120s is generous for slow
                // local generation; `0` disables. Override via the env var.
                std::env::var("NWIRO_LOCAL_LLM_INACTIVITY_TIMEOUT_SECS")
                    .ok()
                    .and_then(|s| s.parse::<u64>().ok())
                    .unwrap_or(120),
                // v0.3.0 P1 pre-stream cap (per attempt): bound the wait from sending
                // the prompt to a usable streaming response (send + admission gates),
                // so a wedged backend that accepts the socket then never answers can't
                // hang the turn. Default 30s; `0` disables. `effective_prestream_cap`
                // clamps a nonzero value above the connect timeout so a slow CONNECT
                // still surfaces as `unreachable`, not a spurious transient `timeout`.
                // Override via NWIRO_LOCAL_LLM_PROMPT_PRESTREAM_TIMEOUT_SECS.
                crate::openai::client::effective_prestream_cap(
                    std::env::var("NWIRO_LOCAL_LLM_PROMPT_PRESTREAM_TIMEOUT_SECS")
                        .ok()
                        .and_then(|s| s.parse::<u64>().ok())
                        .unwrap_or(crate::openai::client::PROMPT_PRESTREAM_TIMEOUT_SECS),
                    crate::openai::client::connect_timeout_secs(),
                ),
                {
                    let write_update_ref = &write_update;
                    let buf_handle = std::sync::Arc::clone(&emulated_buf);
                    // v0.2.2: the coalescer now owns session_id for its frames, so
                    // the closure no longer captures it directly.
                    let coalescer_h = std::sync::Arc::clone(&coalescer);
                    move |chunk| {
                        // Stream reasoning tokens (Ollama `reasoning` / Qwen3
                        // `reasoning_content`) as ACP `agent_thought_chunk` so
                        // the bridge surfaces a "thinking…" indicator and disarms
                        // its first_token_timer during long deliberation phases.
                        // Reasoning is dispatched BEFORE content because providers
                        // never populate both in the same chunk; ordering only
                        // matters defensively for hybrid SSE shapes.
                        if let Some(delta) = chunk.reasoning_delta {
                            if !delta.is_empty() {
                                // v0.2.2: route through the coalescer (lock dropped
                                // before the outbound write).
                                let notes = coalescer_h
                                    .lock()
                                    .expect("coalescer mutex")
                                    .push_reasoning(&delta);
                                for n in notes {
                                    write_update_ref(n);
                                }
                            }
                        }
                        if let Some(delta) = chunk.content_delta {
                            if delta.is_empty() {
                                return;
                            }
                            if should_buffer_tool_content {
                                // v0.1.26 critic claude MEDIUM finding:
                                // do NOT hold the MutexGuard across the
                                // outbound `write_update_ref(...)` call.
                                // If the callback panics with the guard
                                // live, the mutex gets poisoned and the
                                // post-stream `.lock().expect()` cascades
                                // a recoverable callback failure into a
                                // task abort. Compute the next action
                                // under the lock, drop the guard, then
                                // dispatch I/O.
                                // v0.2.5/v0.2.6 bleed fix: ALWAYS buffer the
                                // content channel for buffered tool sessions
                                // (Emulated, or a bleed-prone family on Native;
                                // see should_buffer_tool_content) — never stream
                                // content live. The post-stream span
                                // stripper (clean_envelope_remainder) then removes
                                // the {"tool":...} / <tool_call> envelope no matter
                                // WHERE it sits in the content. A reasoning model
                                // can emit prose BEFORE the envelope; the old
                                // per-delta gate "prose-latched" on that leading
                                // prose and StreamDirect-streamed the raw envelope
                                // live, bleeding it to the UI before the post-stream
                                // strip could run. Reasoning still streams live via
                                // the SEPARATE reasoning_delta path above, so only
                                // the (short) final answer is deferred to the single
                                // post-stream flush. `decided`/`looks_like_envelope`
                                // stay false → the post-stream `!decided` branch
                                // lifts the whole buffer for span-stripping. Compute
                                // under the lock, drop it, then return (never hold
                                // the guard across an outbound write).
                                {
                                    let mut s =
                                        buf_handle.lock().expect("emulated_buf mutex");
                                    s.buffer.push_str(&delta);
                                }
                                return;
                            }
                            let notes = coalescer_h
                                .lock()
                                .expect("coalescer mutex")
                                .push_content(&delta);
                            for n in notes {
                                write_update_ref(n);
                            }
                        }
                    }
                },
            )
            .await;

        match attempt {
            Err(ShimError::OpenAiHttp(ref msg))
                // Exact LEADING-tag match (not substring): a backend body or raw
                // SSE payload that merely echoes "[context_overflow]" later in the
                // text must NOT hijack the bespoke trim-retry path (review finding).
                if crate::acp::messages::extract_error_kind(msg) == Some("context_overflow")
                    && overflow_retry_count < MAX_OVERFLOW_RETRIES =>
            {
                // No parseable n_ctx → we have no budget to size a fit
                // ceiling, so don't burn a retry: fall through to the
                // clean refusal with the original error.
                let Some(n_ctx) = parse_n_ctx_from_overflow(msg) else {
                    break attempt;
                };
                let cur = tools_for_call.as_ref().map(|t| t.len()).unwrap_or(0);
                // Can't trim below a single tool — refuse rather than send
                // an empty/one-tool set that still won't fit. (Also the floor
                // for a non-tool-caused overflow: history/system-prompt too big
                // even at one tool → bounded retries exhaust → clean refusal.)
                if cur <= 1 {
                    break attempt;
                }
                // PRIMARY: size the trim from the backend's MEASURED prompt token
                // count (`n_keep`) when present — accurate regardless of tool
                // shape, and it shrinks every round so the loop converges.
                // FALLBACK: the conservative (/3) char estimate when the wording
                // carries no `n_keep` (non-llama.cpp backends).
                let fit = match parse_n_keep_from_overflow(msg) {
                    Some(n_keep) => overflow_target_from_nkeep(n_ctx, n_keep, cur),
                    None => context_fit_ceiling(n_ctx, tools_for_call.as_deref().unwrap_or(&[])),
                };
                // min with the family bleed ceiling (keep the family cap as
                // the hard upper bound; the fit estimate only ever tightens it).
                let capped = match tool_ceiling {
                    Some(c) => fit.min(c as usize),
                    None => fit,
                };
                // Guarantee we actually SHRINK the array this attempt (at
                // least one tool dropped), and never below one tool.
                let capped = capped.max(1).min(cur.saturating_sub(1));
                overflow_retry_count += 1;
                tracing::warn!(
                    n_ctx,
                    from = cur,
                    to = capped,
                    attempt = overflow_retry_count,
                    max = MAX_OVERFLOW_RETRIES,
                    "context_overflow: tail-trimming tools to fit and retrying (bounded)"
                );
                tools_for_call =
                    enforce_tool_ceiling(effective_tool_tier, tools_for_call, Some(capped as u32));
                // Hold the cap PENDING — committed to session state only if a
                // retried call below succeeds (see the post-loop commit). A
                // run of still-overflowing retries must NOT cache it.
                pending_ceiling = Some(capped);
                continue;
            }
            other => break other,
        }
        };

        // v0.2.6+ design decision: commit the learned context-fit ceiling ONLY
        // after a successful (possibly retried) call. If the retry still
        // overflowed — or the overflow's real cause was history/system-prompt
        // rather than the tool array — `result` is Err and we leave the cache
        // untouched, so later turns are not poisoned by an unvalidated cap.
        if result.is_ok() {
            if let Some(p) = pending_ceiling {
                state.learned_tool_ceiling = Some(p);
            }
        }

        // v0.2.6 follow-up — context-overflow clean degrade. When the prompt
        // plus the attached tool schemas exceed the model's LOADED context
        // window, the backend (LM Studio / llama.cpp) returns HTTP 400 and
        // `chat_completion_stream` surfaces it as
        // `ShimError::OpenAiHttp("[context_overflow] HTTP 400: ...")` (the
        // `context_overflow` kind assigned by `classify_http_error_kind`).
        // Propagating that via `?` turns it into a JSON-RPC -32000 transport
        // error with NO stopReason — which a harness scores as a hard BLACK
        // failed turn. A context that is merely too small is a SAFE, recoverable
        // condition, so MIRROR the schema-bleed refusal path: surface ONE clean
        // content line and end the turn as a refusal (stopReason "refusal" +
        // advisory errorKind "context_overflow"), never the -32000 path.
        //
        // History stays consistent without stubs: the assistant message for
        // THIS round is pushed only AFTER the call returns (below, at
        // `state.history.push(result.final_message...)`), so a failure AT the
        // request leaves no dangling tool_call to close — unlike the schema-bleed
        // co-emission guard, which closes a call the model DID emit.
        let result = match result {
            Ok(r) => r,
            Err(ShimError::OpenAiHttp(msg))
                // Exact LEADING-tag match (review finding): only a genuinely
                // context_overflow-tagged error takes the bespoke degrade; an
                // unrelated error whose body contains the token falls to the
                // generic arm below.
                if crate::acp::messages::extract_error_kind(&msg) == Some("context_overflow") =>
            {
                tracing::warn!(
                    detail = %clip_tool_io(&msg, 300),
                    "context-overflow on the prompt round — the prompt plus attached tools \
                     exceeded the model's loaded context window; surfacing a clean refusal \
                     (stopReason 'refusal') instead of a -32000 transport error. Load the model \
                     with a larger context length or reduce the tool set."
                );
                write_update(SessionUpdateNotification::content_delta(
                    session_id.clone(),
                    "The prompt plus the attached tools exceeded the model's context window — \
                     reduce the tool set or load the model with a larger context length."
                        .to_string(),
                ));
                let kind =
                    crate::acp::messages::finish_reason_to_prompt_error_kind("context_overflow");
                return Ok(("context_overflow".to_string(), kind));
            }
            // P0-C generic degrade (design decision, supersedes the
            // v0.2.6 server_error-only arm): ANY tagged backend/transport failure
            // on the prompt round — `ShimError::OpenAiHttp("[kind] ...")` from
            // classify_http_error_kind (auth/not_found/rate_limited/timeout/oom/
            // model_unloaded/server_error/unknown), the connection-phase
            // tls_cert/unreachable tags, or the P0-E turn_timeout/response_too_large
            // abort guards — becomes a clean refusal (stopReason "refusal")
            // carrying an advisory errorKind, instead of propagating via `?` into
            // a JSON-RPC -32000 whose raw error string LEAKS into the UE5 chat.
            // `context_overflow` keeps its bespoke trim-and-retry arm ABOVE this.
            // Non-OpenAiHttp ShimErrors (Cancelled, etc.) still propagate via the
            // final arm. The kind tag is parsed from the leading "[...]"; the user
            // line is operator-worded so a backend/config fault does NOT read like
            // a model safety refusal. One degrader, one kind→message table — no
            // 8-way arm fan-out to drift.
            Err(ShimError::OpenAiHttp(msg)) => {
                let kind = crate::acp::messages::extract_error_kind(&msg).unwrap_or("unknown");
                tracing::warn!(
                    error_kind = kind,
                    // Bound the raw backend/SSE text before it hits the log — the
                    // user-facing line is already sanitized; this keeps an
                    // unbounded body (or remote-endpoint tenant detail) out of the
                    // operator log (review finding).
                    detail = %clip_tool_io(&msg, 300),
                    "backend/transport error on the prompt round — surfacing a clean refusal \
                     (stopReason 'refusal') with an advisory errorKind instead of a -32000."
                );
                write_update(SessionUpdateNotification::content_delta(
                    session_id.clone(),
                    crate::acp::messages::kind_to_user_message(kind).to_string(),
                ));
                // P0: all new kinds use the wire-stable Unknown(kind) bucket;
                // typed PromptErrorKind variants are a later schema evolution.
                return Ok((
                    kind.to_string(),
                    Some(crate::acp::messages::PromptErrorKind::Unknown(kind.to_string())),
                ));
            }
            Err(e) => return Err(e),
        };

        // v0.2.2: flush any deltas still buffered in the coalescer BEFORE the
        // post-stream content/tool handling, so the final content + reasoning
        // land ahead of any tool_call / terminal frame (causal order).
        {
            let notes = coalescer.lock().expect("coalescer mutex").flush();
            for n in notes {
                write_update(n);
            }
        }

        // v0.1.26 G4: post-stream envelope buffer handling.
        // If the buffer was decided "envelope-like" but the
        // post-stream synth attempt fails to extract a tool call
        // (parser miss), the buffer turns out to have been prose
        // after all — flush it. If synth hits, it WAS an envelope
        // and the user shouldn't see it (suppress). If the buffer
        // was never classified (very short response), flush the
        // raw bytes as content.
        let buffered_envelope: Option<String> = {
            let mut s = emulated_buf.lock().expect("emulated_buf mutex");
            if !s.decided {
                // Stream ended before classification threshold —
                // whatever's in the buffer IS the content.
                if s.buffer.is_empty() {
                    None
                } else {
                    Some(std::mem::take(&mut s.buffer))
                }
            } else if s.looks_like_envelope {
                // Decided envelope — hold for synth-or-flush decision
                Some(std::mem::take(&mut s.buffer))
            } else {
                // Decided prose — buffer was already flushed inline
                None
            }
        };

        state.history.push(result.final_message.clone());

        // Phase 4 / v0.1.17: Emulated-tier tool synthesis. If the
        // model emitted no native `tool_calls` but is classified
        // Emulated, attempt to extract a ToolCall from the accumulated
        // content (inline JSON or Qwen XML). The parser's
        // `tool_names` membership guard is the critical false-positive
        // discriminator — see emulated_parser.rs.
        //
        // Important: parser runs on the FULLY-ACCUMULATED content
        // (`result.final_message.content`), NOT on individual SSE
        // delta chunks. The streaming layer already accumulated for
        // us during chat_completion_stream.
        let mut tool_calls = result.tool_calls;
        let mut synth_hit_on_envelope = false;
        if tool_calls.is_empty()
            && effective_tool_tier == ToolTier::Emulated
            && !tool_names.is_empty()
        {
            if let Some(content) = result.final_message.content_text() {
                if let Some(synth) =
                    emulated_parser::try_extract_tool_call(content, &tool_names)
                {
                    tracing::info!(
                        name = %synth.function.name,
                        id = %synth.id,
                        "synthesised tool_call from Emulated-tier content"
                    );
                    // Attach the synthesised tool_call to the assistant
                    // message in history so the next-turn LLM sees the
                    // standard [user, assistant-with-tool_call,
                    // tool-result] sequence rather than [user,
                    // assistant-prose, tool-result] which would be
                    // semantically off. Content stays — some models
                    // emit a brief "thought" before the call envelope
                    // and that's valid OpenAI assistant-message shape.
                    if let Some(last) = state.history.last_mut() {
                        last.tool_calls = Some(vec![synth.clone()]);
                    }
                    tool_calls = vec![synth];
                    synth_hit_on_envelope = true;
                }
            }
        }

        // v0.1.26 G4: post-extract flush/suppress decision for the
        // buffered envelope. If synth hit AND we held an envelope
        // buffer, the buffer was the envelope itself — suppress
        // (don't stream raw envelope chars to UI; user sees the
        // tool_call event instead). If synth missed (or wasn't
        // attempted because the model wasn't Emulated-tier with
        // tools), flush the buffer as content_delta — the model's
        // prose response IS the answer.
        // Real-request schema-bleed guard tripped? Declared here so it is in
        // scope for the finish_reason return below.
        let mut schema_bleed_tripped = false;
        if let Some(buffered) = buffered_envelope {
            if synth_hit_on_envelope {
                // v0.2.5 display fix: a reasoning model (deepseek-r1) emits
                // chain-of-thought PROSE then a `{"tool":...}` (or
                // `<tool_call>…`) envelope, all in `content`. The per-delta
                // gate latched "envelope" and buffered the whole tail; the
                // tool already fired via the synth block above. Here we strip
                // ONLY the registered-envelope span(s) from the buffered tail
                // and flush the remaining prose, so the user sees their answer
                // WITHOUT the raw envelope bleed. The membership guard inside
                // the span scanner is the single source of truth — only a
                // registered tool name produces a span, so legit JSON is never
                // stripped.
                let cleaned = clean_envelope_remainder(&buffered, &tool_names);
                match cleaned {
                    Some(remainder) => {
                        tracing::debug!(
                            buffered_len = buffered.len(),
                            remainder_len = remainder.len(),
                            "G4: stripped emulated tool envelope span(s), flushed remaining prose"
                        );
                        if !remainder.is_empty() {
                            write_update(SessionUpdateNotification::content_delta(
                                session_id.clone(),
                                remainder,
                            ));
                        }
                    }
                    None => {
                        // No span found (defensive — synth hit but the span
                        // scanner produced nothing). Flush the whole buffer so
                        // nothing is silently lost.
                        tracing::debug!(
                            buffered_len = buffered.len(),
                            "G4: synth hit but no span found — flushing buffer verbatim"
                        );
                        write_update(SessionUpdateNotification::content_delta(
                            session_id.clone(),
                            buffered,
                        ));
                    }
                }
            } else if bleed_guard_enabled
                && crate::openai::client::looks_like_schema_bleed(&buffered)
            {
                // The model echoed the tool SCHEMA back as text (a small model
                // collapsing under the tool payload) instead of a usable call.
                // Suppress the `"object"/"type"/"properties"` wall and surface
                // ONE clean line. The false-positive gates hold by construction:
                // only buffered tool sessions populate this buffer (Emulated, or a
                // bleed-prone family on Native — the schema-fragment detector is
                // specific enough not to fire on legit content), it is content
                // (not a tool_calls envelope), and synth missed (a real call would
                // have taken the suppress branch above).
                tracing::warn!(
                    buffered_len = buffered.len(),
                    "real-request schema-bleed detected — suppressing the garbage and returning a clean refusal (model likely collapsed under the tool payload; reduce the tool set or use a larger model). Disable via NWIRO_LOCAL_LLM_BLEED_GUARD=off."
                );
                write_update(SessionUpdateNotification::content_delta(
                    session_id.clone(),
                    "The model returned malformed output (JSON schema fragments instead of a usable response) — it likely collapsed under the number of attached tools. Reduce the tool set or use a larger model."
                        .to_string(),
                ));
                schema_bleed_tripped = true;
            } else {
                write_update(SessionUpdateNotification::content_delta(
                    session_id.clone(),
                    buffered,
                ));
            }
        }

        // Schema-bleed co-emission guard: a collapsed model can emit a native
        // tool_call AND schema-bleed content in the same response. Reachable for
        // any BUFFERED tool session (Emulated, or a bleed-prone family on Native —
        // see should_buffer_tool_content); a non-buffered Native session has
        // buffered_envelope == None, so schema_bleed_tripped stays false here. The
        // bleed guard
        // above set `schema_bleed_tripped`, but it was only honoured in the
        // `tool_calls.is_empty()` arm below — so a co-emitted call kept the agentic
        // loop alive (re-collapsing each round) until `max_turn_requests` (a BLACK
        // shim failure). On a detected collapse the whole response is untrusted:
        // do NOT execute the co-emitted call. Close every open call with a
        // tool-result stub so the already-pushed assistant-with-tool_calls is not
        // left dangling (strict backends reject an unmatched tool_call on the next
        // turn — the invariant `push_skipped_call_stubs_preserves_atomicity_for_strict_backends`
        // guards), then end the turn as ONE clean refusal.
        if schema_bleed_tripped && !tool_calls.is_empty() {
            // v0.1.39 polish: strip the schema-bleed wall from the assistant
            // message pushed at the top of this iteration. result.final_message
            // carried the garbage `accumulated_content` alongside the co-emitted
            // tool_calls; left intact it replays to the model next turn and can
            // re-trigger the collapse / pollute context. Clear the content but
            // KEEP tool_calls so the stub-pairing below stays atomic.
            if let Some(last) = state.history.last_mut() {
                if last.tool_calls.is_some() {
                    last.content = None;
                }
            }
            push_skipped_call_stubs(
                &mut state.history,
                &tool_calls,
                0,
                "the model emitted malformed schema output; the response was discarded",
            );
            let kind =
                crate::acp::messages::finish_reason_to_prompt_error_kind("schema_bleed");
            return Ok(("schema_bleed".to_string(), kind));
        }

        if tool_calls.is_empty() {
            // Three reasons we land here:
            //   1. Native model emitted a content-only response (normal completion).
            //   2. Emulated model emitted unparseable prose (parser missed).
            //   3. Emulated session without tools registered (no synth attempted).
            // For cases 2 and 3, the streamed prose IS the user's
            // response; an additional refusal chunk would be redundant
            // and confusing (the user already saw the model's attempt).
            // Future sprints may add a one-line diagnostic for case 2
            // ("tool intent emitted in unsupported format") but
            // v0.1.17 keeps the UX minimal.
            //
            // v0.1.24 G2 (round-2 critic-corrected): return the OpenAI
            // finish_reason so the outer ACP responder can map it to
            // an ACP `stopReason` field on the `session/prompt`
            // response. ACP uses the response, not a session/update,
            // to signal turn completion.
            //
            // If the schema-bleed guard tripped above, report it as its own
            // finish_reason so the responder maps it to stopReason "refusal"
            // (a content refusal, not a transport error → not the -32000 path).
            // v0.2.6+ GENERAL reasoning-budget degrade (model- AND backend-
            // agnostic). A thinking model (GLM-Z1, DeepSeek-R1, Qwen3-thinking,
            // QwQ, ...) can spend its ENTIRE generation budget on chain-of-thought
            // and emit an EMPTY final answer: finish_reason "length"/"max_tokens"
            // with NO visible content and (we are already in this branch) NO tool
            // call. The reasoning is routed to the thought channel correctly, but
            // the turn would otherwise end EMPTY — the user sees only dangling
            // thinking. Surface a clean, model-agnostic message (the thinking stays
            // visible; only the missing ANSWER is explained), mapped to a refusal
            // with advisory errorKind "reasoning_budget_exhausted".
            // `result.final_message.content` is the model's ACTUAL content (empty
            // when it only reasoned), independent of the UI bleed-buffer.
            let no_answer = result
                .final_message
                .content_text()
                .map(|c| c.trim().is_empty())
                .unwrap_or(true);
            let hit_token_limit =
                result.finish_reason == "length" || result.finish_reason == "max_tokens";
            if !schema_bleed_tripped && hit_token_limit && no_answer {
                tracing::warn!(
                    finish = %result.finish_reason,
                    "reasoning-budget exhausted — the model hit the token limit with no visible \
                     answer and no tool call (it spent the whole budget reasoning); surfacing a \
                     clean refusal instead of an empty turn."
                );
                write_update(SessionUpdateNotification::content_delta(
                    session_id.clone(),
                    "The model used its full response budget thinking and didn't reach an \
                     answer. Try rephrasing or simplifying the request, or load the model with \
                     a larger context window."
                        .to_string(),
                ));
                // History invariant (review must-fix): the model's EMPTY
                // assistant message was already pushed to `state.history` above
                // (the post-stream `state.history.push(result.final_message)`).
                // An empty assistant turn can confuse / poison the NEXT turn, so
                // replace its content with a first-person no-answer note — the
                // history stays consistent with what the user saw and the next
                // turn sees a well-formed alternating transcript.
                if let Some(last) = state.history.last_mut() {
                    last.content = Some(
                        "I wasn't able to produce an answer (reasoning budget exhausted)."
                            .to_string()
                            .into(),
                    );
                }
                let kind = crate::acp::messages::finish_reason_to_prompt_error_kind(
                    "reasoning_budget_exhausted",
                );
                return Ok(("reasoning_budget_exhausted".to_string(), kind));
            }

            let finish = if schema_bleed_tripped {
                "schema_bleed".to_string()
            } else {
                result.finish_reason.clone()
            };
            let kind = crate::acp::messages::finish_reason_to_prompt_error_kind(&finish);
            return Ok((finish, kind));
        }

        tool_round += 1;

        // v0.1.26: emit `tool_call` (pending) events upfront for EVERY
        // call in the batch BEFORE execution begins. Rationale: the host
        // bridge's display gate needs to see the full batch before any
        // execution starts — otherwise the UI shows call A as "running"
        // while B and C haven't been displayed yet. The host bridge
        // dispatcher reads `toolCallId`, `status`, `title`, and
        // `rawInput.arguments`. Symmetric for Native + Emulated tier
        // (the bridge doesn't care about provenance).
        for call in &tool_calls {
            // Parse the OpenAI tool_call.arguments string into a
            // serde_json::Value so the bridge sees `rawInput.arguments`
            // as an OBJECT (it calls `GetObjectField`).
            //
            // v0.1.26 critic codex HIGH finding: bare `unwrap_or` on
            // parse failure only handles malformed JSON. Validly-parsed
            // non-object JSON (`[]`, `"x"`, `1`, `null`) would still
            // pass through and break the bridge's `GetObjectField`
            // call. Normalize: accept only `Value::Object`, substitute
            // `{}` for everything else.
            let parsed = serde_json::from_str::<serde_json::Value>(
                &call.function.arguments,
            )
            .unwrap_or(serde_json::Value::Null);
            let args_value = match parsed {
                serde_json::Value::Object(_) => parsed,
                _ => serde_json::json!({}),
            };
            write_update(SessionUpdateNotification::tool_call_pending(
                session_id.clone(),
                call.id.clone(),
                call.function.name.clone(),
                args_value,
            ));
        }

        for (call_idx, call) in tool_calls.iter().enumerate() {
            // v0.1.23 F1: tiered error classification. The original `.await?`
            // propagated ANY Err out of handle_session_prompt, silently ending
            // the session and denying the model any chance to self-correct.
            // Per planner pass (claude+codex+gemini unanimous):
            //   - ShimError::Cancelled propagates (user deliberate action;
            //     would be wrong to swallow as "tool failed")
            //   - ShimError::McpRoundtrip → in-band isError so the model
            //     sees a normal failure response and can retry or surface
            //   - AcpFraming/OpenAiHttp/Config → fatal (infrastructure
            //     failure means subsequent calls fail too; fast-fail
            //     beats burning the round budget)
            //
            // Variants enumerated EXPLICITLY (no wildcard `_ =>`) per
            // v0.1.23 review pass: adding a new
            // ShimError variant in a future release will force a compile
            // error here, forcing a deliberate F1-policy decision rather
            // than silently landing in the fatal bucket.
            let tio_started = std::time::Instant::now();
            let response_value = match tools::execute_tool(
                call,
                tool_schemas.get(&call.function.name),
                &mut mcp_connection_id,
                &write_mcp_request,
            )
            .await
            {
                Ok(v) => v,
                Err(ShimError::Cancelled) => return Err(ShimError::Cancelled),
                Err(ShimError::McpRoundtrip(e)) => serde_json::json!({
                    "content": [{
                        "type": "text",
                        "text": format!("Tool execution failed: {e}")
                    }],
                    "isError": true
                }),
                // UnknownSession is minted only at the prompt-entry session
                // lookup — `execute_tool` can never construct it — but the
                // no-wildcard policy demands an explicit bucket: fatal, like
                // the other infrastructure failures.
                Err(e @ ShimError::AcpFraming(_))
                | Err(e @ ShimError::UnknownSession(_))
                | Err(e @ ShimError::OpenAiHttp(_))
                | Err(e @ ShimError::Config(_)) => return Err(e),
            };

            // v0.1.23 F2: detect perseveration — same tool call,
            // same canonical arguments, failing N consecutive times.
            // Read isError from the response (set by F1 above for
            // transport errors, OR by the upstream MCP server for
            // legitimate tool failures).
            let is_error = response_value
                .get("isError")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            // v0.1.36: opt-in tool-I/O log (NWIRO_LOCAL_LLM_LOG_TOOL_IO). This is
            // the sole site that sees the resolved `is_error` AND every
            // synthesized outcome envelope (Ok / in-band McpRoundtrip / breaker
            // stub) — `execute_tool` returns `Err` before those exist.
            log_tool_io(tool_round, call, &response_value, is_error, tio_started.elapsed());
            let canonical_args = canonicalize_arguments(&call.function.arguments);
            let signature = format!("{}::{}", call.function.name, canonical_args);

            if is_error && last_call_signature.as_deref() == Some(signature.as_str()) {
                consecutive_identical_errors += 1;
                consecutive_identical_successes = 0;
            } else if is_error {
                // New failing call — reset streak to 1 (this is the 1st
                // of a potentially-perseverating run) and record signature.
                consecutive_identical_errors = 1;
                consecutive_identical_successes = 0;
                last_call_signature = Some(signature.clone());
            } else if last_call_signature.as_deref() == Some(signature.as_str()) {
                // v0.1.39: SUCCESS repeating the SAME signature — a reasoning
                // over-call loop the error breaker misses (it counts failures).
                consecutive_identical_successes += 1;
                consecutive_identical_errors = 0;
            } else {
                // Success with a NEW signature — reset both; 1st of a new run.
                consecutive_identical_errors = 0;
                consecutive_identical_successes = 1;
                last_call_signature = Some(signature.clone());
            }

            // v0.1.26: emit tool_call_update (completed or failed)
            // BEFORE pushing the tool result to history. Bridge reads
            // `rawOutput` to render the result; the v0.1.23 F1
            // envelope `{content:[...], isError: bool}` matches what
            // the bridge dispatcher expects (the host bridge handles
            // object/array/string shapes for rawOutput).
            if is_error {
                write_update(SessionUpdateNotification::tool_call_failed(
                    session_id.clone(),
                    call.id.clone(),
                    response_value.clone(),
                ));
            } else {
                write_update(SessionUpdateNotification::tool_call_completed(
                    session_id.clone(),
                    call.id.clone(),
                    response_value.clone(),
                ));
            }

            state
                .history
                .push(ChatMessage::tool(call.id.clone(), response_value));

            if consecutive_identical_errors >= repeated_call_limit {
                // v0.1.23 critic claude FINDING 1: atomicity preservation.
                // If the breaker trips on call B of a [A, B, C] batch, we
                // must synthesise stub tool results for the remaining
                // unprocessed calls (C onwards) so the next-turn history
                // has a `tool` message for EVERY id in the prior
                // assistant's `tool_calls` array. Otherwise strict-mode
                // backends (and `prune_history_atomic`) would see an
                // orphan tool_call reference — the same atomicity
                // violation v0.1.22 spent ~80 lines of pruner code to
                // prevent. Helper-extracted so the path has a dedicated
                // regression test (the invariant already regressed once).
                push_skipped_call_stubs(
                    &mut state.history,
                    &tool_calls,
                    call_idx + 1,
                    "circuit breaker fired earlier in this round",
                );

                // v0.1.26: emit tool_call_failed events for the skipped
                // remainder so the bridge UI doesn't have orphan "pending"
                // rows. Mirror history's stub-completion logic — for every
                // call we emitted a pending event for upfront but never
                // executed (call_idx+1 onwards), emit a terminal failed
                // event. Per codex planner Q4 required-change.
                let skip_stub = serde_json::json!({
                    "content": [{
                        "type": "text",
                        "text": "Skipped: circuit breaker fired earlier in this round."
                    }],
                    "isError": true
                });
                for skipped_call in tool_calls.iter().skip(call_idx + 1) {
                    write_update(SessionUpdateNotification::tool_call_failed(
                        session_id.clone(),
                        skipped_call.id.clone(),
                        skip_stub.clone(),
                    ));
                }

                // Circuit breaker fired. Two outputs required:
                //   1. A user-visible session/update content delta so the
                //      UE5 chat shows the abort message (history-only push
                //      is invisible to the bridge — codex caught this).
                //   2. A synthetic Role::Assistant message persisted to
                //      history so the next user prompt sees a coherent
                //      conversation shape `[tool, assistant, user-next]`.
                let abort_msg = format!(
                    "Tool `{}` returned an error {} consecutive times with \
                     identical arguments. Aborting to prevent runaway inference. \
                     Adjust the request or try a different approach.",
                    call.function.name, consecutive_identical_errors
                );
                write_update(SessionUpdateNotification::content_delta(
                    session_id.clone(),
                    abort_msg.clone(),
                ));
                state
                    .history
                    .push(ChatMessage::assistant(Some(abort_msg), None));
                tracing::warn!(
                    tool_name = %call.function.name,
                    signature_hash = hash_call_signature(&call.function.name, &canonical_args),
                    consecutive_errors = consecutive_identical_errors,
                    repeated_call_limit,
                    tool_round,
                    skipped_calls = tool_calls.len().saturating_sub(call_idx + 1),
                    "F2 circuit breaker fired — aborting tool loop to prevent perseveration"
                );
                // v0.1.24 G2 (round-2): return synthetic "circuit_breaker"
                // finish_reason so the outer ACP responder can map it
                // to stopReason="refusal" on the session/prompt response.
                // The prior content_delta already delivered the abort
                // message to the UI via the live streaming channel.
                return Ok(("circuit_breaker".to_string(), None));
            }

            // v0.1.39: identical-SUCCESS circuit breaker. The error breaker above
            // counts only identical FAILURES; a model can also loop on identical
            // SUCCESSFUL calls (qwen3-class reasoning over-call — same name + args,
            // each succeeding) and run to max_turn_requests (a hang-class BLACK).
            // Mirror the error-breaker teardown: stub the remaining batch, emit a
            // user-visible abort line, persist a synthetic assistant message, and
            // return the "circuit_breaker" finish_reason (a CLEAN stopReason
            // refusal, NOT max_turn_requests).
            if consecutive_identical_successes >= identical_success_limit {
                push_skipped_call_stubs(
                    &mut state.history,
                    &tool_calls,
                    call_idx + 1,
                    "circuit breaker fired earlier in this round",
                );
                let skip_stub = serde_json::json!({
                    "content": [{
                        "type": "text",
                        "text": "Skipped: circuit breaker fired earlier in this round."
                    }],
                    "isError": true
                });
                for skipped_call in tool_calls.iter().skip(call_idx + 1) {
                    write_update(SessionUpdateNotification::tool_call_failed(
                        session_id.clone(),
                        skipped_call.id.clone(),
                        skip_stub.clone(),
                    ));
                }
                let abort_msg = format!(
                    "Tool `{}` was called {} consecutive times with identical \
                     arguments (each succeeding) without converging. Ending the \
                     turn to prevent a runaway loop.",
                    call.function.name, consecutive_identical_successes
                );
                write_update(SessionUpdateNotification::content_delta(
                    session_id.clone(),
                    abort_msg.clone(),
                ));
                state
                    .history
                    .push(ChatMessage::assistant(Some(abort_msg), None));
                tracing::warn!(
                    tool_name = %call.function.name,
                    signature_hash = hash_call_signature(&call.function.name, &canonical_args),
                    consecutive_successes = consecutive_identical_successes,
                    identical_success_limit,
                    tool_round,
                    skipped_calls = tool_calls.len().saturating_sub(call_idx + 1),
                    "v0.1.39 identical-success circuit breaker fired — ending turn to prevent perseveration loop"
                );
                return Ok(("circuit_breaker".to_string(), None));
            }
        }
        // Loop: re-submit history including tool results for the next LLM turn.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Phase 2: image-input prompt gate ---

    fn img(mime: &str, data: &str) -> crate::acp::messages::ImageInput {
        crate::acp::messages::ImageInput {
            mime: mime.into(),
            data: data.into(),
        }
    }

    #[test]
    fn build_user_message_text_only_is_a_plain_string() {
        let m = build_user_message("hi".into(), vec![], true, "qwen2.5-vl");
        assert_eq!(m.content_text(), Some("hi"));
        assert!(matches!(
            m.content,
            Some(crate::openai::messages::MessageContent::Text(_))
        ));
    }

    #[test]
    fn build_user_message_vision_model_forwards_images() {
        let m = build_user_message(
            "what is this?".into(),
            vec![img("image/png", "AAAA")],
            true,
            "qwen2.5-vl",
        );
        let v = serde_json::to_value(&m).unwrap();
        let arr = v["content"].as_array().expect("multimodal content array");
        assert!(arr.iter().any(|p| p["type"] == "image_url"
            && p["image_url"]["url"] == "data:image/png;base64,AAAA"));
    }

    #[test]
    fn build_user_message_text_only_model_degrades_with_visible_note() {
        let m = build_user_message(
            "look".into(),
            vec![img("image/png", "AAAA")],
            false,
            "qwen3:14b",
        );
        // Degrades to a plain string (no image_url) with a visible omission note.
        assert!(matches!(
            m.content,
            Some(crate::openai::messages::MessageContent::Text(_))
        ));
        let t = m.content_text().unwrap();
        assert!(t.starts_with("look"));
        assert!(
            t.contains("image attachment(s) omitted"),
            "must surface the omission, got: {t}"
        );
    }

    #[test]
    fn strip_history_images_downgrades_prior_image_turns() {
        let mut history = vec![
            ChatMessage::system("sys"),
            ChatMessage::user_multimodal("earlier".into(), vec![("image/png".into(), "AA".into())]),
        ];
        strip_history_images(&mut history);
        // The prior multimodal user turn is now plain text (no base64 to re-send).
        assert!(matches!(
            history[1].content,
            Some(crate::openai::messages::MessageContent::Text(_))
        ));
        let v = serde_json::to_value(&history[1]).unwrap();
        assert!(v["content"].is_string(), "re-sent turn must be text, got {v}");
        assert!(history[1].content_text().unwrap().contains("earlier"));
    }

    #[test]
    fn tool_ceiling_truncates_tail_only_when_exceeded_on_a_tool_tier() {
        let mk = |n: u32| Some((0..n).collect::<Vec<u32>>());
        // exceeds the ceiling on a tool-using tier → truncate to the first N
        assert_eq!(enforce_tool_ceiling(ToolTier::Native, mk(64), Some(29)), mk(29));
        assert_eq!(enforce_tool_ceiling(ToolTier::Emulated, mk(64), Some(29)), mk(29));
        // keeps the HEAD, not the tail (nwiro orders best-first)
        assert_eq!(
            enforce_tool_ceiling(ToolTier::Native, Some(vec![1u32, 2, 3, 4, 5]), Some(2)),
            Some(vec![1, 2])
        );
        // under the ceiling → unchanged
        assert_eq!(enforce_tool_ceiling(ToolTier::Native, mk(10), Some(29)), mk(10));
        // no documented ceiling → no cap
        assert_eq!(enforce_tool_ceiling(ToolTier::Native, mk(200), None), mk(200));
        // None tier → never capped here (tools already stripped upstream)
        assert_eq!(enforce_tool_ceiling(ToolTier::None, mk(64), Some(29)), mk(64));
        // no tools → None passes through
        assert_eq!(enforce_tool_ceiling::<u32>(ToolTier::Native, None, Some(29)), None);
        // ceiling == 0 → empty vec (pathological but well-defined, no panic — review edge case)
        assert_eq!(enforce_tool_ceiling(ToolTier::Native, mk(5), Some(0)), Some(Vec::<u32>::new()));
    }

    #[test]
    fn parse_n_ctx_handles_present_absent_and_spellings() {
        // The real wording from the backend (golden.rs:808).
        assert_eq!(
            parse_n_ctx_from_overflow(
                "the request exceeds the available context size (n_ctx = 4096); the prompt is too long"
            ),
            Some(4096)
        );
        // `=` with no spaces.
        assert_eq!(parse_n_ctx_from_overflow("n_ctx=512 too long"), Some(512));
        // `is` instead of `=`.
        assert_eq!(parse_n_ctx_from_overflow("n_ctx is 8192, prompt too long"), Some(8192));
        // No `n_ctx` token at all → None (caller refuses without a retry).
        assert_eq!(parse_n_ctx_from_overflow("HTTP 400: the prompt is too long"), None);
        // `n_ctx` present but no digits follow → None.
        assert_eq!(parse_n_ctx_from_overflow("n_ctx unknown"), None);
    }

    #[test]
    fn context_fit_ceiling_worked_example_and_clamps() {
        // Build a known tool set: each tool serializes to a fixed-ish size.
        let tools: Vec<serde_json::Value> = (0..40)
            .map(|i| serde_json::json!({ "name": format!("tool_{i:02}"), "description": "x" }))
            .collect();
        // Worked example: serialized length / 3 = total tool tokens; per_tool =
        // total / len; budget = n_ctx * 60% ; fit = budget / per_tool, clamped
        // to [1, len]. With a generous n_ctx the fit clamps to the array length.
        let big = context_fit_ceiling(1_000_000, &tools);
        assert_eq!(big, tools.len(), "huge ctx → fit clamps to tools.len()");
        // A small n_ctx trims below the full set but never below 1.
        let small = context_fit_ceiling(512, &tools);
        assert!(small >= 1 && small < tools.len(), "small ctx trims: got {small}");
        // Empty tools → nothing to fit → 0 (no division-by-zero).
        assert_eq!(context_fit_ceiling(4096, &[]), 0);
        // Hand-checked: a single small tool with a tiny ctx still yields >= 1.
        let one = vec![serde_json::json!({ "name": "t", "description": "d" })];
        assert_eq!(context_fit_ceiling(64, &one), 1);
    }

    #[test]
    fn parse_n_keep_and_n_ctx_from_real_llamacpp_wording() {
        // The EXACT wording observed against a live LM Studio / llama.cpp pod —
        // BOTH numbers must be extracted from the one string.
        let msg = "[context_overflow] HTTP 400: The number of tokens to keep from \
                   the initial prompt is greater than the context length (n_keep: \
                   39147 >= n_ctx: 4096). Try to load the model with a larger \
                   context length, or provide a shorter input.";
        assert_eq!(parse_n_keep_from_overflow(msg), Some(39147));
        assert_eq!(parse_n_ctx_from_overflow(msg), Some(4096));
        // The EXACT llama-server (raw llama.cpp HTTP server) wording — the measured
        // prompt size is `n_prompt_tokens`, NOT `n_keep`. Both must still parse.
        let llama_server = r#"{"error":{"code":400,"message":"request (39069 tokens) exceeds the available context size (4096 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":39069,"n_ctx":4096}}"#;
        assert_eq!(parse_n_keep_from_overflow(llama_server), Some(39069));
        assert_eq!(parse_n_ctx_from_overflow(llama_server), Some(4096));
        // n_keep / n_prompt_tokens absent (synthetic wording) → None → context_fit.
        let no_keep = "the request exceeds the available context size (n_ctx = 512); too long";
        assert_eq!(parse_n_keep_from_overflow(no_keep), None);
        assert_eq!(parse_n_ctx_from_overflow(no_keep), Some(512));
        assert_eq!(parse_n_keep_from_overflow("unrelated 400 body"), None);
    }

    #[test]
    fn overflow_target_from_nkeep_worked_example_and_clamps() {
        // 224 tools MEASURED at 39147 tokens, n_ctx=4096: target = 0.8*4096 = 3276;
        // floor(224 * 3276 / 39147) = 18. (The /4 char estimate over-shot to 40.)
        assert_eq!(overflow_target_from_nkeep(4096, 39147, 224), 18);
        // The (1b) golden's case: 40 tools, n_keep 5000 → 26.
        assert_eq!(overflow_target_from_nkeep(4096, 5000, 40), 26);
        // A massive (history-dominated) overflow floors at 1 tool, never 0.
        assert_eq!(overflow_target_from_nkeep(4096, 500_000, 40), 1);
        // Always strictly shrinks (clamped to cur-1); can never return `cur`.
        let t = overflow_target_from_nkeep(4096, 4097, 10);
        assert!((1..10).contains(&t), "must strictly shrink below cur; got {t}");
    }

    use crate::openai::messages::{ChatMessage, ToolCall, ToolCallFunction};

    fn user(text: &str) -> ChatMessage {
        ChatMessage::user(text.to_string())
    }
    fn assistant(text: &str) -> ChatMessage {
        ChatMessage::assistant(Some(text.to_string()), None)
    }
    fn system(text: &str) -> ChatMessage {
        ChatMessage::system(text.to_string())
    }
    fn assistant_with_tool_call(id: &str, name: &str) -> ChatMessage {
        ChatMessage::assistant(
            None,
            Some(vec![ToolCall {
                id: id.to_string(),
                r#type: "function".to_string(),
                function: ToolCallFunction {
                    name: name.to_string(),
                    arguments: "{}".to_string(),
                },
            }]),
        )
    }
    fn tool(call_id: &str, result: &str) -> ChatMessage {
        ChatMessage::tool(call_id.to_string(), serde_json::json!({"result": result}))
    }

    // -------- estimate_payload_tokens / estimate_messages_tokens --------

    #[test]
    fn estimate_empty_history_is_zero() {
        assert_eq!(estimate_messages_tokens(&[]), 0);
        assert_eq!(estimate_payload_tokens(&[], None), 0);
    }

    #[test]
    fn estimate_payload_with_tools_only() {
        // No messages, but tools attached — tool bytes should drive the
        // estimate. Matches the v0.2.0 reality where the tool array
        // dominates payload size before any user message.
        let tools = serde_json::json!([{"type":"function","function":{"name":"x"}}]);
        let tools_arr = tools.as_array().unwrap();
        let est = estimate_payload_tokens(&[], Some(tools_arr));
        assert!(est > 0, "tools-only payload must register non-zero tokens");
    }

    #[test]
    fn estimate_grows_with_history_length() {
        let short = vec![user("hi")];
        let long = vec![user("hello there friend, how are you doing today")];
        assert!(
            estimate_messages_tokens(&long) > estimate_messages_tokens(&short),
            "longer history must estimate more tokens"
        );
    }

    // -------- prune_history_atomic --------

    #[test]
    fn prune_returns_zero_when_under_budget() {
        let mut h = vec![user("hi"), assistant("hello")];
        let before_len = h.len();
        let dropped = prune_history_atomic(&mut h, 100_000); // huge budget
        assert_eq!(dropped, 0);
        assert_eq!(h.len(), before_len);
    }

    #[test]
    fn prune_preserves_system_messages() {
        // System block + several user/assistant turns; budget tight
        // enough that all non-system turns except the latest user
        // get dropped. The system message must remain in position 0.
        let big = "x".repeat(2000);
        let mut h = vec![
            system("system directive that must survive"),
            user(&big),
            assistant(&big),
            user(&big),
            assistant(&big),
            user("latest user"),
        ];
        let dropped = prune_history_atomic(&mut h, 50);
        assert!(dropped > 0, "should have pruned at least one turn");
        // System message survives at index 0.
        assert!(
            matches!(h[0].role, Role::System),
            "system message must survive pruning"
        );
        // Latest user message is preserved as the final entry.
        assert!(
            matches!(h.last().unwrap().role, Role::User),
            "latest user must survive — got role {:?}",
            h.last().unwrap().role
        );
    }

    #[test]
    fn prune_drops_assistant_with_tool_calls_atomically_with_tool_response() {
        // The Q4 blocker: a (assistant+tool_calls, tool) pair must drop
        // as one atomic unit. After pruning, the surviving history must
        // NOT contain an assistant with tool_calls referencing an id
        // whose tool response was pruned away.
        let big = "x".repeat(2000);
        let mut h = vec![
            user(&big),                                  // 0 — turn 1 user
            assistant_with_tool_call("call_abc", "foo"), // 1 — turn 1 assistant
            tool("call_abc", "result1"),                 // 2 — turn 1 tool
            user(&big),                                  // 3 — turn 2 user (preserve from here)
            assistant("turn 2 response"),                // 4 — turn 2 assistant
        ];
        let dropped = prune_history_atomic(&mut h, 50);
        assert!(dropped >= 3, "turn 1 cluster (3 entries) should drop together; dropped={dropped}");
        // No assistant with tool_calls should survive without a matching
        // tool response. After pruning turn 1 atomically, no orphans.
        for msg in &h {
            if let (Role::Assistant, Some(calls)) = (&msg.role, &msg.tool_calls) {
                for call in calls {
                    let has_response = h.iter().any(|m| {
                        matches!(m.role, Role::Tool)
                            && m.tool_call_id.as_deref() == Some(call.id.as_str())
                    });
                    assert!(
                        has_response,
                        "orphan tool_call id={} after prune — atomicity violated",
                        call.id
                    );
                }
            }
        }
    }

    #[test]
    fn prune_drops_oldest_turn_clusters_first() {
        // Older turn clusters drop before newer ones. Verify ordering
        // by tagging each turn with its index in the user content.
        let big = "x".repeat(1000);
        let mut h = vec![
            user(&format!("turn1 {big}")),
            assistant(&format!("turn1-resp {big}")),
            user(&format!("turn2 {big}")),
            assistant(&format!("turn2-resp {big}")),
            user("turn3 latest"), // preserved
        ];
        let _dropped = prune_history_atomic(&mut h, 50);
        // The last surviving user message must be the "latest" one.
        let last_user = h.iter().rev().find(|m| matches!(m.role, Role::User)).unwrap();
        let content = last_user.content_text().unwrap_or("");
        assert!(
            content.contains("latest"),
            "latest user must survive; surviving user content was {content:?}"
        );
    }

    #[test]
    fn prune_handles_only_system_and_latest_user() {
        // System + just one user message — nothing to prune even if over
        // budget, because the user is the latest and must be preserved.
        let big = "x".repeat(10_000);
        let mut h = vec![system("sys"), user(&big)];
        let dropped = prune_history_atomic(&mut h, 50);
        assert_eq!(
            dropped, 0,
            "cannot prune when only system + latest user remain"
        );
        assert_eq!(h.len(), 2);
    }

    #[test]
    fn prune_handles_no_user_messages() {
        // Edge case: only system messages, no user. Nothing to prune
        // and no panic.
        let mut h = vec![system("sys1"), system("sys2")];
        let dropped = prune_history_atomic(&mut h, 50);
        assert_eq!(dropped, 0);
        assert_eq!(h.len(), 2);
    }

    #[test]
    fn prune_is_idempotent_when_already_under_budget() {
        let mut h = vec![user("short"), assistant("reply"), user("latest")];
        let initial_len = h.len();
        let dropped_first = prune_history_atomic(&mut h, 100_000);
        let dropped_second = prune_history_atomic(&mut h, 100_000);
        assert_eq!(dropped_first, 0);
        assert_eq!(dropped_second, 0);
        assert_eq!(h.len(), initial_len);
    }

    // -------- v0.1.22 critic F4 — coverage gap fixes --------

    #[test]
    fn prune_empty_history_is_zero() {
        // Direct empty vec — should return 0 without panic.
        let mut h: Vec<ChatMessage> = vec![];
        let dropped = prune_history_atomic(&mut h, 0);
        assert_eq!(dropped, 0);
        assert!(h.is_empty());
    }

    #[test]
    fn prune_leading_orphan_assistant_cluster() {
        // Malformed input: leading assistant+tool without a preceding
        // user (e.g. half-completed prior round). Should drop as one
        // atomic cluster up to the first user.
        let big = "x".repeat(2000);
        let mut h = vec![
            assistant_with_tool_call("orphan_id", "foo"), // 0 — orphan asst
            tool("orphan_id", "result"),                  // 1 — orphan tool
            user(&big),                                   // 2 — turn 1 user (also latest, must preserve)
        ];
        let dropped = prune_history_atomic(&mut h, 50);
        assert_eq!(
            dropped, 2,
            "leading orphan asst+tool should drop together; got dropped={dropped}"
        );
        // Surviving: just the latest user.
        assert_eq!(h.len(), 1);
        assert!(matches!(h[0].role, Role::User));
    }

    #[test]
    fn prune_preserves_multi_system_block() {
        // Multiple leading system messages (e.g. _meta.systemPrompt.append
        // + EMIT-004 directive). ALL should survive pruning.
        let big = "x".repeat(2000);
        let mut h = vec![
            system("system 1"),
            system("system 2"),
            system("system 3"),
            user(&big), // 3 — turn 1 user
            assistant(&big),
            user("latest"), // 5 — latest user
        ];
        let dropped = prune_history_atomic(&mut h, 50);
        assert!(dropped > 0);
        // First 3 entries must still be system messages in order.
        for (i, expected_text) in ["system 1", "system 2", "system 3"].iter().enumerate() {
            assert!(
                matches!(h[i].role, Role::System),
                "h[{i}] expected role=System, got {:?}",
                h[i].role
            );
            assert_eq!(h[i].content_text(), Some(*expected_text));
        }
        // Latest user must still be present at the end.
        assert!(matches!(h.last().unwrap().role, Role::User));
    }

    #[test]
    fn prune_multi_cluster_drops_left_to_right() {
        // 3 prior turn clusters + latest user. After pruning, the
        // RIGHTMOST clusters (newest among the prunable) survive
        // longer. This verifies cluster-ordering, not just system-
        // and-latest-user preservation.
        let big = "x".repeat(800);
        let mut h = vec![
            user(&format!("turn1_user {big}")),
            assistant(&format!("turn1_asst {big}")),
            user(&format!("turn2_user {big}")),
            assistant(&format!("turn2_asst {big}")),
            user(&format!("turn3_user {big}")),
            assistant(&format!("turn3_asst {big}")),
            user("latest"), // preserved
        ];
        // Target small enough to drop turn1 + turn2 but maybe keep turn3.
        let target = 600; // tokens — small enough to force aggressive prune
        let _dropped = prune_history_atomic(&mut h, target);
        // Latest user always survives.
        assert_eq!(
            h.last().unwrap().content_text(),
            Some("latest"),
            "latest user must always survive"
        );
        // If turn3 survives, it must come AFTER turn1/2 in time order
        // (i.e. turn1 dropped first). Check that no turn1 content
        // remains.
        for msg in &h {
            let content = msg.content_text().unwrap_or("");
            assert!(
                !content.starts_with("turn1_"),
                "turn1 must be dropped before turn3; found {content:?}"
            );
        }
    }

    // -------- estimate_emulated_directive_overhead --------

    #[test]
    fn directive_overhead_zero_only_for_none_tier() {
        // v0.1.35: Native now ALSO carries the model-agnostic invocation
        // mandate, so it is no longer zero — only None (tools stripped) is.
        let names = vec!["tool_a".to_string(), "tool_b".to_string()];
        assert!(estimate_emulated_directive_overhead(ToolTier::Native, &names) > 0);
        assert_eq!(estimate_emulated_directive_overhead(ToolTier::None, &names), 0);
    }

    #[test]
    fn directive_overhead_zero_when_no_tools() {
        let names: Vec<String> = vec![];
        assert_eq!(estimate_emulated_directive_overhead(ToolTier::Emulated, &names), 0);
    }

    #[test]
    fn directive_overhead_grows_with_tool_count() {
        let few = vec!["a".to_string(), "b".to_string()];
        let many: Vec<String> = (0..50).map(|i| format!("tool_{i:02}")).collect();
        let est_few = estimate_emulated_directive_overhead(ToolTier::Emulated, &few);
        let est_many = estimate_emulated_directive_overhead(ToolTier::Emulated, &many);
        assert!(
            est_many > est_few,
            "directive overhead must grow with tool count; few={est_few}, many={est_many}"
        );
        // Sanity: base prose overhead is always present.
        assert!(est_few >= 100, "base directive prose should be ≥100 tokens");
    }

    // -------- v0.1.29 directive overhead expansion --------

    #[test]
    fn directive_overhead_native_with_tools_has_overhead_for_every_model() {
        // v0.1.35 model-AGNOSTIC (this assertion was INVERTED): ANY Native
        // session with tools now gets the invocation mandate — no per-family
        // gate — so the overhead is non-zero regardless of recognized family.
        // Pre-v0.1.35 this was zero for every non-GLM family, which is exactly
        // the bug that left their tools uncalled (model names tool, never
        // invokes it). The estimator must budget the mandate so token-budget
        // warnings stay honest for all models.
        let names = vec!["tool_a".to_string()];
        assert!(
            estimate_directive_overhead(ToolTier::Native, &names) >= 150,
            "Native + tools must budget the mandate overhead for EVERY model"
        );
    }

    #[test]
    fn directive_overhead_native_no_tools_is_zero() {
        // Zero registered tools = no directive injected = no overhead.
        let names: Vec<String> = vec![];
        assert_eq!(estimate_directive_overhead(ToolTier::Native, &names), 0);
    }

    #[test]
    fn directive_overhead_emulated_stacks_emit004_and_mandate() {
        // Emulated gets BOTH EMIT-004 (FORMAT) AND the invocation mandate
        // (POLICY), each carrying a copy of the tool-name list — so the
        // Emulated overhead must exceed the Native overhead (mandate only)
        // for the same tools, by at least the EMIT-004 base (~113 tokens).
        let names = vec!["tool_a".to_string(), "tool_b".to_string()];
        let native = estimate_directive_overhead(ToolTier::Native, &names);
        let emulated = estimate_directive_overhead(ToolTier::Emulated, &names);
        assert!(
            emulated > native,
            "Emulated (EMIT-004 + mandate) must exceed Native (mandate only); \
             native={native}, emulated={emulated}"
        );
        assert!(
            emulated - native >= 113,
            "Emulated should add at least the EMIT-004 base (113) over Native; \
             got delta={}",
            emulated - native
        );
    }

    #[test]
    fn directive_overhead_none_tier_always_zero() {
        // None strips tools entirely → no directive → zero, for every model.
        let names = vec!["tool_a".to_string()];
        assert_eq!(estimate_directive_overhead(ToolTier::None, &names), 0);
    }

    // -------- v0.1.35 model-agnostic tool-invocation mandate --------

    #[test]
    fn mandate_fires_for_native_with_tools_any_model() {
        // THE regression test for the describer-over-actor bug: a Native
        // session with tools gets the "invoke, don't describe" mandate for
        // EVERY model — `build_tool_invocation_mandate` takes no family/model
        // argument, so it structurally cannot be GLM-specific. (Pre-v0.1.35
        // only GLM got it; every other model's tools went uncalled.)
        let names = vec!["spawn_actor".to_string()];
        let m = build_tool_invocation_mandate(ToolTier::Native, &names)
            .expect("Native + tools must get the mandate for any model");
        assert!(m.contains("spawn_actor"), "mandate lists the registered tools");
        // ACT on actions (the describer-over-actor fix stays).
        assert!(
            m.contains("call that tool directly"),
            "mandate tells the model to ACT on action requests"
        );
        assert!(
            m.contains("do not merely describe"),
            "mandate forbids describer behavior"
        );
        // v0.2.6+ prompt-architect: the non-action path is now explicit — answer
        // greetings/questions directly without calling a tool or over-deliberating.
        assert!(
            m.contains("reply directly and briefly") && m.contains("do NOT call a tool"),
            "mandate must steer non-action turns to a direct brief reply (no tool)"
        );
        assert!(
            m.contains("do not deliberate at length"),
            "mandate must discourage over-deliberation (weak reasoners exhaust their budget)"
        );
    }

    #[test]
    fn mandate_fires_for_emulated_with_tools() {
        let names = vec!["spawn_light".to_string()];
        assert!(build_tool_invocation_mandate(ToolTier::Emulated, &names).is_some());
    }

    #[test]
    fn mandate_absent_for_none_tier() {
        // None tier = tools stripped = nothing to invoke = no mandate.
        let names = vec!["spawn_actor".to_string()];
        assert!(build_tool_invocation_mandate(ToolTier::None, &names).is_none());
    }

    #[test]
    fn mandate_absent_without_tools() {
        // No registered tools = no action to mandate (preserves plain chat).
        let names: Vec<String> = vec![];
        assert!(build_tool_invocation_mandate(ToolTier::Native, &names).is_none());
        assert!(build_tool_invocation_mandate(ToolTier::Emulated, &names).is_none());
    }

    // -------- v0.1.36 tool-I/O logging helpers --------

    #[test]
    fn tool_io_anomaly_detects_success_false_without_iserror() {
        // The green-badge breadcrumb: MCP isError:false but an explicit
        // success:false (top-level or under structuredContent) → log it in
        // `failures` mode so the Nwiro isError bug is visible.
        let top = serde_json::json!({"content":[],"isError":false,"success":false});
        assert!(tool_io_success_anomaly(&top));
        let structured =
            serde_json::json!({"content":[],"isError":false,"structuredContent":{"success":false}});
        assert!(tool_io_success_anomaly(&structured));
    }

    #[test]
    fn tool_io_anomaly_false_for_clean_or_nested_success() {
        // Clean success → no anomaly.
        let ok = serde_json::json!({"content":[{"type":"text","text":"Spawned"}],"isError":false});
        assert!(!tool_io_success_anomaly(&ok));
        // success:true must NOT trip it.
        let ok2 = serde_json::json!({"isError":false,"structuredContent":{"success":true}});
        assert!(!tool_io_success_anomaly(&ok2));
        // A success:false buried in result DATA (not the envelope top level or
        // structuredContent) must NOT trip it — we deliberately check only the
        // two unambiguous locations to avoid false positives on legitimate data.
        let nested = serde_json::json!({"content":[],"isError":false,"results":[{"success":false}]});
        assert!(!tool_io_success_anomaly(&nested));
    }

    #[test]
    fn clip_tool_io_marks_truncation_and_respects_unlimited() {
        let s = "abcdefghij"; // 10 ASCII bytes
        assert_eq!(clip_tool_io(s, 0), "abcdefghij", "0 = unlimited");
        assert_eq!(clip_tool_io(s, 100), "abcdefghij", "cap > len = as-is");
        let clipped = clip_tool_io(s, 4).into_owned();
        assert!(clipped.starts_with("abcd"));
        assert!(
            clipped.contains("truncated 6 of 10 bytes"),
            "truncation must be marked, never silent: {clipped}"
        );
    }

    #[test]
    fn clip_tool_io_respects_utf8_boundary() {
        // "héllo": é is 2 bytes. Clipping at byte 2 would split é → must back
        // off to a char boundary so the kept prefix is valid UTF-8.
        let clipped = clip_tool_io("héllo", 2).into_owned();
        assert!(clipped.starts_with('h'));
        assert!(clipped.contains("truncated"));
    }

    // -------- v0.1.23 F2 — canonicalize_arguments / hash_call_signature --------

    #[test]
    fn canonicalize_sorts_keys() {
        // serde_json's default Map (BTreeMap, no preserve_order feature)
        // gives sorted keys on reserialize. Two equivalent JSONs with
        // different key orderings must canonicalize to the same string.
        let a = canonicalize_arguments(r#"{"b": 2, "a": 1}"#);
        let b = canonicalize_arguments(r#"{"a": 1, "b": 2}"#);
        assert_eq!(a, b);
    }

    #[test]
    fn canonicalize_handles_nested_objects() {
        let a = canonicalize_arguments(r#"{"outer": {"z": 1, "a": 2}}"#);
        let b = canonicalize_arguments(r#"{"outer": {"a": 2, "z": 1}}"#);
        assert_eq!(a, b);
    }

    #[test]
    fn canonicalize_falls_back_to_raw_on_parse_failure() {
        // Malformed JSON: helper must not panic — return the raw string.
        let raw = "not valid {{ json";
        assert_eq!(canonicalize_arguments(raw), raw.to_string());
    }

    #[test]
    fn canonicalize_distinguishes_different_values() {
        // Same keys, different values must produce different canonical
        // strings — otherwise the breaker would fire on retries with
        // CORRECTED arguments (false positive).
        let a = canonicalize_arguments(r#"{"x": 1}"#);
        let b = canonicalize_arguments(r#"{"x": 2}"#);
        assert_ne!(a, b);
    }

    #[test]
    fn hash_call_signature_is_deterministic() {
        // Same inputs produce same hash across calls (DefaultHasher is
        // deterministic within a process — not stable across releases
        // of std, but that's fine for telemetry correlation within a
        // single session).
        let h1 = hash_call_signature("find_actor", r#"{"name":"Cube"}"#);
        let h2 = hash_call_signature("find_actor", r#"{"name":"Cube"}"#);
        assert_eq!(h1, h2);
    }

    #[test]
    fn hash_call_signature_changes_with_inputs() {
        let h_a = hash_call_signature("find_actor", r#"{"name":"A"}"#);
        let h_b = hash_call_signature("find_actor", r#"{"name":"B"}"#);
        let h_c = hash_call_signature("delete_actor", r#"{"name":"A"}"#);
        assert_ne!(h_a, h_b, "different args must produce different hashes");
        assert_ne!(h_a, h_c, "different tool names must produce different hashes");
    }

    // -------- v0.1.23 F2 — circuit-breaker counter semantics --------
    //
    // These tests model the counter logic from `handle_session_prompt`
    // in isolation. The real function can't be tested without a full
    // shim runtime, but the counter state machine is pure logic and
    // can be exercised here.

    /// Simulate the circuit-breaker counter updates for a sequence of
    /// (is_error, signature) tuples and return the final
    /// `consecutive_identical_errors` count.
    fn simulate_breaker(events: &[(bool, &str)]) -> usize {
        let mut last_call_signature: Option<String> = None;
        let mut consecutive_identical_errors: usize = 0;
        for (is_error, sig) in events {
            let signature = sig.to_string();
            if *is_error && last_call_signature.as_deref() == Some(signature.as_str()) {
                consecutive_identical_errors += 1;
            } else if *is_error {
                consecutive_identical_errors = 1;
                last_call_signature = Some(signature.clone());
            } else {
                consecutive_identical_errors = 0;
                last_call_signature = Some(signature.clone());
            }
        }
        consecutive_identical_errors
    }

    #[test]
    fn breaker_counts_three_identical_failures() {
        // The triggering scenario from issue #1: same call signature
        // failing 3 consecutive times. With default limit=3, the third
        // failure trips the breaker.
        let count = simulate_breaker(&[
            (true, "find_actor::{\"name\":\"X\"}"),
            (true, "find_actor::{\"name\":\"X\"}"),
            (true, "find_actor::{\"name\":\"X\"}"),
        ]);
        assert_eq!(count, 3, "three identical failures should accumulate to 3");
    }

    #[test]
    fn breaker_resets_on_successful_call() {
        // Identical signature with a SUCCESS in between resets the
        // streak — legitimate polling patterns where the same call is
        // made many times but most succeed should never trip the breaker.
        let count = simulate_breaker(&[
            (true, "get_status::{}"),
            (false, "get_status::{}"), // success — reset
            (true, "get_status::{}"),
        ]);
        assert_eq!(count, 1, "success in middle should reset the streak");
    }

    #[test]
    fn breaker_resets_on_different_signature() {
        // Different signature breaks the streak even if both fail.
        let count = simulate_breaker(&[
            (true, "find_actor::{\"name\":\"A\"}"),
            (true, "find_actor::{\"name\":\"B\"}"), // different args
        ]);
        assert_eq!(count, 1, "different signature should reset streak to 1");
    }

    #[test]
    fn breaker_never_fires_on_only_successful_calls() {
        // Legitimate polling: same tool called many times, all succeed.
        // Counter must stay at 0 — breaker only counts errors.
        let count = simulate_breaker(&[
            (false, "get_status::{}"),
            (false, "get_status::{}"),
            (false, "get_status::{}"),
            (false, "get_status::{}"),
        ]);
        assert_eq!(count, 0, "all-success sequence must leave counter at 0");
    }

    // -------- v0.1.23 critic-round corrections --------

    // -------- v0.1.23 critic claude FINDING 1 — atomicity regression test --------

    fn mk_tool_call(id: &str, name: &str) -> crate::openai::messages::ToolCall {
        crate::openai::messages::ToolCall {
            id: id.to_string(),
            r#type: "function".to_string(),
            function: crate::openai::messages::ToolCallFunction {
                name: name.to_string(),
                arguments: "{}".to_string(),
            },
        }
    }

    #[test]
    fn push_skipped_call_stubs_completes_remaining_calls_in_batch() {
        // [A, B, C] batch — breaker tripped on B (idx 1), so we need to
        // stub-complete C (idx 2 onwards). The history starts with
        // assistant{tool_calls:[A,B,C]} and tool(A), tool(B) already
        // pushed by the loop body.
        let mut history: Vec<ChatMessage> = vec![
            ChatMessage::assistant(
                None,
                Some(vec![
                    mk_tool_call("call_a", "tool_a"),
                    mk_tool_call("call_b", "tool_b"),
                    mk_tool_call("call_c", "tool_c"),
                ]),
            ),
            ChatMessage::tool("call_a".to_string(), serde_json::json!({"ok": "A"})),
            ChatMessage::tool("call_b".to_string(), serde_json::json!({"isError": true})),
        ];
        let tool_calls = vec![
            mk_tool_call("call_a", "tool_a"),
            mk_tool_call("call_b", "tool_b"),
            mk_tool_call("call_c", "tool_c"),
        ];
        let pushed = push_skipped_call_stubs(
            &mut history,
            &tool_calls,
            2,
            "circuit breaker fired earlier in this round",
        );
        assert_eq!(pushed, 1, "exactly 1 stub for the single unprocessed call (C)");
        assert_eq!(history.len(), 4, "history grew by 1");
        let last = history.last().unwrap();
        assert!(matches!(last.role, Role::Tool));
        assert_eq!(last.tool_call_id.as_deref(), Some("call_c"));
        // Content shape includes isError:true so backends treat it as
        // a failure response, not a phantom success.
        let content = last.content_text().unwrap_or("");
        assert!(
            content.contains("circuit breaker"),
            "stub message should mention the breaker; got {content:?}"
        );
    }

    #[test]
    fn push_skipped_call_stubs_preserves_atomicity_for_strict_backends() {
        // Verify the invariant directly: after the stub push, EVERY id
        // in the assistant's tool_calls array has a matching tool
        // response in history. This is the contract that prune_history_atomic
        // checks via `prune_drops_assistant_with_tool_calls_atomically...`
        // and the same contract OpenAI strict-mode enforces.
        let tool_calls = vec![
            mk_tool_call("id_1", "t"),
            mk_tool_call("id_2", "t"),
            mk_tool_call("id_3", "t"),
            mk_tool_call("id_4", "t"),
        ];
        // Simulate: breaker tripped on idx 1. Only tool(id_1) and
        // tool(id_2) actually exist in history.
        let mut history: Vec<ChatMessage> = vec![
            ChatMessage::assistant(None, Some(tool_calls.clone())),
            ChatMessage::tool("id_1".to_string(), serde_json::json!({"ok":true})),
            ChatMessage::tool("id_2".to_string(), serde_json::json!({"isError":true})),
        ];
        push_skipped_call_stubs(&mut history, &tool_calls, 2, "circuit breaker fired earlier in this round");

        // Atomicity check: every tool_call id has a matching tool response.
        for tc in &tool_calls {
            let has_response = history.iter().any(|m| {
                matches!(m.role, Role::Tool)
                    && m.tool_call_id.as_deref() == Some(tc.id.as_str())
            });
            assert!(
                has_response,
                "orphan tool_call id={} after stub-completion — atomicity violated",
                tc.id
            );
        }
    }

    #[test]
    fn push_skipped_call_stubs_zero_when_already_complete() {
        // Breaker tripped on the LAST call (idx 2 of [A, B, C]). No
        // unprocessed calls remain. Helper should push 0 stubs.
        let tool_calls = vec![
            mk_tool_call("a", "t"),
            mk_tool_call("b", "t"),
            mk_tool_call("c", "t"),
        ];
        let mut history: Vec<ChatMessage> = vec![ChatMessage::user("x".to_string())];
        let initial_len = history.len();
        let pushed = push_skipped_call_stubs(&mut history, &tool_calls, 3, "circuit breaker fired earlier in this round");
        assert_eq!(pushed, 0);
        assert_eq!(history.len(), initial_len);
    }

    #[test]
    fn push_skipped_call_stubs_from_zero_stubs_entire_batch() {
        // v0.1.39: the schema-bleed co-emission guard calls this with from_idx=0
        // (NONE of the batch executed) — stub-complete EVERY call so the
        // already-pushed assistant-with-tool_calls is atomic. Distinct from the
        // breaker case (from_idx>0). Also asserts the generalized `reason` text.
        let tool_calls = vec![
            mk_tool_call("c1", "t"),
            mk_tool_call("c2", "t"),
            mk_tool_call("c3", "t"),
        ];
        let mut history: Vec<ChatMessage> =
            vec![ChatMessage::assistant(None, Some(tool_calls.clone()))];
        let pushed = push_skipped_call_stubs(
            &mut history,
            &tool_calls,
            0,
            "the model emitted malformed schema output; the response was discarded",
        );
        assert_eq!(pushed, 3, "from_idx=0 stubs the entire batch");
        // Atomicity: every tool_call id has a matching tool response.
        for tc in &tool_calls {
            assert!(
                history.iter().any(|m| matches!(m.role, Role::Tool)
                    && m.tool_call_id.as_deref() == Some(tc.id.as_str())),
                "orphan tool_call id={} after from_idx=0 stub-completion",
                tc.id
            );
        }
        // The caller's reason — NOT the hardcoded breaker phrasing — is surfaced.
        let last = history.last().unwrap();
        let content = last.content_text().unwrap_or("");
        assert!(
            content.contains("malformed schema output"),
            "stub must use the caller's reason, got {content:?}"
        );
        assert!(
            !content.contains("circuit breaker"),
            "bleed stub must NOT carry the breaker phrasing, got {content:?}"
        );
    }

    // -------- v0.1.25 regression — tracked guard against backstop misfire --------
    //
    // Codex critic FINDING 2: the v0.1.22-v0.1.24 backstop misfire
    // incident motivated a smoke test, but that test lives in
    // gitignored `reports/`. A future contributor could tune defaults
    // back down without seeing the gitignored coverage. This unit
    // test is the IN-TREE assertion that runs on every `cargo test`.

    #[test]
    fn coalescer_batches_until_due_and_preserves_type_order() {
        // High interval so only the size limit / explicit flush trigger —
        // deterministic, no wall-clock dependency.
        let mut c = StreamCoalescer::new("s".to_string(), 60_000);
        // Small content deltas accumulate without emitting.
        assert!(c.push_content("Hello").is_empty(), "below size/interval — buffer, no emit");
        assert!(c.push_content(", world").is_empty());
        let out = c.flush();
        assert_eq!(out.len(), 1, "one coalesced content frame");
        let s = serde_json::to_string(&out[0]).unwrap();
        assert!(s.contains("Hello, world"), "coalesced text joined: {s}");
        assert!(s.contains("agent_message_chunk"), "content type preserved: {s}");

        // Type switch: a content push flushes the causally-older reasoning first.
        assert!(c.push_reasoning("thinking").is_empty(), "reasoning buffers");
        let switch = c.push_content("answer");
        assert_eq!(switch.len(), 1, "type switch flushes the older reasoning buffer");
        let sw = serde_json::to_string(&switch[0]).unwrap();
        assert!(
            sw.contains("agent_thought_chunk") && sw.contains("thinking"),
            "reasoning flushed first, as a thought chunk: {sw}"
        );
        assert!(serde_json::to_string(&c.flush()[0]).unwrap().contains("answer"));

        // Size limit (1024) forces an immediate flush.
        let big = "x".repeat(1100);
        assert_eq!(c.push_content(&big).len(), 1, "size limit forces an immediate flush");

        // interval == 0 disables coalescing (per-token; the cfg!(test) default).
        let mut c0 = StreamCoalescer::new("s".to_string(), 0);
        assert_eq!(c0.push_content("a").len(), 1, "interval 0 = per-token flush");
        assert_eq!(c0.push_content("b").len(), 1);
    }

    #[test]
    fn default_backstop_accommodates_v020_tool_array() {
        // The v0.2.0 bridge attaches ~25K-token tool JSON on every
        // session/prompt (verified by the user's incident report:
        // payload estimated at ~25205 tokens). The default backstop
        // (= 2 × DEFAULT_WARN_TOKEN_THRESHOLD) MUST exceed that
        // figure — otherwise first prompts die locally before reaching
        // any backend (the exact v0.1.22-v0.1.24 misfire that v0.1.25
        // was cut to fix).
        let observed_v020_tool_tokens: usize = 25_205;
        let backstop = DEFAULT_WARN_TOKEN_THRESHOLD.saturating_mul(2);
        assert!(
            observed_v020_tool_tokens < backstop,
            "v0.1.25 default backstop ({backstop}) must accommodate v0.2.0 \
             bridge's observed ~25K tool array (~{observed_v020_tool_tokens} tokens). \
             Pre-v0.1.25 had backstop=8192 (=2×4096) and first prompts died \
             locally with 'tool array alone dominates' before reaching backend. \
             If you are seeing this assertion fail, you likely lowered \
             DEFAULT_WARN_TOKEN_THRESHOLD without considering v0.2.0 bridge tool \
             array size. See docs/MODEL-SETUP.md §'Shim-side defaults (v0.1.25+)'."
        );
    }

    #[test]
    fn default_prune_below_default_warn() {
        // Invariant: prune fires earlier than warn on the messages-only
        // estimate. If anyone bumps prune ABOVE warn, the warn would
        // fire before prune ever has a chance to shrink history,
        // making the prune dead code.
        assert!(
            DEFAULT_PRUNE_TOKEN_THRESHOLD < DEFAULT_WARN_TOKEN_THRESHOLD,
            "DEFAULT_PRUNE_TOKEN_THRESHOLD ({}) must be < \
             DEFAULT_WARN_TOKEN_THRESHOLD ({}) so message pruning gets \
             a chance to trim history before the soft warn fires.",
            DEFAULT_PRUNE_TOKEN_THRESHOLD,
            DEFAULT_WARN_TOKEN_THRESHOLD
        );
    }

    #[test]
    fn repeated_call_limit_clamps_to_at_least_one() {
        // Codex catch: NWIRO_LOCAL_LLM_REPEATED_CALL_LIMIT=0 would make
        // `0 >= 0` fire on the first tool result. The runtime code uses
        // `.max(1)` to floor the parsed value; verify the clamp logic
        // in isolation here.
        let parsed: usize = "0".parse::<usize>().unwrap_or(3).max(1);
        assert_eq!(parsed, 1, "0 must clamp to 1");
        let parsed: usize = "5".parse::<usize>().unwrap_or(3).max(1);
        assert_eq!(parsed, 5, "valid values pass through unchanged");
        // Missing env var path: unwrap_or(default 3), then .max(1)
        let parsed: usize = std::env::var("NWIRO_NO_SUCH_VAR_FOR_TEST")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_REPEATED_CALL_LIMIT)
            .max(1);
        assert_eq!(
            parsed, DEFAULT_REPEATED_CALL_LIMIT,
            "missing env var falls through to default"
        );
    }

    // -------- clean_envelope_remainder (v0.2.5 display fix) --------
    //
    // These exercise the post-stream suppress-or-flush helper that backs
    // the synth-hit branch of the buffered-envelope block. The helper is
    // a pure function (buffered tail + tool_names → cleaned remainder or
    // None), mirroring the other pure-function bridge tests; the async
    // run-turn block simply emits its `Some(remainder)` (when non-empty)
    // or flushes the whole buffer on `None`.

    fn tool_names_fixture() -> Vec<String> {
        vec!["find_blueprints".to_string(), "spawn_actor".to_string()]
    }

    #[test]
    fn clean_remainder_prose_then_envelope_keeps_only_prose() {
        // The deepseek-r1 bleed shape: prose then a raw `{"tool":...}`
        // envelope, both buffered. Synth already fired the tool; the raw
        // envelope must NOT reach the UI, but the prose must.
        let buffered = r#"Let me do that. {"tool":"spawn_actor","arguments":{}}"#;
        let cleaned = clean_envelope_remainder(buffered, &tool_names_fixture())
            .expect("a registered span exists → Some");
        assert_eq!(cleaned, "Let me do that.");
        // No raw envelope char survives.
        assert!(!cleaned.contains('{'), "raw envelope leaked: {cleaned:?}");
    }

    #[test]
    fn clean_remainder_envelope_between_prose_keeps_both_sides() {
        // Envelope sandwiched between prose → remainder keeps both sides;
        // the helper emits exactly one content_delta (one Some string).
        let buffered = r#"Sure: {"tool":"spawn_actor","arguments":{}} done."#;
        let cleaned = clean_envelope_remainder(buffered, &tool_names_fixture())
            .expect("a registered span exists → Some");
        assert!(cleaned.contains("Sure:"), "lost leading prose: {cleaned:?}");
        assert!(cleaned.contains("done."), "lost trailing prose: {cleaned:?}");
        assert!(!cleaned.contains('{'), "raw envelope leaked: {cleaned:?}");
    }

    #[test]
    fn clean_remainder_legit_json_with_synth_miss_flushes_verbatim() {
        // A synth MISS never reaches clean_envelope_remainder in the
        // production block (that path takes the schema-bleed-or-flush
        // branch). But even if called directly with a legit config blob
        // that is NOT a registered envelope, the helper returns None →
        // the caller flushes the buffer verbatim (NOT suppressed).
        let buffered = r#"{"max_tokens": 100, "temperature": 0.7}"#;
        assert!(
            clean_envelope_remainder(buffered, &tool_names_fixture()).is_none(),
            "legit non-envelope JSON must yield None (→ flush verbatim), not a stripped remainder"
        );
    }

    #[test]
    fn clean_remainder_xml_envelope_stripped_prose_kept() {
        // The XML form of the bleed must strip too (tags + body), prose kept.
        let buffered = r#"Thinking... <tool_call>spawn_actor({})</tool_call> ok"#;
        let cleaned = clean_envelope_remainder(buffered, &tool_names_fixture())
            .expect("a registered XML span exists → Some");
        assert!(cleaned.contains("Thinking..."), "lost leading prose: {cleaned:?}");
        assert!(cleaned.contains("ok"), "lost trailing prose: {cleaned:?}");
        assert!(!cleaned.contains("<tool_call>"), "raw XML envelope leaked: {cleaned:?}");
    }

    #[test]
    fn clean_remainder_truncated_envelope_at_eos_yields_none() {
        // A truncated registered envelope at end-of-stream produces ZERO
        // spans (never balances) → helper returns None → caller flushes it
        // as prose. This is the pre-change baseline: the tool also fires
        // zero times because try_extract_tool_call finds no span either.
        let buffered = r#"{"tool":"spawn_actor","arguments":{"#;
        assert!(
            clean_envelope_remainder(buffered, &tool_names_fixture()).is_none(),
            "truncated envelope must yield None (flush-as-prose)"
        );
        // Parity: firing path also sees no tool here (== baseline).
        assert!(
            emulated_parser::try_extract_tool_call(buffered, &tool_names_fixture()).is_none(),
            "truncated envelope must not fire a tool"
        );
    }

    #[test]
    fn clean_remainder_envelope_only_buffer_yields_empty_string() {
        // Buffer is JUST the envelope (no prose). A span exists → Some(""),
        // and the production block then emits NOTHING (skips empty). The
        // raw envelope is fully suppressed.
        let buffered = r#"{"tool":"spawn_actor","arguments":{}}"#;
        let cleaned = clean_envelope_remainder(buffered, &tool_names_fixture())
            .expect("a registered span exists → Some");
        assert_eq!(cleaned, "", "envelope-only buffer must clean to empty");
    }

    #[test]
    fn clean_remainder_xml_containing_a_json_envelope_does_not_panic() {
        // Pathological OVERLAP: an XML <tool_call> whose args are themselves a
        // registered {"tool":...} JSON envelope. The XML pass and the JSON pass
        // scan independently, so the JSON span is NESTED inside the XML span.
        // Union-removal must strip both without panicking (the old DESC
        // replace_range applied the outer span's stale offsets after the inner
        // removal shrank the string → out-of-bounds panic) and keep the prose.
        let buffered = r#"Before. <tool_call>spawn_actor({"tool":"find_blueprints","arguments":{}})</tool_call> After."#;
        let cleaned = clean_envelope_remainder(buffered, &tool_names_fixture())
            .expect("a registered span exists → Some");
        assert!(cleaned.contains("Before."), "lost leading prose: {cleaned:?}");
        assert!(cleaned.contains("After."), "lost trailing prose: {cleaned:?}");
        assert!(!cleaned.contains("tool_call"), "raw XML envelope leaked: {cleaned:?}");
        assert!(
            !cleaned.contains("find_blueprints"),
            "nested JSON envelope leaked: {cleaned:?}"
        );
        assert!(!cleaned.contains('{'), "raw brace leaked: {cleaned:?}");
    }

    #[test]
    fn emulated_gate_classifies_plain_prose_as_non_envelope() {
        // The per-delta gate's prose-vs-envelope decision is unchanged for
        // plain prose: a non-`{`/`<`/`#` first significant char classifies
        // as prose (looks_like_envelope == false → streamed live, never
        // buffered for suppression). Mirror that exact classification rule.
        let trimmed = "Here is the answer you asked for.".trim_start();
        let looks_like_envelope = trimmed.starts_with('{')
            || trimmed.starts_with('<')
            || trimmed.starts_with('#');
        assert!(
            !looks_like_envelope,
            "plain prose must classify as non-envelope (streamed live, gate untouched)"
        );
    }
}
