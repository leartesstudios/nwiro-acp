use crate::acp::messages::{ToolTier, WarmupResult};
use crate::model_family::ModelFamily;
use crate::openai::messages::{
    ChatMessage, ChatResult, Delta, StreamChunk, StreamingResponse, ToolCall,
    ToolCallFunction,
};
use crate::ShimError;
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use reqwest::header::{HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde_json::json;
use std::collections::BTreeMap;
use tokio_util::sync::CancellationToken;
use tracing::instrument;

/// Cloneable HTTP client. `reqwest::Client` is already `Arc`-wrapped internally,
/// so cloning is cheap (just bumps refcounts). The clone-derive is required by
/// `acp/server.rs::handle_session_prompt` which clones the client to avoid a
/// simultaneous &mut self.sessions + &self.client borrow pattern.
#[derive(Clone)]
pub struct Client {
    base_url: String,
    model: String,
    api_key: Option<crate::ApiKey>,
    http: reqwest::Client,
}

/// Default time budget for the one-shot tool-capability probe issued
/// after a successful warmup. 5s comfortably covers a single non-streaming
/// completion against a freshly-warmed local model; on cold-load the
/// probe is allowed to fail (→ `ToolTier::None`) rather than block.
///
/// Overridable at runtime via the `NWIRO_LOCAL_LLM_PROBE_TIMEOUT_SECS`
/// environment variable — set to a higher value (e.g. `30`) when warming
/// partial-offload models that page weights from CPU RAM on first token
/// (Kimi-K2-style 1T MoE quants). Unparseable / unset values fall back
/// to this default, preserving fast-path UX for the common case.
// v0.1.20: bumped 5 → 10 to match the new max_tokens: 256 (design
// decision). Rationale: at ~30 tok/s typical inference, 256
// tokens = ~8.5s. The previous 5s timeout would fire BEFORE the
// model finished emitting the call envelope for any slower-than-
// 51-tok/s backend, defeating the max_tokens bump. 10s gives ~2s
// of headroom on a 30 tok/s baseline. Override via
// NWIRO_LOCAL_LLM_PROBE_TIMEOUT_SECS env var for partial-offload
// configurations (Kimi-K2 etc.) that need even more.
const PROBE_TIMEOUT_SECS: u64 = 10;

// v0.2.1: total timeout for the warmup model-load request. The shared
// reqwest client caps CONNECT only (10s default) — on a serverless cold
// start (RunPod scale-from-zero) the connect succeeds instantly and the
// response then blocks for the ENTIRE model cold load, hanging the UE5
// "Load Model"/Save spinner unboundedly (docs/RUNNING.md "Known gap").
// 300s is generously above any healthy local load (multi-GB GGUF cold
// load is 30-60s; large partial-offload a few minutes) while finally
// bounding the pathological hang with a diagnosable `timeout` errorKind.
// Override via NWIRO_LOCAL_LLM_WARMUP_TIMEOUT_SECS; `0` disables the cap.
const WARMUP_TIMEOUT_SECS: u64 = 300;

/// Extract a human-readable error message from a parsed JSON value, covering
/// the envelope shapes observed across LM Studio, Ollama, llama.cpp server,
/// and FastAPI-based backends. Returns `None` if no plausible message field
/// is present, in which case callers should fall back to surfacing the raw
/// body.
///
/// Priority order (first match wins, by early return):
/// 1. `{"error": {"message": "..."}}` — nested LM Studio / OpenAI shape
/// 2. `{"error": "..."}` — flat-string envelope (some Ollama builds)
/// 3. `{"message": "..."}` — top-level (in-stream LM Studio context-overflow)
/// 4. `{"detail": "..."}` — FastAPI default error shape
///
/// v0.1.21: added to unify the SSE-fallback site with the already-existing
/// extraction patterns in `chat_completion_stream` (Content-Type guard) and
/// `warmup` (status-error path). For v0.1.21 we only call this helper at
/// the NEW SSE-fallback site to avoid functional expansion of the two
/// already-working sites — they'll be unified in a future release once
/// regression tests pin their current behavior.
fn extract_backend_error_message(v: &serde_json::Value) -> Option<String> {
    if let Some(msg) = v.pointer("/error/message").and_then(|m| m.as_str()) {
        return Some(msg.to_string());
    }
    if let Some(msg) = v.pointer("/error").and_then(|e| e.as_str()) {
        return Some(msg.to_string());
    }
    if let Some(msg) = v.pointer("/message").and_then(|m| m.as_str()) {
        return Some(msg.to_string());
    }
    if let Some(msg) = v.pointer("/detail").and_then(|m| m.as_str()) {
        return Some(msg.to_string());
    }
    None
}

/// Classify a prompt-path backend failure into a machine-readable kind so the
/// bridge can branch (auth prompt vs rate-limit notice vs context-overflow)
/// instead of only ever seeing a flat `-32000` (P0-C — the prompt path had no
/// errorKind, unlike the warmup taxonomy). The body is consulted first because
/// the real cause of a 200/4xx envelope often lives in its text (LM Studio
/// "model unloaded", a context-length overflow) rather than the status code.
fn classify_http_error_kind(status: u16, body: &str) -> &'static str {
    let b = body.to_lowercase();
    // Context/token-limit overflow (LM Studio / llama.cpp phrasings). A prompt
    // that no longer fits the model's loaded context window is a CLEAN, safe
    // degrade — the bridge maps this kind to a refusal (stopReason "refusal")
    // rather than letting it surface as a flat -32000 transport failure (which a
    // harness scores as a hard BLACK rather than a recoverable RED). The match
    // set is deliberately specific to context/token-limit wording so an
    // unrelated 400 (bad JSON, unknown field) does NOT get mis-tagged.
    if b.contains("context length")
        || b.contains("context window")
        || b.contains("context the overflows")
        || b.contains("n_ctx")
        || b.contains("exceed context")
        || b.contains("exceeds context")
        || b.contains("prompt is too long")
        || b.contains("maximum context")
        || b.contains("too many tokens")
        || b.contains("tokens exceed")
    {
        return "context_overflow";
    }
    if b.contains("model unloaded") || b.contains("model not loaded") {
        return "model_unloaded";
    }
    // OOM / weight-allocation failure (CUDA / Metal / llama.cpp phrasings). FM-08
    // — the warmup path already sniffs these; mirror it on the prompt path so an
    // out-of-VRAM failure surfaces as `oom` (operator: free memory / smaller
    // model) instead of a generic 5xx → server_error.
    if b.contains("out of memory")
        || b.contains("cuda out of memory")
        || b.contains("failed to allocate")
        || b.contains("cudamalloc")
    {
        return "oom";
    }
    match status {
        401 | 403 => "auth",
        404 => "not_found",
        408 => "timeout",
        429 => "rate_limited",
        500..=599 => "server_error",
        _ => "unknown",
    }
}

/// Derive an HTTP-style status from an in-band `error` object so
/// `classify_http_error_kind` can tag it precisely (Gap-5 hardening, audit
/// MAJOR-B/C). OpenRouter usually echoes the upstream HTTP status as a NUMERIC
/// `code`, but OpenAI-style errors instead carry a STRING `code`
/// (e.g. `"rate_limit_exceeded"`) or only a `type` (`"insufficient_quota"`,
/// `"invalid_request_error"`). Reading the numeric form only (as the original
/// Gap-5 site did) defaulted those to status 200 → body-text classification →
/// a `[unknown]` tag for a real rate-limit/auth failure. Returns `None` when no
/// recognizable discriminator is present, so the caller falls back to body-text
/// classification — never a worse tag than before.
fn http_status_from_error_object(err: &serde_json::Value) -> Option<u16> {
    // Numeric code: the upstream HTTP status, used verbatim.
    if let Some(n) = err.get("code").and_then(|c| c.as_u64()) {
        return Some(n as u16);
    }
    // String code or type: map the common OpenAI/OpenRouter discriminators.
    let tag = err
        .get("code")
        .and_then(|c| c.as_str())
        .or_else(|| err.get("type").and_then(|t| t.as_str()))
        .unwrap_or("")
        .to_lowercase();
    if tag.is_empty() {
        return None;
    }
    if tag.contains("rate_limit") || tag.contains("insufficient_quota") || tag.contains("quota") {
        Some(429)
    } else if tag.contains("invalid_api_key")
        || tag.contains("authentication")
        || tag.contains("unauthorized")
    {
        Some(401)
    } else if tag.contains("permission") || tag.contains("forbidden") {
        Some(403)
    } else if tag.contains("not_found") || tag.contains("model_not_found") {
        Some(404)
    } else if tag.contains("timeout") {
        Some(408)
    } else if tag.contains("server_error") || tag.contains("internal") {
        Some(500)
    } else {
        None
    }
}

/// CEILING on total prompt-path attempts (initial + retries). v0.3.0 P1: raised from
/// 2 to 3 (two retries) now that each attempt's PRE-STREAM phase is time-bounded by
/// `effective_prestream_cap`. With the DEFAULT cap the worst case is safe —
/// `3 × 30s + backoffs ≈ 91s`, well inside nwiro's ~300s first-token watchdog. BUT
/// `effective_prestream_cap` can INFLATE the per-attempt cap above the raw default (it
/// clamps above the connect timeout, and the docs tell cloud operators to raise
/// CONNECT_TIMEOUT to 120s → a 121s cap), so a fixed 3 attempts could blow the
/// watchdog (3×121s ≈ 363s). `effective_max_prestream_attempts` therefore DERIVES the
/// effective count per call, reducing it so `attempts × cap` stays within
/// `PRESTREAM_TOTAL_BUDGET_SECS`. This const is only the ceiling. Transient classes
/// ONLY (`is_transient_kind`); hard / operator-actionable kinds never retry.
const MAX_PROMPT_ATTEMPTS: u32 = 3;

/// Maximum added backoff sleep, in ms, the retry layer may introduce — a hard
/// ceiling so a server-advertised `Retry-After` (or a large exponential step) can
/// never erode the host's first-token watchdog budget.
const MAX_RETRY_BACKOFF_MS: u64 = 2000;

/// Default per-attempt PRE-STREAM timeout (secs): the wall-clock cap from sending
/// the prompt request to the backend producing a usable streaming response. It
/// covers `send()` (time-to-response-headers) AND the small admission-gate body
/// reads that follow it — the non-2xx error body and the LM Studio `200 +
/// application/json` "model unloaded" envelope — but NOT the SSE generation after
/// `break resp`. `connect_timeout` bounds only the TCP/TLS connect; the wait for
/// response headers (and a header-then-stall-body) is otherwise UNBOUNDED, so a
/// wedged backend that accepts the socket then never answers would hang forever.
/// On elapse the attempt is classified `timeout` (transient) and retried. 30s lets
/// a busy local box start (post-warmup the model is already loaded, so real
/// first-header latency is normally sub-second) while still failing a truly wedged
/// backend in well under the watchdog. Override:
/// NWIRO_LOCAL_LLM_PROMPT_PRESTREAM_TIMEOUT_SECS (`0` disables the cap entirely).
pub(crate) const PROMPT_PRESTREAM_TIMEOUT_SECS: u64 = 30;

/// The reqwest connect-timeout (secs) — bounds ONLY the TCP/TLS connect, not the
/// wait for response headers. Read via this helper (shared by `Client::new` and the
/// bridge's pre-stream-cap clamp) so both agree on the default. Override:
/// NWIRO_LOCAL_LLM_CONNECT_TIMEOUT_SECS.
pub(crate) fn connect_timeout_secs() -> u64 {
    std::env::var("NWIRO_LOCAL_LLM_CONNECT_TIMEOUT_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(10)
}

/// Resolve the effective pre-stream cap from a raw (operator-configured or default)
/// value, clamping a NONZERO cap up to at least `connect_timeout + 1s`. Without this
/// clamp, an operator who set the cap BELOW the connect timeout would turn a slow
/// TCP *connect* into a spurious transient `timeout` (which is retried) instead of
/// the correct hard `unreachable` (which is not) — the connect timeout must be able
/// to fire FIRST. `0` (disabled) passes through unchanged. Pure so it is unit-tested
/// directly without racing a process-global env var.
pub(crate) fn effective_prestream_cap(raw_secs: u64, connect_timeout: u64) -> u64 {
    if raw_secs == 0 {
        0
    } else {
        raw_secs.max(connect_timeout.saturating_add(1))
    }
}

/// Total PRE-STREAM wall-clock budget (secs) the bounded retry loop may consume across
/// ALL attempts, sized to stay under nwiro's documented ~300s first-token watchdog with
/// margin for backoffs and the final refusal write. The pre-stream retry loop runs
/// entirely before any token is emitted, so the host's first-token timer is unguarded
/// the whole time — `MAX_PROMPT_ATTEMPTS × cap` must not exceed this.
const PRESTREAM_TOTAL_BUDGET_SECS: u64 = 240;

/// Effective max attempts for a given per-attempt pre-stream `cap_secs`: at most
/// `MAX_PROMPT_ATTEMPTS`, but REDUCED so `attempts × cap` stays within
/// `PRESTREAM_TOTAL_BUDGET_SECS` (never below 1 — one bounded attempt always runs).
/// This reconciles the retry budget with `effective_prestream_cap`'s upward clamp: at
/// the documented cloud config (CONNECT_TIMEOUT=120 → cap 121s) it yields a single
/// attempt (121s ≪ 300s) instead of three (363s > 300s) — so the diagnosable
/// `[timeout]` refusal still reaches the user before the host watchdog fires. The
/// default local config (cap 30s) is unaffected (240/30 = 8, clamped to 3). `cap == 0`
/// (the per-attempt timeout disabled) keeps the full ceiling: without a cap the only
/// retries are FAST HTTP-error retries, which don't consume wall-clock budget. Pure →
/// unit-tested directly.
pub(crate) fn effective_max_prestream_attempts(cap_secs: u64) -> u32 {
    if cap_secs == 0 {
        return MAX_PROMPT_ATTEMPTS;
    }
    ((PRESTREAM_TOTAL_BUDGET_SECS / cap_secs) as u32).clamp(1, MAX_PROMPT_ATTEMPTS)
}

/// Read a response body to a `String`, bounded by the per-attempt pre-stream `deadline`
/// (`None` = unbounded). On a STALLED body — the deadline elapses mid-read — return a
/// placeholder (NOT an error) so the caller classifies the failure by the already-known
/// HTTP status rather than collapsing a header-known FATAL response into a transient
/// `timeout` (review finding). The placeholder is deliberately free of any
/// `classify_http_error_kind` keyword, so a 401 stays `auth` and a 5xx stays
/// `server_error`.
async fn read_body_bounded(
    resp: reqwest::Response,
    deadline: Option<tokio::time::Instant>,
) -> String {
    let fut = resp.text();
    match deadline {
        Some(dl) => match tokio::time::timeout_at(dl, fut).await {
            Ok(Ok(text)) => text,
            Ok(Err(_)) => "<unreadable body>".to_string(),
            Err(_) => "<no response body within the pre-stream cap>".to_string(),
        },
        None => fut.await.unwrap_or_else(|_| "<unreadable body>".to_string()),
    }
}

/// Transient HTTP error kinds that warrant the single pre-stream retry (P0-C).
/// Per the error-taxonomy design: `rate_limited` (429), `timeout` (408) and
/// `server_error` (5xx) only — NOT `unreachable`/`auth`/`not_found`/`tls_cert`/
/// `model_unloaded`, which are hard or operator-actionable failures where an
/// automatic retry would only delay diagnosis. `context_overflow` has its own
/// trim-and-retry path. Mid-stream and post-byte failures are never retried.
fn is_transient_kind(kind: &str) -> bool {
    matches!(kind, "rate_limited" | "timeout" | "server_error")
}

/// Parse a `Retry-After` header in delta-seconds form (the common 429 case). The
/// HTTP-date form is ignored (None) — a local backend emitting an absolute date
/// is vanishingly rare and the policy caps the wait regardless.
fn parse_retry_after_secs(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
}

/// Backoff for a transient retry, EXPONENTIAL in the 0-indexed retry number
/// (`retry_num = attempt - 1`: first retry = 0, second = 1). Non-`rate_limited`
/// kinds use a 250ms base; `rate_limited` WITHOUT a `Retry-After` uses 500ms; both
/// scale as `base * 2^retry_num`. `rate_limited` WITH a delta-seconds `Retry-After`
/// honors that hint directly (NOT exponential — the server told us how long to
/// wait). A small time-derived jitter (0-99ms) de-synchronizes retries without an
/// RNG dependency. Everything is clamped to `MAX_RETRY_BACKOFF_MS` so a server hint
/// (or a large exponent) can never erode the host's first-token watchdog.
fn retry_backoff(kind: &str, retry_after_secs: Option<u64>, retry_num: u32) -> std::time::Duration {
    if kind == "rate_limited" {
        if let Some(secs) = retry_after_secs {
            return std::time::Duration::from_millis(
                secs.saturating_mul(1000).min(MAX_RETRY_BACKOFF_MS),
            );
        }
    }
    let base_ms: u64 = if kind == "rate_limited" { 500 } else { 250 };
    // Cap the exponent so `1 << retry_num` cannot overflow on a pathological count.
    let exp_ms = base_ms.saturating_mul(1u64 << retry_num.min(10));
    let jitter_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| (d.subsec_nanos() % 100) as u64)
        .unwrap_or(0);
    std::time::Duration::from_millis(exp_ms.saturating_add(jitter_ms).min(MAX_RETRY_BACKOFF_MS))
}

/// Flatten a std error and its `source()` chain into one diagnostic string —
/// the cert/DNS/refused cause often lives in a SOURCE of the reqwest error,
/// not its top-level Display.
fn error_chain(e: &dyn std::error::Error) -> String {
    let mut chain = format!("{e}");
    let mut src = e.source();
    while let Some(s) = src {
        chain.push_str(" -> ");
        chain.push_str(&s.to_string());
        src = s.source();
    }
    chain
}

/// Detect a TLS certificate-trust failure in a connection-error chain. rustls
/// trusts the BUNDLED webpki-roots, NOT the OS store, so a corporate
/// TLS-intercepting proxy (Zscaler/Netskope) presents a root rustls does not
/// trust → "invalid peer certificate: UnknownIssuer". This must be classified
/// distinctly (errorKind=tls_cert), not as "unreachable" — which implies the
/// backend is down and points the operator at the wrong knob (G-NET-1).
fn is_tls_cert_error(chain: &str) -> bool {
    let c = chain.to_lowercase();
    c.contains("certificate")
        || c.contains("unknownissuer")
        || c.contains("invalid peer cert")
        || c.contains("webpki")
}

#[derive(Default)]
struct ToolCallAccum {
    id: String,
    name: String,
    arguments: String,
}

/// Probe outcome carrying everything `warmup()` needs to decide whether
/// to accept the session, refuse with family-specific guidance, or fall
/// through to the existing tier-based behavior.
///
/// v0.1.28 design verdict: the bare `ToolTier`
/// return value was insufficient for the GLM-family chat-template
/// problem. The probe sees the symptom (schema-bleed content) and can
/// detect the family from the model name, but warmup() had no way to
/// receive both facts in a single return value. Promoting the return
/// type to a struct keeps the existing tier-based decisions intact
/// while adding two new fields the warmup gate consumes.
///
/// Fields are populated as the probe progresses; an early-error /
/// inconclusive path (network failure, non-2xx, non-JSON) returns
/// `ProbeAssessment::failed(model)` which (v0.1.35) fails OPEN to
/// `tier = Emulated` so tools are not stripped, with
/// `schema_bleed_detected = false` and family-from-name. Family is
/// always set from the model name regardless of network outcome —
/// that's the user's intent (`detect()` is pure).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProbeAssessment {
    /// Tool-call tier classification: Native / Emulated / None.
    pub(crate) tier: ToolTier,
    /// Recognized model family from `ModelFamily::detect(model)`. Set
    /// even on probe-network-error paths so warmup can still emit
    /// family-specific guidance if other signals indicate the gate
    /// should fire.
    pub(crate) family: Option<ModelFamily>,
    /// `true` when probe response content matched
    /// `looks_like_schema_bleed`. The warmup gate fires only when this
    /// is true AND `family.is_some()` — see warmup() implementation.
    pub(crate) schema_bleed_detected: bool,
}

impl ProbeAssessment {
    /// Construct an assessment for early-error / inconclusive probe paths
    /// (network failure, non-2xx, non-JSON). v0.1.35: fails OPEN to
    /// `ToolTier::Emulated`, NOT `None` — a transient probe miss must not
    /// strip tools from every subsequent request for ANY model. Emulated is
    /// a safety superset of None: it still attaches the tools, injects
    /// EMIT-004 + the (now model-agnostic) invocation mandate, and runs the
    /// prose parser; the runtime schema-bleed guard contains any garbage a
    /// genuinely tool-incapable model might emit. Schema-bleed is false (no
    /// content to analyze). Family is detected from the model name regardless
    /// of whether the network call succeeded. Backend-DOWN paths (warmup
    /// send-fail / non-2xx) deliberately stay `None` — that is an
    /// availability signal, not a capability verdict.
    fn failed(model: &str) -> Self {
        Self {
            tier: ToolTier::Emulated,
            family: ModelFamily::detect(model),
            schema_bleed_detected: false,
        }
    }

    /// Definitive "this model has no tool support" verdict — e.g. Ollama's
    /// HTTP 400 `"<model> does not support tools"`. Unlike `failed()` (a
    /// transient / inconclusive miss that fails OPEN to Emulated), this is a
    /// capability verdict: classify `None` so the bridge strips the tools array
    /// and the model degrades to a clean tool-free response, instead of
    /// re-sending the tools and surfacing the backend 400 as a raw `-32000`.
    fn no_tool_support(model: &str) -> Self {
        Self {
            tier: ToolTier::None,
            family: ModelFamily::detect(model),
            schema_bleed_detected: false,
        }
    }
}

/// v0.1.27 telemetry: detect probe responses that look like the model
/// echoed the tool JSON schema back as literal content tokens (the
/// GLM-4.5-air + LM Studio failure mode where the chat template
/// doesn't process OpenAI tool_calls). Used by `probe_tool_capability`
/// to emit a diagnostic `tracing::warn!` when the symptom is detected.
///
/// Heuristic signature (all must hold):
/// 1. Content length ≥ 50 chars (filter out short prose responses)
/// 2. ≥5 occurrences of schema keywords (`object`, `"type"`, `properties`)
/// 3. > 50% of characters are structural (`"`, `:`, `{`, `}`, `[`, `]`,
///    `,`, space)
///
/// The 50% threshold was calibrated against the v0.1.27 user-reported
/// GLM-4.5-air sample (LM Studio + broken chat template), which measured
/// 54% structural with 25 schema-keyword hits. Normal assistant prose
/// sits around 15–20% structural — the gap is wide enough that 50%
/// catches the symptom without firing on healthy output. Conservative
/// design: false positives cost only a misleading log line, not
/// user-visible behavior change.
pub(crate) fn looks_like_schema_bleed(content: &str) -> bool {
    // v0.1.27 review finding 4: both length and ratio gates must
    // count Unicode scalars, not bytes. content.len() returns BYTES, which
    // would let a 50-byte CJK-prefixed string (≈17 chars) clear the length
    // gate spuriously. chars().count() aligns with the gate-3 denominator.
    let char_count = content.chars().count();
    if char_count < 50 {
        return false;
    }
    let object_count = content.matches("object").count();
    let type_count = content.matches("\"type\"").count();
    let properties_count = content.matches("properties").count();
    let schema_keyword_total = object_count + type_count + properties_count;
    if schema_keyword_total < 5 {
        return false;
    }
    let structural: usize = content
        .chars()
        .filter(|c| matches!(c, '"' | ':' | '{' | '}' | '[' | ']' | ',' | ' '))
        .count();
    // char_count was computed by the gate-1 length check above and is
    // guaranteed ≥ 50 here. Reuse it instead of re-traversing content.
    structural * 100 / char_count > 50
}

impl Client {
    #[instrument(skip(api_key))]
    pub fn new(base_url: String, model: String, api_key: Option<crate::ApiKey>) -> Self {
        // IPv4 normalization: on Windows, reqwest's hyper DNS resolves
        // `localhost` and tries IPv6 (::1) before IPv4 (127.0.0.1). Most
        // local LLM servers (Ollama default, llama.cpp default) bind to
        // 127.0.0.1 only — IPv6 connect fails immediately, the request
        // surfaces as `connection error: error sending request`. curl
        // works because it does happy-eyeballs and falls back to v4.
        // We force v4 here by rewriting the host portion. Users who
        // explicitly want IPv6 can write `[::1]` and it stays untouched.
        let base_url = base_url
            .replacen("://localhost:", "://127.0.0.1:", 1)
            .replacen("://localhost/", "://127.0.0.1/", 1);
        // Trailing-slash variant of the bare-host case: `http://localhost`
        let base_url = if base_url.ends_with("://localhost") {
            base_url.replacen("://localhost", "://127.0.0.1", 1)
        } else {
            base_url
        };

        // Connect-phase timeout: a hung TCP-accept (e.g. backend process
        // crashed mid-listen, firewall silently dropping) otherwise leaks
        // the prompt until the user clicks Cancel. Read timeout intentionally
        // left default — local model generation can legitimately take
        // tens of seconds, and the existing 30s MCP timeout + the bridge's
        // first-token-timeout already cap end-to-end waits.
        // Override via NWIRO_LOCAL_LLM_CONNECT_TIMEOUT_SECS (matching the
        // existing env-var pattern in main.rs).
        let http = reqwest::ClientBuilder::new()
            .use_rustls_tls()
            .connect_timeout(std::time::Duration::from_secs(connect_timeout_secs()))
            // SEC-KEY-2 / SEC-SSRF (defense-in-depth): do NOT follow HTTP
            // redirects. reqwest's default chases up to 10 (stripping Authorization
            // on a cross-host hop), but a backend that 3xx-redirects
            // /chat/completions is anomalous, and following one lets a hostile or
            // compromised endpoint bounce the request — carrying the prompt + tool
            // schemas, and on the same host the Authorization bearer — to an
            // arbitrary target (an SSRF vector). With no-follow, a 3xx surfaces as a
            // clean degrade and the request never reaches the redirect target.
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .expect("failed to construct rustls reqwest client");
        Self { base_url, model, api_key, http }
    }

    /// Read-only accessor for the configured endpoint URL. Used by `acp::server`
    /// when it needs to fall back to the client's base_url after an ACP
    /// `initialize` request omitted `params.localLlm.baseUrl`.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Read-only accessor for the configured model name. Used by `acp::server`
    /// for the same reason as `base_url()` — fallback when `initialize` omits
    /// the field.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Stream a chat completion, calling `on_chunk` synchronously for every
    /// content/tool delta. Cancels cleanly if `cancel` is triggered.
    ///
    /// `model` is taken as an explicit argument (NOT read from `self.model`)
    /// so per-session model overrides from `session/set_config_option` flow
    /// through correctly. The Client's `self.model` is the startup default
    /// from initialize; per-session updates live on `SessionState.current_model`.
    /// (Caught by reviewer audit — Bug C in SAVE-CONFIG-DEBUG.md.)
    ///
    /// `tools` is accepted as raw `serde_json::Value` rather than a typed
    /// `Tool` struct — they originate from the bridge's MCP layer and are
    /// already shaped per the OpenAI tool spec. Treating them as opaque JSON
    /// avoids a redundant deserialize+reserialize round-trip and keeps the
    /// shim agnostic to MCP-specific extensions to the tool schema.
    // The orthogonal per-turn guards (cancel, max_response_bytes, deadline,
    // inactivity_secs, prestream_timeout_secs) push this past clippy's 7-arg
    // threshold. They are independent runtime knobs, not a cohesive value object, so
    // folding them into a params struct would obscure each call site more than it
    // clarifies; the threshold is suppressed deliberately.
    #[allow(clippy::too_many_arguments)]
    #[instrument(skip(self, on_chunk), fields(model = %model))]
    pub async fn chat_completion_stream(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<serde_json::Value>>,
        cancel: CancellationToken,
        // Runaway/repetition guard (P0-E): abort once the accumulated response
        // (content + tool-call args) exceeds this many bytes. 0 disables.
        max_response_bytes: usize,
        // Per-turn wall-clock deadline (P0-E, design-validated): the absolute
        // instant at which the turn aborts with `turn_timeout`. `None` disables
        // it. This is the ONLY bound that stops a continuously-emitting
        // repetition loop (the read timeout is off; the host first-token timer
        // disarms after the first token).
        deadline: Option<tokio::time::Instant>,
        // SEC-DOS-1 inactivity guard: abort if the backend emits NO SSE event for
        // this many seconds (a silent STALL — complements the wall-clock `deadline`,
        // which bounds runaway EMISSION). Resets on every received event. `0`
        // disables; the bridge defaults it to 120s.
        inactivity_secs: u64,
        // v0.3.0 P1 pre-stream cap: the per-attempt wall-clock bound (secs) from
        // sending the prompt to a usable streaming response — `send()` plus the
        // small admission-gate body reads, NOT the SSE generation that follows.
        // `0` disables. The bridge resolves this from
        // NWIRO_LOCAL_LLM_PROMPT_PRESTREAM_TIMEOUT_SECS via `effective_prestream_cap`
        // (clamped above the connect timeout); tests pass it directly. See
        // `PROMPT_PRESTREAM_TIMEOUT_SECS`.
        prestream_timeout_secs: u64,
        mut on_chunk: impl FnMut(StreamChunk),
    ) -> crate::Result<ChatResult> {
        // base_url is expected to end at the OpenAI-compat versioned root —
        // e.g. `http://localhost:11434/v1` for Ollama, `http://localhost:1234/v1`
        // for LM Studio. Format only adds `/chat/completions` so we don't double
        // the `/v1` segment (caught by reviewer audit — Bug A in SAVE-CONFIG-DEBUG.md).
        let url = format!(
            "{}/chat/completions",
            self.base_url.trim_end_matches('/')
        );

        let mut body = json!({
            "model": model,
            "messages": messages,
            "stream": true,
            // Ollama-specific: hold the model loaded for 15 minutes after
            // the request, so a normal back-and-forth conversation doesn't
            // pay the cold-load penalty between turns. The default 5min
            // unloads during a single thinking pause for slow reasoning
            // models. LM Studio + cloud passthrough silently ignore unknown
            // fields per the OpenAI permissive-schema convention, so this
            // is safe to send unconditionally for our local backends.
            // (Design decision 3, 2026-05-09.)
            "keep_alive": "15m",
        });

        // v0.2.6+ configurable generation cap (design guardrail). A reasoning
        // model at a large context can fill the WHOLE window with chain-of-thought
        // (latency + waste) before answering — or never answer. This bounds the
        // generation; the `reasoning_budget_exhausted` degrade still handles the
        // no-answer case. Default 16384 is a RUNAWAY guardrail, not a routine
        // ceiling: it clears any normal answer (incl. long code/blueprint dumps)
        // and is a NO-OP for <=16384-context models (the context limits generation
        // first); only truly runaway large-context reasoning is bounded. A
        // genuinely longer answer that needs more can be uncapped with `0`, or the
        // ceiling raised, via NWIRO_LOCAL_LLM_MAX_TOKENS.
        let max_tokens = std::env::var("NWIRO_LOCAL_LLM_MAX_TOKENS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(16384);
        if max_tokens > 0 {
            body["max_tokens"] = serde_json::json!(max_tokens);
        }

        if let Some(tools) = tools {
            if !tools.is_empty() {
                body["tools"] = serde_json::Value::Array(tools);
            }
        }

        let mut req = self.http.post(&url).header(CONTENT_TYPE, "application/json");

        if let Some(ref key) = self.api_key {
            let hval = HeaderValue::from_str(&format!("Bearer {}", key.as_str()))
                .map_err(|e| ShimError::OpenAiHttp(format!("invalid api key header: {e}")))?;
            req = req.header(AUTHORIZATION, hval);
        }

        // v0.3.0 P1: a BOUNDED, per-attempt-TIMED automatic retry for TRANSIENT
        // pre-stream failures (rate_limited / timeout / server_error). Up to
        // MAX_PROMPT_ATTEMPTS attempts; each attempt's PRE-STREAM phase — the request
        // send PLUS the status / Content-Type admission gates (incl. the small error
        // and 200-JSON-envelope body reads) — is wrapped in a `prestream_timeout_secs`
        // cap. This is STRICTLY before any SSE byte is consumed (the stream is built
        // from `response` below), so a retry can never duplicate or tear streamed
        // output. The cap is `tokio::time::timeout`, NOT reqwest's request
        // `.timeout()`, which would also kill the long SSE body. Hard /
        // operator-actionable kinds (auth, not_found, tls_cert, unreachable,
        // model_unloaded) are NOT retried; an automatic retry would only delay
        // diagnosis. `req.try_clone()` succeeds because the JSON body is attached per
        // attempt via `.json(body_ref)`, so the builder itself carries no stream body.
        //
        // The send + admission gates run inside one async block whose result is
        // turned into a value (`PreStream`); the retry/return decision is taken
        // OUTSIDE so `continue`/`return`/`break` are not stranded inside the block.
        enum PreStream {
            // Headers received, 2xx, event-stream: hand the body to the SSE loop.
            Stream(reqwest::Response),
            // A real HTTP error status — eligible for the transient retry by `kind`.
            HttpError {
                kind: &'static str,
                status: String,
                retry_after: Option<u64>,
                body: String,
            },
            // A non-retryable, already-formatted failure (tls_cert / unreachable /
            // the LM Studio 200-JSON "model unloaded" envelope).
            Fatal(ShimError),
            // The pre-stream cap elapsed (a wedged backend, or headers-then-stalled
            // body): classified `timeout` (transient) so it re-enters the retry path.
            Timeout,
        }

        // Derive the effective attempt count from the cap so `attempts × cap` stays
        // within the pre-stream wall-clock budget (an inflated cloud cap reduces
        // attempts; the default 30s cap keeps the full 3). See
        // `effective_max_prestream_attempts`.
        let max_attempts = effective_max_prestream_attempts(prestream_timeout_secs);
        let response = {
            let mut attempt: u32 = 1;
            loop {
                let this_req = req.try_clone().ok_or_else(|| {
                    ShimError::OpenAiHttp(
                        "[unknown] could not clone the request to send".to_string(),
                    )
                })?;
                // Borrow `body` through a Copy reference so the `async move` block
                // captures the REFERENCE (not `body` itself), leaving `body` intact
                // for the next attempt.
                let body_ref = &body;

                // Per-attempt ABSOLUTE deadline, shared by send() AND the admission-gate
                // body reads (`0` = disabled → no deadline). Bounding each await with the
                // SAME deadline — rather than one wrapper around the whole block — is what
                // lets us tell a genuine pre-HEADER stall (status unknown → retriable
                // `timeout`) apart from a stalled BODY read on a response whose status is
                // ALREADY known: a fatal 4xx must NOT be reclassified as a transient
                // `timeout` and retried just because its diagnostic body is slow
                // (review finding).
                let attempt_deadline = (prestream_timeout_secs > 0).then(|| {
                    tokio::time::Instant::now()
                        + std::time::Duration::from_secs(prestream_timeout_secs)
                });

                let outcome = async move {
                    // send() resolves at response HEADERS. Elapsed here → we know NOTHING
                    // about the response → a genuine pre-stream stall (retriable). On
                    // elapse the in-flight request future is dropped, closing the socket;
                    // no SSE byte was consumed, so a retry is safe.
                    let send_fut = this_req.json(body_ref).send();
                    let send_res = match attempt_deadline {
                        Some(dl) => match tokio::time::timeout_at(dl, send_fut).await {
                            Ok(res) => res,
                            Err(_) => return PreStream::Timeout,
                        },
                        None => send_fut.await,
                    };
                    let resp = match send_res {
                        Ok(resp) => resp,
                        Err(e) => {
                            // Strip the request URL from the error BEFORE walking the
                            // source chain. `reqwest::Error::Display` (v0.12.28
                            // src/error.rs:267-269) unconditionally appends
                            // " for url ({url})" when `inner.url` is `Some`. For local
                            // defaults this is harmless, but for user-configured remote
                            // endpoints (Azure OpenAI, etc.) the URL can carry tenant or
                            // deployment identifiers that don't belong in bridge logs.
                            // `without_url()` is the upstream-sanctioned API for this —
                            // see error.rs:14-16. The bearer token is never in the chain
                            // (Inner has no headers field — error.rs:23-27). A
                            // connection failure (DNS / refused / cert) is NOT transient
                            // under the P0 policy: a hard-down or misconfigured host is
                            // surfaced immediately, not masked behind a retry.
                            let e = e.without_url();
                            let chain = error_chain(&e);
                            if is_tls_cert_error(&chain) {
                                return PreStream::Fatal(ShimError::OpenAiHttp(format!(
                                    "[tls_cert] TLS certificate not trusted: {chain} — the shim \
                                     trusts bundled roots (rustls/webpki-roots), not the OS store; a \
                                     corporate TLS-intercepting proxy fails here. Use a direct \
                                     endpoint or trust the proxy root."
                                )));
                            }
                            return PreStream::Fatal(ShimError::OpenAiHttp(format!(
                                "[unreachable] connection error: {chain}"
                            )));
                        }
                    };

                    // Headers are known now. The body read is bounded by the SAME
                    // deadline; if it STALLS, `read_body_bounded` yields a placeholder so
                    // `classify_http_error_kind` keys off the STATUS, not a transient
                    // timeout — a fatal 401/404 stays fatal, a 5xx stays server_error.
                    if !resp.status().is_success() {
                        let status = resp.status();
                        // Capture Retry-After BEFORE consuming the body with text().
                        let retry_after = parse_retry_after_secs(resp.headers());
                        let body_text = read_body_bounded(resp, attempt_deadline).await;
                        let kind = classify_http_error_kind(status.as_u16(), &body_text);
                        return PreStream::HttpError {
                            kind,
                            status: status.to_string(),
                            retry_after,
                            body: body_text,
                        };
                    }

                    // LM Studio returns HTTP 200 with `Content-Type: application/json`
                    // and an error envelope when the requested model is downloaded but
                    // not loaded (e.g. `{"error":{"message":"Model unloaded."}}`).
                    // Branch BEFORE the event-stream parser so the JSON envelope gets a
                    // clean, named surface — classify the body so the bridge sees the
                    // real kind (model_unloaded / not_found / ...) instead of a hostile
                    // "SSE parse error: missing field `choices`". NOT retried: a 200
                    // non-stream response is a hard failure regardless of body content.
                    let content_type = resp
                        .headers()
                        .get(reqwest::header::CONTENT_TYPE)
                        .and_then(|v| v.to_str().ok())
                        .unwrap_or("")
                        .to_ascii_lowercase();
                    if content_type.contains("application/json") {
                        let body_text = read_body_bounded(resp, attempt_deadline).await;
                        // Lift `error.message` out of the envelope so the log carries
                        // the human-readable cause rather than a raw JSON blob.
                        let pretty = serde_json::from_str::<serde_json::Value>(&body_text)
                            .ok()
                            .and_then(|v| {
                                v.get("error")
                                    .and_then(|e| e.get("message"))
                                    .and_then(|m| m.as_str())
                                    .map(|s| s.to_string())
                                    .or_else(|| {
                                        v.get("error").and_then(|e| e.as_str()).map(|s| s.to_string())
                                    })
                            })
                            .unwrap_or_else(|| body_text.clone());
                        let kind = classify_http_error_kind(200, &body_text);
                        return PreStream::Fatal(ShimError::OpenAiHttp(format!(
                            "[{kind}] Local LLM did not stream a response: {pretty}"
                        )));
                    }

                    PreStream::Stream(resp)
                }
                .await;

                match outcome {
                    PreStream::Stream(resp) => break resp,
                    PreStream::Fatal(err) => return Err(err),
                    PreStream::HttpError {
                        kind,
                        status,
                        retry_after,
                        body,
                    } => {
                        if is_transient_kind(kind) && attempt < max_attempts {
                            let backoff = retry_backoff(kind, retry_after, attempt - 1);
                            tracing::warn!(
                                error_kind = kind,
                                attempt,
                                backoff_ms = backoff.as_millis() as u64,
                                "transient backend error on the prompt round — retrying before \
                                 degrading to a clean refusal"
                            );
                            tokio::time::sleep(backoff).await;
                            attempt += 1;
                            continue;
                        }
                        return Err(ShimError::OpenAiHttp(format!(
                            "[{kind}] HTTP {status}: {body}"
                        )));
                    }
                    PreStream::Timeout => {
                        if attempt < max_attempts {
                            let backoff = retry_backoff("timeout", None, attempt - 1);
                            tracing::warn!(
                                error_kind = "timeout",
                                attempt,
                                backoff_ms = backoff.as_millis() as u64,
                                prestream_timeout_secs,
                                "prompt round stalled before streaming (no response within the \
                                 pre-stream cap) — retrying"
                            );
                            tokio::time::sleep(backoff).await;
                            attempt += 1;
                            continue;
                        }
                        return Err(ShimError::OpenAiHttp(format!(
                            "[timeout] backend accepted the connection but sent no response within \
                             {prestream_timeout_secs}s across {max_attempts} attempt(s) \
                             (wedged backend?) — raise NWIRO_LOCAL_LLM_PROMPT_PRESTREAM_TIMEOUT_SECS \
                             or set it to 0 to disable"
                        )));
                    }
                }
            }
        };

        let mut accumulated_content = String::new();
        // G-MODEL-1: strip `<think>...</think>` chain-of-thought that some
        // backends emit inside `delta.content` so it never streams to the UI as
        // the answer. Stateful across chunks (a tag may split across deltas).
        let mut think_stripper = ThinkStripper::default();
        let mut tool_call_accum: BTreeMap<usize, ToolCallAccum> = BTreeMap::new();
        let mut finish_reason = "stop".to_string();

        let mut stream = response.bytes_stream().eventsource();

        // Per-turn wall-clock deadline (P0-E, design-validated): built once (the
        // deadline is absolute). `None` (cap disabled) parks forever via
        // `pending()` so the select arm never fires.
        let turn_deadline = async {
            match deadline {
                Some(d) => tokio::time::sleep_until(d).await,
                None => std::future::pending::<()>().await,
            }
        };
        tokio::pin!(turn_deadline);

        // SEC-DOS-1 (per-token INACTIVITY guard): an ABSOLUTE deadline that RESETS
        // on every received SSE event. Distinct from the wall-clock `deadline`
        // (which bounds runaway EMISSION) and the response-size ceiling — this
        // catches a silent STALL (the backend goes quiet mid-turn). `0` disables.
        let mut inactivity_deadline =
            tokio::time::Instant::now() + std::time::Duration::from_secs(inactivity_secs);

        loop {
            // Fresh inactivity future each iteration, from the (possibly just-reset)
            // absolute deadline. `move` copies the Instant so the stream arm below
            // can still reassign `inactivity_deadline`.
            let dl = inactivity_deadline;
            let inactivity_guard = async move {
                if inactivity_secs > 0 {
                    tokio::time::sleep_until(dl).await
                } else {
                    std::future::pending::<()>().await
                }
            };
            tokio::pin!(inactivity_guard);
            tokio::select! {
                biased;
                _ = cancel.cancelled() => {
                    // Dropping stream cancels the underlying TCP connection.
                    return Err(ShimError::Cancelled);
                }
                _ = &mut turn_deadline => {
                    // Precedence: cancel first, then the wall-clock deadline, then
                    // the inactivity stall guard, then stream events — so a runaway
                    // (or a stall) can never starve the guard arms.
                    return Err(ShimError::OpenAiHttp(
                        "[turn_timeout] turn exceeded the wall-clock cap and was aborted \
                         (runaway/stall guard); raise NWIRO_LOCAL_LLM_MAX_TURN_DURATION_SECS \
                         or set 0 to disable"
                            .to_string(),
                    ));
                }
                _ = &mut inactivity_guard => {
                    return Err(ShimError::OpenAiHttp(format!(
                        "[stream_inactivity_timeout] backend emitted no token for {inactivity_secs}s \
                         and was aborted (stalled endpoint); raise \
                         NWIRO_LOCAL_LLM_INACTIVITY_TIMEOUT_SECS or set 0 to disable"
                    )));
                }
                event = stream.next() => {
                    let event = match event {
                        None => break,
                        Some(Err(e)) => {
                            // Mid-stream transport drop. Tag so the bridge degrades
                            // to a clean refusal + errorKind rather than a raw
                            // -32000 (P0-C). NOT retried — bytes may already have
                            // streamed; a whole-turn retry could tear output.
                            return Err(ShimError::OpenAiHttp(format!(
                                "[unknown] SSE stream error: {e}"
                            )));
                        }
                        Some(Ok(e)) => e,
                    };
                    // SEC-DOS-1: a received event is activity → push the deadline out.
                    if inactivity_secs > 0 {
                        inactivity_deadline = tokio::time::Instant::now()
                            + std::time::Duration::from_secs(inactivity_secs);
                    }

                    if event.data == "[DONE]" {
                        break;
                    }

                    let raw: StreamingResponse = match serde_json::from_str(&event.data) {
                        Ok(r) => r,
                        Err(e) => {
                            // v0.1.21: a backend can emit an error envelope
                            // INSIDE the SSE stream (HTTP 200 + Content-Type:
                            // text/event-stream, then a single `data:` event
                            // whose body is an error JSON rather than a
                            // streaming chunk). The Content-Type guard above
                            // can't catch this — admission control fires
                            // AFTER stream headers are sent. Try to lift the
                            // inner backend message before giving up so the
                            // user sees "The number of tokens to keep ..."
                            // instead of "SSE parse error: missing field
                            // `choices`".
                            if let Ok(v) =
                                serde_json::from_str::<serde_json::Value>(&event.data)
                            {
                                if let Some(msg) = extract_backend_error_message(&v) {
                                    // MAJOR-B: classify from the embedded error
                                    // `code`/`type` instead of a hard-coded 200, so
                                    // a choices-less error chunk carrying a 429/etc.
                                    // is tagged precisely, not [unknown].
                                    let code = v
                                        .pointer("/error")
                                        .and_then(|e| http_status_from_error_object(e))
                                        .unwrap_or(200);
                                    let kind = classify_http_error_kind(code, &msg);
                                    return Err(ShimError::OpenAiHttp(format!(
                                        "[{kind}] backend error mid-stream: {msg}"
                                    )));
                                }
                            }
                            return Err(ShimError::OpenAiHttp(format!(
                                "[unknown] SSE parse error: {e} — raw: {}",
                                &event.data
                            )));
                        }
                    };

                    // Gap 5 (OpenRouter/OpenAI in-band streaming error): a chunk
                    // may carry a top-level `error` object instead of — or, on
                    // OpenRouter, alongside a `finish_reason:"error"` choice —
                    // normal deltas. Such a chunk deserializes cleanly (choices
                    // present), so WITHOUT this guard the error is dropped and the
                    // turn ends as a normal completion, silently masking the
                    // provider failure and any billable retry. Mirror the
                    // choices-less envelope path above: lift the human message,
                    // classify by the embedded HTTP-style `code` the OpenRouter
                    // error object carries, and surface a tagged transport error so
                    // the bridge degrades to a clean refusal + errorKind (NOT
                    // retried — bytes may already have streamed this turn).
                    if let Some(err) = &raw.error {
                        let msg = extract_backend_error_message(err)
                            .unwrap_or_else(|| err.to_string());
                        // MAJOR-C: read a string `code`/`type`, not just a numeric
                        // `code`, so an OpenAI-style discriminator tags correctly
                        // instead of defaulting to 200 -> [unknown].
                        let code = http_status_from_error_object(err).unwrap_or(200);
                        let kind = classify_http_error_kind(code, &msg);
                        return Err(ShimError::OpenAiHttp(format!(
                            "[{kind}] backend error mid-stream: {msg}"
                        )));
                    }

                    for mut choice in raw.choices {
                        // Strip in-content `<think>` markup before it is
                        // accumulated or forwarded; an empty result means this
                        // chunk carried only reasoning, so emit no content.
                        if let Some(c) = choice.delta.content.take() {
                            let kept = think_stripper.push(&c);
                            choice.delta.content =
                                if kept.is_empty() { None } else { Some(kept) };
                        }
                        accumulate_delta(
                            &choice.delta,
                            &mut accumulated_content,
                            &mut tool_call_accum,
                        );

                        if let Some(ref reason) = choice.finish_reason {
                            finish_reason = reason.clone();
                        }

                        // Notify caller with content / reasoning / first tool-call delta.
                        // `reasoning_token()` coalesces Ollama's `delta.reasoning`
                        // and DeepSeek/Qwen3's `delta.reasoning_content` into one
                        // optional string — a single SSE chunk uses one or the
                        // other depending on the provider, never both.
                        let has_content = choice
                            .delta
                            .content
                            .as_deref()
                            .is_some_and(|s| !s.is_empty());
                        let reasoning_delta = choice
                            .delta
                            .reasoning_token()
                            .map(|s| s.to_string());
                        let first_tc = choice
                            .delta
                            .tool_calls
                            .as_ref()
                            .and_then(|tc| tc.first().cloned());

                        if has_content
                            || reasoning_delta.is_some()
                            || first_tc.is_some()
                            || choice.finish_reason.is_some()
                        {
                            on_chunk(StreamChunk {
                                content_delta: choice.delta.content,
                                reasoning_delta,
                                tool_call_delta: first_tc,
                                finish_reason: choice.finish_reason,
                            });
                        }
                    }

                    // Runaway/repetition guard (P0-E): a local model stuck in a
                    // repetition loop streams forever — reqwest's read timeout is
                    // off and the host's first-token timer disarms after the
                    // first token, so nothing else bounds an actively-emitting
                    // stream and `accumulated_content` grows until the editor
                    // OOMs. Abort cleanly once the response crosses the cap.
                    if max_response_bytes > 0 {
                        let response_bytes = accumulated_content.len()
                            + tool_call_accum
                                .values()
                                .map(|a| a.arguments.len())
                                .sum::<usize>();
                        if response_bytes > max_response_bytes {
                            // P0-C: tag with the dedicated kind so the runaway
                            // guard finally surfaces `errorKind=response_too_large`
                            // through the bridge's generic degrader (closing the
                            // SEC-DOS-1 errorKind-surfacing gap) instead of a raw
                            // -32000. The stream is already bounded by the guard
                            // itself; this only fixes the surface.
                            return Err(ShimError::OpenAiHttp(format!(
                                "[response_too_large] response exceeded {max_response_bytes} bytes \
                                 and was aborted (runaway/repetition guard); raise \
                                 NWIRO_LOCAL_LLM_MAX_RESPONSE_BYTES or set 0 to disable"
                            )));
                        }
                    }
                }
            }
        }

        // Emit any content the stripper held back as a possible partial tag that
        // turned out to be real (an unterminated think block is dropped).
        let tail = think_stripper.flush();
        if !tail.is_empty() {
            accumulated_content.push_str(&tail);
            on_chunk(StreamChunk {
                content_delta: Some(tail),
                reasoning_delta: None,
                tool_call_delta: None,
                finish_reason: None,
            });
        }

        let tool_calls: Vec<ToolCall> = finalize_tool_calls(tool_call_accum);

        let final_message = ChatMessage::assistant(
            if accumulated_content.is_empty() {
                None
            } else {
                Some(accumulated_content)
            },
            if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls.clone())
            },
        );

        Ok(ChatResult { final_message, tool_calls, finish_reason })
    }

    /// Pre-load the model into the backend's working memory ("warm-up").
    ///
    /// Why this exists: cold-loading a multi-GB GGUF takes 30-60s. Doing it
    /// inline at the first chat send hides the latency and risks tripping
    /// the bridge's first_token_timeout. By warming explicitly (at config
    /// Save or via a "Load Model" button) the wait becomes transparent and
    /// failure modes (`not_found`, `oom`, `unreachable`, `model_unloaded`)
    /// become diagnosable instead of collapsing into "300s timeout".
    ///
    /// Strategy (v0.1.12 onward): POST `/v1/chat/completions` with a
    /// single placeholder token (`"."`), `max_tokens: 1`, `stream: false`,
    /// and the keep_alive field. This works identically on Ollama (which
    /// honours `keep_alive` here just like on `/api/generate`) AND on
    /// LM Studio (which has no `/api/generate` equivalent but answers
    /// every chat-completions request). Critically, this path *verifies*
    /// the model is inference-ready — previous /api/generate probe
    /// returned 200 even when the LM Studio model wasn't loaded, giving
    /// a false "loaded" status. The placeholder is non-empty because
    /// Ollama rejects empty messages on /v1/chat/completions with
    /// "messages must not be empty" — `"."` costs one token and is
    /// effectively invisible.
    ///
    /// Failure modes categorised in `error_kind`:
    /// - `not_found`     : model name unknown / not pulled
    /// - `model_unloaded`: LM Studio's "Model unloaded." (downloaded but
    ///                     not loaded into VRAM via the UI)
    /// - `oom`           : resource limits / failed to load weights
    /// - `unreachable`   : connection refused / DNS failed
    /// - `auth`          : 401 / 403
    /// - `timeout`       : the load request exceeded
    ///                     `NWIRO_LOCAL_LLM_WARMUP_TIMEOUT_SECS` (default
    ///                     300s; 0 disables) — typically a serverless
    ///                     cold start still loading the model
    /// - `unknown`       : everything else
    ///
    /// `keep_alive` accepts Ollama's syntax: `"15m"`, `"-1"` (forever),
    /// `"0"` (unload immediately). LM Studio silently ignores it. The
    /// shim passes through whatever the bridge sends; bridge default is
    /// `"15m"`. (Reviewed rewrite — 2026-05-10.)
    #[instrument(skip(self), fields(model = %model, keep_alive = %keep_alive))]
    pub async fn warmup(&self, model: &str, keep_alive: &str) -> WarmupResult {
        // Mechanical-coverage guarantee: this binding is declared BEFORE
        // every early-return path so all 4 WarmupResult literals capture
        // it via field shorthand. Adding a future return site that misses
        // this binding is a compile error (missing field on struct
        // literal) per the corrected plan §8 item 2.
        let recommended_tool_ceiling: Option<u32> = ModelFamily::detect(model)
            .and_then(|f| f.recommended_tool_ceiling());
        let start = std::time::Instant::now();
        let url = format!(
            "{}/chat/completions",
            self.base_url.trim_end_matches('/')
        );
        // `"."` placeholder: Ollama rejects empty `messages` content on the
        // OpenAI-compat endpoint, and we never see this token in the UI —
        // the chat-completions response on warmup is discarded.
        let body = json!({
            "model": model,
            "messages": [{"role": "user", "content": "."}],
            "max_tokens": 1,
            "stream": false,
            "keep_alive": keep_alive,
        });

        let mut req = self
            .http
            .post(&url)
            .header(CONTENT_TYPE, "application/json");
        if let Some(ref key) = self.api_key {
            if let Ok(hval) = HeaderValue::from_str(&format!("Bearer {}", key.as_str())) {
                req = req.header(AUTHORIZATION, hval);
            }
        }

        // v0.2.1: bound the load request end-to-end (connect + headers +
        // body). Per-request, so chat/probe paths are untouched; `0`
        // preserves the old unbounded behavior for operators who want it.
        let warmup_timeout_secs: u64 = std::env::var("NWIRO_LOCAL_LLM_WARMUP_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(WARMUP_TIMEOUT_SECS);
        if warmup_timeout_secs > 0 {
            req = req.timeout(std::time::Duration::from_secs(warmup_timeout_secs));
        }

        // v0.3.0: surface the warmup wait at START (previously the timeout value was
        // visible only on failure). This is the ONE setting the user actively waits on
        // — the editor spinner blocks on this model load — so it gets an explicit line;
        // the other NWIRO_LOCAL_LLM_* knobs stay env-only (read silently, not logged).
        if warmup_timeout_secs > 0 {
            tracing::info!(
                %model,
                "warming up model (loading into memory) — the editor spinner blocks until \
                 this finishes, up to {warmup_timeout_secs}s (raise or disable the cap via \
                 NWIRO_LOCAL_LLM_WARMUP_TIMEOUT_SECS)"
            );
        } else {
            tracing::info!(
                %model,
                "warming up model (loading into memory) — the editor spinner blocks until \
                 this finishes (no time cap; NWIRO_LOCAL_LLM_WARMUP_TIMEOUT_SECS=0)"
            );
        }

        let send_result = req.json(&body).send().await;
        let resp = match send_result {
            Ok(r) => r,
            Err(e) => {
                // Same URL-stripping rationale as chat_completion_stream
                // above — warmup is even more sensitive because it runs
                // on Save (i.e. user-visible) and the resulting message
                // is shown directly to the end-user via WarmupResult.
                let e = e.without_url();
                // v0.2.1: a timed-out load is its own diagnosable failure —
                // the backend ACCEPTED the connection but did not finish
                // loading in time (serverless cold start). Distinct from
                // `unreachable` (connect refused / DNS) so the operator
                // message can point at the right knob. The `!is_connect()`
                // guard matters: reqwest reports a CONNECT timeout (10s
                // connect_timeout, unroutable/filtered host) with
                // is_timeout()=true too — that case is an availability
                // failure where the connection was never accepted, so it
                // must keep flowing to `unreachable` below, not claim
                // "backend accepted the connection".
                if e.is_timeout() && !e.is_connect() {
                    tracing::warn!(
                        warmup_timeout_secs,
                        "warmup: load request timed out — returning ToolTier::None (availability verdict). Raise NWIRO_LOCAL_LLM_WARMUP_TIMEOUT_SECS for slow cold starts, or set 0 to disable the cap."
                    );
                    return WarmupResult {
                        status: "failed".to_string(),
                        elapsed_ms: start.elapsed().as_millis() as u64,
                        tool_tier: ToolTier::None,
                        model_size_bytes: None,
                        error_kind: Some("timeout".to_string()),
                        // Phase-agnostic phrasing (review NIT): with a cap set
                        // BELOW the connect window the total timeout can fire
                        // while still connecting (is_connect()==false on
                        // reqwest's total-timeout error regardless of phase),
                        // so don't claim the connection was accepted.
                        message: Some(format!(
                            "warmup timed out after {warmup_timeout_secs}s — the backend did not finish answering the model-load request in time (typically a serverless cold start still loading the model). Raise NWIRO_LOCAL_LLM_WARMUP_TIMEOUT_SECS, or set it to 0 to disable the cap."
                        )),
                        recommended_tool_ceiling,
                    };
                }
                tracing::warn!(error = %e, "warmup: send failed — returning ToolTier::None WITHOUT running the probe (backend unreachable). Tools stay stripped until a successful warmup.");
                // v0.1.35 BOUNDARY: backend-DOWN deliberately stays None (an
                // availability verdict). Do NOT route this through the probe's
                // `ProbeAssessment::failed()`, which fails OPEN to Emulated —
                // attaching tools to an unreachable backend has no benefit and
                // erases the operator's "backend down" diagnostic.
                let chain = error_chain(&e);
                let (kind, msg) = if is_tls_cert_error(&chain) {
                    (
                        "tls_cert",
                        format!(
                            "TLS certificate not trusted: {chain} — the shim trusts bundled \
                             roots (rustls/webpki-roots), not the OS store, so a corporate \
                             TLS-intercepting proxy fails here. Use a direct endpoint or trust \
                             the proxy root."
                        ),
                    )
                } else {
                    ("unreachable", format!("connection error: {chain}"))
                };
                return WarmupResult {
                    status: "failed".to_string(),
                    elapsed_ms: start.elapsed().as_millis() as u64,
                    tool_tier: ToolTier::None,
                    model_size_bytes: None,
                    error_kind: Some(kind.to_string()),
                    message: Some(msg),
                    recommended_tool_ceiling,
                };
            }
        };

        let status = resp.status();
        // Read the body once so we can branch on both status + content.
        // LM Studio returns HTTP 200 + JSON error body when the model is
        // not loaded — status.is_success() alone is unreliable here.
        // v0.2.1: a body-read failure must NOT masquerade as success. The
        // old `.unwrap_or_default()` turned a read-phase timeout (or a
        // mid-body connection reset) after 200-headers into an empty body
        // -> no error field -> "loaded" — a false-success on a backend
        // that never finished answering.
        let body_text = match resp.text().await {
            Ok(t) => t,
            Err(e) => {
                let e = e.without_url();
                let (kind, msg) = if e.is_timeout() {
                    (
                        "timeout",
                        format!(
                            "warmup timed out after {warmup_timeout_secs}s while reading the response body — the backend started answering but did not finish in time. Raise NWIRO_LOCAL_LLM_WARMUP_TIMEOUT_SECS, or set it to 0 to disable the cap."
                        ),
                    )
                } else {
                    ("unknown", format!("warmup response body read failed: {e}"))
                };
                tracing::warn!(error_kind = %kind, error = %e, "warmup: body read failed — returning ToolTier::None (availability verdict)");
                return WarmupResult {
                    status: "failed".to_string(),
                    elapsed_ms: start.elapsed().as_millis() as u64,
                    tool_tier: ToolTier::None,
                    model_size_bytes: None,
                    error_kind: Some(kind.to_string()),
                    message: Some(msg),
                    recommended_tool_ceiling,
                };
            }
        };
        let parsed: serde_json::Value = serde_json::from_str(&body_text)
            .unwrap_or(serde_json::Value::Null);
        let has_error_field = parsed.get("error").is_some();

        if status.is_success() && !has_error_field {
            // Real inference-ready response. Either the response has
            // `choices` (full chat completion) or a single token came
            // back — both are proof the model loaded and inferred.
            //
            // Issue a one-shot tool-capability probe against the same
            // model now that we know it's loaded. v0.1.35: if the probe is
            // inconclusive for any reason it fails OPEN to `ToolTier::Emulated`
            // (not None) — see `ProbeAssessment::failed` — so tools are never
            // silently stripped. Warmup succeeds either way.
            let assessment = self.probe_tool_capability(model).await;

            // v0.1.35: the GLM-family `forces_emulated_tier()` warmup
            // override was DELETED here. It promoted a (false-positive)
            // Native/None GLM probe to Emulated. That job is now universal —
            // an inconclusive probe fails OPEN to Emulated for EVERY model
            // (`ProbeAssessment::failed` + the terminal probe return below),
            // so no per-family override is needed. The
            // `NWIRO_LOCAL_LLM_FORCE_TOOL_TIER` escape hatch remains for the
            // rare false-positive-Native-on-manual-template case.

            // v0.1.28 Phase 2: family + bleed combo gate.
            // Refuse warmup when a known template-sensitive family is
            // detected AND the probe response shows schema-bleed
            // signature — operator's backend is misconfigured and the
            // session would otherwise stream garbage to UE5.
            //
            // Bypass via env var because the existing
            // NWIRO_LOCAL_LLM_FORCE_TOOL_TIER fires only at
            // session/prompt (post-warmup) and so cannot help if warmup
            // itself refuses. The bypass exists for troubleshooting
            // and for operators who knowingly want to proceed despite
            // detection (e.g. testing a fix in progress).
            let bypass_gate = std::env::var("NWIRO_LOCAL_LLM_BYPASS_TEMPLATE_GATE")
                .ok()
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            if let Some(family) = assessment.family {
                if assessment.schema_bleed_detected && !bypass_gate {
                    tracing::warn!(
                        family = ?family,
                        model = %model,
                        "warmup REFUSED — known template-sensitive family + \
                         schema-bleed detected; returning broken_chat_template \
                         error. Bypass via NWIRO_LOCAL_LLM_BYPASS_TEMPLATE_GATE=1."
                    );
                    return WarmupResult {
                        status: "failed".to_string(),
                        elapsed_ms: start.elapsed().as_millis() as u64,
                        // Keep tool_tier as the probe-derived value so
                        // downstream code that inspects it post-failure
                        // still sees the truth — the session is failed
                        // but the tier signal is preserved.
                        //
                        // v0.1.28 review round-2 NEW-B: this
                        // is a deliberate exception to the otherwise-
                        // uniform `failed → ToolTier::None` pattern
                        // used by other failure paths in this function
                        // (unreachable / not_found / oom / auth /
                        // unknown all return None). The asymmetry is
                        // intentional: for `broken_chat_template`
                        // specifically the probed tier is meaningful
                        // diagnostic data (e.g. "the model emitted a
                        // Native tool_call envelope BUT bleed in
                        // content") that an operator scanning the
                        // failure structure benefits from seeing.
                        // Consumers MUST gate on status="failed"
                        // before reading tool_tier; treating the
                        // tier as authoritative on the failure path
                        // is a consumer-side bug.
                        tool_tier: assessment.tier,
                        model_size_bytes: None,
                        error_kind: Some(family.error_kind().to_string()),
                        message: Some(family.template_guidance().to_string()),
                        recommended_tool_ceiling,
                    };
                }
                if assessment.schema_bleed_detected && bypass_gate {
                    tracing::warn!(
                        family = ?family,
                        model = %model,
                        "schema-bleed detected on known family — gate bypassed \
                         via NWIRO_LOCAL_LLM_BYPASS_TEMPLATE_GATE; proceeding \
                         with degraded session (expect garbage output)."
                    );
                }
            }

            return WarmupResult {
                status: "loaded".to_string(),
                elapsed_ms: start.elapsed().as_millis() as u64,
                tool_tier: assessment.tier,
                model_size_bytes: None,
                error_kind: None,
                message: None,
                recommended_tool_ceiling,
            };
        }

        // From here we are in error territory: either status != 2xx OR
        // 200-with-error-body (LM Studio's "Model unloaded." pattern).
        let code = status.as_u16();
        let body_lower = body_text.to_lowercase();
        let extracted_message = parsed
            .get("error")
            .and_then(|e| e.get("message"))
            .and_then(|m| m.as_str())
            .map(|s| s.to_string())
            .or_else(|| {
                parsed
                    .get("error")
                    .and_then(|e| e.as_str())
                    .map(|s| s.to_string())
            });

        let error_kind = if code == 401 || code == 403 {
            "auth"
        } else if body_lower.contains("model unloaded")
            || body_lower.contains("model is not loaded")
            || body_lower.contains("no model loaded")
        {
            "model_unloaded"
        } else if body_lower.contains("not found") || body_lower.contains("no such model") {
            "not_found"
        } else if body_lower.contains("memory")
            || body_lower.contains("oom")
            || body_lower.contains("resource limit")
            || body_lower.contains("failed to load")
        {
            "oom"
        } else {
            "unknown"
        };

        tracing::warn!(
            error_kind = %error_kind,
            body_excerpt = %body_text.chars().take(300).collect::<String>(),
            "warmup: backend returned an error (non-2xx or 200+error-body) — returning ToolTier::None WITHOUT running the probe. Tools stay stripped until a successful warmup."
        );
        // v0.1.35 BOUNDARY: backend-error stays None (availability verdict), NOT
        // the probe's fail-open Emulated — see the send-failure path above.
        WarmupResult {
            status: "failed".to_string(),
            elapsed_ms: start.elapsed().as_millis() as u64,
            tool_tier: ToolTier::None,
            model_size_bytes: None,
            error_kind: Some(error_kind.to_string()),
            message: Some(
                extracted_message
                    .unwrap_or_else(|| format!("HTTP {code}: {body_text}")),
            ),
            recommended_tool_ceiling,
        }
    }

    /// One-shot probe to classify a model's tool-calling capability.
    ///
    /// Called on the success path of `warmup`. The probe POSTs a fixed
    /// chat-completion request with a single `find_blueprints` tool spec
    /// and a directive prompt. Classification (priority order):
    ///   1. `choices[0].message.tool_calls[0].function.name == "find_blueprints"` → `Native`
    ///   2. `content` parses as a valid tool call via the runtime
    ///      Emulated parser (`emulated_parser::try_extract_tool_call`)
    ///      → `Emulated`. v0.1.27 review finding 5: replaces
    ///      the old loose substring check (`contains("find_blueprints")
    ///      && contains('{') && contains('}')`) which allowed
    ///      schema-bleeding output (literal `"object"` repetition that
    ///      happened to contain the tool name + braces from the schema)
    ///      to misclassify as Emulated. The probe contract now matches
    ///      the runtime parser contract: if the parser can't extract a
    ///      call from probe output, it won't extract one at runtime
    ///      either, so falling through to `None` is correct.
    ///   3. anything else (empty tool_calls, prose, network error, non-200,
    ///      non-JSON, timeout) → `None`
    ///
    /// Probe failures are non-fatal to warmup: any error path returns
    /// `ToolTier::None` so the caller observes a populated field without
    /// having to handle a separate probe-error type.
    ///
    /// `stream: false` is mandatory — the classifier reads the final
    /// `choices[0].message` envelope, which SSE chunks don't supply
    /// directly.
    #[instrument(skip(self), fields(model = %model))]
    pub(crate) async fn probe_tool_capability(&self, model: &str) -> ProbeAssessment {
        // v0.1.28: detect family from model name FIRST. This is a pure
        // function of the configured model id and does not depend on
        // the probe network call succeeding. warmup() uses this even
        // on early-error paths.
        let family = ModelFamily::detect(model);
        // Base probe body. `tool_choice` is added PER-ATTEMPT below so a backend
        // that rejects one form can be retried with another.
        let base_body = json!({
            "model": model,
            "messages": [{
                "role": "user",
                // v0.1.20: prepend `/no_think` — Qwen 3's marker to skip
                // chain-of-thought reasoning for this turn. Qwen 3 is
                // reasoning-by-default; without this marker, even the
                // 128-token probe budget gets consumed entirely by the
                // `reasoning` field before the model emits any
                // `tool_calls`, producing `finish_reason: "length"` with
                // empty content and a false `ToolTier::None`. Empirically
                // confirmed via direct curl against Ollama qwen3:14b.
                //
                // `/no_think` is a no-op for non-Qwen-3 models — they see
                // it as literal text and ignore it. Safe to send
                // unconditionally.
                "content": "/no_think Call find_blueprints with searchTerm 'test'"
            }],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "find_blueprints",
                    "description": "Search Blueprint assets",
                    "parameters": {
                        "type": "object",
                        "properties": { "searchTerm": { "type": "string" } },
                        "required": ["searchTerm"]
                    }
                }
            }],
            // 256 (raised from 128 in v0.1.20): even with the `/no_think`
            // marker above, reasoning-mode models (Qwen 3, DeepSeek-R1)
            // may still emit ~100 tokens of reasoning preamble before
            // the tool-call envelope. 128 was insufficient — Qwen 3 14B
            // empirically consumed 64+ reasoning tokens without
            // `/no_think` and the envelope was truncated. 256 gives
            // ~150 tokens of reasoning headroom plus ~100 for the call,
            // covering Qwen 3 even when `/no_think` is unsupported by
            // the backend (some non-Ollama backends pass the marker
            // through to the model verbatim where it's harmless prose).
            "max_tokens": 256,
            "stream": false
        });

        let url = format!(
            "{}/chat/completions",
            self.base_url.trim_end_matches('/')
        );
        // Resolve the probe timeout: env var takes precedence over the
        // compiled-in default. Lets campaigns testing partial-offload
        // models (Kimi-K2 et al.) set a higher value without a rebuild,
        // while preserving the 5s fast-path for the common case.
        let probe_timeout_secs = std::env::var("NWIRO_LOCAL_LLM_PROBE_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(PROBE_TIMEOUT_SECS);
        // v0.2.6 tool_choice compatibility + native-capability force.
        // Try `tool_choice: "required"` first (forces a call to the only probe
        // tool — equivalent to the OpenAI object-form here, but UNIVERSALLY
        // supported). LM Studio / llama.cpp servers reject the object-form with
        // HTTP 400 ("Invalid tool_choice type: 'object'. Supported string values:
        // none, auto, required"); the old probe treated that 400 as transient and
        // FAILED OPEN to Emulated, mis-classifying every native-capable LM Studio
        // model (Hermes-3, Qwen3-30B) as Emulated — which then drove the EMIT-004
        // directive path whose tool-name re-listing overflowed a 4096-ctx model
        // at 30 tools into a downstream 400 BLACK. If a backend ALSO rejects the
        // `"required"` string, retry ONCE without any tool_choice (the prompt
        // already asks the model to call find_blueprints), so a tool_choice
        // incompatibility NEVER fail-opens a native model to Emulated.
        let body_text = {
            let mut found: Option<String> = None;
            for tool_choice in [Some("required"), None] {
                let mut body = base_body.clone();
                if let Some(tc) = tool_choice {
                    body["tool_choice"] = json!(tc);
                }
                let mut req = self
                    .http
                    .post(&url)
                    .header(CONTENT_TYPE, "application/json")
                    .timeout(std::time::Duration::from_secs(probe_timeout_secs));
                if let Some(ref key) = self.api_key {
                    if let Ok(hval) = HeaderValue::from_str(&format!("Bearer {}", key.as_str())) {
                        req = req.header(AUTHORIZATION, hval);
                    }
                }
                let resp = match req.json(&body).send().await {
                    Ok(r) => r,
                    Err(e) => {
                        let e = e.without_url();
                        tracing::warn!(error = %e, "probe: send failed — failing OPEN to ToolTier::Emulated — transient transport failure, NOT a capability verdict; tools are NOT stripped");
                        return ProbeAssessment::failed(model);
                    }
                };
                let status = resp.status();
                if status.is_success() {
                    match resp.text().await {
                        Ok(t) => {
                            found = Some(t);
                            break;
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "probe: response body read failed — failing OPEN to ToolTier::Emulated — transient; tools are NOT stripped");
                            return ProbeAssessment::failed(model);
                        }
                    }
                }
                let err_body = resp.text().await.unwrap_or_default();
                let err_lower = err_body.to_lowercase();
                // Some backends (Ollama) reject the tools array for models with no
                // function-calling support with HTTP 400 "<model> does not support
                // tools". That is a DEFINITIVE capability verdict — classify None so
                // the bridge strips tools and the model answers tool-free, instead of
                // failing OPEN to Emulated (which re-sends tools on the real prompt ->
                // the same 400 -> a raw -32000).
                if err_lower.contains("does not support tools") {
                    tracing::warn!(
                        status = %status,
                        body_excerpt = %err_body.chars().take(300).collect::<String>(),
                        "probe: model does not support tools — classifying ToolTier::None (tools stripped; clean tool-free degrade) instead of failing open to Emulated"
                    );
                    return ProbeAssessment::no_tool_support(model);
                }
                // A tool_choice-shaped 400 is a BACKEND COMPATIBILITY issue, NOT a
                // capability verdict: retry without tool_choice (next loop arm)
                // rather than fail-opening a possibly-native model to Emulated.
                if status.as_u16() == 400
                    && err_lower.contains("tool_choice")
                    && tool_choice.is_some()
                {
                    tracing::warn!(
                        status = %status,
                        body_excerpt = %err_body.chars().take(300).collect::<String>(),
                        "probe: backend rejected tool_choice — retrying probe WITHOUT tool_choice (compatibility fallback, NOT a capability verdict)"
                    );
                    continue;
                }
                let body_excerpt: String = err_body.chars().take(300).collect();
                tracing::warn!(status = %status, body_excerpt = %body_excerpt, "probe: non-2xx status — failing OPEN to ToolTier::Emulated — transient, NOT a capability verdict; tools are NOT stripped");
                return ProbeAssessment::failed(model);
            }
            match found {
                Some(t) => t,
                None => {
                    tracing::warn!("probe: exhausted tool_choice fallbacks — failing OPEN to ToolTier::Emulated");
                    return ProbeAssessment::failed(model);
                }
            }
        };
        let v: serde_json::Value = match serde_json::from_str(&body_text) {
            Ok(v) => v,
            Err(e) => {
                let excerpt: String = body_text.chars().take(300).collect();
                tracing::warn!(error = %e, body_excerpt = %excerpt, "probe: response not valid JSON — failing OPEN to ToolTier::Emulated (v0.1.35; was None) — transient; tools are NOT stripped");
                return ProbeAssessment::failed(model);
            }
        };

        // Priority 1/2: name match on the first tool_call. Per spec, the
        // name being correct is the strongest signal — we do NOT reject
        // on an arguments-parse failure (some models emit partial JSON).
        if let Some(name) = v
            .pointer("/choices/0/message/tool_calls/0/function/name")
            .and_then(|n| n.as_str())
        {
            if name == "find_blueprints" {
                // v0.1.28 review round-1 defect 2: a backend
                // returning BOTH a valid tool_calls envelope AND
                // bleed-shaped text in `content` would skip the gate
                // if we hardcoded bleed=false here. Empirically rare
                // (most backends emit one or the other) but real:
                // some LM Studio configurations succeed at extracting
                // a partial tool_calls JSON from the first tokens
                // while leaving the rest as bleed in content. The
                // gate policy is "broken template ⇒ refuse" not
                // "broken template ⇒ refuse unless we got lucky with
                // one parse." Carry the signal so warmup() can apply
                // it consistently.
                let content_bleed = v
                    .pointer("/choices/0/message/content")
                    .and_then(|c| c.as_str())
                    .map(looks_like_schema_bleed)
                    .unwrap_or(false);
                return ProbeAssessment {
                    tier: ToolTier::Native,
                    family,
                    schema_bleed_detected: content_bleed,
                };
            }
        }

        // Priority 2: try to extract a real tool call from the response
        // content using the same parser that runs downstream in
        // `bridge::handle_session_prompt`. Only classify as Emulated
        // if the parser SUCCEEDS — meaning at runtime the shim could
        // actually synthesize a ToolCall from this model's prose.
        // (v0.1.27 review round-3 finding 7: inline label was "Priority 3"
        // — stale from a pre-v0.1.27 numbering. Aligned with the docstring
        // enumeration above: 1=tool_calls, 2=Emulated parser, 3=None.)
        //
        // v0.1.27 review-corrected: the previous check was a loose
        // substring match (`contains("find_blueprints") && contains("{") &&
        // contains("}")`). Models with broken chat templates (e.g.
        // GLM-4.5-air via LM Studio without proper Jinja template)
        // autoregressively echo the tool schema back as literal text
        // tokens — `"object", "object", ": "object"` etc. That output
        // contains `find_blueprints` (it's in the schema), contains
        // `{` and `}` (schema structure), but the EMIT parsers can't
        // extract anything coherent from it. The loose substring check
        // misclassified that as Emulated → downstream G4 buffer didn't
        // help → ~25K tokens of schema fragments streamed to UI as
        // content. Tightening the probe to match the runtime parser's
        // actual contract closes that gap.
        //
        // v0.1.17-v0.1.19 implemented Emulated-tier execution via
        // `src/bridge/emulated_parser.rs` (Qwen XML, inline JSON,
        // Markdown headers). v0.1.19 EMIT-004 added the per-call
        // system-prompt directive guiding models to emit a clean
        // inline-JSON envelope.
        // v0.1.28: track schema-bleed flag across the content analysis
        // so we can propagate it into the ProbeAssessment whether the
        // path falls through to Emulated or None.
        let mut schema_bleed_detected = false;
        if let Some(content) = v
            .pointer("/choices/0/message/content")
            .and_then(|c| c.as_str())
        {
            let probe_tool_names = vec!["find_blueprints".to_string()];
            if crate::bridge::emulated_parser::try_extract_tool_call(
                content,
                &probe_tool_names,
            )
            .is_some()
            {
                // v0.1.28: even on the Emulated success path, capture
                // schema-bleed signal. Some models emit BOTH a valid
                // emulated tool call AND surrounding schema garbage —
                // warmup() can decide whether bleed-with-Emulated still
                // warrants a refusal for known-family models.
                return ProbeAssessment {
                    tier: ToolTier::Emulated,
                    family,
                    schema_bleed_detected: looks_like_schema_bleed(content),
                };
            }

            // v0.1.27 telemetry (Option A, light): if the content
            // smells like schema fragments rather than coherent
            // output, emit a tracing::warn! so operators reading
            // logs see the diagnostic when their model + chat
            // template combination is broken. v0.1.28: ALSO capture
            // the boolean so warmup() can gate on it.
            if looks_like_schema_bleed(content) {
                schema_bleed_detected = true;
                tracing::warn!(
                    content_excerpt = %&content.chars().take(120).collect::<String>(),
                    "warmup probe response looks like JSON schema fragments — \
                     check inference backend's chat template (e.g. LM Studio \
                     'Prompt Template' must match the loaded model's tool-call \
                     spec). See docs/MODEL-SETUP.md for recommended templates."
                );
            }
        }

        // Probe observability: we reached here with a 200 + valid JSON but
        // neither a native `tool_calls` envelope (checked at the top) nor
        // emulated-parseable content. Historically this None was a black box —
        // operators (and workflow diagnostics) had to GUESS whether it
        // meant "model truncated its reasoning", "tool_calls arrived in an
        // unexpected shape", or "genuine no-tool-support". Capture the raw
        // shape so the classification is self-explaining in the trace file.
        let finish_reason = v
            .pointer("/choices/0/finish_reason")
            .and_then(|f| f.as_str())
            .unwrap_or("<none>");
        let tool_calls_present = v
            .pointer("/choices/0/message/tool_calls")
            .map(|tc| !tc.is_null())
            .unwrap_or(false);
        // Qwen3 / DeepSeek-R1 put chain-of-thought in a separate `reasoning`
        // (Ollama) / `reasoning_content` (LM Studio) field and leave `content`
        // empty — the exact shape a direct curl confirmed returns valid
        // tool_calls. Flag both explicitly so this None is unambiguous in the
        // trace: "reasoning filled + content empty + tool_calls missing" is a
        // reasoning-model envelope quirk, NOT genuine no-tool-support.
        let reasoning_present = v
            .pointer("/choices/0/message/reasoning")
            .and_then(|r| r.as_str())
            .map(|s| !s.is_empty())
            .unwrap_or(false)
            || v
                .pointer("/choices/0/message/reasoning_content")
                .and_then(|r| r.as_str())
                .map(|s| !s.is_empty())
                .unwrap_or(false);
        let content_empty = v
            .pointer("/choices/0/message/content")
            .and_then(|c| c.as_str())
            .map(|s| s.is_empty())
            .unwrap_or(true);
        let body_excerpt: String = body_text.chars().take(400).collect();
        tracing::warn!(
            finish_reason = %finish_reason,
            tool_calls_present = tool_calls_present,
            reasoning_present = reasoning_present,
            content_empty = content_empty,
            schema_bleed = schema_bleed_detected,
            body_excerpt = %body_excerpt,
            "probe: inconclusive 200 response (no native tool_calls, no emulated-parseable content) — failing OPEN to ToolTier::Emulated (v0.1.35; was None) so tools are NOT stripped; raw response captured for diagnosis"
        );

        ProbeAssessment {
            tier: ToolTier::Emulated,
            family,
            schema_bleed_detected,
        }
    }

    // `warmup_noop_via_models` (the old /v1/models fallback) was removed
    // in v0.1.12 — the unified /v1/chat/completions probe covers Ollama,
    // LM Studio, and any future OpenAI-compat backend without needing a
    // backend-detection heuristic. Kept this comment as a tombstone so a
    // future contributor doesn't reintroduce the dual-path complexity.
}

fn accumulate_delta(
    delta: &Delta,
    content: &mut String,
    accum: &mut BTreeMap<usize, ToolCallAccum>,
) {
    if let Some(ref c) = delta.content {
        content.push_str(c);
    }
    if let Some(ref tcs) = delta.tool_calls {
        for tc in tcs {
            let entry = accum.entry(tc.index).or_default();
            if let Some(ref id) = tc.id {
                if entry.id.is_empty() {
                    entry.id = id.clone();
                }
            }
            if let Some(ref func) = tc.function {
                if let Some(ref name) = func.name {
                    if entry.name.is_empty() {
                        entry.name = name.clone();
                    }
                }
                if let Some(ref args) = func.arguments {
                    entry.arguments.push_str(args);
                }
            }
        }
    }
}

/// Convert the index-keyed streaming accumulation into the final ordered
/// tool-call list. Extracted from `chat_completion_stream` so the keep/drop
/// rule is unit-testable (it decides whether a model's tool call actually
/// reaches the host).
fn finalize_tool_calls(accum: BTreeMap<usize, ToolCallAccum>) -> Vec<ToolCall> {
    accum
        .into_values()
        // A real tool call always has a NAME. vLLM / llama.cpp streaming deltas
        // often omit the `id`, so filtering on a non-empty id silently dropped
        // genuine calls (P0-D / Finding M2). Keep any accum that has a name and
        // synthesize an id when the backend didn't send one; drop only truly
        // empty (nameless) slots — stray deltas, not calls. An id-without-name
        // is not dispatchable, so it is correctly dropped too.
        .filter(|acc| !acc.name.is_empty())
        .map(|acc| ToolCall {
            id: if acc.id.is_empty() {
                format!("call_{}", uuid::Uuid::new_v4().simple())
            } else {
                acc.id
            },
            r#type: "function".to_string(),
            function: ToolCallFunction {
                name: acc.name,
                arguments: acc.arguments,
            },
        })
        .collect()
}

/// Strips `<think>...</think>` chain-of-thought markup that some backends
/// (notably llama.cpp without a reasoning-format split) emit INSIDE
/// `delta.content` instead of the structured `reasoning_content` field. Without
/// this the model's private reasoning streams to the UI as the assistant's
/// answer (G-MODEL-1). Stateful across stream chunks: a tag split across two
/// deltas is handled by holding back a possible-partial-tag tail in `pending`.
#[derive(Default)]
struct ThinkStripper {
    in_think: bool,
    pending: String,
}

impl ThinkStripper {
    const OPEN: &'static str = "<think>";
    const CLOSE: &'static str = "</think>";

    /// Feed one content delta; return the portion safe to emit as real content
    /// (everything outside a `<think>` block). Content inside a block is dropped.
    fn push(&mut self, delta: &str) -> String {
        let mut buf = std::mem::take(&mut self.pending);
        buf.push_str(delta);
        let mut out = String::new();
        loop {
            if self.in_think {
                if let Some(pos) = buf.find(Self::CLOSE) {
                    buf.replace_range(..pos + Self::CLOSE.len(), "");
                    self.in_think = false;
                } else {
                    // Still inside a think block: hold a tail that might be a
                    // partial "</think>", drop the rest (it is reasoning).
                    self.pending = hold_partial_suffix(&buf, Self::CLOSE);
                    return out;
                }
            } else {
                // Not inside a think block. Act on whichever tag comes first: an
                // OPEN enters a block; a BARE CLOSE (`</think>` whose open was
                // routed through another channel — e.g. delta.reasoning, or a
                // backend reasoning-format that splits the close into content) is
                // a stray marker that must be DROPPED, not emitted as the answer.
                // The old code searched only for OPEN, so a lone `</think>` fell
                // through and leaked verbatim to the UI (deepseek-r1 BLACK-risk).
                let open = buf.find(Self::OPEN);
                let close = buf.find(Self::CLOSE);
                match (open, close) {
                    // Both tags present: act on whichever comes first.
                    (Some(o), Some(c)) if o < c => {
                        out.push_str(&buf[..o]);
                        buf.replace_range(..o + Self::OPEN.len(), "");
                        self.in_think = true;
                    }
                    (Some(_), Some(c)) => {
                        // CLOSE precedes OPEN: drop the stray close, keep the open
                        // for the next loop turn.
                        out.push_str(&buf[..c]);
                        buf.replace_range(..c + Self::CLOSE.len(), "");
                    }
                    // Only an OPEN: emit text before it, enter the think block.
                    (Some(o), None) => {
                        out.push_str(&buf[..o]);
                        buf.replace_range(..o + Self::OPEN.len(), "");
                        self.in_think = true;
                    }
                    // Only a bare CLOSE (open routed elsewhere): emit text before
                    // it, drop the marker, stay outside the block.
                    (None, Some(c)) => {
                        out.push_str(&buf[..c]);
                        buf.replace_range(..c + Self::CLOSE.len(), "");
                    }
                    // Neither tag present: emit all but a tail that could begin
                    // EITHER tag, so a boundary-split open OR close is still held.
                    (None, None) => {
                        let keep_open = hold_partial_suffix(&buf, Self::OPEN);
                        let keep_close = hold_partial_suffix(&buf, Self::CLOSE);
                        let keep = if keep_close.len() > keep_open.len() {
                            keep_close
                        } else {
                            keep_open
                        };
                        let emit_to = buf.len() - keep.len();
                        out.push_str(&buf[..emit_to]);
                        self.pending = keep;
                        return out;
                    }
                }
            }
        }
    }

    /// At end-of-stream, emit any held content that turned out not to be a tag.
    /// An unterminated `<think>` block is dropped (it was reasoning).
    fn flush(&mut self) -> String {
        if self.in_think {
            self.pending.clear();
            String::new()
        } else {
            std::mem::take(&mut self.pending)
        }
    }
}

/// The longest suffix of `buf` that is a strict prefix of `tag` — a possible
/// partial tag straddling the next delta. UTF-8-safe (skips non-boundaries, so
/// a multibyte char before a stray `<` can never panic the slice).
fn hold_partial_suffix(buf: &str, tag: &str) -> String {
    let max = (tag.len() - 1).min(buf.len());
    for n in (1..=max).rev() {
        let start = buf.len() - n;
        if !buf.is_char_boundary(start) {
            continue;
        }
        let suffix = &buf[start..];
        if tag.starts_with(suffix) {
            return suffix.to_string();
        }
    }
    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::messages::{ToolCallDelta, ToolCallFunctionDelta};
    use serde_json::json;

    #[test]
    fn empty_id_tool_call_with_name_is_kept_not_dropped() {
        // P0-D / Finding M2: vLLM and llama.cpp streaming deltas frequently omit
        // the tool_call `id` (or send a non-OpenAI-shaped one). The model DID
        // call a tool — but the finalize filter dropped any accum whose id
        // stayed empty (`!acc.id.is_empty()`), so the call silently vanished and
        // the user saw prose / a stall. An Ollama-only matrix never caught it
        // (Ollama populates ids). A real call always has a NAME; finalize must
        // keep it and synthesize an id, not discard it.
        let mut accum: BTreeMap<usize, ToolCallAccum> = BTreeMap::new();
        accum.insert(
            0,
            ToolCallAccum {
                id: String::new(), // backend never sent an id
                name: "find_blueprints".to_string(),
                arguments: r#"{"query":"door"}"#.to_string(),
            },
        );

        let calls = finalize_tool_calls(accum);

        assert_eq!(calls.len(), 1, "an id-less tool call WITH a name must survive");
        assert_eq!(calls[0].function.name, "find_blueprints");
        assert_eq!(calls[0].function.arguments, r#"{"query":"door"}"#);
        assert!(
            !calls[0].id.is_empty(),
            "finalize must synthesize a non-empty id for an id-less call"
        );
    }

    #[test]
    fn truly_empty_accum_slot_is_dropped() {
        // Boundary guard for the fix: a slot with NEITHER id NOR name is garbage
        // (a stray empty delta), not a real call — it must still be dropped so
        // the synthesize-id fix cannot resurrect noise into a phantom tool call.
        let mut accum: BTreeMap<usize, ToolCallAccum> = BTreeMap::new();
        accum.insert(0, ToolCallAccum::default()); // all empty
        assert!(
            finalize_tool_calls(accum).is_empty(),
            "a nameless, id-less slot is not a real call and must be dropped"
        );
    }

    // Build a Delta carrying exactly one streaming tool-call fragment.
    fn tc_delta(index: usize, id: Option<&str>, name: Option<&str>, args: Option<&str>) -> Delta {
        Delta {
            tool_calls: Some(vec![ToolCallDelta {
                index,
                id: id.map(String::from),
                r#type: None,
                function: Some(ToolCallFunctionDelta {
                    name: name.map(String::from),
                    arguments: args.map(String::from),
                }),
            }]),
            ..Default::default()
        }
    }

    #[test]
    fn accumulate_delta_concatenates_fragmented_arguments() {
        // P0-A / Finding M3: OpenAI-compatible backends stream tool-call
        // arguments split across many SSE chunks, often character-by-character
        // (the shape codex-acp 0.64.0 regressed on vs local OpenAI-compat
        // backends — every JSON parse failed, zero tools executed). id + name
        // arrive on the first chunk for an index; args append across chunks.
        let mut content = String::new();
        let mut accum: BTreeMap<usize, ToolCallAccum> = BTreeMap::new();

        accumulate_delta(&tc_delta(0, Some("call_1"), Some("search"), None), &mut content, &mut accum);
        for ch in r#"{"q":"door"}"#.chars() {
            accumulate_delta(&tc_delta(0, None, None, Some(&ch.to_string())), &mut content, &mut accum);
        }

        let calls = finalize_tool_calls(accum);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].function.name, "search");
        assert_eq!(
            calls[0].function.arguments, r#"{"q":"door"}"#,
            "character-fragmented arguments must concatenate into valid JSON"
        );
        assert!(content.is_empty(), "no content deltas were sent");
    }

    #[test]
    fn accumulate_delta_keeps_parallel_calls_index_separated() {
        // Two parallel tool calls streamed with interleaved, out-of-order
        // fragments. The index keys the accumulation, so args must partition
        // per index with zero cross-contamination, and finalize returns them
        // ordered by index.
        let mut content = String::new();
        let mut accum: BTreeMap<usize, ToolCallAccum> = BTreeMap::new();

        accumulate_delta(&tc_delta(1, Some("call_b"), Some("beta"), None), &mut content, &mut accum);
        accumulate_delta(&tc_delta(0, Some("call_a"), Some("alpha"), None), &mut content, &mut accum);
        accumulate_delta(&tc_delta(0, None, None, Some(r#"{"a":1}"#)), &mut content, &mut accum);
        accumulate_delta(&tc_delta(1, None, None, Some(r#"{"b":2}"#)), &mut content, &mut accum);

        let calls = finalize_tool_calls(accum);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "alpha");
        assert_eq!(calls[0].function.arguments, r#"{"a":1}"#);
        assert_eq!(calls[1].function.name, "beta");
        assert_eq!(calls[1].function.arguments, r#"{"b":2}"#);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn runaway_stream_is_aborted_at_the_response_byte_cap() {
        // P0-E: a local model stuck in a repetition loop streams content forever.
        // The read timeout is off and the host's first-token timer disarms after
        // the first token, so the byte cap is the only bound. With a tiny cap and
        // a body that overflows it, the stream must abort with the guard message
        // rather than accumulate without limit.
        let mut mock = mockito::Server::new_async().await;
        let mut body = String::new();
        for _ in 0..20 {
            body.push_str("data: {\"choices\":[{\"delta\":{\"content\":\"spam\"}}]}\n\n");
        }
        body.push_str("data: [DONE]\n\n");
        let _m = mock
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(body)
            .create_async()
            .await;

        let client = Client::new(mock.url(), "m".to_string(), None);
        let err = client
            .chat_completion_stream(
                "m",
                vec![ChatMessage::user("hi")],
                None,
                CancellationToken::new(),
                16, // tiny cap → overflow after the first few "spam" deltas
                None,
                0, // SEC-DOS-1 inactivity disabled in this test
                0, // P1 pre-stream cap disabled in this test
                |_chunk| {},
            )
            .await
            .expect_err("a stream exceeding the byte cap must abort with an error");

        let msg = err.to_string();
        assert!(
            msg.contains("runaway") || msg.contains("exceeded"),
            "abort must mention the runaway/size guard; got: {msg}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn normal_stream_under_the_cap_completes_cleanly() {
        // Boundary: a normal short reply under the cap must complete (the guard
        // must not abort legitimate responses).
        let mut mock = mockito::Server::new_async().await;
        let body = "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n\
                    data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
                    data: [DONE]\n\n";
        let _m = mock
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(body)
            .create_async()
            .await;

        let client = Client::new(mock.url(), "m".to_string(), None);
        let result = client
            .chat_completion_stream(
                "m",
                vec![ChatMessage::user("hi")],
                None,
                CancellationToken::new(),
                8 * 1024 * 1024, // realistic cap
                None,
                0, // SEC-DOS-1 inactivity disabled in this test
                0, // P1 pre-stream cap disabled in this test
                |_chunk| {},
            )
            .await
            .expect("a short reply under the cap must complete");
        assert_eq!(result.finish_reason, "stop");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn openrouter_midstream_error_chunk_surfaces_as_error_not_clean_finish() {
        // Gap 5: OpenRouter delivers a mid-stream failure as an HTTP-200
        // text/event-stream chunk carrying a top-level `error` object AND a
        // `choices` array with finish_reason:"error". Before the fix this
        // deserialized cleanly and ended the turn as a normal completion, silently
        // dropping the error (and masking billable retries). The fix surfaces a
        // tagged transport error. MUTATION CHECK: remove the `raw.error` guard in
        // chat_completion_stream and this returns Ok(finish_reason:"error").
        let mut mock = mockito::Server::new_async().await;
        let body = "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"error\":{\"code\":429,\"message\":\"Rate limit exceeded\"},\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"error\"}]}\n\n\
                    data: [DONE]\n\n";
        let _m = mock
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(body)
            .create_async()
            .await;

        let client = Client::new(mock.url(), "openai/gpt-4o-mini".to_string(), None);
        let err = client
            .chat_completion_stream(
                "openai/gpt-4o-mini",
                vec![ChatMessage::user("hi")],
                None,
                CancellationToken::new(),
                8 * 1024 * 1024,
                None,
                0,
                0,
                |_chunk| {},
            )
            .await
            .expect_err("an in-band error chunk must surface as Err, not a clean finish");
        let msg = err.to_string();
        assert!(msg.contains("rate_limited"), "must classify the embedded 429 code: {msg}");
        assert!(msg.contains("Rate limit exceeded"), "must surface the provider message: {msg}");
        assert!(msg.contains("mid-stream"), "must tag it as a mid-stream backend error: {msg}");
    }

    #[test]
    fn http_status_from_error_object_reads_numeric_string_and_type() {
        use serde_json::json;
        // Numeric code: used verbatim (OpenRouter echoes the upstream HTTP status).
        assert_eq!(http_status_from_error_object(&json!({"code": 429})), Some(429));
        assert_eq!(http_status_from_error_object(&json!({"code": 401})), Some(401));
        // String code (OpenAI-style discriminator): mapped to a status.
        assert_eq!(
            http_status_from_error_object(&json!({"code": "rate_limit_exceeded"})),
            Some(429)
        );
        assert_eq!(
            http_status_from_error_object(&json!({"code": "invalid_api_key"})),
            Some(401)
        );
        // `type` is consulted when `code` is absent.
        assert_eq!(
            http_status_from_error_object(&json!({"type": "insufficient_quota"})),
            Some(429)
        );
        // Unrecognized / absent discriminator -> None (caller falls back to body).
        assert_eq!(http_status_from_error_object(&json!({"message": "boom"})), None);
        assert_eq!(http_status_from_error_object(&json!({"code": "weird_unknown"})), None);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn midstream_error_with_string_code_classifies_rate_limited_not_unknown() {
        // Gap-5 hardening (audit MAJOR-C): OpenRouter/OpenAI often deliver the
        // in-band error `code` as a STRING ("rate_limit_exceeded") rather than a
        // numeric 429. The original site read `code` as u64 only, defaulted to
        // status 200, and tagged the failure [unknown]. It must now classify the
        // string code as rate_limited.
        let mut mock = mockito::Server::new_async().await;
        let body = "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"error\":{\"code\":\"rate_limit_exceeded\",\"message\":\"Slow down\"},\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"error\"}]}\n\n\
                    data: [DONE]\n\n";
        let _m = mock
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(body)
            .create_async()
            .await;

        let client = Client::new(mock.url(), "openai/gpt-4o-mini".to_string(), None);
        let err = client
            .chat_completion_stream(
                "openai/gpt-4o-mini",
                vec![ChatMessage::user("hi")],
                None,
                CancellationToken::new(),
                8 * 1024 * 1024,
                None,
                0,
                0,
                |_chunk| {},
            )
            .await
            .expect_err("a string-code in-band error must still surface as Err");
        let msg = err.to_string();
        assert!(
            msg.contains("rate_limited"),
            "string code rate_limit_exceeded must classify rate_limited, not unknown: {msg}"
        );
        assert!(msg.contains("mid-stream"), "must tag it as a mid-stream backend error: {msg}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn probe_classifies_native_when_backend_only_accepts_string_tool_choice() {
        // v0.2.6 regression guard. LM Studio / llama.cpp servers REJECT the
        // OpenAI object-form `tool_choice: {type:function, function:{name}}` with
        // HTTP 400 ("Invalid tool_choice type: 'object'. Supported string values:
        // none, auto, required") but accept the string `"required"`. The old probe
        // sent the object form, treated the 400 as transient, and FAILED OPEN to
        // Emulated — mis-classifying every native-capable LM Studio model
        // (Hermes-3, Qwen3-30B) as Emulated, which then drove the EMIT-004
        // directive path into a downstream context-overflow BLACK at 30 tools.
        //
        // The two mocks match DISJOINT request shapes (object-form vs the
        // `"required"` string), so this is robust to mock-ordering. MUTATION
        // CHECK: revert the probe to the object-form tool_choice and this FAILS —
        // the object mock returns 400, the probe fails open, tier == Emulated.
        let mut mock = mockito::Server::new_async().await;
        // Object-form tool_choice -> the LM Studio 400 (what the OLD probe sent).
        let _obj = mock
            .mock("POST", "/chat/completions")
            .match_body(mockito::Matcher::Regex(r#""tool_choice":\s*\{"#.to_string()))
            .with_status(400)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{"error":"Invalid tool_choice type: 'object'. Supported string values: none, auto, required"}"#,
            )
            .create_async()
            .await;
        // String "required" -> a clean native tool_call (what the FIXED probe sends).
        let _req = mock
            .mock("POST", "/chat/completions")
            .match_body(mockito::Matcher::Regex(
                r#""tool_choice":\s*"required""#.to_string(),
            ))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{"choices":[{"message":{"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"find_blueprints","arguments":"{\"searchTerm\":\"test\"}"}}]},"finish_reason":"tool_calls"}]}"#,
            )
            .create_async()
            .await;

        let client = Client::new(mock.url(), "hermes-3-llama-3.1-8b".to_string(), None);
        let assessment = client.probe_tool_capability("hermes-3-llama-3.1-8b").await;
        assert_eq!(
            assessment.tier,
            ToolTier::Native,
            "a backend that rejects object-form tool_choice but accepts the \"required\" string must probe Native, not fail-open to Emulated"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn probe_retries_without_tool_choice_when_string_form_is_also_rejected() {
        // v0.2.6: belt-and-suspenders. A backend that rejects EVERY tool_choice
        // form (object AND the "required" string) must NOT fail-open to Emulated —
        // the probe drops tool_choice entirely and retries (the prompt already
        // asks the model to call find_blueprints). First mock (carries a
        // tool_choice key) 400s; the retry omits the key and succeeds.
        let mut mock = mockito::Server::new_async().await;
        let _rej = mock
            .mock("POST", "/chat/completions")
            .match_body(mockito::Matcher::Regex(r#""tool_choice""#.to_string()))
            .with_status(400)
            .with_header("content-type", "application/json")
            .with_body(r#"{"error":"unsupported parameter: tool_choice"}"#)
            .create_async()
            .await;
        // The no-tool_choice retry (match-all fallback) -> native tool_call.
        let _ok = mock
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{"choices":[{"message":{"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"find_blueprints","arguments":"{\"searchTerm\":\"test\"}"}}]},"finish_reason":"tool_calls"}]}"#,
            )
            .create_async()
            .await;

        let client = Client::new(mock.url(), "some-model".to_string(), None);
        let assessment = client.probe_tool_capability("some-model").await;
        assert_eq!(
            assessment.tier,
            ToolTier::Native,
            "rejection of every tool_choice form must trigger a no-tool_choice retry, not fail-open to Emulated"
        );
    }

    #[test]
    fn client_new_normalizes_localhost_to_ipv4_but_leaves_explicit_ipv6() {
        // G-NET-2: on Windows, reqwest/hyper resolves `localhost` to ::1 first,
        // but Ollama/llama.cpp default to binding 127.0.0.1 only — so a user who
        // types `localhost` in the Endpoint URL gets `connection error` on an
        // otherwise-correct config. Client::new rewrites the host to 127.0.0.1;
        // an explicit `[::1]` is left untouched for users who genuinely want v6.
        let cases = [
            ("http://localhost:11434/v1", "http://127.0.0.1:11434/v1"),
            ("http://localhost/v1", "http://127.0.0.1/v1"),
            ("http://localhost", "http://127.0.0.1"),
            ("http://[::1]:11434/v1", "http://[::1]:11434/v1"),
            ("http://192.168.1.50:1234/v1", "http://192.168.1.50:1234/v1"),
        ];
        for (input, expected) in cases {
            let c = Client::new(input.to_string(), "m".to_string(), None);
            assert_eq!(c.base_url(), expected, "normalizing {input}");
        }
    }

    #[test]
    fn http_error_kind_classifies_common_prompt_path_failures() {
        // P0-C: the prompt path used to collapse every backend HTTP failure to a
        // flat -32000 with no machine-readable kind. Classify so the bridge can
        // branch (show "check API key" on auth, "rate limited" on 429, etc.).
        assert_eq!(classify_http_error_kind(401, ""), "auth");
        assert_eq!(classify_http_error_kind(403, ""), "auth");
        assert_eq!(classify_http_error_kind(404, ""), "not_found");
        assert_eq!(classify_http_error_kind(429, ""), "rate_limited");
        assert_eq!(classify_http_error_kind(500, ""), "server_error");
        assert_eq!(classify_http_error_kind(503, ""), "server_error");
        // Body-driven kinds — the cause is in the text, not the status code:
        assert_eq!(
            classify_http_error_kind(400, "This model's maximum context length is 4096 tokens"),
            "context_overflow"
        );
        assert_eq!(
            classify_http_error_kind(200, "Model unloaded. Please reload the model."),
            "model_unloaded"
        );
        // Body cause wins even when the status alone would say something else.
        assert_eq!(
            classify_http_error_kind(400, "context window exceeded"),
            "context_overflow"
        );
        assert_eq!(classify_http_error_kind(418, "teapot"), "unknown");
    }

    #[test]
    fn tls_cert_error_is_distinguished_from_plain_unreachable() {
        // G-NET-1: rustls trusts bundled webpki-roots, not the OS store, so a
        // corporate TLS-intercepting proxy yields an UnknownIssuer cert error.
        // That must classify as tls_cert (actionable), NOT "unreachable" (which
        // implies the backend is down — the wrong cause).
        assert!(is_tls_cert_error(
            "error sending request -> invalid peer certificate: UnknownIssuer"
        ));
        assert!(is_tls_cert_error("the server certificate was not trusted"));
        assert!(!is_tls_cert_error(
            "tcp connect error: Connection refused (os error 111)"
        ));
        assert!(!is_tls_cert_error(
            "dns error: failed to lookup address information"
        ));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn turn_deadline_already_reached_aborts_with_turn_timeout() {
        // P0-E (wall-clock, design-validated): the per-turn deadline is the only
        // bound that stops a CONTINUOUSLY-EMITTING repetition loop (the read
        // timeout is off and the host first-token timer disarms after the first
        // token). A deadline that has already elapsed proves the select arm fires
        // and takes precedence over further streaming.
        let mut mock = mockito::Server::new_async().await;
        let body = "data: {\"choices\":[{\"delta\":{\"content\":\"loop loop loop\"}}]}\n\n";
        let _m = mock
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(body)
            .create_async()
            .await;

        let client = Client::new(mock.url(), "m".to_string(), None);
        // Make the deadline robustly in the PAST — well beyond tokio's ~1ms timer
        // granularity — so sleep_until is unconditionally Ready on the first poll.
        // A same-tick `now()` capture could otherwise be Pending on first poll and
        // let the already-ready mock body win the (biased) select instead.
        let deadline = tokio::time::Instant::now();
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        let err = client
            .chat_completion_stream(
                "m",
                vec![ChatMessage::user("hi")],
                None,
                CancellationToken::new(),
                8 * 1024 * 1024,
                Some(deadline),
                0, // inactivity disabled — isolating the wall-clock deadline
                0, // P1 pre-stream cap disabled in this test
                |_chunk| {},
            )
            .await
            .expect_err("a turn past its deadline must abort");
        assert!(
            err.to_string().contains("turn_timeout"),
            "abort must be a turn_timeout; got: {err}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn no_turn_deadline_lets_a_normal_reply_complete() {
        // Boundary: deadline = None (NWIRO_LOCAL_LLM_MAX_TURN_DURATION_SECS=0)
        // must never abort a legitimate reply.
        let mut mock = mockito::Server::new_async().await;
        let body = "data: {\"choices\":[{\"delta\":{\"content\":\"done\"}}]}\n\n\
                    data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
                    data: [DONE]\n\n";
        let _m = mock
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(body)
            .create_async()
            .await;

        let client = Client::new(mock.url(), "m".to_string(), None);
        let result = client
            .chat_completion_stream(
                "m",
                vec![ChatMessage::user("hi")],
                None,
                CancellationToken::new(),
                8 * 1024 * 1024,
                None, // deadline disabled
                0, // inactivity disabled
                0, // P1 pre-stream cap disabled in this test
                |_chunk| {},
            )
            .await
            .expect("no deadline must let the reply complete");
        assert_eq!(result.finish_reason, "stop");
        assert_eq!(result.final_message.content_text(), Some("done"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stalled_stream_aborts_with_inactivity_timeout() {
        // SEC-DOS-1: a backend that sends ONE content chunk then goes SILENT (holds
        // the connection open, emits nothing more) must be aborted by the inactivity
        // guard. The wall-clock deadline is disabled and the response is tiny, so
        // ONLY the per-token inactivity timeout can catch this stall. A mockito mock
        // can't model a mid-stream hang (it closes after the body), so we hand-roll
        // a TCP server that writes one SSE event, then sleeps. Mutation-valid: with
        // inactivity disabled (0) the call would hang until the server closes.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            if let Ok((mut sock, _)) = listener.accept().await {
                use tokio::io::{AsyncReadExt, AsyncWriteExt};
                let mut buf = [0u8; 8192];
                let _ = sock.read(&mut buf).await; // drain the request head
                let _ = sock
                    .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n")
                    .await;
                let _ = sock
                    .write_all(b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n")
                    .await;
                let _ = sock.flush().await;
                // Then STALL: hold the socket open, emit nothing well past the cap.
                tokio::time::sleep(std::time::Duration::from_secs(10)).await;
            }
        });

        let client = Client::new(format!("http://{addr}/v1"), "m".to_string(), None);
        let err = client
            .chat_completion_stream(
                "m",
                vec![ChatMessage::user("hi")],
                None,
                CancellationToken::new(),
                8 * 1024 * 1024,
                None, // wall-clock deadline disabled — isolate the inactivity guard
                1,    // inactivity = 1s
                0, // P1 pre-stream cap disabled in this test
                |_chunk| {},
            )
            .await
            .expect_err("a stalled stream must abort via the inactivity guard");
        assert!(
            err.to_string().contains("stream_inactivity_timeout"),
            "expected a stream_inactivity_timeout abort; got: {err}"
        );
        server.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn inactivity_deadline_resets_on_each_event_so_a_paced_stream_completes() {
        // SEC-DOS-1 RESET branch (the `inactivity_deadline` re-arm on each received
        // event): a stream that keeps emitting with gaps SHORTER than the cap must
        // complete even when the TOTAL elapsed exceeds the cap. The existing stall test
        // only proves the FIRST period fires; this proves the per-event RE-ARM — the
        // load-bearing branch a v0.3.0 gap-analysis flagged as un-asserted. Mutation-
        // valid: pin the deadline once (drop the reset) and the two 0.6s gaps (1.2s
        // total > the 1s cap) trip the guard at ~1.0s, before `[DONE]` at ~1.2s ->
        // spurious abort -> this `.expect()` fails.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            if let Ok((mut sock, _)) = listener.accept().await {
                let mut buf = [0u8; 8192];
                let _ = sock.read(&mut buf).await; // drain the request head
                let _ = sock
                    .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n")
                    .await;
                let _ = sock
                    .write_all(b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n")
                    .await;
                let _ = sock.flush().await;
                // 0.6s gap (< 1s cap): a working reset re-arms the deadline here.
                tokio::time::sleep(std::time::Duration::from_millis(600)).await;
                let _ = sock
                    .write_all(
                        b"data: {\"choices\":[{\"delta\":{\"content\":\" there\"},\"finish_reason\":\"stop\"}]}\n\n",
                    )
                    .await;
                let _ = sock.flush().await;
                // Another 0.6s gap — total 1.2s > the 1s cap; only a re-armed deadline
                // survives to here.
                tokio::time::sleep(std::time::Duration::from_millis(600)).await;
                let _ = sock.write_all(b"data: [DONE]\n\n").await;
                let _ = sock.flush().await;
            }
        });

        let client = Client::new(format!("http://{addr}/v1"), "m".to_string(), None);
        let result = client
            .chat_completion_stream(
                "m",
                vec![ChatMessage::user("hi")],
                None,
                CancellationToken::new(),
                8 * 1024 * 1024,
                None, // wall-clock deadline disabled — isolate the inactivity reset
                1,    // inactivity = 1s; each gap is 0.6s, total elapsed 1.2s
                0,    // pre-stream cap disabled
                |_chunk| {},
            )
            .await
            .expect("a paced stream (each gap < the cap) must complete — the deadline re-arms");
        assert_eq!(result.finish_reason, "stop");
        assert_eq!(result.final_message.content_text(), Some("hi there"));
        server.abort();
    }

    // ── v0.3.0 P1: pre-stream timeout cap + exponential backoff ──────────────

    #[test]
    fn effective_prestream_cap_clamps_above_connect_timeout() {
        // `0` (disabled) passes through untouched.
        assert_eq!(effective_prestream_cap(0, 10), 0);
        // A cap at/below the connect timeout is raised to connect+1, so the connect
        // timeout can fire FIRST — a slow CONNECT stays `unreachable`, not a spurious
        // transient `timeout` that would be wrongly retried.
        assert_eq!(effective_prestream_cap(5, 10), 11);
        assert_eq!(effective_prestream_cap(10, 10), 11);
        // A cap already above the connect timeout is left as the operator set it.
        assert_eq!(effective_prestream_cap(30, 10), 30);
        // The clamp tracks a non-default connect timeout.
        assert_eq!(effective_prestream_cap(3, 20), 21);
    }

    #[test]
    fn effective_max_prestream_attempts_fits_the_watchdog_budget() {
        // Default local cap (30s): the full ceiling — 240/30 = 8, clamped to 3.
        assert_eq!(effective_max_prestream_attempts(30), MAX_PROMPT_ATTEMPTS);
        // Documented cloud config (CONNECT_TIMEOUT=120 → effective cap 121s): a fixed
        // 3 attempts would be ~363s > the 300s host watchdog, so the count drops to ONE
        // (121s ≪ 300s) — the diagnosable [timeout] refusal still beats the watchdog.
        assert_eq!(effective_max_prestream_attempts(121), 1);
        // Mid-range: 240/100 = 2.
        assert_eq!(effective_max_prestream_attempts(100), 2);
        // Never below 1, even when a single attempt alone exceeds the budget (a
        // pathological connect timeout above the host watchdog — unfixable here, but
        // one bounded attempt still runs rather than zero).
        assert_eq!(effective_max_prestream_attempts(500), 1);
        // CORE INVARIANT: for every cap, either it's a single (minimal) attempt or the
        // product stays within the budget.
        for cap in 1..=300u64 {
            let n = effective_max_prestream_attempts(cap) as u64;
            assert!(
                n == 1 || n * cap <= PRESTREAM_TOTAL_BUDGET_SECS,
                "cap={cap}: {n} attempts × {cap}s exceeds the {PRESTREAM_TOTAL_BUDGET_SECS}s budget"
            );
        }
        // `0` (cap disabled) keeps the full ceiling — only fast HTTP-error retries run.
        assert_eq!(effective_max_prestream_attempts(0), MAX_PROMPT_ATTEMPTS);
    }

    #[test]
    fn retry_backoff_is_exponential_and_honors_retry_after() {
        // server_error: 250ms base, doubling per 0-indexed retry. Jitter (<=99ms)
        // can't invert the order: b0 in [250,350), b1 in [500,600).
        let b0 = retry_backoff("server_error", None, 0);
        let b1 = retry_backoff("server_error", None, 1);
        assert!(
            b0 >= std::time::Duration::from_millis(250)
                && b0 < std::time::Duration::from_millis(350),
            "b0={b0:?}"
        );
        assert!(
            b1 >= std::time::Duration::from_millis(500)
                && b1 < std::time::Duration::from_millis(600),
            "b1={b1:?}"
        );
        assert!(b1 > b0, "exponential growth: b1={b1:?} > b0={b0:?}");
        // rate_limited WITH a Retry-After honors the hint exactly (no exponential,
        // no jitter), clamped to the ceiling.
        assert_eq!(
            retry_backoff("rate_limited", Some(1), 0),
            std::time::Duration::from_millis(1000)
        );
        assert_eq!(
            retry_backoff("rate_limited", Some(60), 2),
            std::time::Duration::from_millis(MAX_RETRY_BACKOFF_MS)
        );
        // A pathological exponent can never exceed the ceiling (no shift overflow).
        assert!(
            retry_backoff("server_error", None, 64)
                <= std::time::Duration::from_millis(MAX_RETRY_BACKOFF_MS)
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn prestream_stall_exhausts_with_timeout_kind() {
        // P1: a backend that ACCEPTS the connection but never sends response headers
        // must fail per-attempt within the pre-stream cap, RETRY up to
        // MAX_PROMPT_ATTEMPTS, then surface a diagnosable `[timeout]`. Mutation-valid:
        // with the cap disabled (0) — the pre-P1 behavior — this hangs forever.
        use std::sync::atomic::{AtomicUsize, Ordering};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let conns = std::sync::Arc::new(AtomicUsize::new(0));
        let conns_srv = conns.clone();
        let server = tokio::spawn(async move {
            let mut held = Vec::new();
            loop {
                if let Ok((sock, _)) = listener.accept().await {
                    conns_srv.fetch_add(1, Ordering::SeqCst);
                    held.push(sock); // hold open, never write a byte
                }
            }
        });

        let client = Client::new(format!("http://{addr}/v1"), "m".to_string(), None);
        let started = std::time::Instant::now();
        let err = client
            .chat_completion_stream(
                "m",
                vec![ChatMessage::user("hi")],
                None,
                CancellationToken::new(),
                8 * 1024 * 1024,
                None, // wall-clock deadline disabled — isolate the pre-stream cap
                0,    // inactivity disabled (never arms; no stream ever starts)
                1,    // pre-stream cap = 1s per attempt
                |_chunk| {},
            )
            .await
            .expect_err("a backend that never answers must abort, not hang");

        assert!(
            err.to_string().contains("[timeout]"),
            "exhausted pre-stream stall must surface [timeout]; got: {err}"
        );
        // The timeout must be RETRIED (initial + at least one retry).
        assert!(
            conns.load(Ordering::SeqCst) >= 2,
            "expected >=2 attempts; saw {} connection(s)",
            conns.load(Ordering::SeqCst)
        );
        // Bounded by the caps + tiny backoffs — never the unbounded pre-P1 hang.
        assert!(
            started.elapsed() < std::time::Duration::from_secs(15),
            "must be bounded by the caps (took {:?})",
            started.elapsed()
        );
        server.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn prestream_stall_then_success_retries_and_streams() {
        // P1: attempt 1 stalls before headers (caught by the cap); the retry reaches a
        // now-responsive backend and streams cleanly. Proves the timeout path RETRIES
        // rather than failing the turn on the first stall.
        use std::sync::atomic::{AtomicUsize, Ordering};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let n = std::sync::Arc::new(AtomicUsize::new(0));
        let n_srv = n.clone();
        let server = tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let mut held = Vec::new();
            loop {
                if let Ok((mut sock, _)) = listener.accept().await {
                    let which = n_srv.fetch_add(1, Ordering::SeqCst);
                    if which == 0 {
                        held.push(sock); // first attempt: hold, never answer
                    } else {
                        let mut buf = [0u8; 8192];
                        let _ = sock.read(&mut buf).await; // drain the request head
                        let _ = sock
                            .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n")
                            .await;
                        let _ = sock
                            .write_all(b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n")
                            .await;
                        let _ = sock
                            .write_all(
                                b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
                            )
                            .await;
                        let _ = sock.write_all(b"data: [DONE]\n\n").await;
                        let _ = sock.flush().await;
                        held.push(sock);
                    }
                }
            }
        });

        let client = Client::new(format!("http://{addr}/v1"), "m".to_string(), None);
        let result = client
            .chat_completion_stream(
                "m",
                vec![ChatMessage::user("hi")],
                None,
                CancellationToken::new(),
                8 * 1024 * 1024,
                None,
                0,
                1, // pre-stream cap = 1s: attempt 1 stalls, attempt 2 succeeds
                |_chunk| {},
            )
            .await
            .expect("the retry must reach the responsive backend");
        assert_eq!(result.finish_reason, "stop");
        assert_eq!(result.final_message.content_text(), Some("hi"));
        assert!(
            n.load(Ordering::SeqCst) >= 2,
            "the stalled first attempt must have been retried"
        );
        server.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fatal_status_with_stalled_body_keeps_kind_and_is_not_retried() {
        // P1 + review fix: a backend that returns a FATAL status (401) instantly
        // then STALLS the body must (a) be bounded by the pre-stream cap (not hang on
        // resp.text()), AND (b) surface the header-known kind [auth] WITHOUT retrying — a
        // slow diagnostic body must NOT reclassify a fatal response as a transient
        // `timeout`. Mutation-valid: the single-wrapper design returned [timeout] and
        // retried 3×; a send-only cap would hang on the body read.
        use std::sync::atomic::{AtomicUsize, Ordering};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let conns = std::sync::Arc::new(AtomicUsize::new(0));
        let conns_srv = conns.clone();
        let server = tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let mut held = Vec::new();
            loop {
                if let Ok((mut sock, _)) = listener.accept().await {
                    conns_srv.fetch_add(1, Ordering::SeqCst);
                    let mut buf = [0u8; 8192];
                    let _ = sock.read(&mut buf).await;
                    // 401 headers arrive instantly; the promised 4096-byte body never fully
                    // arrives, so resp.text() would hang without a bounded read.
                    let _ = sock
                        .write_all(
                            b"HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\nContent-Length: 4096\r\n\r\n{\"error\":\"",
                        )
                        .await;
                    let _ = sock.flush().await;
                    held.push(sock); // hold open: never send the remaining body bytes
                }
            }
        });

        let client = Client::new(format!("http://{addr}/v1"), "m".to_string(), None);
        let started = std::time::Instant::now();
        let err = client
            .chat_completion_stream(
                "m",
                vec![ChatMessage::user("hi")],
                None,
                CancellationToken::new(),
                8 * 1024 * 1024,
                None,
                0,
                1, // pre-stream cap = 1s — bounds the stalled body read
                |_chunk| {},
            )
            .await
            .expect_err("a 401 must surface as an error");
        assert!(
            err.to_string().contains("[auth]"),
            "a fatal 401 with a stalled body must keep its [auth] kind, not become [timeout]; got: {err}"
        );
        assert_eq!(
            conns.load(Ordering::SeqCst),
            1,
            "a fatal status must NOT be retried even when its body stalls; saw {} connection(s)",
            conns.load(Ordering::SeqCst)
        );
        assert!(
            started.elapsed() < std::time::Duration::from_secs(10),
            "the stalled body read must be bounded by the cap (took {:?})",
            started.elapsed()
        );
        server.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn retriable_status_with_stalled_body_keeps_kind_and_is_bounded() {
        // Counterpart to the fatal case: a 503 with a stalled body stays the RETRIABLE
        // `server_error` kind (classified from status, body unavailable), is retried to
        // exhaustion, and surfaces [server_error] — NOT a generic [timeout]. Proves the
        // body read is bounded (no hang) AND status classification is preserved for the
        // transient class too.
        use std::sync::atomic::{AtomicUsize, Ordering};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let conns = std::sync::Arc::new(AtomicUsize::new(0));
        let conns_srv = conns.clone();
        let server = tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let mut held = Vec::new();
            loop {
                if let Ok((mut sock, _)) = listener.accept().await {
                    conns_srv.fetch_add(1, Ordering::SeqCst);
                    let mut buf = [0u8; 8192];
                    let _ = sock.read(&mut buf).await;
                    let _ = sock
                        .write_all(
                            b"HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nContent-Length: 4096\r\n\r\n{\"error\":\"",
                        )
                        .await;
                    let _ = sock.flush().await;
                    held.push(sock);
                }
            }
        });

        let client = Client::new(format!("http://{addr}/v1"), "m".to_string(), None);
        let started = std::time::Instant::now();
        let err = client
            .chat_completion_stream(
                "m",
                vec![ChatMessage::user("hi")],
                None,
                CancellationToken::new(),
                8 * 1024 * 1024,
                None,
                0,
                1, // pre-stream cap = 1s per attempt
                |_chunk| {},
            )
            .await
            .expect_err("an exhausted 503 must surface an error");
        assert!(
            err.to_string().contains("[server_error]"),
            "a 503 with a stalled body must keep its [server_error] kind, not become [timeout]; got: {err}"
        );
        assert!(
            conns.load(Ordering::SeqCst) >= 2,
            "a transient status must be RETRIED; saw {} connection(s)",
            conns.load(Ordering::SeqCst)
        );
        assert!(
            started.elapsed() < std::time::Duration::from_secs(15),
            "the retried stalled body reads must be bounded by the cap (took {:?})",
            started.elapsed()
        );
        server.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn disabled_prestream_cap_does_not_arm_a_timeout() {
        // Review should-fix: with the cap DISABLED (0), a slow-to-respond
        // backend must still complete — the disable branch must not arm any timeout
        // (no `timeout_at`, no `Duration::from_secs(0)` that would fire instantly). The
        // server delays HEADERS ~1.5s (well past a typical small cap) before streaming a
        // clean reply; with cap=0 the call succeeds. Mutation-valid: an always-armed cap
        // (or a 0→instant-timeout bug) would abort this before the headers arrive.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let mut held = Vec::new();
            if let Ok((mut sock, _)) = listener.accept().await {
                let mut buf = [0u8; 8192];
                let _ = sock.read(&mut buf).await;
                // Slow headers: nothing is sent for ~1.5s.
                tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
                let _ = sock
                    .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n")
                    .await;
                let _ = sock
                    .write_all(b"data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n")
                    .await;
                let _ = sock
                    .write_all(b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
                    .await;
                let _ = sock.write_all(b"data: [DONE]\n\n").await;
                let _ = sock.flush().await;
                held.push(sock);
            }
        });

        let client = Client::new(format!("http://{addr}/v1"), "m".to_string(), None);
        let result = client
            .chat_completion_stream(
                "m",
                vec![ChatMessage::user("hi")],
                None,
                CancellationToken::new(),
                8 * 1024 * 1024,
                None,
                0,
                0, // pre-stream cap DISABLED — the 1.5s header delay must NOT abort
                |_chunk| {},
            )
            .await
            .expect("cap=0 must not arm a timeout on a slow-but-healthy backend");
        assert_eq!(result.finish_reason, "stop");
        assert_eq!(result.final_message.content_text(), Some("ok"));
        server.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn json_envelope_with_stalled_body_stays_fatal_and_is_not_retried() {
        // Review should-fix: a 200 + application/json "did not stream" envelope
        // whose BODY stalls must stay FATAL (not retried) — a 200 non-stream response is
        // a hard failure regardless of body content. Bounded by the cap; surfaces the
        // clean "did not stream a response" failure in exactly one attempt.
        use std::sync::atomic::{AtomicUsize, Ordering};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let conns = std::sync::Arc::new(AtomicUsize::new(0));
        let conns_srv = conns.clone();
        let server = tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let mut held = Vec::new();
            loop {
                if let Ok((mut sock, _)) = listener.accept().await {
                    conns_srv.fetch_add(1, Ordering::SeqCst);
                    let mut buf = [0u8; 8192];
                    let _ = sock.read(&mut buf).await;
                    let _ = sock
                        .write_all(
                            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 4096\r\n\r\n{\"error\":\"",
                        )
                        .await;
                    let _ = sock.flush().await;
                    held.push(sock); // hold open: body never completes
                }
            }
        });

        let client = Client::new(format!("http://{addr}/v1"), "m".to_string(), None);
        let started = std::time::Instant::now();
        let err = client
            .chat_completion_stream(
                "m",
                vec![ChatMessage::user("hi")],
                None,
                CancellationToken::new(),
                8 * 1024 * 1024,
                None,
                0,
                1, // pre-stream cap = 1s — bounds the stalled envelope body read
                |_chunk| {},
            )
            .await
            .expect_err("a 200 non-stream JSON envelope must surface as an error");
        assert!(
            err.to_string().contains("did not stream a response"),
            "a stalled 200-JSON envelope must stay the fatal 'did not stream' failure; got: {err}"
        );
        assert_eq!(
            conns.load(Ordering::SeqCst),
            1,
            "a 200 non-stream response is fatal and must NOT be retried; saw {} connection(s)",
            conns.load(Ordering::SeqCst)
        );
        assert!(
            started.elapsed() < std::time::Duration::from_secs(10),
            "the stalled envelope body read must be bounded by the cap (took {:?})",
            started.elapsed()
        );
        server.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancel_takes_precedence_over_a_passed_deadline() {
        // Precedence contract (the `biased` select: cancel, then deadline, then
        // stream): a token cancelled before the deadline fires must yield
        // Cancelled, not turn_timeout. Pins the documented ordering so a future
        // arm reorder can't silently flip it.
        let mut mock = mockito::Server::new_async().await;
        let _m = mock
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body("data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n")
            .create_async()
            .await;

        let client = Client::new(mock.url(), "m".to_string(), None);
        let cancel = CancellationToken::new();
        cancel.cancel(); // already cancelled before the call
        let err = client
            .chat_completion_stream(
                "m",
                vec![ChatMessage::user("hi")],
                None,
                cancel,
                8 * 1024 * 1024,
                Some(tokio::time::Instant::now()), // deadline also ready
                0, // inactivity disabled
                0, // P1 pre-stream cap disabled in this test
                |_chunk| {},
            )
            .await
            .expect_err("a cancelled turn must abort");
        assert!(
            matches!(err, ShimError::Cancelled),
            "cancel must take precedence over the deadline; got: {err}"
        );
    }

    #[test]
    fn think_stripper_removes_inline_think_block() {
        let mut s = ThinkStripper::default();
        let out = s.push("<think>secret reasoning</think>the answer");
        assert_eq!(out, "the answer");
        assert_eq!(s.flush(), "");
    }

    #[test]
    fn think_stripper_handles_tag_split_across_chunks() {
        // The open AND close tags arrive split across deltas — the exact shape
        // that defeats a stateless per-chunk strip.
        let mut s = ThinkStripper::default();
        let mut emitted = String::new();
        for d in ["before <thi", "nk>hidden", " more</thi", "nk>after"] {
            emitted.push_str(&s.push(d));
        }
        emitted.push_str(&s.flush());
        assert_eq!(emitted, "before after");
    }

    #[test]
    fn think_stripper_passes_plain_content_unchanged() {
        let mut s = ThinkStripper::default();
        let mut emitted = String::new();
        for d in ["Hello ", "world", "! 2 < 3 is ", "true"] {
            emitted.push_str(&s.push(d));
        }
        emitted.push_str(&s.flush());
        // A literal `<` that is not a think tag must survive intact.
        assert_eq!(emitted, "Hello world! 2 < 3 is true");
    }

    #[test]
    fn think_stripper_drops_a_lone_close_tag_in_content() {
        // A backend can route the `<think>` OPEN through delta.reasoning (or split
        // the close into content), surfacing a bare `</think>` in delta.content
        // with no matching open. It must be dropped, not emitted as the answer
        // (the deepseek-r1 reasoning-leak BLACK-risk). The old stripper searched
        // only for OPEN, so this lone close leaked verbatim.
        let mut s = ThinkStripper::default();
        assert_eq!(s.push("</think>The door is open."), "The door is open.");
        assert_eq!(s.flush(), "");
    }

    #[test]
    fn think_stripper_drops_lone_close_before_emulated_envelope() {
        // Emulated-tier co-emission: a leaked `</think>` is stripped while the
        // emulated tool envelope that follows is left intact for the parser.
        let mut s = ThinkStripper::default();
        let out = s.push("</think>\n\n{\"tool\": \"spawn_actor\", \"arguments\": {}}");
        assert_eq!(out, "\n\n{\"tool\": \"spawn_actor\", \"arguments\": {}}");
        assert_eq!(s.flush(), "");
    }

    #[test]
    fn think_stripper_holds_a_lone_close_split_across_chunks() {
        // A bare close split at a delta boundary must be held + dropped, not
        // partially emitted (the !in_think tail now holds a partial CLOSE too).
        let mut s = ThinkStripper::default();
        assert_eq!(s.push("answer </th"), "answer ");
        assert_eq!(s.push("ink> more"), " more");
        assert_eq!(s.flush(), "");
    }

    #[test]
    fn think_stripper_drops_unterminated_block() {
        // A think block with no closing tag (stream ended mid-thought) must not
        // leak — flush drops it rather than emitting raw reasoning.
        let mut s = ThinkStripper::default();
        let out = s.push("answer<think>still thinking when the stream ended");
        assert_eq!(out, "answer");
        assert_eq!(s.flush(), "");
    }

    #[test]
    fn think_stripper_is_utf8_safe_on_partial_tag_boundary() {
        // A multibyte char immediately before a partial "<" must not panic the
        // boundary slicing in hold_partial_suffix.
        let mut s = ThinkStripper::default();
        let mut emitted = String::new();
        emitted.push_str(&s.push("café<"));
        emitted.push_str(&s.push("think>x</think>!"));
        emitted.push_str(&s.flush());
        assert_eq!(emitted, "café!");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn in_content_think_markup_is_stripped_from_the_answer() {
        // G-MODEL-1: a model on llama.cpp without a reasoning-format split emits
        // `<think>...</think>` INSIDE delta.content. The shim must strip it so the
        // chain-of-thought never becomes the assistant's visible answer.
        let mut mock = mockito::Server::new_async().await;
        let body = "data: {\"choices\":[{\"delta\":{\"content\":\"<think>plan: open the door\"}}]}\n\n\
                    data: {\"choices\":[{\"delta\":{\"content\":\"</think>The door is open.\"}}]}\n\n\
                    data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
                    data: [DONE]\n\n";
        let _m = mock
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(body)
            .create_async()
            .await;

        let client = Client::new(mock.url(), "m".to_string(), None);
        let result = client
            .chat_completion_stream(
                "m",
                vec![ChatMessage::user("open the door")],
                None,
                CancellationToken::new(),
                8 * 1024 * 1024,
                None,
                0, // SEC-DOS-1 inactivity disabled in this test
                0, // P1 pre-stream cap disabled in this test
                |_chunk| {},
            )
            .await
            .expect("stream completes");
        assert_eq!(
            result.final_message.content_text(),
            Some("The door is open."),
            "the <think> chain-of-thought must be stripped from the answer"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stream_ending_without_done_sentinel_terminates_cleanly() {
        // CC-03: Ollama and llama.cpp-server close the socket WITHOUT emitting
        // the `data: [DONE]` sentinel that OpenAI/vLLM send. The turn must still
        // terminate cleanly with the streamed finish_reason and full content,
        // not hang waiting for a sentinel that never arrives.
        let mut mock = mockito::Server::new_async().await;
        let body = "data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\n\n\
                    data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n\
                    data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n";
        // No `data: [DONE]` line — the stream just ends (socket close).
        let _m = mock
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(body)
            .create_async()
            .await;

        let client = Client::new(mock.url(), "m".to_string(), None);
        let result = client
            .chat_completion_stream(
                "m",
                vec![ChatMessage::user("hi")],
                None,
                CancellationToken::new(),
                8 * 1024 * 1024,
                None,
                0, // SEC-DOS-1 inactivity disabled in this test
                0, // P1 pre-stream cap disabled in this test
                |_chunk| {},
            )
            .await
            .expect("a stream that closes without [DONE] must still complete");
        assert_eq!(result.finish_reason, "stop");
        assert_eq!(result.final_message.content_text(), Some("hello"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reasoning_only_reply_yields_empty_content_and_clean_stop() {
        // CC-04: a reply that is ALL reasoning (no content) — a thinking model
        // that deliberates then stops — must end cleanly with NO assistant
        // content, not hang or fabricate an empty assistant message.
        let mut mock = mockito::Server::new_async().await;
        let body = "data: {\"choices\":[{\"delta\":{\"reasoning\":\"thinking...\"}}]}\n\n\
                    data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
                    data: [DONE]\n\n";
        let _m = mock
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(body)
            .create_async()
            .await;

        let client = Client::new(mock.url(), "m".to_string(), None);
        let result = client
            .chat_completion_stream(
                "m",
                vec![ChatMessage::user("hi")],
                None,
                CancellationToken::new(),
                8 * 1024 * 1024,
                None,
                0, // SEC-DOS-1 inactivity disabled in this test
                0, // P1 pre-stream cap disabled in this test
                |_chunk| {},
            )
            .await
            .expect("a reasoning-only reply must complete");
        assert_eq!(result.finish_reason, "stop");
        assert_eq!(result.final_message.content, None);
        assert!(result.tool_calls.is_empty());
    }

    #[test]
    fn probe_failure_fails_open_to_emulated_not_none() {
        // v0.1.35 model-agnostic: a transient / inconclusive probe must NOT
        // strip tools. `failed()` fails OPEN to Emulated (a safety superset of
        // None — tools stay attached, EMIT-004 + the mandate inject, the prose
        // parser runs) for EVERY model, so a single probe hiccup never leaves a
        // model unable to call tools.
        let a = ProbeAssessment::failed("some-unknown-model-7b");
        assert_eq!(
            a.tier,
            ToolTier::Emulated,
            "probe failure must fail OPEN to Emulated, not strip tools"
        );
        assert!(!a.schema_bleed_detected);
    }

    #[test]
    fn probe_no_tool_support_classifies_none_not_emulated() {
        // A DEFINITIVE "does not support tools" backend verdict (Ollama 400)
        // must classify None — so the bridge strips tools and the model
        // degrades to a clean tool-free response — NOT fail-open to Emulated
        // (which would re-send tools, 400 again, and surface a raw -32000).
        let a = ProbeAssessment::no_tool_support("gemma2:9b-instruct-q4_K_M");
        assert_eq!(
            a.tier,
            ToolTier::None,
            "a definitive no-tool-support verdict must classify None (strip tools)"
        );
        assert!(!a.schema_bleed_detected);
    }

    // v0.2.1 — the warmup load-request timeout (docs/RUNNING.md "Known gap").
    // A backend that ACCEPTS the TCP connection but never answers (the
    // serverless cold-start shape: connect succeeds instantly, response
    // blocks for the whole model load) must fail warmup within the cap with
    // a diagnosable `timeout` errorKind — not hang unboundedly.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn warmup_times_out_with_diagnosable_timeout_error_kind() {
        struct EnvGuard(&'static str);
        impl Drop for EnvGuard {
            fn drop(&mut self) {
                std::env::remove_var(self.0);
            }
        }
        // Process-global env var, held for ~1s (the cap) — same house
        // pattern as golden.rs EnvGuards. The only other warmup-reaching
        // test (golden lazy-probe) answers via mockito in ms, so the 1s
        // window overlapping a >1s-stalled warmup is a accepted-residual
        // flake vector (review MINOR, documented here).
        std::env::set_var("NWIRO_LOCAL_LLM_WARMUP_TIMEOUT_SECS", "1");
        let _g = EnvGuard("NWIRO_LOCAL_LLM_WARMUP_TIMEOUT_SECS");

        // Accept-and-hold server: connections succeed, no byte is ever
        // written back, sockets are kept alive so the client sees neither
        // a refusal (-> unreachable) nor a reset (-> unknown) — only the
        // per-request timeout can end the wait.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind ephemeral port");
        let addr = listener.local_addr().expect("local addr");
        tokio::spawn(async move {
            let mut held = Vec::new();
            loop {
                if let Ok((sock, _)) = listener.accept().await {
                    held.push(sock);
                }
            }
        });

        let client = Client::new(
            format!("http://{addr}/v1"),
            "test-model".to_string(),
            None,
        );
        let started = std::time::Instant::now();
        let r = client.warmup("test-model", "15m").await;

        assert_eq!(r.status, "failed", "timed-out warmup must report failed");
        assert_eq!(
            r.error_kind.as_deref(),
            Some("timeout"),
            "must be the diagnosable timeout kind, not unreachable/unknown; got {:?} / {:?}",
            r.error_kind,
            r.message
        );
        assert_eq!(
            r.tool_tier,
            ToolTier::None,
            "availability verdict: tools stay stripped until a successful warmup"
        );
        assert!(
            started.elapsed() < std::time::Duration::from_secs(10),
            "the 1s cap must bound the wait (took {:?}) — unbounded hang is the bug",
            started.elapsed()
        );
    }

    // v0.1.21 — extract_backend_error_message tests.
    //
    // IMPORTANT (per review defect 2): each test fixture MUST exercise
    // exactly one extraction path. The helper's early-return ordering
    // means a fixture with BOTH /error/message AND /message would only
    // exercise the first match, leaving downstream paths un-tested even
    // though the test name implies otherwise. Hence each test below
    // uses a fixture that ONLY contains the field the test name targets.

    #[test]
    fn extract_nested_lm_studio_error_message() {
        let v = json!({"error": {"message": "model unloaded"}});
        assert_eq!(
            extract_backend_error_message(&v),
            Some("model unloaded".to_string())
        );
    }

    #[test]
    fn extract_flat_error_string() {
        // Ollama-style flat envelope. No nested message field.
        let v = json!({"error": "context length exceeded"});
        assert_eq!(
            extract_backend_error_message(&v),
            Some("context length exceeded".to_string())
        );
    }

    #[test]
    fn extract_top_level_message_only() {
        // ISOLATED fixture: no /error key, so the first two branches
        // miss and the /message branch fires. This is the path that
        // surfaces a clean message for the in-stream LM Studio
        // context-overflow envelope from the user's screenshot.
        let v = json!({"message": "context overflow"});
        assert_eq!(
            extract_backend_error_message(&v),
            Some("context overflow".to_string())
        );
    }

    #[test]
    fn extract_fastapi_detail_only() {
        // ISOLATED fixture: no /error or /message, so the /detail
        // branch is the only one that can fire.
        let v = json!({"detail": "validation failed"});
        assert_eq!(
            extract_backend_error_message(&v),
            Some("validation failed".to_string())
        );
    }

    #[test]
    fn extract_returns_none_on_unrelated_json() {
        // Guard against false positives: a streaming chunk shape
        // (choices array) must NOT be mis-classified as an error.
        let v = json!({"choices": [{"delta": {"content": "hi"}}]});
        assert_eq!(extract_backend_error_message(&v), None);
    }

    #[test]
    fn extract_precedence_nested_wins_over_top_level() {
        // The user's screenshot has BOTH /error/message AND /message.
        // Path 1 must win — verify the early-return ordering matches
        // the doc comment claim.
        let v = json!({
            "error": {"message": "nested takes priority"},
            "message": "top-level should be skipped"
        });
        assert_eq!(
            extract_backend_error_message(&v),
            Some("nested takes priority".to_string())
        );
    }

    // v0.1.27 — looks_like_schema_bleed tests.
    //
    // Each test isolates ONE gate of the heuristic so that a future
    // tightening of any threshold trips exactly one failure rather
    // than cascading. The positive sample uses the actual GLM-4.5-air
    // output from the user's screenshot (a real GLM schema-leak
    // sample) so future probe-path edits cannot regress the
    // exact symptom that motivated this work.

    #[test]
    fn schema_bleed_rejects_short_content() {
        // Gate 1: content.chars().count() < 50 short-circuits before any
        // analysis. (v0.1.27 review note: was incorrectly
        // documented as `content.len() < 50` — that's bytes; the impl
        // counts Unicode scalars.) Even a fragment that would otherwise
        // score 100% structural must return false.
        let short = r#":""object":"#;
        assert!(!looks_like_schema_bleed(short));
    }

    #[test]
    fn schema_bleed_rejects_prose() {
        // Normal user-facing assistant prose. Long enough to clear gate 1,
        // but has zero schema keywords and a low structural ratio.
        let prose = "Hello! I can help you find blueprints in your project. \
            What would you like to look for? I have access to a few tools \
            that let me search the asset library and report back the matches.";
        assert!(!looks_like_schema_bleed(prose));
    }

    #[test]
    fn schema_bleed_rejects_low_keyword_count() {
        // Gate 2: schema_keyword_total < 5 short-circuits. This fixture
        // is structural-heavy (passes gate 3) but only has 1 schema
        // keyword, so the function must return false.
        // Padded with quotes/braces/colons so gate-3 would pass if gate-2
        // didn't trip first — that's the point: we're isolating gate-2.
        let low_kw = r#"{ "name": "x", "object": "y" } { "" : "" : "" : "" }"#;
        // Sanity: long enough for gate 1. v0.1.27 review note:
        // was using .len() (bytes); impl gate uses chars().count(). Aligned
        // here so the assertion guards the same invariant production code
        // checks, preserving the byte-vs-char regression catch.
        let char_count = low_kw.chars().count();
        assert!(
            char_count >= 50,
            "fixture must clear gate 1; got {} chars",
            char_count
        );
        // object appears 1x, "type" 0x, properties 0x → total = 1 < 5.
        assert!(!looks_like_schema_bleed(low_kw));
    }

    #[test]
    fn schema_bleed_rejects_low_structural_ratio() {
        // Gate 3: structural ratio ≤ 50% (impl is `> 50`, so the
        // rejection boundary is anything at or below 50%). This fixture
        // has enough schema keywords to clear gate 2 but the surrounding
        // prose pushes the structural-character ratio below the threshold.
        // v0.1.27 review defect 2: was incorrectly documented as
        // "≤ 60%" — the actual impl threshold is 50, not 60.
        let with_prose = "The schema for this tool requires an object \
            with specific properties. The type field on each object \
            indicates whether the object is a leaf or contains other \
            properties. We use object composition with a type tag and \
            optional properties to describe each parameter cleanly here.";
        // Sanity: gate 2 should pass (≥5 schema keywords).
        let total = with_prose.matches("object").count()
            + with_prose.matches("\"type\"").count()
            + with_prose.matches("properties").count();
        assert!(total >= 5, "fixture must clear gate 2; got {total}");
        // But gate 3 should fail → overall false.
        assert!(!looks_like_schema_bleed(with_prose));
    }

    #[test]
    fn schema_bleed_detects_glm_user_sample() {
        // Verbatim GLM-4.5-air output from the user's screenshot
        // (LM Studio with broken chat template). This is the canonical
        // positive sample — if this test fails after a future change,
        // the v0.1.27 telemetry no longer covers the symptom that
        // shipped the feature.
        let glm_bleed = r#":
        " " " "object: "object
        "":" " : object
        " : "object",
        " ":"s": " "object : "object "object" ] "object" "object": " },
        object " : "object } object": "object": " : "": "object":
        "object": "object" : "object": "object", "type": "object",
        "properties": { "type": "object" }, "object": "object""#;
        assert!(
            looks_like_schema_bleed(glm_bleed),
            "GLM-4.5-air bleed sample must be detected; \
             this is the regression anchor for v0.1.27"
        );
    }

    #[test]
    fn schema_bleed_empty_string_is_safe() {
        // Defensive: empty content must not panic and must return false.
        // (Gate 1 catches it; this pins the contract.)
        assert!(!looks_like_schema_bleed(""));
    }
}
