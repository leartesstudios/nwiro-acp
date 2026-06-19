//! Emulated-tier tool-call parser (Phase 4 / v0.1.17).
//!
//! When a model is classified as `ToolTier::Emulated` by the warmup
//! probe at `client.rs::probe_tool_capability`, it knows about tools
//! conceptually (its system prompt enumerates them) but emits its
//! invocation intent in the `content` field rather than the OpenAI
//! streaming `tool_calls` field. The Phase 3 transport at v0.1.15
//! can execute a real `ToolCall` end-to-end but the bridge guard
//! previously refused these sessions outright.
//!
//! This module synthesises a real `ToolCall` from the accumulated
//! content so the existing transport can take over.
//!
//! # Formats supported in v0.1.17
//!
//! 1. **Inline JSON** — `{ "tool": "name", "arguments": {...} }`
//!    (variants: `"tool"` or `"name"`; `"arguments"` or `"args"`).
//!    Handles `{...}` blocks embedded in prose; brace-balanced scan
//!    finds candidate object boundaries.
//! 2. **Qwen XML** — `<tool_call>name(args_json)</tool_call>`. The
//!    native format emitted by Qwen 2.5 7B and several Mistral
//!    variants.
//!
//! # Deferred to v0.1.18
//!
//! - Markdown header format (`Tool: name\nParams:\n  - key: value`)
//!   — false-positive risk against natural prose about tools is too
//!   high to ship without a richer unit-test scaffold (the
//!   false-positive risk drove this deferral).
//!
//! # The false-positive discriminator
//!
//! Every successful extraction is gated on `tool_names` membership:
//! the extracted tool name must match an entry in the caller-supplied
//! `tool_names` list. **Without this check, the parser would mis-fire
//! on any model response that legitimately discusses JSON** (e.g. the
//! user asks "show me a config" and the model emits a JSON example
//! that happens to contain a `"name"` key). The membership guard is
//! the single most important correctness invariant in the module —
//! it is enforced inside both extractor helpers AND short-circuited
//! at the public entry point.
//!
//! # Call-site contract
//!
//! `content` MUST be the fully-accumulated final assistant text from
//! `ChatResult.final_message.content`, NOT individual SSE delta chunks.
//! Partial JSON in a single delta would silently parse to garbage and
//! bypass the discriminator.

use crate::openai::messages::{ToolCall, ToolCallFunction};
use uuid::Uuid;

/// A tool call extracted from free-form content, paired with the BYTE
/// span it occupies in the original `content`. The span lets the bridge
/// strip the raw envelope from displayed content without disturbing the
/// surrounding prose (v0.2.5 display fix — the reasoning-model "bleed").
///
/// `span` is a byte range into the SAME `&str` that was passed to
/// `extract_tool_calls_with_spans`; ranges are non-overlapping and each
/// covers exactly one envelope (the whole `{...}` object or the whole
/// `<tool_call>...</tool_call>` element including both tags).
#[derive(Debug, Clone)]
pub(crate) struct ExtractedToolCall {
    pub call: ToolCall,
    pub span: std::ops::Range<usize>,
}

/// Attempt to extract a `ToolCall` from the model's free-form `content`
/// when the tier-Emulated model expressed tool intent as prose rather
/// than native OpenAI `tool_calls`.
///
/// Returns `Some(ToolCall)` only when:
///   1. A recognised format is matched (inline JSON or Qwen XML).
///   2. The extracted name is in `tool_names` (membership guard).
///
/// Format precedence: **Qwen XML → inline JSON → Markdown headers**.
/// Most-distinctive first. XML tags are unambiguous (a `<tool_call>`
/// substring almost never appears in natural prose); inline JSON is
/// next because brace-balanced objects are structurally unique; the
/// Markdown header format ranks last because it has the highest
/// overlap with natural prose about tools — so
/// XML and JSON get the first shot if the content carries both.
///
/// The synthesised `ToolCall.id` carries a `synth_` prefix so the
/// rest of the shim can distinguish a synthesised call from a
/// model-native one in logs.
pub fn try_extract_tool_call(content: &str, tool_names: &[String]) -> Option<ToolCall> {
    if tool_names.is_empty() {
        // No registered tools → no legitimate name could match.
        // Bail before any parsing work to avoid false positives on
        // arbitrary JSON in the content.
        return None;
    }

    // v0.2.5: the XML→JSON span scanner is now the single extractor for
    // those two formats; `.next()` reproduces the original first-match,
    // XML-before-JSON precedence. Markdown stays a separate, line-based
    // fallback (firing-only — it is never span-stripped from displayed
    // content because it overlaps too readily with natural prose).
    if let Some(extracted) = extract_tool_calls_with_spans(content, tool_names)
        .into_iter()
        .next()
    {
        return Some(extracted.call);
    }
    if let Some(tc) = try_extract_markdown(content, tool_names) {
        return Some(tc);
    }
    None
}

/// Resilient multi-candidate SPAN scanner: find EVERY registered tool
/// envelope in `content` (XML and inline-JSON forms) and return each as
/// an [`ExtractedToolCall`] carrying its byte span.
///
/// Precedence: XML spans are collected and returned FIRST, then JSON
/// spans, so `.next()` on the result preserves the original
/// `try_extract_tool_call` ordering (XML beats JSON). The list is NOT
/// globally sorted — callers that need positional order (e.g. the bridge
/// span-stripper) sort locally.
///
/// The `tool_names` membership guard remains the single source of truth:
/// a span is produced ONLY for an envelope whose name is registered, so
/// legitimate JSON in the content (config blobs, examples) is never
/// stripped. Empty `tool_names` short-circuits to `vec![]`.
pub(crate) fn extract_tool_calls_with_spans(
    content: &str,
    tool_names: &[String],
) -> Vec<ExtractedToolCall> {
    if tool_names.is_empty() {
        // Mirror the public-entry short-circuit: no registered tool could
        // match, so no span can be produced.
        return vec![];
    }

    let mut spans: Vec<ExtractedToolCall> = Vec::new();
    let bytes = content.as_bytes();

    // ── XML spans first (preserve XML-before-JSON precedence) ──────────
    let open_tag = "<tool_call>";
    let close_tag = "</tool_call>";
    let mut cursor = 0usize;
    while let Some(rel_open) = content[cursor..].find(open_tag) {
        let start = cursor + rel_open;
        let body_start = start + open_tag.len();
        // Reuse the exact try_extract_qwen_xml body logic on this hit.
        if let Some(rel_end) = content[body_start..].find(close_tag) {
            let body = content[body_start..body_start + rel_end].trim();
            let span_end = body_start + rel_end + close_tag.len();
            if let Some(call) = try_parse_qwen_xml_body(body, tool_names) {
                spans.push(ExtractedToolCall {
                    call,
                    span: start..span_end,
                });
                // Advance past the whole consumed element.
                cursor = span_end;
                continue;
            }
        }
        // Not a valid+registered element — advance past THIS `<tool_call>`
        // occurrence so the scan doesn't loop forever on it.
        cursor = body_start;
    }

    // ── JSON spans second ──────────────────────────────────────────────
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] != b'{' {
            i += 1;
            continue;
        }

        // Brace-balanced walk (string- and escape-aware), lifted from the
        // old try_extract_inline_json walker.
        let mut depth: i64 = 1;
        let mut j = i + 1;
        let mut in_string = false;
        let mut escape = false;
        while j < bytes.len() && depth > 0 {
            let b = bytes[j];
            if escape {
                escape = false;
            } else if in_string {
                if b == b'\\' {
                    escape = true;
                } else if b == b'"' {
                    in_string = false;
                }
            } else if b == b'"' {
                in_string = true;
            } else if b == b'{' {
                depth += 1;
            } else if b == b'}' {
                depth -= 1;
            }
            j += 1;
        }

        if depth != 0 {
            // This `{` never balanced before EOS. Do NOT discard prior
            // spans and do NOT consume forward — advance ONE byte and keep
            // scanning so a balanced envelope NESTED-after or LATER in the
            // buffer is still found (the early-unbalanced-then-valid case).
            // A genuinely truncated trailing `{` simply never balances on
            // any later start either → it produces no span and the bridge
            // flushes it as prose, never mistaking it for a call.
            i += 1;
            continue;
        }

        let candidate = &content[i..j];
        if let Some(call) = try_parse_inline_candidate(candidate, tool_names) {
            // Registered envelope → record its span and jump PAST the whole
            // object so back-to-back envelopes are each found.
            spans.push(ExtractedToolCall { call, span: i..j });
            i = j;
        } else {
            // Balanced but unregistered/unparseable. Skip the WHOLE object
            // (i = j, not i += 1) so (a) a later valid envelope is still
            // found and (b) a registered name nested inside this object's
            // string values is never independently matched.
            i = j;
        }
    }

    spans
}

/// Build a synthesised `ToolCall` with a UUID-stamped id. Centralised
/// so the `synth_` prefix policy lives in exactly one place.
fn make_synth_tool_call(name: String, arguments: String) -> ToolCall {
    ToolCall {
        id: format!("synth_{}", Uuid::new_v4()),
        r#type: "function".to_string(),
        function: ToolCallFunction { name, arguments },
    }
}

// ─── Qwen XML extractor ────────────────────────────────────────────

/// Qwen 2.5 7B native format: `<tool_call>name(args_json)</tool_call>`.
///
/// - Tag content is `name` followed by `(args_json)`.
/// - Args inside the parens parse as a JSON object (or are empty,
///   defaulting to `{}`).
/// - Whitespace around `name` and `args_json` is tolerated.
///
/// Returns `None` on any structural mismatch — malformed XML, missing
/// parens, malformed args JSON, or name-not-in-`tool_names`.
///
/// `body` is the already-extracted, already-trimmed text BETWEEN the
/// `<tool_call>` and `</tool_call>` tags (i.e. `name(args_json)`). It is
/// factored out of the tag-finding logic so the span scanner can reuse
/// the exact membership + args-validation rules per matched element.
fn try_parse_qwen_xml_body(body: &str, tool_names: &[String]) -> Option<ToolCall> {
    // body should be: name(args_json) — split on first '(' to grab name.
    let paren = body.find('(')?;
    let name = body[..paren].trim().to_string();
    if !tool_names.iter().any(|n| n == &name) {
        return None;
    }

    // Args between the first '(' and the LAST ')' (`rfind` so nested
    // parens inside JSON strings don't truncate prematurely).
    let after_paren = &body[paren + 1..];
    let close_paren_rel = after_paren.rfind(')')?;
    let args_raw = after_paren[..close_paren_rel].trim();

    let arguments = if args_raw.is_empty() {
        "{}".to_string()
    } else {
        // Validate as JSON — silently drop on parse failure rather
        // than dispatch garbage args to the MCP server.
        serde_json::from_str::<serde_json::Value>(args_raw).ok()?;
        args_raw.to_string()
    };

    Some(make_synth_tool_call(name, arguments))
}

// ─── Inline JSON extractor ─────────────────────────────────────────

/// Scan `content` for `{ ... }` object substrings; for each balanced
/// candidate, check if it parses as JSON and matches the expected
/// tool-invocation shape `{ "tool"|"name": <name>, "arguments"|"args": {...} }`.
///
/// Brace-balance walker treats string-internal braces correctly (a `{`
/// inside a JSON string doesn't increment depth). Backslash-escaped
/// quotes inside strings are respected.
///
/// Returns the FIRST candidate whose name passes the `tool_names`
/// membership check. Later candidates are ignored — multi-call
/// prose is rare enough in Emulated tier output that the first-match
/// policy is acceptable for v0.1.17.
///
/// v0.2.5: the byte walker that found `{...}` candidate boundaries was
/// lifted into [`extract_tool_calls_with_spans`] (which now backs both
/// firing and display-stripping). The per-candidate shape/membership
/// discriminator below stays the single source of truth and is shared.
///
/// Try to interpret a single brace-balanced substring as a tool
/// invocation envelope.
pub(crate) fn try_parse_inline_candidate(
    candidate: &str,
    tool_names: &[String],
) -> Option<ToolCall> {
    let v: serde_json::Value = serde_json::from_str(candidate).ok()?;
    let obj = v.as_object()?;

    // Accept either "tool" or "name" as the name key. Different model
    // families settle on different conventions; we accept both.
    let name = obj
        .get("tool")
        .or_else(|| obj.get("name"))
        .and_then(|n| n.as_str())?
        .to_string();
    if !tool_names.iter().any(|n| n == &name) {
        return None;
    }

    // Accept "arguments" (OpenAI-spec) or "args" (some models). Absent
    // → empty args object. The value can be an object OR a string
    // (some models emit args as a JSON-encoded string already — we
    // pass either through unchanged in serialised form).
    let args_value = obj
        .get("arguments")
        .or_else(|| obj.get("args"))
        .cloned()
        .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
    let arguments = match args_value {
        serde_json::Value::String(s) => s,
        other => serde_json::to_string(&other).ok()?,
    };

    Some(make_synth_tool_call(name, arguments))
}

// ─── Markdown header extractor ────────────────────────────────────
//
// Empirically observed (Screenshot 2 from the user's bug report):
//
//     Tool: spawn_actor
//     Params:
//       - class_name: "PointLight"
//       - location: [0, 0, 0]
//
// This format is the highest-false-positive risk in the parser
// suite — natural prose frequently uses the word "Tool:" followed
// by a description of a tool. The minimum-viable mitigation is to
// require ALL of:
//
//   1. A `Tool: <name>` line where `<name>` matches `tool_names`
//      (case-insensitive `Tool:` prefix; whitespace tolerated).
//   2. A `Params:` header line somewhere AFTER the Tool line
//      (case-insensitive).
//   3. At least ONE `- key: value` entry between the Params line
//      and the next non-list-item, non-empty line.
//
// If any of those three are absent, the extractor returns None.
// The false-positive risk made this minimum mandatory; it's the
// discriminator that keeps prose
// like "Tool: this is interesting…" from synthesising a tool_call.
fn try_extract_markdown(content: &str, tool_names: &[String]) -> Option<ToolCall> {
    let lines: Vec<&str> = content.lines().collect();

    // ─ Step 1: find a `Tool: <name>` line whose name is registered ─
    let mut tool_line_idx: Option<usize> = None;
    let mut name: Option<String> = None;
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        // Case-insensitive prefix check for "Tool:" — accept "Tool:",
        // "tool:", "TOOL:", with the colon required.
        let lower = trimmed.to_ascii_lowercase();
        if let Some(rest) = lower.strip_prefix("tool:") {
            // The actual name lives at the same offset in the
            // original (case-preserving) line.
            let name_offset = trimmed.len() - rest.len();
            let raw_name = trimmed[name_offset..].trim();
            // Strip surrounding quotes if the model wrapped the name.
            let extracted = raw_name
                .trim_matches(|c: char| c == '"' || c == '\'')
                .to_string();
            if !extracted.is_empty()
                && tool_names.iter().any(|n| n == &extracted)
            {
                tool_line_idx = Some(i);
                name = Some(extracted);
                break;
            }
        }
    }

    let tool_idx = tool_line_idx?;
    let name = name?;

    // ─ Step 2: find a `Params:` header after the Tool line ─
    let params_idx = lines[tool_idx + 1..]
        .iter()
        .position(|l| {
            l.trim()
                .to_ascii_lowercase()
                .starts_with("params:")
        })
        .map(|p| tool_idx + 1 + p)?;

    // ─ Step 3: collect `- key: value` entries after Params: ─
    let mut args = serde_json::Map::new();
    for line in &lines[params_idx + 1..] {
        let trimmed = line.trim_start();
        if let Some(entry) = trimmed.strip_prefix('-') {
            // Format: `- key: value`. Whitespace around `-` and the
            // colon is tolerated.
            let entry = entry.trim_start();
            if let Some(colon) = entry.find(':') {
                let key = entry[..colon].trim().to_string();
                let value_str = entry[colon + 1..].trim();
                if !key.is_empty() {
                    // Try parsing the value as JSON (handles strings
                    // with quotes, numbers, arrays, objects, bools).
                    // Fall back to treating it as a bare string.
                    let value: serde_json::Value =
                        serde_json::from_str(value_str).unwrap_or_else(|_| {
                            serde_json::Value::String(value_str.to_string())
                        });
                    args.insert(key, value);
                }
            }
        } else if !trimmed.is_empty() {
            // Non-list-item, non-empty line → Params block ended.
            // Stop collecting.
            break;
        }
        // Blank line: tolerate (some models pad with blanks). Continue.
    }

    if args.is_empty() {
        // Architect's risk mitigation: refuse the synthesis if no
        // params were collected. A bare "Tool: foo\nParams:" sequence
        // followed by prose is too prose-like to trust.
        return None;
    }

    let arguments = serde_json::to_string(&serde_json::Value::Object(args)).ok()?;
    Some(make_synth_tool_call(name, arguments))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names() -> Vec<String> {
        vec!["find_blueprints".into(), "spawn_actor".into()]
    }

    // ── public-entry behaviour ─────────────────────────────────────

    #[test]
    fn empty_content_returns_none() {
        assert!(try_extract_tool_call("", &names()).is_none());
    }

    #[test]
    fn empty_tool_names_short_circuits_to_none() {
        // Even a valid-looking inline JSON should be ignored if no
        // tools are registered for this prompt.
        let content = r#"{ "tool": "find_blueprints", "arguments": {} }"#;
        assert!(try_extract_tool_call(content, &[]).is_none());
    }

    #[test]
    fn prose_only_returns_none() {
        // No JSON wrapper, no XML — just a model talking about tools.
        let content = "I would call find_blueprints with searchTerm box.";
        assert!(try_extract_tool_call(content, &names()).is_none());
    }

    // ── inline JSON ────────────────────────────────────────────────

    #[test]
    fn inline_json_with_tool_key_hits() {
        let content = r#"{ "tool": "find_blueprints", "arguments": {"searchTerm":"box"} }"#;
        let tc = try_extract_tool_call(content, &names()).expect("should parse");
        assert_eq!(tc.function.name, "find_blueprints");
        assert!(tc.function.arguments.contains("searchTerm"));
        assert!(tc.id.starts_with("synth_"));
    }

    #[test]
    fn inline_json_with_name_key_hits() {
        // Some models use "name" instead of "tool".
        let content = r#"{ "name": "spawn_actor", "args": {"class":"PointLight"} }"#;
        let tc = try_extract_tool_call(content, &names()).expect("should parse");
        assert_eq!(tc.function.name, "spawn_actor");
        assert!(tc.function.arguments.contains("PointLight"));
    }

    #[test]
    fn inline_json_with_unregistered_name_returns_none() {
        // CRITICAL FALSE-POSITIVE GUARD: the model emits a structurally
        // valid envelope but for a tool we don't have registered. We
        // must NOT synthesise — otherwise any JSON discussion that
        // happens to use `tool`/`name` keys would mis-fire.
        let content = r#"{ "tool": "delete_universe", "arguments": {} }"#;
        assert!(try_extract_tool_call(content, &names()).is_none());
    }

    #[test]
    fn inline_json_embedded_in_prose_is_extracted() {
        // The user's Screenshot 1 shape — two JSON blocks in prose.
        // First-match policy: we take the first valid one.
        let content = r#"Sure, here's what I'll do:
{ "tool": "spawn_actor", "class": "PointLight", "location": "0,0,0", "arguments": {"x": 0} }
Then:
{ "tool": "find_blueprints", "arguments": {"searchTerm": "box"} }"#;
        let tc = try_extract_tool_call(content, &names()).expect("should parse");
        assert_eq!(tc.function.name, "spawn_actor");
    }

    #[test]
    fn inline_json_malformed_returns_none() {
        // Trailing comma — strict JSON parser rejects.
        let content = r#"{ "tool": "find_blueprints", "arguments": {}, }"#;
        assert!(try_extract_tool_call(content, &names()).is_none());
    }

    #[test]
    fn inline_json_with_no_name_keys_returns_none() {
        // Looks like a config object, not a tool envelope.
        let content = r#"{ "max_tokens": 100, "temperature": 0.7 }"#;
        assert!(try_extract_tool_call(content, &names()).is_none());
    }

    #[test]
    fn inline_json_args_as_string_is_passed_through() {
        // OpenAI's tool_calls.function.arguments is ALWAYS a string.
        // Some models pre-serialise; we honour that.
        let content = r#"{ "tool": "find_blueprints", "arguments": "{\"searchTerm\":\"box\"}" }"#;
        let tc = try_extract_tool_call(content, &names()).expect("should parse");
        assert_eq!(tc.function.arguments, r#"{"searchTerm":"box"}"#);
    }

    #[test]
    fn inline_json_nested_braces_in_string_dont_break_balance() {
        // Brace inside a JSON string must not increment depth.
        let content = r#"{ "tool": "find_blueprints", "arguments": {"query":"{not a brace}"} }"#;
        let tc = try_extract_tool_call(content, &names()).expect("should parse");
        assert_eq!(tc.function.name, "find_blueprints");
    }

    // ── Qwen XML ───────────────────────────────────────────────────

    #[test]
    fn qwen_xml_hits() {
        let content = r#"<tool_call>find_blueprints({"searchTerm":"box"})</tool_call>"#;
        let tc = try_extract_tool_call(content, &names()).expect("should parse");
        assert_eq!(tc.function.name, "find_blueprints");
        assert_eq!(tc.function.arguments, r#"{"searchTerm":"box"}"#);
        assert!(tc.id.starts_with("synth_"));
    }

    #[test]
    fn qwen_xml_with_empty_args_defaults_to_empty_object() {
        let content = "<tool_call>find_blueprints()</tool_call>";
        let tc = try_extract_tool_call(content, &names()).expect("should parse");
        assert_eq!(tc.function.arguments, "{}");
    }

    #[test]
    fn qwen_xml_with_unregistered_name_returns_none() {
        let content = r#"<tool_call>delete_universe({})</tool_call>"#;
        assert!(try_extract_tool_call(content, &names()).is_none());
    }

    #[test]
    fn qwen_xml_with_malformed_args_returns_none() {
        // Args between parens but not valid JSON.
        let content = r#"<tool_call>find_blueprints(not-json)</tool_call>"#;
        assert!(try_extract_tool_call(content, &names()).is_none());
    }

    #[test]
    fn qwen_xml_inside_prose_is_extracted() {
        let content = r#"I'll do that. <tool_call>spawn_actor({"class":"PointLight"})</tool_call> Done."#;
        let tc = try_extract_tool_call(content, &names()).expect("should parse");
        assert_eq!(tc.function.name, "spawn_actor");
    }

    #[test]
    fn qwen_xml_takes_precedence_over_inline_json() {
        // If both formats are present, the XML extractor runs first.
        let content = r#"<tool_call>spawn_actor({"x":1})</tool_call>
also: { "tool": "find_blueprints", "arguments": {} }"#;
        let tc = try_extract_tool_call(content, &names()).expect("should parse");
        assert_eq!(tc.function.name, "spawn_actor");
    }

    // ── id uniqueness ──────────────────────────────────────────────

    #[test]
    fn synthesised_ids_are_unique_across_calls() {
        let content = r#"<tool_call>find_blueprints()</tool_call>"#;
        let a = try_extract_tool_call(content, &names()).unwrap();
        let b = try_extract_tool_call(content, &names()).unwrap();
        assert_ne!(a.id, b.id);
    }

    // ── Markdown header (v0.1.19 EMIT-008) ─────────────────────────

    #[test]
    fn markdown_hit_single_param() {
        let content = "Tool: find_blueprints\nParams:\n  - searchTerm: \"box\"";
        let tc = try_extract_tool_call(content, &names()).expect("should parse");
        assert_eq!(tc.function.name, "find_blueprints");
        assert!(tc.function.arguments.contains("searchTerm"));
        assert!(tc.function.arguments.contains("box"));
        assert!(tc.id.starts_with("synth_"));
    }

    #[test]
    fn markdown_hit_multiple_params_with_json_value_types() {
        // The user's Screenshot 2 shape — multiple params, one a JSON
        // array, one a quoted string.
        let content = "Tool: spawn_actor\nParams:\n  - class_name: \"PointLight\"\n  - location: [0, 0, 0]";
        let tc = try_extract_tool_call(content, &names()).expect("should parse");
        assert_eq!(tc.function.name, "spawn_actor");
        // Args is a JSON-encoded object; check both keys are present
        let parsed: serde_json::Value = serde_json::from_str(&tc.function.arguments).unwrap();
        let obj = parsed.as_object().unwrap();
        assert_eq!(obj.get("class_name").and_then(|v| v.as_str()), Some("PointLight"));
        let loc = obj.get("location").and_then(|v| v.as_array()).unwrap();
        assert_eq!(loc.len(), 3);
        assert_eq!(loc[0].as_i64(), Some(0));
    }

    #[test]
    fn markdown_unregistered_name_returns_none() {
        // CRITICAL FALSE-POSITIVE GUARD for Markdown — same logic as
        // JSON and XML variants. A structurally valid Markdown
        // envelope for an unregistered tool must NOT synthesise.
        let content = "Tool: delete_universe\nParams:\n  - confirm: true";
        assert!(try_extract_tool_call(content, &names()).is_none());
    }

    #[test]
    fn markdown_without_params_header_returns_none() {
        // Pure prose with "Tool: <name>" but no Params: structure —
        // looks like a Markdown comment, not an invocation.
        let content = "Tool: spawn_actor is the one you want for that use case.";
        assert!(try_extract_tool_call(content, &names()).is_none());
    }

    #[test]
    fn markdown_with_empty_params_returns_none() {
        // Architect's mitigation: refuse when Params: has no entries.
        // Just "Tool:" + "Params:" + blank lines doesn't constitute a
        // legitimate invocation envelope.
        let content = "Tool: find_blueprints\nParams:\n\nThis is just text.";
        assert!(try_extract_tool_call(content, &names()).is_none());
    }

    #[test]
    fn markdown_with_bare_value_falls_back_to_string() {
        // `- searchTerm: box` (no quotes) → value treated as string.
        let content = "Tool: find_blueprints\nParams:\n  - searchTerm: box";
        let tc = try_extract_tool_call(content, &names()).expect("should parse");
        let parsed: serde_json::Value = serde_json::from_str(&tc.function.arguments).unwrap();
        assert_eq!(
            parsed.get("searchTerm").and_then(|v| v.as_str()),
            Some("box")
        );
    }

    #[test]
    fn markdown_precedence_lower_than_xml_and_json() {
        // Content has all three formats present. XML wins because
        // it's first in the precedence chain.
        let content = "Tool: find_blueprints\nParams:\n  - searchTerm: \"text\"\n\n<tool_call>spawn_actor({\"class\":\"P\"})</tool_call>";
        let tc = try_extract_tool_call(content, &names()).expect("should parse");
        assert_eq!(tc.function.name, "spawn_actor"); // XML wins
    }

    #[test]
    fn markdown_case_insensitive_headers() {
        // `tool:` / `TOOL:` / `Params:` / `PARAMS:` should all work
        // — different model families use different capitalisations.
        let content = "TOOL: find_blueprints\nPARAMS:\n  - searchTerm: \"x\"";
        let tc = try_extract_tool_call(content, &names()).expect("should parse");
        assert_eq!(tc.function.name, "find_blueprints");
    }

    // ── span scanner (v0.2.5 display fix) ──────────────────────────

    #[test]
    fn spans_back_to_back_json_envelopes_yield_two_non_overlapping() {
        // Two adjacent envelopes → TWO spans, each covering its own
        // object, non-overlapping, in order.
        let content =
            r#"{"tool":"spawn_actor","arguments":{}}{"tool":"find_blueprints","arguments":{}}"#;
        let spans = extract_tool_calls_with_spans(content, &names());
        assert_eq!(spans.len(), 2, "expected two spans, got {}", spans.len());
        assert_eq!(spans[0].call.function.name, "spawn_actor");
        assert_eq!(spans[1].call.function.name, "find_blueprints");
        // Non-overlapping and covering both objects.
        assert!(spans[0].span.end <= spans[1].span.start, "spans overlap");
        assert_eq!(&content[spans[0].span.clone()], r#"{"tool":"spawn_actor","arguments":{}}"#);
        assert_eq!(
            &content[spans[1].span.clone()],
            r#"{"tool":"find_blueprints","arguments":{}}"#
        );
    }

    #[test]
    fn spans_early_unbalanced_then_valid_yields_one() {
        // First `{` never closes before the second envelope opens. The
        // truncated `{` must NOT abort the scan, and must be in NO span;
        // only the second, balanced+registered envelope produces a span.
        let content = "{\"tool\": \"spawn_actor\"\n{\"tool\":\"find_blueprints\",\"arguments\":{}}";
        let spans = extract_tool_calls_with_spans(content, &names());
        assert_eq!(spans.len(), 1, "expected one span, got {}", spans.len());
        assert_eq!(spans[0].call.function.name, "find_blueprints");
        // The matched span starts at the SECOND `{`, not the first.
        let second_brace = content[1..].find('{').map(|p| p + 1).unwrap();
        assert_eq!(spans[0].span.start, second_brace);
    }

    #[test]
    fn spans_unregistered_then_registered_yields_one_and_skips_unregistered() {
        // The unregistered envelope is skipped WHOLE (i = j), so the
        // later registered envelope is still found; the unregistered
        // object is NOT in any span.
        let content =
            r#"{"tool":"delete_universe","arguments":{}}{"tool":"spawn_actor","arguments":{}}"#;
        let spans = extract_tool_calls_with_spans(content, &names());
        assert_eq!(spans.len(), 1, "expected one span, got {}", spans.len());
        assert_eq!(spans[0].call.function.name, "spawn_actor");
        // Span covers ONLY the registered object, not the unregistered one.
        assert_eq!(&content[spans[0].span.clone()], r#"{"tool":"spawn_actor","arguments":{}}"#);
    }

    #[test]
    fn spans_registered_name_inside_string_value_yields_zero() {
        // No top-level "tool"/"name" KEY; the registered name lives only
        // inside a string VALUE. The string-aware walker must never
        // produce a span from braces inside that string.
        let content = r#"{"example": "to invoke, emit {\"tool\": \"spawn_actor\"}"}"#;
        let spans = extract_tool_calls_with_spans(content, &names());
        assert_eq!(spans.len(), 0, "string-internal name must not match");
    }

    #[test]
    fn spans_nested_registered_name_in_args_string_yields_one_outer() {
        // The OUTER object is a valid spawn_actor envelope; find_blueprints
        // appears only inside an args string value and must NOT match
        // independently. Exactly ONE span, covering the whole outer object.
        let content =
            r#"{"tool":"spawn_actor","arguments":{"note":"{\"tool\":\"find_blueprints\"}"}}"#;
        let spans = extract_tool_calls_with_spans(content, &names());
        assert_eq!(spans.len(), 1, "expected one span, got {}", spans.len());
        assert_eq!(spans[0].call.function.name, "spawn_actor");
        assert_eq!(&content[spans[0].span.clone()], content);
    }

    #[test]
    fn spans_split_key_reassembled_yields_one() {
        // Simulate a delta-split key reassembled into the full buffer
        // (the fn always runs on accumulated content).
        let part_a = "{\"too";
        let part_b = "l\": \"spawn_actor\", \"arguments\": {}}";
        let content = format!("{part_a}{part_b}");
        let spans = extract_tool_calls_with_spans(&content, &names());
        assert_eq!(spans.len(), 1, "expected one span, got {}", spans.len());
        assert_eq!(spans[0].call.function.name, "spawn_actor");
    }

    #[test]
    fn spans_xml_multi_match_yields_two_with_close_tags() {
        // Two XML elements → two spans; each span includes its closing
        // tag in the covered range.
        let content =
            r#"<tool_call>spawn_actor({})</tool_call> ... <tool_call>find_blueprints({})</tool_call>"#;
        let spans = extract_tool_calls_with_spans(content, &names());
        assert_eq!(spans.len(), 2, "expected two spans, got {}", spans.len());
        assert_eq!(spans[0].call.function.name, "spawn_actor");
        assert_eq!(spans[1].call.function.name, "find_blueprints");
        assert_eq!(&content[spans[0].span.clone()], "<tool_call>spawn_actor({})</tool_call>");
        assert_eq!(
            &content[spans[1].span.clone()],
            "<tool_call>find_blueprints({})</tool_call>"
        );
    }

    #[test]
    fn spans_empty_tool_names_yields_zero() {
        // Mirror the public-entry short-circuit.
        let content = r#"{"tool":"spawn_actor","arguments":{}}"#;
        assert!(extract_tool_calls_with_spans(content, &[]).is_empty());
    }

    #[test]
    fn delegation_parity_first_span_matches_legacy_extract() {
        // The refactored try_extract_tool_call delegates to the span
        // scanner's .next(); the existing precedence/result must hold.
        // XML beats JSON.
        let content = "<tool_call>spawn_actor({\"x\":1})</tool_call>\nalso: { \"tool\": \"find_blueprints\", \"arguments\": {} }";
        let tc = try_extract_tool_call(content, &names()).expect("should parse");
        assert_eq!(tc.function.name, "spawn_actor");
        let spans = extract_tool_calls_with_spans(content, &names());
        // XML span returned first.
        assert_eq!(spans[0].call.function.name, "spawn_actor");
    }
}
