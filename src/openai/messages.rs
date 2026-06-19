use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ChatMessage {
    pub role: Role,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    /// Construct a role=system ChatMessage. Used for per-session system
    /// prompts supplied by the bridge via `session/new._meta.systemPrompt.append`.
    /// The shim pushes this as `history[0]` at session creation; the OpenAI
    /// request then naturally leads with the system message on every turn.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn assistant(content: Option<String>, tool_calls: Option<Vec<ToolCall>>) -> Self {
        Self {
            role: Role::Assistant,
            content,
            tool_calls,
            tool_call_id: None,
        }
    }

    /// Construct a role=tool message. The model needs the tool's *result text*,
    /// not the MCP transport envelope — `mcp_result_to_text` extracts the
    /// `content[].text` (and annotates `isError`) so a weak local model can read
    /// it. The full envelope is kept for the UI via the separate `rawOutput`
    /// channel (`tool_call_completed`); only the model-facing `content` is
    /// normalised here.
    pub fn tool(tool_call_id: impl Into<String>, content: serde_json::Value) -> Self {
        Self {
            role: Role::Tool,
            content: Some(mcp_result_to_text(&content)),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
        }
    }
}

/// Normalise an MCP `CallToolResult` envelope into the plain result text an
/// OpenAI-compatible model expects in a `role:"tool"` message.
///
/// MCP returns `{"content":[{"type":"text","text":"..."}], "isError": bool}`
/// (optionally `structuredContent`). Forwarding that wrapper verbatim is a known
/// anti-pattern: the OpenAI tool role is text-only, and weak local models key off
/// surface tokens like `isError` and misread a *success* envelope as a failure
/// (observed with GLM-4-9B reporting "could not be spawned" on an `isError:false`
/// result). Every real MCP host (langchain-mcp-adapters, openai-agents,
/// pydantic-ai) extracts `content[].text` and treats `isError` as host-side
/// control flow, never a model-visible field. We mirror that:
///   - prefer `structuredContent` (data-bearing tools), serialised as JSON;
///   - else join every `content[].text` block with newlines;
///   - empty / non-text-only → a fallback so nothing is silently dropped;
///   - `isError:true` → prefix `"Tool execution failed: …"` (natural language,
///     never the raw boolean); success stays pristine (no prefix);
///   - a non-envelope Value (a bare value / test stub) degrades to `to_string()`.
pub(crate) fn mcp_result_to_text(value: &serde_json::Value) -> String {
    let content_arr = value.get("content").and_then(|c| c.as_array());
    let structured = value.get("structuredContent");
    let has_is_error = value.get("isError").is_some();

    // Not an MCP CallToolResult envelope — preserve the prior behaviour.
    if content_arr.is_none() && structured.is_none() && !has_is_error {
        return value.to_string();
    }

    let is_error = value
        .get("isError")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let mut body = if let Some(sc) = structured {
        sc.to_string()
    } else {
        content_arr
            .map(|blocks| {
                blocks
                    .iter()
                    .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .unwrap_or_default()
    };

    if body.is_empty() {
        body = match content_arr {
            // Non-text blocks (image / resource): stringify so nothing is
            // silently dropped on a text-only endpoint.
            Some(blocks) if !blocks.is_empty() => {
                serde_json::Value::Array(blocks.clone()).to_string()
            }
            _ => "(tool returned no textual output)".to_string(),
        };
    }

    if is_error {
        format!("Tool execution failed: {body}")
    } else {
        body
    }
}

// v0.1.24 C1: typed Tool / ToolFunction structs + Tool::function constructor
// removed. The shim forwards tool definitions as raw `serde_json::Value`
// from `SessionPromptParams.tools` through to the OpenAI request body
// (see `chat_completion_stream` in `src/openai/client.rs`). Avoids a
// redundant deserialize → reserialize round-trip and keeps the bridge's
// chosen JSON shape authoritative. These structs were flagged as STALE
// in the v0.1.23 "what's left" review pass.

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ToolCall {
    pub id: String,
    // NOTE: kept for serde roundtrip. OpenAI's spec defines this as
    // an open enum (currently always "function"), and stripping it
    // would break re-serialization compatibility with backends that
    // strictly validate. Read-only from the shim's perspective.
    #[allow(dead_code)]
    pub r#type: String,
    pub function: ToolCallFunction,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: String,
}

/// Per-chunk delta for one tool call during streaming. index is the stable
/// identifier across chunks for the same call; id/name arrive only on the first
/// chunk for that index.
#[derive(Deserialize, Clone, Debug, Default)]
pub struct ToolCallDelta {
    pub index: usize,
    pub id: Option<String>,
    // NOTE: kept for serde roundtrip. Same rationale as ToolCall.type
    // above — OpenAI defines this as an open enum (always "function"
    // in current models) and stripping it would break backends that
    // strict-validate the streaming shape.
    #[allow(dead_code)]
    pub r#type: Option<String>,
    pub function: Option<ToolCallFunctionDelta>,
}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct ToolCallFunctionDelta {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

#[derive(Debug, Clone)]
pub struct StreamChunk {
    pub content_delta: Option<String>,
    /// Reasoning-stream delta (Ollama `reasoning` or DeepSeek/Qwen3
    /// `reasoning_content` — coalesced via `Delta::reasoning_token()`).
    /// Forwarded by the bridge dispatch closure as an ACP `agent_thought_chunk`
    /// so the UE5 plugin can surface a "thinking…" indicator without conflating
    /// chain-of-thought with the assistant's actual response.
    pub reasoning_delta: Option<String>,
    /// First tool-call delta from this SSE chunk (index-keyed accumulation happens
    /// in client.rs directly from the raw StreamingResponse).
    // TODO: forward as ACP session/update tool_call_delta event when
    // bridge UI surfaces in-flight tool-call deltas. Tracked alongside
    // G3 (write_update placement) for the same release window.
    #[allow(dead_code)]
    pub tool_call_delta: Option<ToolCallDelta>,
    /// Set on the last chunk of a stream when present in the SSE
    /// payload. Coalesced into ChatResult.finish_reason for downstream
    /// G2 forwarding (see ChatResult below). Per-chunk field is unused
    /// directly — bridge consumes the accumulated ChatResult.finish_reason
    /// at end-of-stream rather than reacting to per-chunk values.
    #[allow(dead_code)]
    pub finish_reason: Option<String>,
}

#[derive(Debug)]
pub struct ChatResult {
    pub final_message: ChatMessage,
    /// Tool calls accumulated from all stream chunks, ordered by delta.index.
    pub tool_calls: Vec<ToolCall>,
    /// Final OpenAI-style finish_reason ("stop" | "length" |
    /// "tool_calls" | "content_filter" | ...) accumulated across
    /// stream chunks. v0.1.24 G2 forwards this as a final
    /// `session/update` notification with `sessionUpdate: "end_of_turn"`
    /// after `handle_session_prompt` finishes a turn, so the UE5
    /// bridge can distinguish clean completions from truncation /
    /// length-cap / content-filter cases.
    pub finish_reason: String,
}

// ---------------------------------------------------------------------------
// Internal streaming deserialization types (pub(crate) — not part of the
// public contract, used directly by client.rs for accumulation)
// ---------------------------------------------------------------------------

#[derive(Deserialize, Debug)]
pub(crate) struct StreamingResponse {
    pub choices: Vec<StreamingChoice>,
    /// OpenAI/OpenRouter in-band STREAMING error. A backend may deliver a
    /// mid-stream failure as a chunk that carries a top-level `error` object.
    /// OpenRouter additionally pairs it with a `choices` array whose
    /// `finish_reason` is `"error"`, so the chunk otherwise deserializes cleanly
    /// and — without capturing this field — the error is silently dropped and the
    /// turn ends as a normal completion (Gap 5). `#[serde(default)]` keeps it
    /// optional so healthy chunks are unaffected; `client.rs` inspects it
    /// immediately after deserialization and surfaces a tagged transport error.
    #[serde(default)]
    pub error: Option<serde_json::Value>,
}

#[derive(Deserialize, Debug)]
pub(crate) struct StreamingChoice {
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Deserialize, Debug, Default)]
pub(crate) struct Delta {
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCallDelta>>,
    /// Ollama emits reasoning tokens via `delta.reasoning` (no underscore).
    /// Captured so chain-of-thought from local reasoning models is forwarded
    /// to the bridge as ACP `agent_thought_chunk` instead of being silently
    /// dropped during deserialization.
    #[serde(default)]
    pub reasoning: Option<String>,
    /// LM Studio / DeepSeek / Qwen3 emit reasoning tokens via
    /// `delta.reasoning_content`. Same role as `reasoning` above; we deserialize
    /// both because providers don't share a convention. In a single chunk only
    /// one is populated — see `Delta::reasoning_token()` for the canonical
    /// accessor.
    #[serde(default)]
    pub reasoning_content: Option<String>,
}

impl Delta {
    /// Coalesce the two reasoning fields into a single canonical token.
    /// Prefers `reasoning_content` (the more common convention used by
    /// DeepSeek-style providers) but falls back to `reasoning` (Ollama).
    /// Both fields populated in one chunk has not been observed in the wild;
    /// if it ever happens, the more specific `_content` form wins.
    pub fn reasoning_token(&self) -> Option<&str> {
        self.reasoning_content
            .as_deref()
            .or(self.reasoning.as_deref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tool_result_extracts_plain_text() {
        let env = json!({"content":[{"type":"text","text":"3 door blueprints"}],"isError":false});
        assert_eq!(mcp_result_to_text(&env), "3 door blueprints");
    }

    #[test]
    fn tool_result_success_leaks_no_envelope_tokens() {
        // The precise GLM-4-9B bug: on a SUCCESS the model must never see the
        // `isError` token or the envelope scaffolding.
        let env = json!({"content":[{"type":"text","text":"Spawned PointLight at (0,0,0)"}],"isError":false});
        let out = mcp_result_to_text(&env);
        assert_eq!(out, "Spawned PointLight at (0,0,0)");
        assert!(!out.contains("isError"), "success must not leak isError: {out}");
        assert!(!out.contains("\"content\""), "success must not leak the envelope: {out}");
    }

    #[test]
    fn tool_result_error_is_prefixed_not_a_boolean() {
        let env = json!({"content":[{"type":"text","text":"actor name not found"}],"isError":true});
        let out = mcp_result_to_text(&env);
        assert!(out.starts_with("Tool execution failed:"), "error must be prefixed: {out}");
        assert!(out.contains("actor name not found"));
        assert!(!out.contains("isError"));
    }

    #[test]
    fn tool_result_joins_multiple_text_blocks() {
        let env = json!({"content":[{"type":"text","text":"line one"},{"type":"text","text":"line two"}],"isError":false});
        assert_eq!(mcp_result_to_text(&env), "line one\nline two");
    }

    #[test]
    fn tool_result_prefers_structured_content() {
        let env = json!({"content":[{"type":"text","text":"ignored"}],"structuredContent":{"spawned":"PointLight_0"},"isError":false});
        assert!(mcp_result_to_text(&env).contains("PointLight_0"));
    }

    #[test]
    fn tool_result_empty_content_does_not_panic() {
        let out = mcp_result_to_text(&json!({"content":[],"isError":false}));
        assert!(!out.is_empty());
        assert!(!out.contains("isError"));
    }

    #[test]
    fn tool_result_non_text_block_is_not_dropped() {
        let out = mcp_result_to_text(
            &json!({"content":[{"type":"image","data":"...","mimeType":"image/png"}],"isError":false}),
        );
        assert!(!out.is_empty());
    }

    #[test]
    fn tool_result_iserror_without_content_does_not_panic() {
        // The in-test stub shape `{ "isError": true }` (no content array).
        let out = mcp_result_to_text(&json!({"isError":true}));
        assert!(out.starts_with("Tool execution failed:"), "{out}");
    }

    #[test]
    fn tool_result_non_envelope_value_degrades_to_string() {
        let v = json!({"result":"x"});
        assert_eq!(mcp_result_to_text(&v), v.to_string());
    }

    #[test]
    fn streaming_response_captures_openrouter_midstream_error() {
        // Gap 5: OpenRouter delivers a mid-stream error as a chunk that ALSO
        // carries a (usually empty / finish_reason:"error") choices array, so the
        // struct must still deserialize WHILE exposing the top-level error object.
        let chunk = r#"{"id":"x","object":"chat.completion.chunk","error":{"code":429,"message":"Rate limit exceeded"},"choices":[{"index":0,"delta":{},"finish_reason":"error"}]}"#;
        let parsed: StreamingResponse =
            serde_json::from_str(chunk).expect("error chunk must still deserialize");
        let err = parsed.error.expect("top-level error must be captured");
        assert_eq!(err.get("code").and_then(|c| c.as_u64()), Some(429));
        assert_eq!(parsed.choices.len(), 1);
        assert_eq!(parsed.choices[0].finish_reason.as_deref(), Some("error"));
    }

    #[test]
    fn streaming_response_normal_chunk_has_no_error() {
        // Regression guard: a normal content chunk leaves `error` None so the
        // Gap-5 surface never fires on healthy output.
        let chunk = r#"{"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}"#;
        let parsed: StreamingResponse =
            serde_json::from_str(chunk).expect("normal chunk must deserialize");
        assert!(parsed.error.is_none());
    }
}
