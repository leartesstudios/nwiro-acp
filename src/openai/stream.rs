use crate::openai::messages::{StreamChunk, StreamingResponse};
use crate::ShimError;

/// Parse the payload of a single SSE `data:` line.
///
/// Returns:
/// - `Ok(None)` for keepalive/empty lines and the `[DONE]` sentinel.
/// - `Ok(Some(chunks))` for real OpenAI delta payloads (one chunk per choice).
/// - `Err` if the JSON is structurally invalid (not an expected parse miss).
// TODO(phase-3): alternative SSE parser kept as fallback if eventsource-stream proves unsuitable.
#[allow(dead_code)]
pub fn parse_sse_data(data: &str) -> crate::Result<Option<Vec<StreamChunk>>> {
    let data = data.trim();

    if data.is_empty() || data == "[DONE]" {
        return Ok(None);
    }

    let response: StreamingResponse = serde_json::from_str(data).map_err(|e| {
        ShimError::OpenAiHttp(format!("SSE JSON parse error: {e} — raw: {data}"))
    })?;

    let chunks: Vec<StreamChunk> = response
        .choices
        .into_iter()
        .map(|choice| {
            let reasoning_delta = choice.delta.reasoning_token().map(|s| s.to_string());
            StreamChunk {
                content_delta: choice.delta.content,
                reasoning_delta,
                tool_call_delta: choice
                    .delta
                    .tool_calls
                    .and_then(|mut tc| if tc.is_empty() { None } else { Some(tc.remove(0)) }),
                finish_reason: choice.finish_reason,
            }
        })
        .collect();

    Ok(Some(chunks))
}
