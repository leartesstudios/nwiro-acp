use crate::openai::messages::ToolCall;
use crate::ShimError;
use serde_json::{json, Value};
use std::future::Future;

/// v0.1.37 (Finding C): true when an MCP-await returned the cancel sentinel —
/// the session was cancelled mid-round-trip. Callers map this to
/// `ShimError::Cancelled` so the turn ends as `stopReason: cancelled`, instead
/// of the generic missing-`result` path which would become an in-band
/// `isError:true` tool failure (a spurious `tool_call_failed` frame). Keying on
/// the distinct sentinel (not just "round-trip cancelled") ensures only a real
/// token cancel takes the cancelled path. See `crate::acp::server::MCP_CANCELLED_SENTINEL`.
fn is_cancel_sentinel(resp: &Value) -> bool {
    resp.pointer("/error/message").and_then(|m| m.as_str())
        == Some(crate::acp::server::MCP_CANCELLED_SENTINEL)
}

/// Schema-aware coercion of double-encoded (stringified) argument values.
///
/// For each TOP-LEVEL property whose current value is a JSON STRING and whose
/// schema declares exactly ONE non-string JSON type among
/// array / object / boolean / number / integer (a single type string, not an
/// array of types), attempt a `serde_json` parse of the string; replace the
/// value ONLY when the parsed type matches the declared type. Everything else
/// is left byte-identical:
///   - `type: "string"` props (protects every code/text-payload parameter),
///   - union types (`type: ["array","string"]`) and `oneOf`/`anyOf` props
///     (the host plugin owns the disambiguation),
///   - props missing from `properties`, or with no `type`,
///   - strings that fail to parse, or parse to a non-matching type (dispatch
///     proceeds untouched; the host plugin's validation owns rejection).
///
/// Returns the coerced property names for the caller's telemetry log.
pub(crate) fn coerce_args_to_schema(
    args: &mut serde_json::Map<String, Value>,
    input_schema: Option<&Value>,
) -> Vec<String> {
    let Some(props) = input_schema
        .and_then(|s| s.get("properties"))
        .and_then(|p| p.as_object())
    else {
        return Vec::new();
    };
    let mut applied: Vec<String> = Vec::new();
    for (key, value) in args.iter_mut() {
        // Only a JSON STRING value can be double-encoded.
        let Some(prop) = props.get(key) else { continue };
        // Union-shaped schemas are never touched — the host plugin owns those.
        if prop.get("oneOf").is_some() || prop.get("anyOf").is_some() {
            continue;
        }
        // `as_str()` is None for `type: [..]` unions and for a missing
        // `type` — both fall through untouched.
        let Some(declared) = prop.get("type").and_then(|t| t.as_str()) else {
            continue;
        };
        let type_matches: fn(&Value) -> bool = match declared {
            "array" => Value::is_array,
            "object" => Value::is_object,
            "boolean" => Value::is_boolean,
            "number" => Value::is_number,
            // `integer` deliberately requires i64/u64 so "1.5" is not
            // coerced into an integer-typed prop.
            "integer" => |v: &Value| v.is_i64() || v.is_u64(),
            // "string", "null", or anything unrecognised: never coerce.
            _ => continue,
        };
        let Some(raw) = value.as_str() else { continue };
        // Parse failure → leave untouched; dispatch proceeds and the host
        // plugin's validation owns rejection.
        let Ok(parsed) = serde_json::from_str::<Value>(raw) else {
            continue;
        };
        if type_matches(&parsed) {
            *value = parsed;
            applied.push(key.clone());
        }
    }
    applied
}

/// Execute one OpenAI tool call via the Nwiro MCP bridge.
///
/// `input_schema` is the tool's `function.parameters` JSON Schema when the
/// bridge forwarded one this turn; used only for stringified-argument
/// coercion (`None` dispatches verbatim, exactly as before).
///
/// `connection_id` is a per-session cache: the first call performs mcp/connect
/// and stores the returned connectionId; subsequent calls reuse it.
///
/// `write_mcp_request` is the async transport provided by the ACP layer.
pub async fn execute_tool<F, Fut>(
    call: &ToolCall,
    input_schema: Option<&Value>,
    connection_id: &mut Option<String>,
    write_mcp_request: &F,
) -> crate::Result<Value>
where
    F: Fn(Value) -> Fut,
    Fut: Future<Output = Value> + Send,
{
    let conn_id = ensure_connection(connection_id, write_mcp_request).await?;

    // Parse the tool-call arguments. A backend can stream a tool call whose
    // arguments get TRUNCATED/garbled — e.g. a mid-stream error leaves
    // `{"class": "PointLight` with no closing quote (observed with GLM-4-32B on
    // llama-server). SILENTLY substituting an empty `{}` here — as this did
    // before — would DISPATCH the WRONG side-effecting action (spawn a generic
    // Actor at the origin instead of the requested PointLight). So:
    //   - empty/whitespace args  → a legitimate no-parameter call → `{}`.
    //   - NON-EMPTY args that do not parse → a clean tool FAILURE: return an
    //     `isError` result the model sees and can retry, and dispatch NOTHING.
    let mut arguments: Value = if call.function.arguments.trim().is_empty() {
        Value::Object(serde_json::Map::new())
    } else {
        match serde_json::from_str::<Value>(&call.function.arguments) {
            Ok(Value::Object(m)) => Value::Object(m),
            // Valid JSON but NOT an object (`[1,2]`, `"x"`, `42`, `true`, `null`).
            // The UE5 bridge reads `arguments` via GetObjectField, so a non-object
            // becomes null and dispatches a NO-arg action — the same wrong-action
            // class. Consistent with the malformed-args policy (and the
            // pending-event normalization at mod.rs ~1945), reject it cleanly.
            Ok(_non_object) => {
                return Ok(json!({
                    "content": [{
                        "type": "text",
                        "text": format!(
                            "Tool call rejected: the `{}` arguments must be a JSON \
                             object, not a bare value. Re-issue with a valid \
                             arguments object.",
                            call.function.name
                        )
                    }],
                    "isError": true
                }));
            }
            Err(e) => {
                return Ok(json!({
                    "content": [{
                        "type": "text",
                        "text": format!(
                            "Tool call rejected: the `{}` arguments were not valid JSON \
                             ({e}). Re-issue the call with a complete, valid JSON \
                             arguments object.",
                            call.function.name
                        )
                    }],
                    "isError": true
                }));
            }
        }
    };

    // v0.4.1: schema-aware argument coercion. Local models frequently
    // double-encode a structured parameter as a JSON STRING (e.g.
    // add_variables: "[{\"name\":\"IsActive\",\"type\":\"bool\"}]"); the host
    // bridge's typed field reader then type-fails that field and skips the op
    // while the rest of the call can still report success. Decode the string
    // ONLY when the tool's inputSchema declares that property as exactly ONE
    // non-string JSON type. Runs strictly AFTER the top-level
    // malformed/non-object guards above — those reject without dispatch,
    // unchanged.
    if let Value::Object(map) = &mut arguments {
        let coerced = coerce_args_to_schema(map, input_schema);
        if !coerced.is_empty() {
            tracing::info!(
                tool = %call.function.name,
                fields = ?coerced,
                "coerced stringified argument(s) to their schema-declared JSON type"
            );
        }
    }

    // Wire shape: the UE5 host bridge reads `params.message` as a full
    // inner JSON-RPC envelope addressed to its in-process MCP server.
    // Sending a flat `params.{method, params}` shape — as v0.1.14 and
    // earlier did — silently took the empty-result branch on the
    // bridge side because `Params->GetObjectField("message")` returned
    // null. Always wrap the tools/call request as the inner message.
    //
    // The `id` field here (0) is a placeholder; the real shim→bridge
    // request id is allocated and stamped on by
    // `acp::server::write_mcp_real` before stdout dispatch.
    let message_req = json!({
        "jsonrpc": "2.0",
        "method": "mcp/message",
        "id": 0,
        "params": {
            "connectionId": conn_id,
            "message": {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": call.function.name,
                    "arguments": arguments
                }
            }
        }
    });

    let resp = write_mcp_request(message_req).await;
    // Finding C: a mid-round-trip cancel surfaces as the distinct sentinel →
    // map to Cancelled (stopReason: cancelled), NOT a tool failure.
    if is_cancel_sentinel(&resp) {
        return Err(ShimError::Cancelled);
    }

    // The bridge proxies the MCP server's response under `result.message`,
    // but that is the MCP server's FULL JSON-RPC response
    // (`{id, jsonrpc, result:{content, isError}}`) — NOT the bare envelope.
    // Unwrap the inner `result` too, so the tool result is the MCP
    // `{content, isError}` envelope that everything downstream expects:
    // the LLM history, the top-level `isError` check in bridge/mod.rs, and
    // nwiro's `rawOutput.content` extractor. Leaving it
    // double-nested makes an errored tool render a GREEN success badge
    // (top-level `isError` reads false) and shows an EMPTY result in the
    // UE5 tool modal (nwiro reads `rawOutput.content`, but content sits at
    // `rawOutput.result.content`). Confirmed via the NWIRO_LOCAL_LLM_LOG_TOOL_IO
    // wire trace. Fall back to the message itself for an MCP-protocol error
    // response (carries `error`, not `result`) or a bridge that ever wraps
    // the bare envelope directly — `{content, isError}` has no `result` key,
    // so the unwrap is a no-op there.
    resp.get("result")
        .and_then(|r| r.get("message"))
        .map(|message| {
            message
                .get("result")
                .cloned()
                .unwrap_or_else(|| message.clone())
        })
        .ok_or_else(|| {
            let err = resp.get("error").cloned().unwrap_or_else(|| resp.clone());
            ShimError::McpRoundtrip(format!("mcp/message error: {err}"))
        })
}

async fn ensure_connection<F, Fut>(
    connection_id: &mut Option<String>,
    write_mcp_request: &F,
) -> crate::Result<String>
where
    F: Fn(Value) -> Fut,
    Fut: Future<Output = Value> + Send,
{
    if let Some(ref id) = connection_id {
        return Ok(id.clone());
    }

    // The `id` field (0) is a placeholder; the real shim→bridge request
    // id is allocated and stamped on by `acp::server::write_mcp_real`
    // before stdout dispatch. The host bridge currently ignores
    // `params.acpId` for connectionId construction (it returns a fixed
    // connection id) but the field is part of the documented protocol —
    // pass through any session-identifying token if a future bridge
    // starts using it.
    let connect_req = json!({
        "jsonrpc": "2.0",
        "method": "mcp/connect",
        "id": 0,
        "params": {}
    });

    let resp = write_mcp_request(connect_req).await;
    // Finding C: a cancel during mcp/connect surfaces as the sentinel → Cancelled.
    if is_cancel_sentinel(&resp) {
        return Err(ShimError::Cancelled);
    }

    let id = resp
        .get("result")
        .and_then(|r| r.get("connectionId"))
        .and_then(|id| id.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| {
            ShimError::McpRoundtrip(format!(
                "mcp/connect response missing connectionId: {resp}"
            ))
        })?;

    *connection_id = Some(id.clone());
    Ok(id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::messages::ToolCallFunction;

    fn make_call(name: &str, args: &str) -> ToolCall {
        ToolCall {
            id: "test-id".to_string(),
            r#type: "function".to_string(),
            function: ToolCallFunction {
                name: name.to_string(),
                arguments: args.to_string(),
            },
        }
    }

    // The bridge proxies the MCP server's FULL JSON-RPC response
    // (`{id, jsonrpc, result:{content, isError}}`) under `result.message`.
    // execute_tool must unwrap BOTH layers to the bare `{content, isError}`
    // envelope — otherwise an errored tool reads `isError:false` at the top
    // level (the green-badge anomaly) and nwiro's `rawOutput.content`
    // extractor finds nothing (empty UI result). Regression for the
    // NWIRO_LOCAL_LLM_LOG_TOOL_IO-confirmed double-nesting bug.
    #[tokio::test]
    async fn execute_tool_unwraps_inner_mcp_result_to_bare_envelope() {
        let write = |req: Value| async move {
            match req.get("method").and_then(|m| m.as_str()).unwrap_or("") {
                "mcp/connect" => json!({ "result": { "connectionId": "test-conn" } }),
                "mcp/message" => json!({
                    "result": { "message": {
                        "id": null,
                        "jsonrpc": "2.0",
                        "result": {
                            "content": [{ "type": "text", "text": "PIE not running" }],
                            "isError": true
                        }
                    }}
                }),
                other => panic!("unexpected mcp method: {other}"),
            }
        };

        let mut conn = None;
        let call = make_call("pie_spawn_actor", r#"{"class_name":"Cube"}"#);
        let result = execute_tool(&call, None, &mut conn, &write).await.unwrap();

        assert!(
            result.get("content").and_then(|c| c.as_array()).is_some(),
            "content must be top-level, not nested under `result`: {result}"
        );
        assert_eq!(
            result.get("isError").and_then(|e| e.as_bool()),
            Some(true),
            "isError must be readable at the top level (drives completed-vs-failed): {result}"
        );
        assert!(result.get("jsonrpc").is_none(), "jsonrpc wrapper leaked: {result}");
        assert!(
            result.get("result").is_none(),
            "inner JSON-RPC `result` wrapper leaked (double-nesting bug): {result}"
        );
    }

    // Defensive fallback: if a bridge ever wraps the BARE `{content, isError}`
    // envelope under `result.message` (no inner `result` key), the unwrap is a
    // no-op and the envelope passes through unchanged.
    #[tokio::test]
    async fn execute_tool_passes_through_a_bare_envelope_unchanged() {
        let write = |req: Value| async move {
            match req.get("method").and_then(|m| m.as_str()).unwrap_or("") {
                "mcp/connect" => json!({ "result": { "connectionId": "c" } }),
                "mcp/message" => json!({
                    "result": { "message": {
                        "content": [{ "type": "text", "text": "ok" }],
                        "isError": false
                    }}
                }),
                other => panic!("unexpected mcp method: {other}"),
            }
        };
        let mut conn = None;
        let call = make_call("get_level_actors", "{}");
        let result = execute_tool(&call, None, &mut conn, &write).await.unwrap();
        assert_eq!(
            result.get("isError").and_then(|e| e.as_bool()),
            Some(false),
            "{result}"
        );
        assert!(
            result.get("content").and_then(|c| c.as_array()).is_some(),
            "{result}"
        );
    }

    // SAFETY GUARD: a tool call whose arguments are non-empty but NOT valid JSON
    // (a backend truncated/garbled the streamed args — e.g. GLM-4-32B on
    // llama-server emitting `{"class": "PointLight` with no closing quote) must
    // NEVER be dispatched with a silently-substituted `{}`, which would spawn the
    // WRONG side-effecting action. It must return a clean tool FAILURE instead.
    //
    // MUTATION CHECK: revert the guard to `.unwrap_or_else(|_| empty Object)` and
    // the malformed call dispatches an `mcp/message` → the `panic!` below fires.
    #[tokio::test]
    async fn execute_tool_rejects_malformed_args_without_dispatch() {
        let write = |req: Value| async move {
            match req.get("method").and_then(|m| m.as_str()).unwrap_or("") {
                "mcp/connect" => json!({ "result": { "connectionId": "c" } }),
                "mcp/message" => {
                    panic!("a tool call with MALFORMED args must NOT be dispatched to the editor")
                }
                other => panic!("unexpected mcp method: {other}"),
            }
        };
        let mut conn = None;
        let call = make_call("spawn_actor", r#"{"class": "PointLight"#);
        let result = execute_tool(&call, None, &mut conn, &write).await.unwrap();
        assert_eq!(
            result.get("isError").and_then(|e| e.as_bool()),
            Some(true),
            "malformed args must yield a clean tool FAILURE, not an empty-args dispatch: {result}"
        );
        let text = result
            .pointer("/content/0/text")
            .and_then(|t| t.as_str())
            .unwrap_or("");
        assert!(text.contains("not valid JSON"), "failure must explain the cause: {result}");
    }

    // EMPTY args are a LEGITIMATE no-parameter call (the guard must only reject
    // non-empty-but-unparseable args) — they dispatch normally as `{}`.
    #[tokio::test]
    async fn execute_tool_empty_args_dispatches_as_no_param_call() {
        let dispatched = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let d = dispatched.clone();
        let write = move |req: Value| {
            let d = d.clone();
            async move {
                match req.get("method").and_then(|m| m.as_str()).unwrap_or("") {
                    "mcp/connect" => json!({ "result": { "connectionId": "c" } }),
                    "mcp/message" => {
                        d.store(true, std::sync::atomic::Ordering::SeqCst);
                        json!({ "result": { "message": {
                            "content": [{ "type": "text", "text": "ok" }], "isError": false } } })
                    }
                    other => panic!("unexpected mcp method: {other}"),
                }
            }
        };
        let mut conn = None;
        let call = make_call("get_level_actors", "");
        let result = execute_tool(&call, None, &mut conn, &write).await.unwrap();
        assert!(
            dispatched.load(std::sync::atomic::Ordering::SeqCst),
            "an empty-args (no-parameter) call must still dispatch"
        );
        assert_eq!(result.get("isError").and_then(|e| e.as_bool()), Some(false), "{result}");
    }

    // Valid JSON but a NON-OBJECT top-level value (`[1,2,3]`, `"x"`, `42`) must
    // ALSO be rejected — the UE5 bridge reads a non-object `arguments` as null and
    // would dispatch a no-arg action (the same wrong-action class as malformed).
    //
    // MUTATION CHECK: revert the accepted arm to `Ok(v) => v` and the array
    // dispatches an `mcp/message` → the `panic!` fires.
    #[tokio::test]
    async fn execute_tool_rejects_non_object_args() {
        let write = |req: Value| async move {
            match req.get("method").and_then(|m| m.as_str()).unwrap_or("") {
                "mcp/connect" => json!({ "result": { "connectionId": "c" } }),
                "mcp/message" => {
                    panic!("a tool call with NON-OBJECT args must NOT be dispatched")
                }
                other => panic!("unexpected mcp method: {other}"),
            }
        };
        let mut conn = None;
        let call = make_call("spawn_actor", "[1, 2, 3]");
        let result = execute_tool(&call, None, &mut conn, &write).await.unwrap();
        assert_eq!(
            result.get("isError").and_then(|e| e.as_bool()),
            Some(true),
            "non-object args must yield a clean tool FAILURE, not a no-arg dispatch: {result}"
        );
    }

    // ---- schema-aware argument coercion (v0.4.1) -------------------------

    /// Recording write closure: captures every dispatched `mcp/message`
    /// request so tests can assert on the ACTUAL dispatched arguments
    /// (`params.message.params.arguments`) — not on mock call counts.
    fn recording_write() -> (
        std::sync::Arc<std::sync::Mutex<Vec<Value>>>,
        impl Fn(Value) -> std::pin::Pin<Box<dyn Future<Output = Value> + Send>>,
    ) {
        let recorded = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let r = recorded.clone();
        let write = move |req: Value| {
            let r = r.clone();
            Box::pin(async move {
                match req.get("method").and_then(|m| m.as_str()).unwrap_or("") {
                    "mcp/connect" => json!({ "result": { "connectionId": "c" } }),
                    "mcp/message" => {
                        r.lock().unwrap().push(req.clone());
                        json!({ "result": { "message": {
                            "content": [{ "type": "text", "text": "ok" }],
                            "isError": false
                        } } })
                    }
                    other => panic!("unexpected mcp method: {other}"),
                }
            }) as std::pin::Pin<Box<dyn Future<Output = Value> + Send>>
        };
        (recorded, write)
    }

    /// Representative host tool schema — covers every prop shape the
    /// coercion gate discriminates on (single non-string type, string type,
    /// `oneOf` union, no declared type). Not a copy of any specific host tool.
    fn representative_schema() -> Value {
        json!({
            "type": "object",
            "properties": {
                "blueprint": { "type": "string" },
                "add_variables": { "type": "array", "items": { "type": "object" } },
                "code": { "type": "string" },
                "mixed": { "oneOf": [ { "type": "array" }, { "type": "string" } ] },
                "untyped": { "description": "no type declared" }
            }
        })
    }

    fn dispatched_arguments(recorded: &std::sync::Arc<std::sync::Mutex<Vec<Value>>>) -> Value {
        let recorded = recorded.lock().unwrap();
        assert_eq!(recorded.len(), 1, "expected exactly one dispatched mcp/message");
        recorded[0]
            .pointer("/params/message/params/arguments")
            .cloned()
            .expect("dispatched mcp/message must carry params.message.params.arguments")
    }

    // THE core coercion case: a model double-encodes an array-typed parameter
    // as a JSON STRING. With the tool's schema declaring that prop as exactly
    // one non-string type (`array`), the shim must dispatch a REAL array —
    // while the string-typed `blueprint` prop passes through untouched.
    #[tokio::test]
    async fn coerces_stringified_array_when_schema_declares_array() {
        let (recorded, write) = recording_write();
        let mut conn = None;
        let args = json!({
            "blueprint": "BP_TestCube",
            "add_variables": "[{\"name\":\"IsActive\",\"type\":\"bool\"}]"
        })
        .to_string();
        let call = make_call("edit_blueprint", &args);
        let schema = representative_schema();
        execute_tool(&call, Some(&schema), &mut conn, &write).await.unwrap();

        let dispatched = dispatched_arguments(&recorded);
        let add_variables = dispatched.get("add_variables").expect("add_variables missing");
        assert!(
            add_variables.is_array(),
            "stringified array must dispatch as a REAL array: {dispatched}"
        );
        assert_eq!(
            add_variables.as_array().map(|a| a.len()),
            Some(1),
            "coerced array must contain the one object: {dispatched}"
        );
        assert_eq!(
            add_variables.pointer("/0/name").and_then(|v| v.as_str()),
            Some("IsActive"),
            "coerced array element must be a real object: {dispatched}"
        );
        assert_eq!(
            dispatched.get("blueprint").and_then(|v| v.as_str()),
            Some("BP_TestCube"),
            "string-typed prop must pass through untouched: {dispatched}"
        );
    }

    // A well-formed call needs NO coercion: the dispatched arguments must be
    // identical to the input.
    #[tokio::test]
    async fn valid_array_passes_byte_identical() {
        let (recorded, write) = recording_write();
        let mut conn = None;
        let input = json!({
            "blueprint": "BP_TestCube",
            "add_variables": [{ "name": "IsActive", "type": "bool" }]
        });
        let call = make_call("edit_blueprint", &input.to_string());
        let schema = representative_schema();
        execute_tool(&call, Some(&schema), &mut conn, &write).await.unwrap();

        assert_eq!(
            dispatched_arguments(&recorded),
            input,
            "already-valid arguments must dispatch unchanged"
        );
    }

    // A string-typed prop whose VALUE happens to look like JSON must NEVER be
    // decoded — code/text payloads legitimately contain JSON-looking text.
    #[tokio::test]
    async fn never_coerces_string_typed_prop() {
        let (recorded, write) = recording_write();
        let mut conn = None;
        let call = make_call("execute_script", r#"{"code": "[1, 2]"}"#);
        let schema = representative_schema();
        execute_tool(&call, Some(&schema), &mut conn, &write).await.unwrap();

        let dispatched = dispatched_arguments(&recorded);
        assert_eq!(
            dispatched.get("code").and_then(|v| v.as_str()),
            Some("[1, 2]"),
            "string-typed prop must stay a string even when it parses as JSON: {dispatched}"
        );
    }

    // Union (`oneOf`) props and props with no declared type are ambiguous —
    // the host plugin owns their disambiguation; the shim must not guess.
    #[tokio::test]
    async fn never_coerces_union_or_schemaless_prop() {
        let (recorded, write) = recording_write();
        let mut conn = None;
        let call = make_call("edit_thing", r#"{"mixed": "[1]", "untyped": "[2]"}"#);
        let schema = representative_schema();
        execute_tool(&call, Some(&schema), &mut conn, &write).await.unwrap();

        let dispatched = dispatched_arguments(&recorded);
        assert_eq!(
            dispatched.get("mixed").and_then(|v| v.as_str()),
            Some("[1]"),
            "oneOf-union prop must never be coerced: {dispatched}"
        );
        assert_eq!(
            dispatched.get("untyped").and_then(|v| v.as_str()),
            Some("[2]"),
            "prop with no declared type must never be coerced: {dispatched}"
        );
    }

    // No schema for the called tool (`None`) → verbatim pass-through, exactly
    // today's behavior. The shim must not guess without a schema.
    #[tokio::test]
    async fn unknown_tool_schema_means_no_coercion() {
        let (recorded, write) = recording_write();
        let mut conn = None;
        let args = json!({
            "add_variables": "[{\"name\":\"IsActive\",\"type\":\"bool\"}]"
        })
        .to_string();
        let call = make_call("edit_blueprint", &args);
        execute_tool(&call, None, &mut conn, &write).await.unwrap();

        let dispatched = dispatched_arguments(&recorded);
        assert_eq!(
            dispatched.get("add_variables").and_then(|v| v.as_str()),
            Some("[{\"name\":\"IsActive\",\"type\":\"bool\"}]"),
            "without a schema the nested string must pass through verbatim: {dispatched}"
        );
    }

    // Guard-ordering regression: a top-level NON-OBJECT argument value is
    // rejected BEFORE coercion ever runs — no dispatch, clean tool failure,
    // even when a schema is present.
    #[tokio::test]
    async fn top_level_nonobject_still_rejected_no_dispatch() {
        let (recorded, write) = recording_write();
        let mut conn = None;
        let call = make_call("edit_blueprint", "[1, 2, 3]");
        let schema = representative_schema();
        let result = execute_tool(&call, Some(&schema), &mut conn, &write).await.unwrap();

        assert!(
            recorded.lock().unwrap().is_empty(),
            "non-object top-level args must NOT be dispatched, schema or not"
        );
        assert_eq!(
            result.get("isError").and_then(|e| e.as_bool()),
            Some(true),
            "non-object args must yield a clean tool FAILURE: {result}"
        );
    }
}
