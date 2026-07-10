"""
v0.4.1 regression — verify schema-aware coercion of double-encoded
(stringified) tool arguments end-to-end through the real shim binary.

Pre-v0.4.1: local models frequently double-encode a structured parameter
as a JSON STRING (e.g. add_variables:
"[{\"name\":\"IsActive\",\"type\":\"bool\"}]"). The shim dispatched the
string verbatim; the host bridge's typed field reader then type-failed
that field and skipped the op while the rest of the call could still
report success.

v0.4.1 fix: per prompt, the bridge builds a tool-name -> parameters-schema
map from session/prompt params.tools; before dispatch, any TOP-LEVEL
argument that is a JSON STRING while its schema declares exactly ONE
non-string type (array/object/boolean/number/integer) is parsed and
replaced ONLY if the parsed type matches. Union-shaped props
(oneOf/anyOf, type arrays) are deliberately untouched — the host tool
owns the disambiguation.

This test (representative host tool schema; no host internals):
1. Mocks an OpenAI backend. Stream 1 emits ONE tool_call for
   `edit_thing` whose arguments carry a STRINGIFIED array-typed field —
   split across 4 SSE delta fragments to exercise streamed tool-call
   argument accumulation (ToolCallAccum reassembly). Stream 2 (after the
   first tool result) emits the SAME stringified payload to `edit_union`
   whose add_variables schema is a oneOf union. Stream 3 is a plain
   final assistant message.
2. session/prompt params.tools declares BOTH tools, so the per-prompt
   schema map contains an array-typed add_variables for edit_thing and a
   oneOf union for edit_union.
3. Mocks the MCP bridge: acknowledges mcp/connect, returns success for
   both tools/call requests, and RECORDS the dispatched params.
4. Asserts:
   - (a) dispatched edit_thing arguments: add_variables is a REAL JSON
     array with the one object {name: IsActive, type: bool} (the
     coercion), and target stays a string.
   - (b) dispatched edit_union arguments: add_variables is STILL A
     STRING (union type -> deliberately untouched).
   - (c) tool_call / tool_call_update events still emit in the pinned
     v0.1.26 order (pending before completed, per call, calls in
     stream order) with status=completed.
   - (d) final response has result.stopReason="end_turn".
"""
import http.server
import json
import os
import queue
import socketserver
import subprocess
import sys
import threading
import time

BIN = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "target",
    "release",
    "local-llm-acp" + (".exe" if os.name == "nt" else ""),
)

MOCK_PORT = 18771

# The double-encoded payload both tool calls carry: an array-typed field
# arriving as a JSON STRING (what a local model actually emits).
STRINGIFIED_ARRAY = "[{\"name\":\"IsActive\",\"type\":\"bool\"}]"
ARGS_FULL = json.dumps({
    "target": "BP_TestCube",
    "add_variables": STRINGIFIED_ARRAY,
})


def _fragment(s, n=4):
    """Split the OpenAI arguments string into n SSE delta fragments so the
    shim must reassemble it via streamed tool-call accumulation."""
    step = max(1, len(s) // n)
    frags = [s[i:i + step] for i in range(0, len(s), step)]
    assert len(frags) >= 3, f"need >=3 fragments, got {len(frags)}"
    return frags


def _drain_stderr_thread(proc):
    proc._stderr_chunks = []

    def _reader():
        try:
            while True:
                chunk = proc.stderr.read1(4096)
                if not chunk:
                    return
                proc._stderr_chunks.append(chunk)
        except Exception:
            return

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    proc._stderr_thread = t


class MockBackend(http.server.BaseHTTPRequestHandler):
    """Mock OpenAI-compatible backend. Returns:
    - warmup probe: trivial completion
    - tool-capability probe (max_tokens=256 + tools): Native tool_calls
      envelope so tier = Native (streamed tool_call deltas are parsed).
    - stream 1 (no tool result yet): Native tool_call for `edit_thing`
      with the arguments string split across 4 delta fragments.
    - stream 2 (one tool result): same fragmented payload to `edit_union`.
    - stream 3 (two tool results): plain final assistant message.
    """

    def log_message(self, fmt, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body_bytes = self.rfile.read(length)
        try:
            body = json.loads(body_bytes.decode("utf-8"))
        except Exception:
            body = {}

        msgs = body.get("messages", [])
        is_warmup = (
            len(msgs) == 1
            and msgs[0].get("content") == "."
            and body.get("max_tokens") == 1
        )
        is_capability_probe = bool(body.get("tools")) and body.get("max_tokens") == 256

        if is_warmup:
            self._json({
                "id": "chatcmpl-w",
                "object": "chat.completion",
                "created": 0,
                "model": body.get("model", "test"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            })
            return
        if is_capability_probe:
            # Return a Native tool_calls envelope so the shim
            # classifies as Native tier.
            self._json({
                "id": "chatcmpl-p",
                "object": "chat.completion",
                "created": 0,
                "model": body.get("model", "test"),
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "call_probe",
                            "type": "function",
                            "function": {
                                "name": "edit_thing",
                                "arguments": "{}"
                            }
                        }]
                    },
                    "finish_reason": "tool_calls"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            })
            return

        # Real prompt — route by how many tool results are in the history.
        tool_results = sum(1 for m in msgs if m.get("role") == "tool")
        if tool_results == 0:
            self._sse_tool_call("call_edit_1", "edit_thing")
        elif tool_results == 1:
            self._sse_tool_call("call_edit_2", "edit_union")
        else:
            self._sse_stream([
                json.dumps({"choices": [{"index": 0, "delta": {"role": "assistant", "content": "Done."}, "finish_reason": None}]}),
                json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
            ])

    def _sse_tool_call(self, call_id, name):
        """Stream ONE tool_call whose arguments string arrives split across
        multiple delta fragments (first delta carries id+name+fragment 0,
        the rest carry only index+arguments)."""
        frags = _fragment(ARGS_FULL)
        lines = [json.dumps({
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "tool_calls": [{
                        "index": 0,
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": frags[0]}
                    }]
                },
                "finish_reason": None
            }]
        })]
        for frag in frags[1:]:
            lines.append(json.dumps({
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "function": {"arguments": frag}
                        }]
                    },
                    "finish_reason": None
                }]
            }))
        lines.append(json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}))
        self._sse_stream(lines)

    def _json(self, body_obj):
        payload = json.dumps(body_obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _sse_stream(self, lines):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        body = b""
        for line in lines:
            body += f"data: {line}\n\n".encode("utf-8")
        body += b"data: [DONE]\n\n"
        self.wfile.write(body)
        self.wfile.flush()


class ReusableTCP(socketserver.TCPServer):
    allow_reuse_address = True


def start_mock_backend():
    srv = ReusableTCP(("127.0.0.1", MOCK_PORT), MockBackend)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv


def send(proc, obj):
    line = json.dumps(obj) + "\n"
    proc.stdin.write(line.encode("utf-8"))
    proc.stdin.flush()


def read_until_response(proc, q, want_id, timeout_s=30.0):
    """Collect ALL frames (notifications + responses) until we see
    the JSON-RPC response with the requested id. Returns the full
    frame list."""
    frames = []
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            line = q.get(timeout=deadline - time.monotonic())
        except queue.Empty:
            break
        try:
            frame = json.loads(line)
        except Exception:
            continue
        frames.append(frame)
        if frame.get("id") == want_id and (
            "result" in frame or "error" in frame
        ):
            return frames
    raise TimeoutError(f"no response for id={want_id} after {timeout_s}s; collected {len(frames)} frames")


def reader_thread(proc, q):
    for line in proc.stdout:
        try:
            q.put(line.decode("utf-8").strip())
        except Exception:
            pass


def main():
    if not os.path.exists(BIN):
        print(f"FAIL: shim binary not found at {BIN}", file=sys.stderr)
        sys.exit(1)

    srv = start_mock_backend()
    time.sleep(0.2)

    env = os.environ.copy()
    env["NWIRO_LOCAL_LLM_API_KEY_localllm"] = "test"
    env["RUST_LOG"] = "error"

    proc = subprocess.Popen(
        [BIN],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    _drain_stderr_thread(proc)

    q: "queue.Queue[str]" = queue.Queue()
    threading.Thread(target=reader_thread, args=(proc, q), daemon=True).start()

    try:
        # initialize
        send(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": 1, "clientCapabilities": {}}})
        read_until_response(proc, q, 1)

        # warmup against mock
        send(proc, {
            "jsonrpc": "2.0", "id": 2, "method": "session/warmup",
            "params": {"model": "test-model", "baseUrl": f"http://127.0.0.1:{MOCK_PORT}/v1", "adapterId": "localllm"},
        })
        read_until_response(proc, q, 2)

        # session/new
        send(proc, {"jsonrpc": "2.0", "id": 3, "method": "session/new", "params": {"cwd": os.getcwd(), "mcpServers": []}})
        new_frames = read_until_response(proc, q, 3)
        new_resp = next((f for f in new_frames if f.get("id") == 3), None)
        session_id = new_resp.get("result", {}).get("sessionId")
        assert session_id, "session/new missing sessionId"

        # session/prompt with TWO tools — a representative host tool schema:
        # `edit_thing` declares add_variables as exactly ONE non-string type
        # (array -> coercion target); `edit_union` declares it as a oneOf
        # union (-> deliberately untouched). The per-prompt schema map the
        # shim coerces against is built from THIS array.
        tool_array = [
            {
                "type": "function",
                "function": {
                    "name": "edit_thing",
                    "description": "Edit a thing (representative host tool)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target": {"type": "string"},
                            "add_variables": {"type": "array", "items": {"type": "object"}},
                        },
                        "required": ["target"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "edit_union",
                    "description": "Edit a thing whose add_variables is union-typed",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target": {"type": "string"},
                            "add_variables": {"oneOf": [{"type": "array"}, {"type": "string"}]},
                        },
                        "required": ["target"],
                    },
                },
            },
        ]
        send(proc, {
            "jsonrpc": "2.0", "id": 4, "method": "session/prompt",
            "params": {
                "sessionId": session_id,
                "prompt": [{"type": "text", "text": "add an IsActive bool variable"}],
                "tools": tool_array,
            }
        })

        # The shim sends mcp/connect + mcp/message as JSON-RPC REQUESTS to
        # us. Respond to those, RECORDING every dispatched tools/call params
        # (params.message.params = {name, arguments}) — the coercion is
        # asserted on what was ACTUALLY dispatched, not on mock call counts.
        frames_collected = []
        dispatched_calls = []
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            try:
                line = q.get(timeout=deadline - time.monotonic())
            except queue.Empty:
                break
            try:
                frame = json.loads(line)
            except Exception:
                continue
            frames_collected.append(frame)

            if frame.get("method") == "mcp/connect" and frame.get("id"):
                send(proc, {
                    "jsonrpc": "2.0", "id": frame["id"],
                    "result": {"connectionId": "conn_test"}
                })
            elif frame.get("method") == "mcp/message" and frame.get("id"):
                inner_params = frame.get("params", {}).get("message", {}).get("params", {})
                dispatched_calls.append(inner_params)
                send(proc, {
                    "jsonrpc": "2.0", "id": frame["id"],
                    "result": {"message": {
                        "jsonrpc": "2.0", "id": 0,
                        "result": {
                            "content": [{"type": "text", "text": "ok: applied"}],
                            "isError": False
                        }
                    }}
                })
            elif frame.get("id") == 4 and ("result" in frame or "error" in frame):
                break

        # Now analyse the collected frames + recorded dispatches
        if len(dispatched_calls) != 2:
            print(f"FAIL: expected 2 dispatched tools/call requests, got {len(dispatched_calls)}: {dispatched_calls}", file=sys.stderr)
            sys.exit(1)

        # (a) edit_thing: stringified array-typed field must dispatch as a
        # REAL JSON array (the v0.4.1 coercion); string-typed target
        # untouched. This also proves ToolCallAccum reassembled the
        # 4-fragment arguments stream correctly.
        d0 = dispatched_calls[0]
        if d0.get("name") != "edit_thing":
            print(f"FAIL: first dispatch name={d0.get('name')!r}, expected 'edit_thing'", file=sys.stderr)
            sys.exit(1)
        args0 = d0.get("arguments")
        if not isinstance(args0, dict):
            print(f"FAIL: edit_thing dispatched arguments not an object: {args0!r}", file=sys.stderr)
            sys.exit(1)
        av0 = args0.get("add_variables")
        if not isinstance(av0, list):
            print(f"FAIL: edit_thing add_variables must be a REAL JSON array after coercion, got {type(av0).__name__}: {av0!r}", file=sys.stderr)
            sys.exit(1)
        if len(av0) != 1 or av0[0] != {"name": "IsActive", "type": "bool"}:
            print(f"FAIL: edit_thing add_variables content wrong: {av0!r}", file=sys.stderr)
            sys.exit(1)
        if not isinstance(args0.get("target"), str) or args0["target"] != "BP_TestCube":
            print(f"FAIL: edit_thing target must stay the string 'BP_TestCube', got {args0.get('target')!r}", file=sys.stderr)
            sys.exit(1)

        # (b) edit_union: SAME stringified payload but a oneOf-union schema
        # -> must dispatch STILL A STRING (the host tool owns union
        # disambiguation; the shim must not guess).
        d1 = dispatched_calls[1]
        if d1.get("name") != "edit_union":
            print(f"FAIL: second dispatch name={d1.get('name')!r}, expected 'edit_union'", file=sys.stderr)
            sys.exit(1)
        args1 = d1.get("arguments")
        if not isinstance(args1, dict):
            print(f"FAIL: edit_union dispatched arguments not an object: {args1!r}", file=sys.stderr)
            sys.exit(1)
        av1 = args1.get("add_variables")
        if not isinstance(av1, str):
            print(f"FAIL: edit_union add_variables must STAY a string (union type), got {type(av1).__name__}: {av1!r}", file=sys.stderr)
            sys.exit(1)
        if av1 != STRINGIFIED_ARRAY:
            print(f"FAIL: edit_union add_variables string mutated: {av1!r} != {STRINGIFIED_ARRAY!r}", file=sys.stderr)
            sys.exit(1)

        # (c) tool_call / tool_call_update events still emit in the pinned
        # order: pending before completed, per call, calls in stream order.
        def updates(kind):
            return [
                f for f in frames_collected
                if f.get("method") == "session/update"
                and f.get("params", {}).get("update", {}).get("sessionUpdate") == kind
            ]
        pending_frames = updates("tool_call")
        completed_frames = updates("tool_call_update")
        if len(pending_frames) != 2:
            print(f"FAIL: expected 2 tool_call (pending) events, got {len(pending_frames)}", file=sys.stderr)
            sys.exit(1)
        if len(completed_frames) != 2:
            print(f"FAIL: expected 2 tool_call_update events, got {len(completed_frames)}", file=sys.stderr)
            sys.exit(1)
        expected = [("call_edit_1", "edit_thing"), ("call_edit_2", "edit_union")]
        for i, (call_id, title) in enumerate(expected):
            pend = pending_frames[i].get("params", {}).get("update", {})
            comp = completed_frames[i].get("params", {}).get("update", {})
            if pend.get("status") != "pending" or pend.get("toolCallId") != call_id or pend.get("title") != title:
                print(f"FAIL: tool_call[{i}] wrong shape: {pend!r}, expected pending/{call_id}/{title}", file=sys.stderr)
                sys.exit(1)
            if comp.get("status") != "completed" or comp.get("toolCallId") != call_id:
                print(f"FAIL: tool_call_update[{i}] wrong shape: {comp!r}, expected completed/{call_id}", file=sys.stderr)
                sys.exit(1)
            p_idx = frames_collected.index(pending_frames[i])
            c_idx = frames_collected.index(completed_frames[i])
            if p_idx >= c_idx:
                print(f"FAIL: tool_call_update[{i}] (idx {c_idx}) arrived before tool_call[{i}] (idx {p_idx})", file=sys.stderr)
                sys.exit(1)

        # (d) final prompt response has result.stopReason = "end_turn"
        final = next((f for f in frames_collected if f.get("id") == 4), None)
        if not final or final.get("result", {}).get("stopReason") != "end_turn":
            print(f"FAIL: id=4 response missing or wrong stopReason: {final}", file=sys.stderr)
            sys.exit(1)

        n_frags = len(_fragment(ARGS_FULL))
        print("PASS: v0.4.1 schema-aware coercion verified end-to-end through the real shim binary.")
        print(f"  (a) edit_thing : add_variables dispatched as REAL array {av0!r}; target stayed string {args0['target']!r}")
        print(f"  (b) edit_union : add_variables stayed STRING {av1!r} (oneOf union untouched)")
        print(f"  (c) event order: pending->completed per call, {len(pending_frames)} calls in stream order")
        print(f"  (d) stopReason : end_turn")
        print(f"  arguments string reassembled from {n_frags} SSE delta fragments per call")

    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        srv.shutdown()
        srv.server_close()


if __name__ == "__main__":
    main()
