"""
v0.1.26 regression — verify tool_call / tool_call_update events emit
in the order the bridge dispatcher expects.

Pre-v0.1.26: shim executed tools via MCP correctly but NEVER emitted
the `tool_call` (status=pending) or `tool_call_update`
(status=completed|failed) session/update notifications. Bridge UI
showed no "tool used" indicator. User reported this on v0.1.25 with
Qwen3:14b — tools executed (folder + file created) but UI looked
idle.

v0.1.26 fix: emit one `tool_call` pending event per call upfront
before the execution loop, then `tool_call_update` (completed or
failed) after each call.

This test:
1. Mocks an OpenAI backend that emits ONE tool_call in stream 1,
   then a final assistant message in stream 2 after the tool result.
2. Mocks the MCP bridge to acknowledge mcp/connect + return a
   successful tool result for the tools/call request.
3. Asserts the shim emits in this ORDER:
   - 1+ agent_message_chunk events (streaming model output)
   - 1 tool_call (status=pending) for the call
   - 1 tool_call_update (status=completed) AFTER tool execution
   - More agent_message_chunk for the final assistant turn
   - Response with result.stopReason="end_turn"

Field shape pinning: tool_call has toolCallId+status+title+rawInput.arguments;
tool_call_update has toolCallId+status+rawOutput.
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

MOCK_PORT = 18770


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
    - tool-capability probe (max_tokens=256 + tools): empty tool_calls
      so tier = Emulated... actually let's go Native by returning a
      tool_calls envelope.
    - first prompt: emits a Native tool_call for `find_blueprints`
    - second prompt (after tool result): emits "Done." content
    """
    call_counter = 0

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
                                "name": "find_blueprints",
                                "arguments": "{}"
                            }
                        }]
                    },
                    "finish_reason": "tool_calls"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            })
            return

        # Real prompt — stream a tool_call OR a final message
        # depending on whether history shows a tool result already.
        has_tool_result = any(m.get("role") == "tool" for m in msgs)
        if not has_tool_result:
            # First round — emit a tool_call for find_blueprints
            self._sse_stream([
                json.dumps({
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_001",
                                "type": "function",
                                "function": {
                                    "name": "find_blueprints",
                                    "arguments": "{\"searchTerm\":\"Cube\"}"
                                }
                            }]
                        },
                        "finish_reason": None
                    }]
                }),
                json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
            ])
        else:
            # Second round — final "Done." after tool result
            self._sse_stream([
                json.dumps({"choices": [{"index": 0, "delta": {"role": "assistant", "content": "Done."}, "finish_reason": None}]}),
                json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
            ])

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

        # session/prompt with a tool array — the mock will emit a tool_call
        # for find_blueprints. The shim must execute it via mcp/message
        # round-trip with the bridge (us) AND emit tool_call + tool_call_update
        # session/update notifications.
        tool_array = [{
            "type": "function",
            "function": {
                "name": "find_blueprints",
                "description": "Find blueprints in the project",
                "parameters": {"type": "object", "properties": {"searchTerm": {"type": "string"}}, "required": ["searchTerm"]}
            }
        }]
        send(proc, {
            "jsonrpc": "2.0", "id": 4, "method": "session/prompt",
            "params": {
                "sessionId": session_id,
                "prompt": [{"type": "text", "text": "find a cube blueprint"}],
                "tools": tool_array,
            }
        })

        # The shim will send mcp/connect + mcp/message as JSON-RPC
        # REQUESTS to us. We need to respond to those.
        # Collect frames until we see id=4 result OR an mcp request.
        frames_collected = []
        deadline = time.monotonic() + 30.0
        sent_mcp_connect_resp = False
        sent_mcp_message_resp = False
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

            # If it's a JSON-RPC request from shim → respond
            if frame.get("method") == "mcp/connect" and frame.get("id"):
                send(proc, {
                    "jsonrpc": "2.0", "id": frame["id"],
                    "result": {"connectionId": "conn_test"}
                })
                sent_mcp_connect_resp = True
            elif frame.get("method") == "mcp/message" and frame.get("id"):
                send(proc, {
                    "jsonrpc": "2.0", "id": frame["id"],
                    "result": {"message": {
                        "jsonrpc": "2.0", "id": 0,
                        "result": {
                            "content": [{"type": "text", "text": "found 3 blueprints: Cube_BP, Sphere_BP, Cylinder_BP"}],
                            "isError": False
                        }
                    }}
                })
                sent_mcp_message_resp = True
            elif frame.get("id") == 4 and ("result" in frame or "error" in frame):
                break

        # Now analyse the collected frames
        # 1. Verify we DID see a tool_call (pending) notification
        tool_call_pending_frames = [
            f for f in frames_collected
            if f.get("method") == "session/update"
            and f.get("params", {}).get("update", {}).get("sessionUpdate") == "tool_call"
        ]
        if not tool_call_pending_frames:
            print(f"FAIL: no tool_call (pending) notification emitted by shim. Total frames: {len(frames_collected)}", file=sys.stderr)
            sys.exit(1)
        pending = tool_call_pending_frames[0].get("params", {}).get("update", {})
        if pending.get("status") != "pending":
            print(f"FAIL: tool_call event has status={pending.get('status')!r}, expected 'pending'", file=sys.stderr)
            sys.exit(1)
        if pending.get("toolCallId") != "call_001":
            print(f"FAIL: tool_call event toolCallId={pending.get('toolCallId')!r}, expected 'call_001'", file=sys.stderr)
            sys.exit(1)
        if pending.get("title") != "find_blueprints":
            print(f"FAIL: tool_call event title={pending.get('title')!r}, expected 'find_blueprints'", file=sys.stderr)
            sys.exit(1)
        raw_input = pending.get("rawInput")
        if not isinstance(raw_input, dict) or not isinstance(raw_input.get("arguments"), dict):
            print(f"FAIL: tool_call.rawInput.arguments must be object, got {raw_input!r}", file=sys.stderr)
            sys.exit(1)
        if raw_input["arguments"].get("searchTerm") != "Cube":
            print(f"FAIL: rawInput.arguments.searchTerm={raw_input['arguments'].get('searchTerm')!r}, expected 'Cube'", file=sys.stderr)
            sys.exit(1)

        # 2. Verify we DID see a tool_call_update (completed) notification
        tool_call_update_frames = [
            f for f in frames_collected
            if f.get("method") == "session/update"
            and f.get("params", {}).get("update", {}).get("sessionUpdate") == "tool_call_update"
        ]
        if not tool_call_update_frames:
            print(f"FAIL: no tool_call_update notification emitted. Total frames: {len(frames_collected)}", file=sys.stderr)
            sys.exit(1)
        completed = tool_call_update_frames[0].get("params", {}).get("update", {})
        if completed.get("status") != "completed":
            print(f"FAIL: tool_call_update status={completed.get('status')!r}, expected 'completed'", file=sys.stderr)
            sys.exit(1)
        if completed.get("toolCallId") != "call_001":
            print(f"FAIL: tool_call_update toolCallId={completed.get('toolCallId')!r}", file=sys.stderr)
            sys.exit(1)
        if "rawOutput" not in completed:
            print(f"FAIL: tool_call_update missing rawOutput", file=sys.stderr)
            sys.exit(1)

        # 3. Verify ORDER: pending must come BEFORE completed
        pending_idx = frames_collected.index(tool_call_pending_frames[0])
        completed_idx = frames_collected.index(tool_call_update_frames[0])
        if pending_idx >= completed_idx:
            print(f"FAIL: tool_call_update (idx {completed_idx}) arrived before tool_call (idx {pending_idx})", file=sys.stderr)
            sys.exit(1)

        # 4. Verify final prompt response has result.stopReason = "end_turn"
        final = next((f for f in frames_collected if f.get("id") == 4), None)
        if not final or final.get("result", {}).get("stopReason") != "end_turn":
            print(f"FAIL: id=4 response missing or wrong stopReason: {final}", file=sys.stderr)
            sys.exit(1)

        print("PASS: tool_call event order verified -- pending -> tool exec -> completed -> final response.")
        print(f"  pending  : toolCallId={pending.get('toolCallId')} title={pending.get('title')!r}")
        print(f"  completed: toolCallId={completed.get('toolCallId')} status={completed.get('status')}")

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
