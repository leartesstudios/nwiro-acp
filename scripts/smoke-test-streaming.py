"""
Smoke test for v0.1.18 STREAM-001 + STREAM-002.

The HTTP mock backend streams content over 5 SSE chunks with 200ms
delays between them. Total stream duration ~1000ms. The driver sends
session/cancel after ~400ms (i.e. after ~2 chunks have been emitted)
and measures how long the shim takes to surface the -32800 cancel
response.

What we're testing
==================

  * STREAM-001 (real-time streaming): at least 2 session/update
    `agent_message_chunk` notifications must arrive BEFORE the cancel
    is sent. Without the per-prompt mpsc drainer task, all 5 updates
    would be buffered and arrive together AFTER the stream completes.

  * STREAM-002 (mid-stream cancel): elapsed time from sending
    session/cancel to receiving the -32800 prompt response must be
    < ~500ms. If the cancel takes ~1000ms+, it indicates the cancel
    only takes effect after the prompt handler completes naturally —
    i.e. the dispatcher is blocked in `handle_session_prompt.await`
    and session/cancel waits in `bridge_rx` until the prompt finishes.

STREAM-002 (mid-stream cancel latency) was debated as a possible no-op
given the frame-router; this test arbitrates by measuring it directly.

Interpretation
==============

| Updates seen | Cancel elapsed | Verdict |
|---|---|---|
| < 2 before cancel | n/a       | STREAM-001 broken — drainer not emitting in real-time |
| ≥ 2 before cancel | < 500ms   | STREAM-002 also works — cancel pre-empts the stream |
| ≥ 2 before cancel | > 800ms   | STREAM-002 needs its own fix — cancel waits for prompt completion |
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

MOCK_PORT = 18758  # avoid collision with other smoke tests


def sse_chunk(obj: dict) -> bytes:
    return f"data: {json.dumps(obj)}\n\n".encode("utf-8")


class MockOpenAIHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body_bytes = self.rfile.read(length)
        try:
            body = json.loads(body_bytes.decode("utf-8"))
        except Exception:
            body = {}

        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return

        is_stream = bool(body.get("stream"))
        has_tools = bool(body.get("tools"))
        is_probe = has_tools and bool(body.get("tool_choice"))

        if not is_stream:
            # warmup or probe
            if is_probe:
                # Classify the test model as Native so the bridge guard
                # allows the prompt without any parser involvement.
                resp = {
                    "id": "p1", "object": "chat.completion",
                    "model": body.get("model", ""),
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant", "content": None,
                            "tool_calls": [{
                                "id": "probe_call", "type": "function",
                                "function": {"name": "find_blueprints", "arguments": "{}"}
                            }]
                        },
                        "finish_reason": "tool_calls"
                    }]
                }
            else:
                resp = {
                    "id": "w1", "object": "chat.completion",
                    "model": body.get("model", ""),
                    "choices": [{"index": 0,
                                 "message": {"role": "assistant", "content": "."},
                                 "finish_reason": "stop"}]
                }
            payload = json.dumps(resp).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        # Streaming path: 5 chunks, 200ms apart, total ~1s.
        # Each chunk flushes immediately so the receiver (the shim's
        # eventsource-stream parser) sees each as a discrete event.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        try:
            for i in range(5):
                chunk = {
                    "id": f"c{i}", "object": "chat.completion.chunk",
                    "model": body.get("model", ""),
                    "choices": [{
                        "index": 0,
                        "delta": {"content": f"part{i} "},
                        "finish_reason": None
                    }]
                }
                self.wfile.write(sse_chunk(chunk))
                self.wfile.flush()
                time.sleep(0.2)
            # Terminating chunk
            self.wfile.write(sse_chunk({
                "id": "c-end", "object": "chat.completion.chunk",
                "model": body.get("model", ""),
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            }))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            # Shim cancelled the stream — dropped the TCP connection.
            # That's the expected behaviour when cancel_token fires
            # inside chat_completion_stream's tokio::select! arm.
            # Windows wraps the disconnect as ConnectionAbortedError
            # (WinError 10053); Linux uses BrokenPipeError. Catch all.
            pass


def _drain_stderr_thread(proc):
    """Daemon thread draining proc.stderr so the OS pipe buffer never fills
    and blocks the shim's tracing writes. Spawn immediately after Popen.
    See smoke-test.py for the shared Windows-pipe diagnosis.
    """
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


def start_http_mock():
    httpd = socketserver.TCPServer(("127.0.0.1", MOCK_PORT), MockOpenAIHandler)
    httpd.allow_reuse_address = True
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd


def send(proc, msg, lock):
    line = (json.dumps(msg) + "\n").encode("utf-8")
    with lock:
        proc.stdin.write(line)
        proc.stdin.flush()


def main():
    if not os.path.exists(BIN):
        print(f"FAIL: binary not found at {BIN}")
        sys.exit(2)

    httpd = start_http_mock()
    failures = []
    proc = None
    try:
        env = os.environ.copy()
        env["NWIRO_LOCAL_LLM_BASE_URL"] = f"http://127.0.0.1:{MOCK_PORT}/v1"
        env["NWIRO_LOCAL_LLM_MODEL"] = "stream-test"

        proc = subprocess.Popen(
            [BIN],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        _drain_stderr_thread(proc)
        stdin_lock = threading.Lock()

        # Single-threaded reader that timestamps every frame on arrival.
        # We don't auto-respond to anything; just collect with timing.
        timestamped: "queue.Queue[tuple[float, dict]]" = queue.Queue()

        def reader():
            buf = b""
            try:
                while True:
                    chunk = proc.stdout.read1(4096)
                    if not chunk:
                        timestamped.put((time.time(), None))  # type: ignore
                        return
                    arrival = time.time()
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            frame = json.loads(line.decode("utf-8"))
                            timestamped.put((arrival, frame))
                        except Exception:
                            continue
            except Exception:
                timestamped.put((time.time(), None))  # type: ignore

        threading.Thread(target=reader, daemon=True).start()

        def recv_until(predicate, timeout_s):
            deadline = time.time() + timeout_s
            collected = []
            while time.time() < deadline:
                try:
                    ts_frame = timestamped.get(timeout=max(0.05, deadline - time.time()))
                except queue.Empty:
                    continue
                if ts_frame[1] is None:
                    break
                collected.append(ts_frame)
                if predicate(ts_frame[1]):
                    return collected
            return collected

        # 1. initialize + warmup + new + set_config_option
        send(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}, stdin_lock)
        recv_until(lambda f: f.get("id") == 1, 5.0)
        print("OK initialize")

        send(proc, {"jsonrpc": "2.0", "id": 2, "method": "session/warmup",
                    "params": {"model": "stream-test", "keepAlive": "15m"}}, stdin_lock)
        warmup_frames = recv_until(lambda f: f.get("id") == 2, 10.0)
        warmup = next((ts_f[1] for ts_f in warmup_frames if ts_f[1].get("id") == 2), None)
        tier = warmup.get("result", {}).get("toolTier") if warmup else None
        print(f"OK warmup toolTier={tier}")
        if tier != "native":
            failures.append(f"expected toolTier=native (probe-as-Native), got {tier!r}")
            return failures

        send(proc, {"jsonrpc": "2.0", "id": 3, "method": "session/new", "params": {}}, stdin_lock)
        new_frames = recv_until(lambda f: f.get("id") == 3, 5.0)
        new_resp = next((ts_f[1] for ts_f in new_frames if ts_f[1].get("id") == 3), None)
        session_id = new_resp["result"]["sessionId"]
        print(f"OK session/new sid={session_id}")

        send(proc, {"jsonrpc": "2.0", "id": 4, "method": "session/set_config_option",
                    "params": {"sessionId": session_id, "configId": "model", "value": "stream-test"}}, stdin_lock)
        recv_until(lambda f: f.get("id") == 4, 5.0)
        print("OK set_config_option")

        # 2. Send prompt at T0 — backend will stream for ~1s.
        T0 = time.time()
        send(proc, {"jsonrpc": "2.0", "id": 5, "method": "session/prompt",
                    "params": {
                        "sessionId": session_id,
                        "prompt": [{"type": "text", "text": "stream me something"}],
                    }}, stdin_lock)

        # 3. Wait for at least 2 updates (~400ms) then send cancel.
        update_arrivals = []
        cancel_sent_at = None
        deadline = T0 + 3.0
        cancel_response = None
        while time.time() < deadline:
            try:
                ts_frame = timestamped.get(timeout=max(0.05, deadline - time.time()))
            except queue.Empty:
                continue
            if ts_frame[1] is None:
                break
            f = ts_frame[1]
            if f.get("method") == "session/update":
                update_arrivals.append(ts_frame)
                if len(update_arrivals) >= 2 and cancel_sent_at is None:
                    # We've seen real-time streaming. Fire cancel.
                    cancel_sent_at = time.time()
                    send(proc, {"jsonrpc": "2.0", "method": "session/cancel",
                                "params": {"sessionId": session_id}}, stdin_lock)
                    print(f"  -- sent cancel at T+{cancel_sent_at-T0:.3f}s "
                          f"(after {len(update_arrivals)} updates)")
            elif f.get("id") == 5:
                cancel_response = ts_frame
                break

        # ── Assessments ─────────────────────────────────────
        # STREAM-001: did we see ≥2 updates before cancel?
        if cancel_sent_at is None:
            failures.append(
                f"STREAM-001 FAIL: fewer than 2 updates arrived before timeout; "
                f"streaming may still be buffered. Got {len(update_arrivals)} updates."
            )
        else:
            print(f"OK STREAM-001: {len(update_arrivals)} updates arrived in real-time "
                  f"BEFORE cancel (real-time mpsc drainer works)")

        # STREAM-002: how fast did the cancel take effect?
        if cancel_response is None:
            failures.append(
                f"STREAM-002 INCONCLUSIVE: id=5 response never arrived. "
                f"Possible deadlock or never-cancel scenario."
            )
        elif cancel_sent_at is not None:
            cancel_elapsed = cancel_response[0] - cancel_sent_at
            response_body = cancel_response[1]
            print(f"  -- prompt response arrived T+{cancel_response[0]-T0:.3f}s "
                  f"({cancel_elapsed*1000:.0f}ms after cancel send)")
            # v0.1.24 G2 round-3: cancel now produces an ACP-compliant
            # `result: {stopReason: "cancelled"}` per ACP prompt-turn
            # spec — NOT a JSON-RPC error -32800. Pre-v0.1.24 used the
            # -32800 error code (non-conformant with ACP).
            #
            # Per JSON-RPC 2.0 spec, a single response MUST carry EITHER
            # `result` OR `error`, never both. Round-3 critic explicitly
            # asked for a no-`error` assertion to pin the spec-compliant
            # shape: if both fields ever appear, that's a regression
            # back to the dual-shape that the round-1 implementation
            # accidentally produced.
            if "error" in response_body:
                failures.append(
                    f"STREAM-002 FAIL: response carries both `result` and `error` — "
                    f"JSON-RPC 2.0 forbids this. error={response_body.get('error')}"
                )
            result = response_body.get("result", {})
            cancelled_via_stop_reason = result.get("stopReason") == "cancelled"
            if cancelled_via_stop_reason:
                if cancel_elapsed < 0.5:
                    print(f"OK STREAM-002: cancel-mid-stream works "
                          f"({cancel_elapsed*1000:.0f}ms — cancel pre-empts the stream) "
                          f"with ACP stopReason='cancelled'")
                else:
                    failures.append(
                        f"STREAM-002 partial: cancel was processed (stopReason='cancelled') "
                        f"but took {cancel_elapsed*1000:.0f}ms — likely after natural stream "
                        f"completion. STREAM-002 is real: cancel waits"
                        f"for dispatcher."
                    )
            else:
                # Some other completion reason landed — cancel may have
                # arrived too late, OR the response shape regressed.
                stop_reason_value = result.get("stopReason", "<missing>")
                failures.append(
                    f"STREAM-002 FAIL: expected stopReason='cancelled' after cancel sent, "
                    f"got stopReason={stop_reason_value!r}. Took {cancel_elapsed*1000:.0f}ms. "
                    f"Full response: {response_body}"
                )

    finally:
        if proc:
            try:
                proc.stdin.close()
            except Exception:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
            if hasattr(proc, "_stderr_thread"):
                proc._stderr_thread.join(timeout=1.0)
            stderr = b"".join(getattr(proc, "_stderr_chunks", []))
            if stderr:
                print(f"STDERR (should be empty):\n{stderr.decode('utf-8', errors='replace')}")
        httpd.shutdown()

    return failures


if __name__ == "__main__":
    failures = main()
    print()
    if failures:
        print(f"=== {len(failures)} FAILURE(S) ===")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("=== ALL CHECKS PASSED ===")
    sys.exit(0)
