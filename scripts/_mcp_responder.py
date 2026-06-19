"""Shim driver for the E2E model matrix (scripts/model-matrix.py).

The caller owns the subprocess; this module provides a single queue-based stdout
reader + the MCP-responder loop. A session/prompt that triggers tool calls can't
be driven by a plain frame-reader: the shim is the JSON-RPC CLIENT for
`mcp/connect` + `mcp/message`, so the harness must ANSWER those requests
mid-stream. Generalised from reports/smoke-test-tool-call-events.py (:305-343) so
the smoke test and the matrix share one responder.

ONE reader per process: start_reader() once after spawn, then route every read
(init, warmup, session, prompt) through the returned queue.
"""
from __future__ import annotations

import json
import queue
import threading
import time


def send(proc, obj: dict) -> None:
    """Write one line-delimited JSON-RPC frame to the shim's stdin."""
    proc.stdin.write((json.dumps(obj) + "\n").encode("utf-8"))
    proc.stdin.flush()


def start_reader(proc) -> "queue.Queue":
    """Daemon thread: each stdout line → the returned queue. ONE per process
    (two readers would race on the single stdout pipe)."""
    q: queue.Queue = queue.Queue()

    def _reader() -> None:
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode("utf-8", "replace").strip()
            if line:
                q.put(line)

    threading.Thread(target=_reader, daemon=True).start()
    return q


def read_until(q: "queue.Queue", want_id, *, timeout_s: float = 30.0) -> list:
    """Collect frames until a response with id==want_id (or timeout). Returns the
    collected frames. Use for init / warmup / session / set_config (no MCP)."""
    frames: list = []
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            line = q.get(timeout=max(0.0, deadline - time.monotonic()))
        except queue.Empty:
            break
        try:
            f = json.loads(line)
        except Exception:
            continue
        frames.append(f)
        if f.get("id") == want_id and ("result" in f or "error" in f):
            break
    return frames


def drive_prompt_with_mcp(proc, q, prompt_id, mcp_result: dict, *, timeout_s: float = 120.0, timings: list = None) -> list:
    """The caller has ALREADY sent the session/prompt with id==prompt_id. Collect
    frames, answering the shim's `mcp/connect` ({connectionId}) and `mcp/message`
    ({result:{message:{result: mcp_result}}}) REQUESTS, until the prompt response
    (id==prompt_id) lands or the timeout fires.

    `mcp_result` is the inner MCP `{content, isError}` envelope the tool returns —
    set `isError: True` to exercise the failure path. Returns the collected frames
    (for the C1-C8 assertions in model-matrix.py).

    If `timings` (a list) is passed, append `time.monotonic()` for EACH collected
    frame (parallel to the returned frames) so the caller can derive first-token
    latency and the max inter-frame gap — the v0.3.0 timing-guard margin telemetry.
    Backward-compatible: callers that omit `timings` are unaffected."""
    frames: list = []
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            line = q.get(timeout=max(0.0, deadline - time.monotonic()))
        except queue.Empty:
            break
        try:
            f = json.loads(line)
        except Exception:
            continue
        frames.append(f)
        if timings is not None:
            timings.append(time.monotonic())
        method = f.get("method")
        if method == "mcp/connect" and f.get("id") is not None:
            send(proc, {"jsonrpc": "2.0", "id": f["id"], "result": {"connectionId": "conn_matrix"}})
        elif method == "mcp/message" and f.get("id") is not None:
            send(
                proc,
                {
                    "jsonrpc": "2.0",
                    "id": f["id"],
                    "result": {"message": {"jsonrpc": "2.0", "id": 0, "result": mcp_result}},
                },
            )
        elif f.get("id") == prompt_id and ("result" in f or "error" in f):
            break
    return frames
