"""
Smoke test for local-llm-acp v0.1.13 — tier-None refusal path.

Drives the shim binary via stdin/stdout JSON-RPC. Ollama does not need to
be running for THIS test — we deliberately exercise the failed-warmup →
tier-None → refusal path. Native/Emulated probe paths need a real backend
and are skipped here (they require a real backend; see docs/RUNNING.md §6).

Verifies:
  - AC-6: session/new inherits last_warmup_tier
  - AC-7: default tier is "none" when no warmup ran (and also when warmup
          failed — same outcome via the failure path that sets None)
  - AC-4 (v0.1.20-updated): tier-None + tools attached + unreachable
          backend → session/prompt returns -32000 error (HTTP connect
          refused). The upfront tier-None refusal was REMOVED in v0.1.20;
          chats are no longer blocked at the gate. The model-call layer
          surfaces the backend failure instead.
  - Wire shape: WarmupResult JSON includes "toolTier" field
"""
import json
import os
import queue
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

# v0.1.20: REFUSAL text is no longer emitted at all — the upfront
# tier-None refusal was removed. Kept here as a NEGATIVE assertion:
# if this string appears anywhere in session/update output, that's a
# regression (the guard came back).
REFUSAL_STR = (
    "This model does not support tool calls. Please switch to a tool-capable "
    "model such as Qwen2.5 14B, Mistral Nemo, or Llama 3.1 70B+."
)


def send(proc, msg):
    line = json.dumps(msg) + "\n"
    proc.stdin.write(line.encode("utf-8"))
    proc.stdin.flush()


def read_n_frames(proc, n, timeout_s=10.0):
    """Read n line-delimited JSON frames from stdout, with a real deadline.

    On Windows, `proc.stdout.read1()` blocks on an empty pipe and the deadline
    check between calls can never fire — when n > frames-emitted (e.g. n=4 but
    the shim only writes 1 response frame) the loop hangs forever. Bridge the
    blocking read through a dedicated reader thread writing into a Queue,
    then consume with Queue.get(timeout=...) so the deadline is enforced
    even while waiting for bytes.

    Ported from smoke-test-model-switch.py:read_frames() in this release.
    """
    if not hasattr(proc, "_chunk_q"):
        proc._chunk_q = queue.Queue()
        proc._buf = b""

        def _reader():
            try:
                while True:
                    chunk = proc.stdout.read1(4096)
                    if not chunk:
                        proc._chunk_q.put(b"")  # EOF sentinel
                        return
                    proc._chunk_q.put(chunk)
            except Exception:
                proc._chunk_q.put(b"")

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        proc._reader_thread = t

    frames = []
    deadline = time.time() + timeout_s
    buf = proc._buf
    while len(frames) < n:
        while b"\n" in buf and len(frames) < n:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line:
                continue
            frames.append(json.loads(line.decode("utf-8")))
        if len(frames) >= n:
            break
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        try:
            chunk = proc._chunk_q.get(timeout=remaining)
        except queue.Empty:
            break
        if not chunk:  # EOF sentinel
            break
        buf += chunk
    proc._buf = buf
    return frames, buf


def _drain_stderr_thread(proc):
    """Spawn a daemon thread that drains the shim's stderr into a list,
    so the OS pipe buffer never fills and blocks the shim's tracing writes.

    Per the shim's v0.1.5 invariant, stderr should be empty in steady state.
    But: tracing-subscriber can emit on shutdown, and any write to a full
    stderr pipe would block the runtime. Drain unconditionally.
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


def main():
    if not os.path.exists(BIN):
        print(f"FAIL: binary not found at {BIN}", file=sys.stderr)
        sys.exit(2)

    env = os.environ.copy()
    # Point at a deliberately-unreachable host so warmup fails fast →
    # tool_tier defaults to None on the failure path.
    env["NWIRO_LOCAL_LLM_BASE_URL"] = "http://127.0.0.1:1/v1"
    env["NWIRO_LOCAL_LLM_MODEL"] = "fake-model"

    proc = subprocess.Popen(
        [BIN],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    # Spawn the stderr drainer IMMEDIATELY after process start — must
    # run for the whole lifetime of `proc`. If we wait until the finally
    # block to drain, any stderr write before then risks blocking the
    # shim's runtime if the OS pipe fills (Windows default ~4KB).
    _drain_stderr_thread(proc)

    failures = []
    try:
        # 1. initialize
        send(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        frames, _ = read_n_frames(proc, 1)
        if not frames or "result" not in frames[0]:
            failures.append(f"initialize response missing result: {frames}")
            print(f"FAIL initialize: {frames}")
            return failures
        print(f"OK initialize: serverInfo={frames[0]['result'].get('serverInfo')}")

        # 2. session/warmup — will fail with "unreachable" (Ollama not at port 1)
        send(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/warmup",
                "params": {"model": "fake-model", "keepAlive": "15m"},
            },
        )
        frames, _ = read_n_frames(proc, 1, timeout_s=15.0)
        if not frames:
            failures.append("session/warmup: no response within 15s")
            return failures
        warmup_result = frames[0].get("result", {})
        print(f"OK session/warmup response: {warmup_result}")
        # Assertions on AC-7 + wire shape:
        if warmup_result.get("status") != "failed":
            failures.append(f"AC: expected status=failed (Ollama unreachable), got {warmup_result.get('status')}")
        if "toolTier" not in warmup_result:
            failures.append(f"AC: WarmupResult missing toolTier field: {warmup_result}")
        if warmup_result.get("toolTier") != "none":
            failures.append(f"AC-7: expected toolTier=none on failed warmup, got {warmup_result.get('toolTier')!r}")

        # 3. session/new — should inherit tier from warmup (none)
        send(proc, {"jsonrpc": "2.0", "id": 3, "method": "session/new", "params": {}})
        frames, _ = read_n_frames(proc, 1)
        if not frames or "result" not in frames[0]:
            failures.append(f"session/new failed: {frames}")
            return failures
        session_id = frames[0]["result"]["sessionId"]
        print(f"OK session/new sessionId={session_id}")

        # 4. session/prompt with tools — must trigger refusal
        send(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "session/prompt",
                "params": {
                    "sessionId": session_id,
                    "prompt": [{"type": "text", "text": "do a tool call"}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "foo",
                                "description": "x",
                                "parameters": {
                                    "type": "object",
                                    "properties": {},
                                },
                            },
                        }
                    ],
                },
            },
        )
        # The upfront tier-None refusal was removed (v0.1.20): the prompt now
        # proceeds to chat_completion_stream against the unreachable backend
        # (port 1), which fails fast at the TCP connect layer. v0.3.0 surfaces
        # that as a CLEAN refusal (stopReason: "refusal" + _meta.errorKind:
        # "unreachable"); older builds surfaced a -32000 JSON-RPC error. Either
        # diagnosable surface is acceptable — what matters is no hang and no
        # upfront tier-None refusal text.
        #
        # Read up to 4 frames (some shim builds may emit a startup log
        # line on stdout via tracing — defensive ceiling). The
        # id=4 response is what we assert on.
        frames, _ = read_n_frames(proc, 4, timeout_s=15.0)
        result_frame = next((f for f in frames if f.get("id") == 4), None)

        # NEGATIVE assertion: REFUSAL text must NOT appear in any
        # session/update frame. The v0.1.20 refusal-removal is the
        # whole point of this change.
        for f in frames:
            if f.get("method") == "session/update":
                update_text = (
                    f.get("params", {})
                    .get("update", {})
                    .get("content", {})
                    .get("text", "")
                )
                if update_text == REFUSAL_STR:
                    failures.append(
                        f"v0.1.20 REGRESSION: refusal text emitted — the upfront "
                        f"tier-None refusal came back."
                    )

        if result_frame is None:
            failures.append(f"AC-4: missing id=4 response frame: {frames}")
        elif "error" in result_frame:
            # Older builds surfaced the unreachable backend as a -32000 error.
            err = result_frame.get("error", {})
            code = err.get("code")
            if code == -32000:
                print(f"OK session/prompt returned -32000 (backend unreachable): {err.get('message', '')[:80]}")
            else:
                failures.append(
                    f"AC-4: expected a diagnosable failure (-32000 or a clean refusal) "
                    f"from the unreachable backend, got code={code}: {err}"
                )
        elif result_frame.get("result") is not None:
            # v0.3.0: an unreachable backend now surfaces as a CLEAN refusal
            # (stopReason: "refusal" + _meta.errorKind) instead of a raw -32000 —
            # a diagnosable surface, not a hang. Accept it as the correct outcome.
            res = result_frame["result"]
            error_kind = (res.get("_meta") or {}).get("errorKind")
            if res.get("stopReason") == "refusal" and error_kind:
                print(f"OK session/prompt returned a clean refusal (errorKind={error_kind}, stopReason=refusal)")
            else:
                failures.append(
                    f"AC-4: expected a diagnosable failure (-32000 or a clean refusal with "
                    f"errorKind) from the unreachable backend, got result: {res!r}"
                )

    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            # After kill, wait briefly so the kernel actually reaps the
            # process and the daemon stderr reader gets its EOF.
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        # The drainer thread has been collecting stderr into a list since
        # process start — no blocking read here. Just join briefly so any
        # final chunks land, then format whatever accumulated. Per the
        # v0.1.5 "silence stderr" invariant, this should be empty.
        if hasattr(proc, "_stderr_thread"):
            proc._stderr_thread.join(timeout=1.0)
        stderr_bytes = b"".join(getattr(proc, "_stderr_chunks", []))
        if stderr_bytes:
            print(
                "STDERR (should be empty per shim invariant!):\n"
                + stderr_bytes.decode("utf-8", errors="replace")
            )

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
