#!/usr/bin/env python3
"""Performance harness for local-llm-acp (publish gate F-PERF-01..07, F-OBS-03).

Drives the REAL release binary as a subprocess against a DETERMINISTIC in-process
mock OpenAI backend (no real model, no network), times every ACP frame at the
shim's stdout, and reports each F-PERF metric against its numeric budget. Exits
non-zero if any HARD budget is violated, so it doubles as a CI regression guard
(the publish bar is "measured + committed"; CI-gating is the P1 follow-up).

  python scripts/perf-harness.py            # full run (incl. the 500-prompt soak)
  python scripts/perf-harness.py --quick    # skip the slow RSS soak (F-PERF-06)
  python scripts/perf-harness.py --json out.json

Dependency-free (stdlib only). RSS uses psutil if present, else PowerShell
Get-Process (Windows) / /proc (Linux). Conventions (mock server, Queue-backed
Windows-safe frame reader, stderr drain, mcp auto-reply) mirror reports/smoke-*.py.
"""
import http.server
import json
import os
import queue
import socketserver
import statistics
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, "..", "target", "release", "local-llm-acp" + (".exe" if os.name == "nt" else ""))
MOCK_PORT = 18799
BASE_URL = f"http://127.0.0.1:{MOCK_PORT}/v1"
COALESCE_MS = 25  # production default (NWIRO_LOCAL_LLM_STREAM_COALESCE_MS)

# ── Deterministic mock OpenAI backend ─────────────────────────────────────────
# A single in-process HTTP server. Each measurement mutates STATE before driving
# the shim; the handler reads STATE to emit a reproducible response. No random
# delays, no adaptive behaviour — same request shape → same bytes every time.
STATE = {
    "mode": "content",     # content | tool
    "tokens": 5,           # number of streamed content tokens
    "spacing_s": 0.0,      # inter-token sleep (server side)
    "first_chunk_delay_s": 0.0,
    "warmup_delay_s": 0.0,  # hang the non-stream (warmup/probe) path this long
}


def _sse(obj):
    return f"data: {json.dumps(obj)}\n\n".encode("utf-8")


class MockHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_):  # silence
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = {}
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except Exception:
            body = {}
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return

        if not body.get("stream"):
            # Warmup / probe path (non-streaming). Optionally hang to exercise the
            # warmup timeout cap (F-PERF-03), else answer instantly.
            if STATE["warmup_delay_s"] > 0:
                time.sleep(STATE["warmup_delay_s"])
            resp = {
                "id": "w1", "object": "chat.completion", "model": body.get("model", ""),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "."}, "finish_reason": "stop"}],
            }
            payload = json.dumps(resp).encode("utf-8")
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                pass  # the shim closed early (e.g. its warmup timeout fired)
            return

        # Streaming path.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            msgs = body.get("messages", [])
            is_followup = any(m.get("role") == "tool" for m in msgs)
            if STATE["mode"] == "tool" and not is_followup:
                # Round 1: a single native tool_call.
                self.wfile.write(_sse({"choices": [{"index": 0, "delta": {"tool_calls": [
                    {"index": 0, "id": "call_1", "type": "function",
                     "function": {"name": "find_blueprints", "arguments": "{\"searchTerm\":\"x\"}"}}]}}]}))
                self.wfile.write(_sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}))
            else:
                # Content stream: STATE["tokens"] deterministic content deltas.
                if STATE["first_chunk_delay_s"] > 0:
                    time.sleep(STATE["first_chunk_delay_s"])
                for i in range(STATE["tokens"]):
                    self.wfile.write(_sse({"choices": [{"index": 0, "delta": {"content": f"t{i} "}, "finish_reason": None}]}))
                    self.wfile.flush()
                    if STATE["spacing_s"] > 0 and i < STATE["tokens"] - 1:
                        time.sleep(STATE["spacing_s"])
                self.wfile.write(_sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass


class _ReusableTCPServer(socketserver.TCPServer):
    # CLASS attribute so SO_REUSEADDR is set during bind() inside __init__. Setting
    # the instance attr after construction (as a plain TCPServer would) is a no-op —
    # the socket is already bound, and an immediate re-run then hits
    # "address already in use" while the port lingers in TIME_WAIT.
    allow_reuse_address = True


def start_mock():
    httpd = _ReusableTCPServer(("127.0.0.1", MOCK_PORT), MockHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


# ── Shim driver ───────────────────────────────────────────────────────────────
class Shim:
    """A launched shim process with a reader thread that timestamps every stdout
    frame, auto-answers mcp/* requests, and queues the rest as (arrival, frame)."""

    def __init__(self, extra_env=None):
        env = os.environ.copy()
        env["NWIRO_LOCAL_LLM_BASE_URL"] = BASE_URL
        env["NWIRO_LOCAL_LLM_MODEL"] = "perf-test-model"
        env["NWIRO_LOCAL_LLM_STREAM_COALESCE_MS"] = str(COALESCE_MS)
        if extra_env:
            env.update(extra_env)
        self.spawned_at = time.monotonic()
        self.proc = subprocess.Popen(
            [BIN], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        self.q = queue.Queue()       # (arrival_monotonic, frame) for non-mcp frames
        self.stdin_lock = threading.Lock()
        self._stderr = []
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        threading.Thread(target=self._reader, daemon=True).start()

    def _drain_stderr(self):
        try:
            while True:
                c = self.proc.stderr.read1(4096)
                if not c:
                    return
                self._stderr.append(c)
        except Exception:
            return

    def _reader(self):
        buf = b""
        try:
            while True:
                c = self.proc.stdout.read1(4096)
                if not c:
                    self.q.put((time.monotonic(), None))
                    return
                # One timestamp per OS read1() chunk — frames decoded from the SAME
                # chunk share it. Fine for the 25ms-spaced coalesced frames (each
                # lands in its own read) and for the wall-clock frame-rate; it bounds
                # the SUB-ms first-token / tool-dispatch numbers to read granularity
                # (a sub-ms value means "within one read" = effectively instant).
                arrival = time.monotonic()
                buf += c
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        f = json.loads(line.decode("utf-8"))
                    except Exception:
                        continue
                    m = f.get("method", "")
                    if m == "mcp/connect":
                        self._reply(f, {"connectionId": "mock"})
                    elif m == "mcp/message":
                        self._reply(f, {"message": {"content": [{"type": "text", "text": "tool result"}], "isError": False}})
                    else:
                        self.q.put((arrival, f))
        except Exception:
            self.q.put((time.monotonic(), None))

    def send(self, msg):
        line = (json.dumps(msg) + "\n").encode("utf-8")
        with self.stdin_lock:
            self.proc.stdin.write(line)
            self.proc.stdin.flush()

    def _reply(self, req, result):
        self.send({"jsonrpc": "2.0", "id": req.get("id"), "result": result})

    def wait_response(self, want_id, timeout_s=30.0):
        """Drain frames until the response to want_id; return (frames, response_arrival)."""
        frames, resp_at = [], None
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                arrival, f = self.q.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                break
            if f is None:
                break
            frames.append((arrival, f))
            if f.get("id") == want_id and ("result" in f or "error" in f):
                resp_at = arrival
                break
        return frames, resp_at

    def initialize(self):
        self.send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        _, at = self.wait_response(1, timeout_s=10.0)
        return at

    def new_session(self):
        self.send({"jsonrpc": "2.0", "id": 2, "method": "session/new", "params": {}})
        frames, _ = self.wait_response(2, timeout_s=5.0)
        for _, f in frames:
            if f.get("id") == 2:
                return f["result"]["sessionId"]
        raise RuntimeError("no sessionId")

    def close(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass


def rss_mb(pid):
    try:
        import psutil
        return psutil.Process(pid).memory_info().rss / (1024 * 1024)
    except Exception:
        pass
    if os.name == "nt":
        try:
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", f"(Get-Process -Id {pid}).WorkingSet64"],
                stderr=subprocess.DEVNULL, text=True)
            return int(out.strip()) / (1024 * 1024)
        except Exception:
            return None
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        return None


def content_frames(frames):
    """(arrival, text) for each agent_message_chunk content frame."""
    out = []
    for arrival, f in frames:
        if f.get("method") == "session/update":
            up = f.get("params", {}).get("update", {})
            if up.get("sessionUpdate") == "agent_message_chunk":
                txt = up.get("content", {}).get("text")
                if txt is not None:
                    out.append((arrival, txt))
    return out


# ── Metrics ───────────────────────────────────────────────────────────────────
def f_perf_07_startup(runs=20):
    """F-PERF-07: Popen → first initialize response, p95 ms. Budget <= 250 ms."""
    samples = []
    for _ in range(runs):
        s = Shim()
        try:
            at = s.initialize()
            if at is not None:
                samples.append((at - s.spawned_at) * 1000.0)
        finally:
            s.close()
    p95 = _p95(samples)
    return {"id": "F-PERF-07", "name": "startup (Popen->initialize)", "value": p95,
            "unit": "ms p95", "budget": "<=250", "ok": p95 is not None and p95 <= 250, "hard": True}


def f_perf_02_first_token(runs=20):
    """F-PERF-02: prompt send → first content frame, p95 ms. Budget <= 50 ms
    (shim overhead; the mock's first chunk is immediate)."""
    STATE.update(mode="content", tokens=5, spacing_s=0.0, first_chunk_delay_s=0.0, warmup_delay_s=0.0)
    samples = []
    for _ in range(runs):
        s = Shim()
        try:
            s.initialize()
            sid = s.new_session()
            sent = time.monotonic()
            s.send({"jsonrpc": "2.0", "id": 4, "method": "session/prompt",
                    "params": {"sessionId": sid, "prompt": [{"type": "text", "text": "hi"}]}})
            frames, _ = s.wait_response(4, timeout_s=10.0)
            cf = content_frames(frames)
            if cf:
                samples.append((cf[0][0] - sent) * 1000.0)
        finally:
            s.close()
    p95 = _p95(samples)
    return {"id": "F-PERF-02", "name": "first-token latency (warm)", "value": p95,
            "unit": "ms p95", "budget": "<=50", "ok": p95 is not None and p95 <= 50, "hard": True}


def f_perf_01_frame_rate():
    """F-PERF-01: frame-emission rate under a fast token flood at coalesce_ms=25.
    Budget <=45/s (target 36-40); FAIL if >100/s (coalescer dead). Also asserts
    content integrity (every input token survives, byte-identical)."""
    n_tokens = 800
    STATE.update(mode="content", tokens=n_tokens, spacing_s=0.003, first_chunk_delay_s=0.0, warmup_delay_s=0.0)
    s = Shim()
    try:
        s.initialize()
        sid = s.new_session()
        t0 = time.monotonic()
        s.send({"jsonrpc": "2.0", "id": 4, "method": "session/prompt",
                "params": {"sessionId": sid, "prompt": [{"type": "text", "text": "flood"}]}})
        frames, resp_at = s.wait_response(4, timeout_s=60.0)
        cf = content_frames(frames)
        wall = (resp_at - t0) if resp_at else (cf[-1][0] - t0 if cf else 0)
        rate = (len(cf) / wall) if wall > 0 else 0
        got = "".join(t for _, t in cf)
        expected = "".join(f"t{i} " for i in range(n_tokens))
        integrity = got == expected
    finally:
        s.close()
    ok = integrity and 0 < rate <= 45
    note = "" if integrity else " (CONTENT MISMATCH!)"
    return {"id": "F-PERF-01", "name": "frame-rate @ coalesce_ms=25" + note, "value": round(rate, 1),
            "unit": f"frames/s ({len(cf)} frames, integrity={'ok' if integrity else 'FAIL'})",
            "budget": "<=45 (target 36-40)", "ok": ok, "hard": True}


def f_perf_04_throughput():
    """F-PERF-04: tokens/sec passthrough on an as-fast-as-possible stream +
    content integrity. Absolute rate is hardware/mock-dependent → INFORMATIONAL
    (the integrity check is the hard part)."""
    n_tokens = 800
    STATE.update(mode="content", tokens=n_tokens, spacing_s=0.0, first_chunk_delay_s=0.0, warmup_delay_s=0.0)
    s = Shim()
    try:
        s.initialize()
        sid = s.new_session()
        t0 = time.monotonic()
        s.send({"jsonrpc": "2.0", "id": 4, "method": "session/prompt",
                "params": {"sessionId": sid, "prompt": [{"type": "text", "text": "fast"}]}})
        frames, resp_at = s.wait_response(4, timeout_s=30.0)
        cf = content_frames(frames)
        wall = (resp_at - t0) if resp_at else 0
        got = "".join(t for _, t in cf)
        expected = "".join(f"t{i} " for i in range(n_tokens))
        integrity = got == expected
        tok_per_s = (n_tokens / wall) if wall > 0 else 0
    finally:
        s.close()
    return {"id": "F-PERF-04", "name": "throughput / integrity", "value": round(tok_per_s, 0),
            "unit": f"tok/s informational (integrity={'ok' if integrity else 'FAIL'})",
            "budget": "integrity must hold", "ok": integrity, "hard": True}


def f_perf_05_tool_dispatch(samples=3):
    """F-PERF-05: shim-side tool-DISPATCH overhead — tool_call(pending) to
    tool_call_update(completed), i.e. the cost of dispatching the (instant) mcp
    round-trip and resuming. NOT the full end-to-end tool latency (the round-2
    backend turn is excluded). Median over `samples` FRESH sessions — a shared
    session accumulates a role:tool history that flips the mock into its followup
    branch, so each sample uses a new process. Budget <= 100 ms. Read-granularity
    bounded (see Shim._reader): a sub-ms value means within one stdout read."""
    STATE.update(mode="tool", tokens=3, spacing_s=0.0, first_chunk_delay_s=0.0, warmup_delay_s=0.0)
    tools = [{"type": "function", "function": {"name": "find_blueprints",
              "parameters": {"type": "object", "properties": {"searchTerm": {"type": "string"}}}}}]
    overheads = []
    for _ in range(samples):
        s = Shim(extra_env={"NWIRO_LOCAL_LLM_FORCE_TOOL_TIER": "native"})
        try:
            s.initialize()
            sid = s.new_session()
            s.send({"jsonrpc": "2.0", "id": 4, "method": "session/prompt",
                    "params": {"sessionId": sid, "prompt": [{"type": "text", "text": "find"}], "tools": tools}})
            frames, _ = s.wait_response(4, timeout_s=20.0)
            pend = next((a for a, f in frames if f.get("params", {}).get("update", {}).get("sessionUpdate") == "tool_call"), None)
            done = next((a for a, f in frames if f.get("params", {}).get("update", {}).get("sessionUpdate") == "tool_call_update"), None)
            if pend is not None and done is not None:
                overheads.append((done - pend) * 1000.0)
        finally:
            s.close()
    val = statistics.median(overheads) if overheads else None
    return {"id": "F-PERF-05", "name": "tool-dispatch overhead (mcp)",
            "value": round(val, 2) if val is not None else None,
            "unit": f"ms median of {len(overheads)}", "budget": "<=100",
            "ok": val is not None and val <= 100, "hard": True}


def f_perf_03_warmup_cap():
    """F-PERF-03: with WARMUP_TIMEOUT_SECS=2 and a 10s-hanging backend, warmup must
    fail fast with errorKind=timeout in ~2-3s (not 10s)."""
    s = Shim(extra_env={"NWIRO_LOCAL_LLM_WARMUP_TIMEOUT_SECS": "2"})
    try:
        s.initialize()
        # Hang ONLY the warmup request, set AFTER initialize so init isn't delayed
        # if the shim ever probes the backend during startup.
        STATE.update(mode="content", warmup_delay_s=10.0)
        sent = time.monotonic()
        s.send({"jsonrpc": "2.0", "id": 9, "method": "session/warmup",
                "params": {"model": "perf-test-model", "baseUrl": BASE_URL}})
        frames, at = s.wait_response(9, timeout_s=12.0)
        elapsed = (at - sent) if at else None
    finally:
        try:
            s.close()
        finally:
            STATE.update(warmup_delay_s=0.0)
    # The 2s cap must fire ~2s, NOT hang to the backend's 10s. Gate matches the
    # displayed budget exactly (no looser hidden bound that silently passes a miss).
    ok = elapsed is not None and 1.9 <= elapsed <= 3.5
    return {"id": "F-PERF-03", "name": "warmup timeout cap (2s)", "value": round(elapsed, 2) if elapsed is not None else None,
            "unit": "s", "budget": "1.9-3.5", "ok": ok, "hard": True}


def f_perf_06_rss_soak(prompts=500, sample_every=50):
    """F-PERF-06: RSS slope over a single-session soak. Budget < 0.5 MB / 100
    prompts (least-squares slope over the samples from prompt 100 on). Skipped
    under --quick."""
    STATE.update(mode="content", tokens=3, spacing_s=0.0, first_chunk_delay_s=0.0, warmup_delay_s=0.0)
    s = Shim()
    samples = []  # (prompt_index, rss_mb)
    try:
        s.initialize()
        sid = s.new_session()
        for i in range(1, prompts + 1):
            s.send({"jsonrpc": "2.0", "id": 1000 + i, "method": "session/prompt",
                    "params": {"sessionId": sid, "prompt": [{"type": "text", "text": "x"}]}})
            s.wait_response(1000 + i, timeout_s=10.0)
            if i % sample_every == 0:
                r = rss_mb(s.proc.pid)
                if r is not None:
                    samples.append((i, r))
    finally:
        s.close()
    slope = None
    pts = [(i, r) for i, r in samples if i >= 100]
    if len(pts) >= 2:
        # Least-squares regression over ALL samples (>= prompt 100), not a 2-point
        # delta — so one transient RSS reading (OS working-set trim, a page-in)
        # can't corrupt the leak signal. slope is MB per 100 prompts.
        try:
            slope = statistics.linear_regression(
                [p[0] for p in pts], [p[1] for p in pts]).slope * 100.0
        except Exception:
            slope = None
    ok = slope is not None and slope < 0.5
    return {"id": "F-PERF-06", "name": "RSS soak slope", "value": round(slope, 3) if slope is not None else None,
            "unit": f"MB/100 prompts ({len(samples)} samples)", "budget": "<0.5", "ok": ok, "hard": True}


def _p95(xs):
    if not xs:
        return None
    xs = sorted(xs)
    k = max(0, int(round(0.95 * (len(xs) - 1))))
    return round(xs[k], 1)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # robust to non-UTF-8 consoles
    except Exception:
        pass
    quick = "--quick" in sys.argv
    out_path = None
    if "--json" in sys.argv:
        out_path = sys.argv[sys.argv.index("--json") + 1]

    if not os.path.exists(BIN):
        print(f"ERROR: release binary not found at {BIN}\n  build it: cargo build --release", file=sys.stderr)
        return 2

    httpd = start_mock()
    print(f"perf-harness: binary={BIN}\n  mock backend on {BASE_URL}, coalesce_ms={COALESCE_MS}\n")
    results = []
    try:
        results.append(f_perf_07_startup())
        results.append(f_perf_02_first_token())
        results.append(f_perf_01_frame_rate())
        results.append(f_perf_04_throughput())
        results.append(f_perf_05_tool_dispatch())
        results.append(f_perf_03_warmup_cap())
        if quick:
            print("(--quick: skipping F-PERF-06 RSS soak)\n")
        else:
            results.append(f_perf_06_rss_soak())
    finally:
        httpd.shutdown()
        httpd.server_close()  # release the port immediately for rapid re-runs

    print(f"{'metric':<12}{'name':<34}{'value':>10}  {'unit':<42}{'budget':<16}{'verdict'}")
    print("-" * 130)
    failed = 0
    for r in results:
        verdict = "PASS" if r["ok"] else ("FAIL" if r["hard"] else "info")
        if r["hard"] and not r["ok"]:
            failed += 1
        val = "n/a" if r["value"] is None else str(r["value"])
        print(f"{r['id']:<12}{r['name']:<34}{val:>10}  {r['unit']:<42}{r['budget']:<16}{verdict}")
    print("-" * 130)
    print(f"{'PASS' if failed == 0 else f'{failed} BUDGET VIOLATION(S)'}  ({len(results)} metrics)")

    if out_path:
        with open(out_path, "w") as fh:
            json.dump({"results": results, "binary": BIN, "coalesce_ms": COALESCE_MS}, fh, indent=2)
        print(f"wrote {out_path}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
