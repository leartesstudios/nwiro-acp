"""
Day-1 campaign driver — runs the T1.x test cases against a live RunPod vLLM
endpoint and writes a manifest the aggregator can consume.

Usage:
    python scripts/run-day1.py \
        --base-url   https://<pod-id>-8000.proxy.runpod.net/v1 \
        --shim-bin   target/release/local-llm-acp.exe \
        --output     reports/runpod-spike-2026-05/raw/day1-input.json \
        --shim-rev   $(git rev-parse HEAD)

Architecture:
    - The driver spawns `local-llm-acp.exe` as a child process for each
      *test case* (clean session per test — cheaper than coding a multi-test
      session orchestrator).
    - For each test, drives ACP JSON-RPC over stdin/stdout: `initialize`,
      `session/warmup`, `session/new`, `session/set_config_option`,
      `session/prompt`. Captures per-frame timing.
    - Computes ttft / chunk_count / chunk_gap_{p50,p99} from `session/update`
      arrival timestamps.
    - Writes one structured result per test to the in-memory manifest,
      flushes to `--output` at the end.
    - The aggregator (scripts/aggregate-day-results.py) consumes the manifest
      and produces the review-payload JSON.

Tests included (Day 1):
    - T1.1: tier classification for Qwen2.5-14B (warmup-only).
    - T1.3: streaming baseline against Qwen2.5-14B (single-turn prompt).
    - T1.4: UTF-8 CJK stress against Qwen2.5-14B.
    - T1.5: refusal path (deliberately point at an unreachable URL so
            warmup fails → toolTier=none → tools prompt → REFUSAL).
    - T1.2 (model switch) is implemented by re-using smoke-test-model-switch
      against the real pod — not in this driver; run as a separate step.
    - T1.6 (16K long-context) is a placeholder — fill in the prompt source.

The driver is intentionally Windows-pipe-aware (queue.Queue + reader thread)
because `proc.stdout.read1()` blocks forever on Windows on an empty pipe.
This is the same pattern the smoke tests use.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REFUSAL = (
    "This model does not support tool calls. Please switch to a tool-capable "
    "model such as Qwen2.5 14B, Mistral Nemo, or Llama 3.1 70B+."
)


# ── ACP frame plumbing (same pattern as smoke-test-model-switch.py) ─────────


def _send(proc: subprocess.Popen, msg: dict) -> None:
    line = json.dumps(msg) + "\n"
    proc.stdin.write(line.encode("utf-8"))  # type: ignore[union-attr]
    proc.stdin.flush()  # type: ignore[union-attr]


def _attach_reader(proc: subprocess.Popen) -> None:
    """One-shot install of a daemon reader thread + queue on the proc."""
    if hasattr(proc, "_chunk_q"):
        return
    proc._chunk_q = queue.Queue()  # type: ignore[attr-defined]
    proc._buf = b""  # type: ignore[attr-defined]

    def _reader() -> None:
        try:
            while True:
                chunk = proc.stdout.read1(4096)  # type: ignore[union-attr]
                if not chunk:
                    proc._chunk_q.put(b"")  # type: ignore[attr-defined]
                    return
                proc._chunk_q.put(chunk)  # type: ignore[attr-defined]
        except Exception:
            proc._chunk_q.put(b"")  # type: ignore[attr-defined]

    threading.Thread(target=_reader, daemon=True).start()


def _read_frames_timed(
    proc: subprocess.Popen,
    n: int,
    timeout_s: float = 30.0,
) -> tuple[list[tuple[float, dict]], bytes]:
    """
    Read up to `n` frames; return list of (monotonic_recv_time, frame).
    Times are captured at the moment the line was fully received, which is
    sufficient resolution for SSE chunk timing analysis.
    """
    _attach_reader(proc)
    frames: list[tuple[float, dict]] = []
    deadline = time.time() + timeout_s
    buf = proc._buf  # type: ignore[attr-defined]
    while len(frames) < n:
        while b"\n" in buf and len(frames) < n:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line:
                continue
            now = time.monotonic()
            try:
                frames.append((now, json.loads(line.decode("utf-8"))))
            except json.JSONDecodeError:
                # malformed line — log+skip; surface in error_summary later
                continue
        if len(frames) >= n:
            break
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        try:
            chunk = proc._chunk_q.get(timeout=remaining)  # type: ignore[attr-defined]
        except queue.Empty:
            break
        if not chunk:
            break
        buf += chunk
    proc._buf = buf  # type: ignore[attr-defined]
    return frames, buf


# ── Per-test data ──────────────────────────────────────────────────────────


@dataclass
class TestCase:
    name: str
    model: str
    kind: str  # "warmup" | "prompt"
    base_url_override: str | None = None  # for T1.5 (unreachable url)
    prompt_text: str | None = None
    tools: list | None = None
    timeout_s: float = 30.0
    expected_tool_tier: str | None = None  # for warmup tests
    expected_refusal: bool = False  # for refusal tests


@dataclass
class TestResult:
    test_name: str
    model: str
    kind: str
    status: str
    # warmup fields
    warmup_latency_ms: int | None = None
    tool_tier: str | None = None
    # prompt fields
    ttft_ms: int | None = None
    total_latency_ms: int | None = None
    chunk_count: int | None = None
    chunk_gap_p50_ms: int | None = None
    chunk_gap_p99_ms: int | None = None
    sse_sample: str | None = None
    acp_frames_summary: list[dict] = field(default_factory=list)
    assertions_failed: list[str] = field(default_factory=list)
    error_code: int | None = None
    error_message: str | None = None
    error_summary: str | None = None
    stderr_lines: int = 0


# ── Runner ─────────────────────────────────────────────────────────────────


def _spawn_shim(shim_bin: Path, base_url: str, model: str, env_extra: dict[str, str]) -> subprocess.Popen:
    env = os.environ.copy()
    env["NWIRO_LOCAL_LLM_BASE_URL"] = base_url
    env["NWIRO_LOCAL_LLM_MODEL"] = model
    # Force tracing OFF on stderr (the v0.1.5 invariant) by not setting
    # NWIRO_LOCAL_LLM_TRACING_FILE. RUST_LOG can be set externally if needed.
    env.pop("NWIRO_LOCAL_LLM_TRACING_FILE", None)
    env.update(env_extra)
    return subprocess.Popen(
        [str(shim_bin)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )


def _run_warmup(proc: subprocess.Popen, tc: TestCase) -> TestResult:
    # init
    _send(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    _read_frames_timed(proc, 1, timeout_s=5.0)

    # warmup
    t0 = time.monotonic()
    _send(proc, {
        "jsonrpc": "2.0", "id": 2, "method": "session/warmup",
        "params": {"model": tc.model, "keepAlive": "15m"},
    })
    frames, _ = _read_frames_timed(proc, 1, timeout_s=tc.timeout_s)
    t1 = time.monotonic()

    r = TestResult(test_name=tc.name, model=tc.model, kind="warmup", status="fail")
    if not frames:
        r.error_summary = "no warmup response within timeout"
        return r
    _recv_at, frame = frames[0]
    result = frame.get("result") or {}
    r.warmup_latency_ms = int(result.get("elapsedMs") or (t1 - t0) * 1000)
    r.tool_tier = result.get("toolTier")
    r.status = result.get("status", "fail")  # "loaded" | "failed"
    r.error_summary = result.get("message")

    if tc.expected_tool_tier and r.tool_tier != tc.expected_tool_tier:
        r.assertions_failed.append(
            f"expected tool_tier={tc.expected_tool_tier!r}, got {r.tool_tier!r}"
        )
        r.status = "fail" if r.status == "loaded" else r.status
    elif r.status == "loaded":
        r.status = "loaded"
    return r


def _run_prompt(proc: subprocess.Popen, tc: TestCase) -> TestResult:
    # init
    _send(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    _read_frames_timed(proc, 1, timeout_s=5.0)

    # warmup first (so the model is loaded + tool_tier resolved)
    _send(proc, {
        "jsonrpc": "2.0", "id": 2, "method": "session/warmup",
        "params": {"model": tc.model, "keepAlive": "15m"},
    })
    _read_frames_timed(proc, 1, timeout_s=tc.timeout_s)

    # session
    _send(proc, {"jsonrpc": "2.0", "id": 3, "method": "session/new", "params": {}})
    frames, _ = _read_frames_timed(proc, 1, timeout_s=5.0)
    if not frames:
        return TestResult(
            test_name=tc.name, model=tc.model, kind="prompt", status="fail",
            error_summary="session/new returned no frame",
        )
    session_id = frames[0][1]["result"]["sessionId"]

    # set model
    _send(proc, {
        "jsonrpc": "2.0", "id": 4, "method": "session/set_config_option",
        "params": {"sessionId": session_id, "configId": "model", "value": tc.model},
    })
    _read_frames_timed(proc, 1, timeout_s=5.0)

    # prompt — measure timings
    prompt_payload: dict[str, Any] = {
        "sessionId": session_id,
        "prompt": [{"type": "text", "text": tc.prompt_text or ""}],
    }
    if tc.tools:
        prompt_payload["tools"] = tc.tools

    t_send = time.monotonic()
    _send(proc, {"jsonrpc": "2.0", "id": 5, "method": "session/prompt", "params": prompt_payload})

    # Pull every frame for up to timeout_s — many session/update + one final result.
    chunks_recv_at: list[float] = []
    sse_sample: str | None = None
    refusal_seen = False
    result_frame: dict | None = None
    deadline = time.time() + tc.timeout_s
    seen_frames: list[tuple[float, dict]] = []
    while time.time() < deadline and result_frame is None:
        frames, _ = _read_frames_timed(proc, 1, timeout_s=max(0.1, deadline - time.time()))
        if not frames:
            break
        for recv_at, frame in frames:
            seen_frames.append((recv_at, frame))
            if frame.get("method") == "session/update":
                chunks_recv_at.append(recv_at)
                # Capture the first chunk's content as our sse_sample for the schema.
                if sse_sample is None:
                    content = (
                        frame.get("params", {})
                        .get("update", {})
                        .get("content", {})
                        .get("text", "")
                    )
                    sse_sample = (content or "")[:200] or None
                # Refusal detection (T1.5)
                text = (
                    frame.get("params", {})
                    .get("update", {})
                    .get("content", {})
                    .get("text", "")
                )
                if text == REFUSAL:
                    refusal_seen = True
            elif frame.get("id") == 5:
                result_frame = frame
                break

    t_done = time.monotonic()

    r = TestResult(test_name=tc.name, model=tc.model, kind="prompt", status="fail")
    r.total_latency_ms = int((t_done - t_send) * 1000)
    r.chunk_count = len(chunks_recv_at)
    if chunks_recv_at:
        r.ttft_ms = int((chunks_recv_at[0] - t_send) * 1000)
        gaps = [int((b - a) * 1000) for a, b in zip(chunks_recv_at, chunks_recv_at[1:])]
        if gaps:
            r.chunk_gap_p50_ms = int(statistics.median(gaps))
            # statistics.quantiles wants n=100 for percentile; for tiny samples this returns crude buckets.
            try:
                r.chunk_gap_p99_ms = int(statistics.quantiles(gaps, n=100)[98])
            except statistics.StatisticsError:
                r.chunk_gap_p99_ms = max(gaps)
    r.sse_sample = sse_sample
    r.acp_frames_summary = _frames_summary(seen_frames)

    if result_frame is None:
        r.assertions_failed.append("no id=5 result frame within timeout")
        r.error_summary = "timeout waiting for prompt result"
    elif "error" in result_frame:
        r.error_code = result_frame["error"].get("code")
        r.error_message = result_frame["error"].get("message")
        r.assertions_failed.append(f"result was an error: {r.error_code} {r.error_message}")
    else:
        r.status = "pass"

    if tc.expected_refusal and not refusal_seen:
        r.assertions_failed.append("expected REFUSAL text but it was not emitted")
        r.status = "fail"
    if (not tc.expected_refusal) and refusal_seen:
        r.assertions_failed.append("REFUSAL emitted unexpectedly (staleness regression?)")
        r.status = "fail"

    if r.assertions_failed:
        r.status = "fail"
    return r


def _frames_summary(seen: list[tuple[float, dict]]) -> list[dict]:
    """Compact every captured frame into a one-line entry for the schema."""
    out: list[dict] = []
    for _at, frame in seen:
        method = frame.get("method")
        has_id = "id" in frame
        if method == "session/update":
            # Collapse runs of session/update into one entry at the end.
            if out and out[-1].get("method") == "session/update":
                out[-1]["note"] = f"agent_message_chunk x{int((out[-1].get('note') or 'x1').split('x')[-1]) + 1}"
                continue
            out.append({"dir": "out", "method": "session/update", "id": None,
                        "status": "ok", "note": "agent_message_chunk x1"})
        elif method:
            out.append({"dir": "out" if not has_id else "in", "method": method,
                        "id": frame.get("id"), "status": "ok", "note": None})
        else:
            kind = "error" if "error" in frame else "ok"
            note = frame.get("error", {}).get("message", "result:{}") if kind == "error" else "result:{}"
            out.append({"dir": "out", "method": None, "id": frame.get("id"),
                        "status": kind, "note": note[:80]})
    return out


# ── Test list ──────────────────────────────────────────────────────────────


def _build_day1_tests(base_url: str) -> list[TestCase]:
    return [
        TestCase(
            name="T1.1 — Qwen2.5-14B tier classification",
            model="Qwen/Qwen2.5-14B-Instruct",
            kind="warmup",
            expected_tool_tier="native",
            timeout_s=120.0,  # cold-load is slow
        ),
        # Add Mistral-Nemo and Llama-3.1-8B-Instruct here when present on the
        # pod. Each requires a fresh shim process so warmup goes through.
        TestCase(
            name="T1.3 — streaming baseline 200 words",
            model="Qwen/Qwen2.5-14B-Instruct",
            kind="prompt",
            prompt_text="Explain function calling in 200 words.",
            timeout_s=60.0,
        ),
        TestCase(
            name="T1.4 — UTF-8 CJK stress",
            model="Qwen/Qwen2.5-14B-Instruct",
            kind="prompt",
            prompt_text="请用中文写一段500字的关于函数调用的解释",
            timeout_s=120.0,
        ),
        # T1.5 — refusal path: unreachable URL forces warmup-fail → tier=none.
        # This test runs against a deliberately bad URL, not the real pod.
        TestCase(
            name="T1.5 — refusal path",
            model="any-model",
            kind="prompt",
            base_url_override="http://127.0.0.1:1/v1",
            prompt_text="Use a tool to find blueprints.",
            tools=[{"type": "function", "function": {"name": "find_blueprints",
                                                       "description": "...",
                                                       "parameters": {"type": "object", "properties": {}}}}],
            expected_refusal=True,
            timeout_s=15.0,
        ),
        # T1.6 (long-context 16K) — fill in real prompt source from a file when ready.
        # Skeleton:
        # TestCase(
        #     name="T1.6 — long-context 16K",
        #     model="Qwen/Qwen2.5-14B-Instruct",
        #     kind="prompt",
        #     prompt_text=Path("tests/data/16k-prompt.txt").read_text(encoding="utf-8"),
        #     timeout_s=180.0,
        # ),
    ]


# ── Main ───────────────────────────────────────────────────────────────────


def _run_one(shim_bin: Path, base_url: str, tc: TestCase) -> TestResult:
    bu = tc.base_url_override or base_url
    proc = _spawn_shim(shim_bin, bu, tc.model, env_extra={})
    try:
        if tc.kind == "warmup":
            r = _run_warmup(proc, tc)
        else:
            r = _run_prompt(proc, tc)
    except Exception as e:
        r = TestResult(test_name=tc.name, model=tc.model, kind=tc.kind, status="fail",
                       error_summary=f"driver exception: {type(e).__name__}: {e}")
    finally:
        try:
            proc.stdin.close()  # type: ignore[union-attr]
        except Exception:
            pass
        try:
            stderr = proc.stderr.read() if proc.stderr else b""  # type: ignore[union-attr]
            r.stderr_lines = len(stderr.splitlines())
            if stderr:
                r.assertions_failed.append(
                    f"shim emitted {len(stderr)} bytes on stderr (v0.1.5 silence invariant)"
                )
                r.status = "fail"
        except Exception:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            pass
    return r


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Day-1 RunPod campaign driver.")
    p.add_argument("--base-url", required=True,
                   help="Pod proxy URL, e.g. https://<pod-id>-8000.proxy.runpod.net/v1")
    p.add_argument("--shim-bin", type=Path,
                   default=Path(__file__).resolve().parent.parent / "target" / "release" / "local-llm-acp.exe",
                   help="Path to the shim binary.")
    p.add_argument("--output", type=Path, required=True,
                   help="Where to write the input manifest for the aggregator.")
    p.add_argument("--shim-rev", default="unknown",
                   help="Shim git rev for the manifest header.")
    p.add_argument("--vllm-image", default="ghcr.io/<org>/<vllm-image>:<tag>",
                   help="vLLM image tag used on the pod.")
    p.add_argument("--gpu-class", default="A40")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    if not args.shim_bin.exists():
        print(f"FATAL: shim binary not found: {args.shim_bin}", file=sys.stderr)
        return 2

    tests = _build_day1_tests(args.base_url)
    results: list[TestResult] = []
    print(f"Running {len(tests)} test cases against {args.base_url}")
    for tc in tests:
        print(f"  > {tc.name} (model={tc.model}, kind={tc.kind})")
        r = _run_one(args.shim_bin, args.base_url, tc)
        status_icon = "OK" if r.status in ("pass", "loaded") else "FAIL"
        print(f"    {status_icon} status={r.status} "
              f"{('tool_tier=' + (r.tool_tier or 'n/a')) if r.kind == 'warmup' else ('ttft_ms=' + str(r.ttft_ms))}")
        if r.assertions_failed:
            for af in r.assertions_failed:
                print(f"      ! {af}")
        results.append(r)

    manifest = {
        "campaign_id": "runpod-spike-2026-05",
        "day": 1,
        "gpu_class": args.gpu_class,
        "shim_version": "0.1.13",
        "shim_git_rev": args.shim_rev,
        "vllm_image": args.vllm_image,
        "review_context": (
            "Architecture A. Windows shim local. vLLM on RunPod A40 (community). "
            "API key redacted by aggregator before publish. "
            "Evaluate ACP correctness, tier classification, streaming, UTF-8, refusal path."
        ),
        "known_intentional_failures": [
            {"surface": "mcp/connect", "code": -32601,
             "reason": "Phase 3 MCP wiring deferred — stub left as-is per session decision 2026-05-16"}
        ],
        "tests": [_result_to_dict(r) for r in results],
        "global_errors": [],
        "expected_mcp_stub_32601_count": 0,
        "unexpected_32601_count": sum(1 for r in results if r.error_code == -32601 and r.kind == "prompt"),
        "budget_estimate_usd": 0.0,  # set by hand or by a wrapping orchestrator script
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote manifest: {args.output} ({len(json.dumps(manifest))} chars)")
    print("Next: python scripts/aggregate-day-results.py "
          f"--input {args.output} --output reports/runpod-spike-2026-05/day1-summary.json")

    n_fail = sum(1 for r in results if r.status not in ("pass", "loaded"))
    return 1 if n_fail else 0


def _result_to_dict(r: TestResult) -> dict:
    """Convert TestResult to the input-manifest shape the aggregator expects."""
    out: dict[str, Any] = {
        "test_name": r.test_name,
        "model": r.model,
        "kind": r.kind,
        "status": r.status,
        "stderr_lines": r.stderr_lines,
    }
    if r.kind == "warmup":
        out["warmup_latency_ms"] = r.warmup_latency_ms
        out["tool_tier"] = r.tool_tier
        out["error_summary"] = r.error_summary
    else:
        out["ttft_ms"] = r.ttft_ms
        out["total_latency_ms"] = r.total_latency_ms
        out["chunk_count"] = r.chunk_count
        out["chunk_gap_p50_ms"] = r.chunk_gap_p50_ms
        out["chunk_gap_p99_ms"] = r.chunk_gap_p99_ms
        out["sse_sample"] = r.sse_sample
        out["acp_frames_summary"] = r.acp_frames_summary
        out["assertions_failed"] = r.assertions_failed
        out["error_code"] = r.error_code
        out["error_message"] = r.error_message
    return out


if __name__ == "__main__":
    sys.exit(main())
