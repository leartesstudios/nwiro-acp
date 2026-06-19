"""
Aggregate one day's test results into the review JSON schema.

The test driver (per-day run script) writes a *flat* manifest of per-test
results to `raw-dir/dayN-input.json`. This script:

  1. groups tests by model,
  2. lifts warmup test results into `models[m].warmup`,
  3. lists prompt tests under `models[m].prompts`,
  4. applies redaction (HF_TOKEN, NWIRO_LOCAL_LLM_API_KEY_*, Bearer …),
  5. hard-caps the output at 20 KB (truncates `sse_sample` first, then drops
     `acp_frames_summary[*].note` content, then refuses to write),
  6. writes the schema-conforming payload to `--output`.

Why structured input instead of parsing RUST_LOG=debug files: the shim's
debug log format is internal and may change. A structured per-test
contract is stable and forces the test driver to extract metrics correctly
at capture time rather than re-parsing them under deadline pressure.

Input manifest shape (`raw/dayN-input.json`):

    {
      "day": 1,
      "gpu_class": "A40",
      "shim_version": "0.1.13",
      "shim_git_rev": "<sha>",
      "vllm_image": "ghcr.io/<org>/<vllm-image>:<tag>",
      "review_context": "Architecture A. Windows shim local. …",
      "tests": [
        {
          "test_name": "T1.1 — tier classification",
          "model": "Qwen/Qwen2.5-14B-Instruct",
          "kind": "warmup",                       // "warmup" | "prompt"
          "status": "pass",                       // "pass" | "fail"
          "warmup_latency_ms": 4321,
          "tool_tier": "native",                  // "native" | "none" | "emulated"
          "error_summary": null,
          "stderr_lines": 0                       // bytes_len on stderr per test
        },
        {
          "test_name": "T1.4 — UTF-8 CJK stress",
          "model": "Qwen/Qwen2.5-14B-Instruct",
          "kind": "prompt",
          "status": "pass",
          "ttft_ms": 740,
          "total_latency_ms": 8210,
          "chunk_count": 142,
          "chunk_gap_p50_ms": 38,
          "chunk_gap_p99_ms": 220,
          "sse_sample": "data: {…}",
          "acp_frames_summary": [ {"dir":"in","method":"session/prompt", ...} ],
          "assertions_failed": [],
          "error_code": null,
          "error_message": null
        }
        // …
      ],
      "global_errors": [],
      "known_intentional_failures": [ { "surface":"mcp/connect", "code":-32601, "reason":"…" } ],
      "expected_mcp_stub_32601_count": 0,
      "unexpected_32601_count": 0,
      "budget_estimate_usd": 5.7
    }

Output schema is the per-day review payload contract: a model-grouped
summary of warmup + prompt test results with redaction and a size cap.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1"
HARD_CAP_BYTES = 20 * 1024  # 20 KB hard cap on output

# ── Redaction ───────────────────────────────────────────────────────────────
#
# Anything that even *looks* like a credential gets replaced with the literal
# string "[REDACTED]" before the payload is written to disk. The patterns are
# intentionally broad — the cost of a false positive (a redacted innocuous
# value) is far smaller than the cost of leaking a real key into a review
# payload that gets emailed around or pasted into a Discord/Slack channel.
REDACT_PATTERNS = [
    re.compile(r"NWIRO_LOCAL_LLM_API_KEY_\w+\s*=\s*[^\s]+"),
    re.compile(r"HF_TOKEN\s*=\s*[^\s]+"),
    re.compile(r"\bhf_[A-Za-z0-9_]{20,}\b"),                   # HF token bare
    re.compile(r"Bearer\s+[A-Za-z0-9._\-+/]{20,}={0,2}"),
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),                     # OpenAI-style
]


def _redact_str(s: str) -> str:
    """Apply every redaction pattern to one string."""
    for pat in REDACT_PATTERNS:
        s = pat.sub("[REDACTED]", s)
    return s


def _redact(obj: Any) -> Any:
    """Walk a JSON-ish structure, redacting strings in-place style."""
    if isinstance(obj, str):
        return _redact_str(obj)
    if isinstance(obj, list):
        return [_redact(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _redact(v) for k, v in obj.items()}
    return obj


# ── Shape conversion ────────────────────────────────────────────────────────


def _model_block(tests: list[dict]) -> dict:
    """Build the per-model schema block from a list of test entries."""
    warmup: dict | None = None
    prompts: list[dict] = []
    stderr_lines = 0

    for t in tests:
        stderr_lines += int(t.get("stderr_lines", 0) or 0)
        if t.get("kind") == "warmup":
            warmup = {
                "status": t.get("status", "unknown"),
                "tool_tier": t.get("tool_tier"),
                "warmup_latency_ms": t.get("warmup_latency_ms"),
                "error_summary": t.get("error_summary"),
            }
        elif t.get("kind") == "prompt":
            prompts.append({
                "test_name": t.get("test_name"),
                "status": t.get("status"),
                "ttft_ms": t.get("ttft_ms"),
                "total_latency_ms": t.get("total_latency_ms"),
                "chunk_count": t.get("chunk_count"),
                "chunk_gap_p50_ms": t.get("chunk_gap_p50_ms"),
                "chunk_gap_p99_ms": t.get("chunk_gap_p99_ms"),
                "sse_sample": (t.get("sse_sample") or "")[:200] or None,
                "acp_frames_summary": t.get("acp_frames_summary", []),
                "assertions_failed": t.get("assertions_failed", []),
                "error_code": t.get("error_code"),
                "error_message": t.get("error_message"),
            })

    pass_count = sum(1 for p in prompts if p["status"] == "pass")
    fail_count = sum(1 for p in prompts if p["status"] == "fail")
    if warmup is not None:
        # Warmup outcome counts toward pass/fail too so the global tally is honest.
        if warmup["status"] == "loaded":
            pass_count += 1
        else:
            fail_count += 1

    return {
        "warmup": warmup,
        "prompts": prompts,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "stderr_lines": stderr_lines,
    }


# ── Hard cap ────────────────────────────────────────────────────────────────


def _serialized_bytes(payload: dict) -> int:
    """Bytes consumed by JSON-serialising `payload` with the standard indent."""
    return len(json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))


def _shrink(payload: dict, *, log: list[str]) -> dict:
    """
    Apply progressively more aggressive truncation until the payload fits
    under `HARD_CAP_BYTES`. Order matters — drop low-value fields first.
    """
    # Pass 1: cap each sse_sample to 120 chars.
    for m in payload["models"].values():
        for p in m.get("prompts", []):
            if p.get("sse_sample") and len(p["sse_sample"]) > 120:
                p["sse_sample"] = p["sse_sample"][:120] + "…[truncated]"
    if _serialized_bytes(payload) <= HARD_CAP_BYTES:
        log.append("shrink: sse_sample capped to 120 chars")
        return payload

    # Pass 2: drop `note` strings from acp_frames_summary entries.
    for m in payload["models"].values():
        for p in m.get("prompts", []):
            for frame in p.get("acp_frames_summary", []):
                if "note" in frame:
                    frame["note"] = frame["note"][:40] + "…" if len(frame.get("note") or "") > 40 else frame.get("note")
    if _serialized_bytes(payload) <= HARD_CAP_BYTES:
        log.append("shrink: acp frame notes capped to 40 chars")
        return payload

    # Pass 3: drop the longest sse_sample entirely.
    candidates = [
        (m_id, p_idx, len(p.get("sse_sample") or ""))
        for m_id, m in payload["models"].items()
        for p_idx, p in enumerate(m.get("prompts", []))
        if p.get("sse_sample")
    ]
    candidates.sort(key=lambda c: -c[2])
    for m_id, p_idx, _ in candidates:
        payload["models"][m_id]["prompts"][p_idx]["sse_sample"] = None
        if _serialized_bytes(payload) <= HARD_CAP_BYTES:
            log.append(f"shrink: dropped sse_sample for {m_id}/{p_idx}")
            return payload

    # If we still don't fit, give up and let the writer surface a fatal error
    # — the caller can manually prune the input file. We do NOT silently drop
    # tests because the review would then be based on an incomplete picture.
    return payload


# ── Main ────────────────────────────────────────────────────────────────────


def aggregate(input_path: Path, *, output_path: Path) -> int:
    """Returns process exit code (0 ok, non-zero fatal)."""
    try:
        raw = json.loads(input_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"FATAL: input file not found: {input_path}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as e:
        print(f"FATAL: input JSON invalid: {e}", file=sys.stderr)
        return 2

    tests = raw.get("tests", [])
    if not isinstance(tests, list) or not tests:
        print("FATAL: input has no 'tests' array (or it is empty)", file=sys.stderr)
        return 2

    # Group tests by model.
    by_model: dict[str, list[dict]] = {}
    for t in tests:
        model = t.get("model")
        if not model:
            print(f"WARN: skipping test without 'model' field: {t.get('test_name')!r}", file=sys.stderr)
            continue
        by_model.setdefault(model, []).append(t)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": raw.get("campaign_id", "runpod-spike-2026-05"),
        "day": int(raw.get("day", 0)),
        "gpu_class": raw.get("gpu_class", "unknown"),
        "shim_version": raw.get("shim_version", "unknown"),
        "shim_git_rev": raw.get("shim_git_rev", "unknown"),
        "vllm_image": raw.get("vllm_image", "unknown"),
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "known_intentional_failures": raw.get("known_intentional_failures", []),
        "review_context": raw.get(
            "review_context",
            "Architecture A. Local shim. Remote vLLM on RunPod pod. Evaluate ACP correctness + latency only.",
        ),
        "models": {m: _model_block(ts) for m, ts in by_model.items()},
        "global_errors": raw.get("global_errors", []),
        "expected_mcp_stub_32601_count": int(raw.get("expected_mcp_stub_32601_count", 0)),
        "unexpected_32601_count": int(raw.get("unexpected_32601_count", 0)),
        "budget_estimate_usd": float(raw.get("budget_estimate_usd", 0.0)),
    }

    # Redact before measuring size; redaction can shrink the payload.
    payload = _redact(payload)

    shrink_log: list[str] = []
    initial_bytes = _serialized_bytes(payload)
    if initial_bytes > HARD_CAP_BYTES:
        payload = _shrink(payload, log=shrink_log)

    final_bytes = _serialized_bytes(payload)
    if final_bytes > HARD_CAP_BYTES:
        print(
            f"FATAL: payload is {final_bytes} bytes after shrink "
            f"(cap {HARD_CAP_BYTES}); prune the input manifest manually.",
            file=sys.stderr,
        )
        return 3

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"OK wrote {output_path} ({final_bytes} bytes, started at {initial_bytes})")
    for line in shrink_log:
        print(f"  {line}")
    return 0


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Aggregate one day's RunPod test results into the review schema.")
    p.add_argument("--input", type=Path, required=True, help="Per-day input manifest (JSON).")
    p.add_argument("--output", type=Path, required=True, help="Where to write the review payload.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    return aggregate(args.input, output_path=args.output)


if __name__ == "__main__":
    sys.exit(main())
