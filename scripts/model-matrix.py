#!/usr/bin/env python3
"""Layer 2 — the E2E model matrix through the shim (docs/MODEL-TEST-PLAN.md §3-5).

Spawns the real shim, warms up each model, drives the canonical UE5 tasks (T-A
spawn, T-D describer-trap) with the MCP-responder (so tool calls actually execute
end-to-end), evaluates C1-C8, and rolls each model up to a verdict
(GREEN/YELLOW/RED/BLACK). Writes a JSON manifest; `--md` regenerates
docs/MODEL-COMPATIBILITY.md.

Reuses the verified harness: `run-day1.py::_spawn_shim`, `_mcp_responder` (the
queue-based driver), `_bleed_oracle` (C4 = the shim's bleed definition), and
`tool-curve.py::_padded_tools` (T-D padding). Requires a LIVE backend (Ollama /
LM Studio) — run on a box with the models pulled. Python 3.11+ (tomllib).

    python scripts/model-matrix.py --only qwen3:14b --out reports/matrix.json
    python scripts/model-matrix.py --out reports/matrix.json --md docs/MODEL-COMPATIBILITY.md
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import tomllib
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from _bleed_oracle import looks_like_schema_bleed  # noqa: E402
from _mcp_responder import drive_prompt_with_mcp, read_until, send, start_reader  # noqa: E402


def _load(filename: str, mod_name: str):
    """importlib-load a hyphenated sibling script (not importable by name).
    Register in sys.modules BEFORE exec so @dataclass can resolve cls.__module__
    (run-day1.py's TestCase/TestResult dataclasses fail to build otherwise)."""
    spec = importlib.util.spec_from_file_location(mod_name, _HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_rd = _load("run-day1.py", "run_day1")  # reuse _spawn_shim
_tc = _load("tool-curve.py", "tool_curve")  # reuse _padded_tools / _gold_tool

# The inner MCP envelope the harness returns for a *successful* tool call.
GOLD_RESULT = {
    "content": [{"type": "text", "text": "Spawned PointLight at (0,0,0)."}],
    "isError": False,
}


def _spawn_actor_tool() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "spawn_actor",
            "description": "Spawn an actor of the given class in the current level.",
            "parameters": {
                "type": "object",
                "properties": {
                    "class": {"type": "string"},
                    "location": {"type": "array", "items": {"type": "number"}},
                },
                "required": ["class"],
            },
        },
    }


def _tasks() -> list[dict]:
    """Canonical tasks (MODEL-TEST-PLAN §2). gold = the tool name expected."""
    # T-D: action verb + ~30 padded tools incl. the spawn_actor gold tool at idx 0.
    t_d_tools = [_spawn_actor_tool()] + [_tc._synth_tool(i) for i in range(1, 30)]
    return [
        {"id": "T-A", "prompt": "Spawn a point light at the origin.",
         "tools": [_spawn_actor_tool()], "gold": "spawn_actor"},
        {"id": "T-D", "prompt": "Spawn a point light actor now.",
         "tools": t_d_tools, "gold": "spawn_actor"},
    ]


# ── C1-C8 (docs/MODEL-TEST-PLAN §3) ─────────────────────────────────────────


def _updates(frames: list, kind: str) -> list:
    return [
        f.get("params", {}).get("update", {})
        for f in frames
        if f.get("method") == "session/update"
        and f.get("params", {}).get("update", {}).get("sessionUpdate") == kind
    ]


def _agent_text(frames: list) -> str:
    chunks = _updates(frames, "agent_message_chunk")
    return "".join(c.get("content", {}).get("text", "") if isinstance(c.get("content"), dict)
                    else str(c.get("content", "")) for c in chunks)


def eval_task(frames: list, gold: str) -> dict:
    """C4-C8 for one driven task. Returns {checks, tool_fired, collapsed}."""
    final = next((f for f in reversed(frames) if "result" in f and f.get("id")), None)
    stop = (final or {}).get("result", {}).get("stopReason")
    text = _agent_text(frames)
    bleed = looks_like_schema_bleed(text)
    pendings = _updates(frames, "tool_call")
    completeds = _updates(frames, "tool_call_update")
    fired = any((p.get("title") == gold or p.get("toolCallId")) for p in pendings) and bool(completeds)
    # C4: no bleed reached UI; a collapse MUST instead be stopReason:refusal.
    c4 = (not bleed) or (stop == "refusal")
    # C5: tool_call(pending) THEN tool_call_update(completed), in order.
    c5 = bool(pendings) and bool(completeds) and (
        frames.index(next(f for f in frames if f.get("params", {}).get("update", {}).get("sessionUpdate") == "tool_call"))
        < frames.index(next(f for f in frames if f.get("params", {}).get("update", {}).get("sessionUpdate") == "tool_call_update"))
    ) if (pendings and completeds) else False
    # C6: correct gold tool name in the pending event.
    c6 = any(p.get("title") == gold for p in pendings)
    # C7: no <think>/reasoning leak in the visible text.
    c7 = "<think>" not in text and "reasoning_content" not in text
    # C8: clean terminal stopReason, no hang/-32000.
    c8 = stop in ("end_turn", "refusal")
    return {
        "stopReason": stop, "bleed": bleed, "tool_fired": fired,
        "C4": c4, "C5": c5, "C6": c6, "C7": c7, "C8": c8,
    }


def verdict(tier: str | None, task_results: list[dict]) -> str:
    """Roll up to GREEN/YELLOW/RED/BLACK (MODEL-TEST-PLAN §0)."""
    any_black = any((not r["C4"]) or (not r["C8"]) for r in task_results)
    if any_black:
        return "BLACK"  # garbage reached UI or a hang — the ONLY shim bug
    fired_clean = all(r["tool_fired"] and r["C5"] and r["C6"] for r in task_results)
    if tier == "native" and fired_clean:
        return "GREEN"
    if tier in ("native", "emulated") and any(r["tool_fired"] for r in task_results):
        return "YELLOW"  # tools fire (maybe emulated/limited) — cap with ToolSelector
    return "RED"  # no tools, but clean refusal held (C4/C8) — safe chat-only


# ── runner ──────────────────────────────────────────────────────────────────


def _frame_telemetry(t0: float, frames: list, timings: list) -> dict:
    """Per-task latency telemetry (ms since the session/prompt send): the first
    session/update frame, the first agent_message_chunk ("first token"), the largest
    gap between consecutive frames, and the total turn duration. Quantifies the MARGIN
    to the v0.3.0 timing guards — first_token_ms vs the 30000ms pre-stream cap and
    max_inter_frame_gap_ms vs the 120000ms inactivity timeout — so a healthy model
    running CLOSE to a cap is visible, not just a binary pass/fail."""
    if not timings:
        return {}
    rel = [round((ts - t0) * 1000) for ts in timings]
    first_update = next(
        (rel[i] for i, f in enumerate(frames) if f.get("method") == "session/update"), None)
    first_token = next(
        (rel[i] for i, f in enumerate(frames)
         if f.get("params", {}).get("update", {}).get("sessionUpdate") == "agent_message_chunk"),
        None)
    gaps = [rel[i] - rel[i - 1] for i in range(1, len(rel))]
    return {
        "first_update_ms": first_update,
        "first_token_ms": first_token,
        "max_inter_frame_gap_ms": max(gaps) if gaps else 0,
        "total_ms": rel[-1] if rel else 0,
    }


def run_model(m: dict, defaults: dict) -> dict:
    proc = _rd._spawn_shim(
        Path(defaults["shim_bin"]), m["base_url"], m["id"],
        {
            "NWIRO_LOCAL_LLM_PROBE_TIMEOUT_SECS": str(defaults.get("probe_timeout_secs", 30)),
            "NWIRO_LOCAL_LLM_HIGH_TOOL_WARN": str(defaults.get("high_tool_warn", 150)),
            "NWIRO_LOCAL_LLM_BLEED_GUARD": str(defaults.get("bleed_guard", "on")),
        },
    )
    try:
        q = start_reader(proc)
        nid = iter(range(1, 10_000))
        # init
        send(proc, {"jsonrpc": "2.0", "id": next(nid), "method": "initialize", "params": {}})
        read_until(q, 1, timeout_s=5.0)
        # warmup -> C1/C2/C3
        wid = next(nid)
        send(proc, {"jsonrpc": "2.0", "id": wid, "method": "session/warmup",
                    "params": {"model": m["id"], "keepAlive": "15m"}})
        wf = read_until(q, wid, timeout_s=m.get("timeout_secs", 120))
        wres = next((f.get("result", {}) for f in wf if f.get("id") == wid), {})
        tier = wres.get("toolTier")
        ceiling = wres.get("recommendedToolCeiling")
        err_kind = wres.get("errorKind")
        c1 = tier in ("native", "emulated")
        c3 = err_kind != "broken_chat_template"
        # session/new + set_config
        sid_n = next(nid)
        send(proc, {"jsonrpc": "2.0", "id": sid_n, "method": "session/new", "params": {}})
        sf = read_until(q, sid_n, timeout_s=5.0)
        session_id = next((f["result"]["sessionId"] for f in sf if f.get("id") == sid_n and "result" in f), None)
        cfg = next(nid)
        send(proc, {"jsonrpc": "2.0", "id": cfg, "method": "session/set_config_option",
                    "params": {"sessionId": session_id, "configId": "model", "value": m["id"]}})
        read_until(q, cfg, timeout_s=5.0)
        # tasks
        task_results = []
        for t in _tasks():
            pid = next(nid)
            t0 = time.monotonic()
            send(proc, {"jsonrpc": "2.0", "id": pid, "method": "session/prompt", "params": {
                "sessionId": session_id, "prompt": [{"type": "text", "text": t["prompt"]}], "tools": t["tools"]}})
            timings: list = []
            frames = drive_prompt_with_mcp(
                proc, q, pid, GOLD_RESULT, timeout_s=m.get("timeout_secs", 120), timings=timings)
            r = eval_task(frames, t["gold"])
            r["task"] = t["id"]
            # v0.3.0 timing-guard margin telemetry (first-token vs the 30s pre-stream
            # cap; max inter-frame gap vs the 120s inactivity timeout).
            r["telemetry"] = _frame_telemetry(t0, frames, timings)
            task_results.append(r)
        v = verdict(tier, task_results)
        return {
            "model": m["id"], "backend": m.get("backend"), "role": m.get("role"),
            "tool_tier": tier, "recommended_ceiling": ceiling, "error_kind": err_kind,
            "C1": c1, "C3": c3, "tasks": task_results, "verdict": v,
        }
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.terminate()


_LEGEND = {"GREEN": "🟢 tool-ready", "YELLOW": "🟡 emulated/limited (cap N)",
           "RED": "🔴 chat-only (safe refusal)", "BLACK": "⬛ broken — file a shim bug"}


def to_markdown(rows: list[dict], version: str) -> str:
    out = ["# Model compatibility matrix",
           f"\n> Generated by `scripts/model-matrix.py` · shim v{version}. Do NOT hand-edit.\n",
           "Legend: " + " · ".join(f"{v}" for v in _LEGEND.values()) + "\n",
           "| Model | Backend | Tier | Ceiling | T-A | T-D | What-shim-does-on-failure | Verdict |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ta = next((t for t in r["tasks"] if t["task"] == "T-A"), {})
        td = next((t for t in r["tasks"] if t["task"] == "T-D"), {})
        ok = lambda t: "✅" if t.get("tool_fired") and t.get("C6") else ("🟡" if t.get("C4") else "❌")
        fail_behavior = "clean refusal" if r["verdict"] in ("RED", "YELLOW") else (
            "calls tools" if r["verdict"] == "GREEN" else "GARBAGE (bug)")
        out.append(
            f"| `{r['model']}` | {r['backend']} | {r['tool_tier']} | "
            f"{r['recommended_ceiling'] or '—'} | {ok(ta)} | {ok(td)} | {fail_behavior} | "
            f"{_LEGEND.get(r['verdict'], r['verdict'])} |")
    return "\n".join(out) + "\n"


def main() -> int:
    # Win consoles default to a non-UTF8 codepage (e.g. cp1254); the success print
    # below emits a U+2192 arrow. Force UTF-8 so output can't crash a completed run.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001 - best-effort; redirected / pre-3.7 stream
        pass
    ap = argparse.ArgumentParser(description="E2E model matrix (Layer 2).")
    ap.add_argument("--models", default=str(_HERE / "models.toml"))
    ap.add_argument("--only", help="comma-separated model ids")
    ap.add_argument("--out", help="write the JSON manifest here")
    ap.add_argument("--md", help="regenerate this MODEL-COMPATIBILITY.md")
    ap.add_argument("--version", default="dev", help="shim version for the md header")
    args = ap.parse_args()

    cfg = tomllib.loads(Path(args.models).read_text(encoding="utf-8"))
    defaults = cfg["defaults"]
    only = set(args.only.split(",")) if args.only else None
    rows = []
    for m in cfg.get("model", []):
        if only and m["id"] not in only:
            continue
        print(f"== {m['id']} ({m.get('backend')}) ==", flush=True)
        try:
            row = run_model(m, defaults)
        except Exception as e:  # noqa: BLE001
            row = {"model": m["id"], "backend": m.get("backend"), "verdict": "ERROR",
                   "error": f"{type(e).__name__}: {e}", "tasks": []}
        print(f"   verdict={row.get('verdict')} tier={row.get('tool_tier')}", flush=True)
        rows.append(row)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {len(rows)} row(s) → {args.out}")
    if args.md:
        Path(args.md).write_text(to_markdown(rows, args.version), encoding="utf-8")
        print(f"regenerated {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
