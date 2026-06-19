#!/usr/bin/env python3
"""Layer 1 — the shim-free tool-count curve (docs/MODEL-TEST-PLAN.md §4).

Hits `{base_url}/chat/completions` DIRECTLY — no shim, no MCP — and walks the
tool-count curve (1/30/80/150/220 by default), recording where each model
collapses into schema-bleed. This is the shim-free tool-count curve: it
isolates raw MODEL capability from shim logic. Uses the SAME bleed
definition as the shim (`scripts/_bleed_oracle.py`, GATE 0).

It replicates the shim's probe message + gold tool (`client.rs`), but sends
`tool_choice: "auto"` (NOT the probe's forced call) so the curve measures the
model's NATURAL collapse as N grows — the probe forces the call for a binary
capability check, a different purpose.

Usage:
    python scripts/tool-curve.py                 # all models in models.toml
    python scripts/tool-curve.py --only qwen3:14b,glm-4-9b-chat
    python scripts/tool-curve.py --out reports/tool-curve.json

Requires Python 3.11+ (tomllib). Stdlib only — no pip install.
"""
from __future__ import annotations

import argparse
import json
import random
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent))
from _bleed_oracle import looks_like_schema_bleed  # GATE 0 — same definition as the shim

GOLD = "find_blueprints"  # the probe keys Native on this name (client.rs:835)


def _gold_tool() -> dict:
    # Verbatim from the shim probe (client.rs:832-843).
    return {
        "type": "function",
        "function": {
            "name": GOLD,
            "description": "Search Blueprint assets",
            "parameters": {
                "type": "object",
                "properties": {"searchTerm": {"type": "string"}},
                "required": ["searchTerm"],
            },
        },
    }


def _synth_tool(i: int) -> dict:
    """A deterministic synthetic tool seeded by index, so the padded array is
    IDENTICAL across runs -> historical comparability (FINDINGS Finding J)."""
    rng = random.Random(i)
    nprops = rng.randint(1, 4)
    props = {
        f"arg{j}": {"type": rng.choice(["string", "number", "boolean", "integer"])}
        for j in range(nprops)
    }
    return {
        "type": "function",
        "function": {
            "name": f"synth_tool_{i:03d}",
            "description": f"Synthetic tool {i} for tool-count padding.",
            "parameters": {"type": "object", "properties": props, "required": list(props)[:1]},
        },
    }


def _padded_tools(n: int) -> list:
    """Gold tool fixed at index 0; n-1 deterministic synthetics after it."""
    return [_gold_tool()] + [_synth_tool(i) for i in range(1, n)]


def _post(base_url: str, model: str, tools: list, timeout: int) -> dict:
    body = {
        "model": model,
        "messages": [
            {"role": "user", "content": "/no_think Call find_blueprints with searchTerm 'test'"}
        ],
        "tools": tools,
        "tool_choice": "auto",  # measure NATURAL collapse (see module docstring)
        "max_tokens": 256,
        "stream": False,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _classify(resp: dict) -> str:
    """One cell of the curve: ok | bleed | length | prose | empty | error."""
    try:
        choice = resp["choices"][0]
    except (KeyError, IndexError, TypeError):
        return "error"
    msg = choice.get("message") or {}
    tool_calls = msg.get("tool_calls") or []
    if tool_calls and (tool_calls[0].get("function") or {}).get("name") == GOLD:
        return "ok"  # native tool call worked
    content = msg.get("content") or ""
    if looks_like_schema_bleed(content):
        return "bleed"  # the collapse — model echoed the schema as text
    if choice.get("finish_reason") == "length":
        return "length"  # ran out of budget before calling
    if content.strip():
        return "prose"  # described instead of calling (describer-over-actor)
    return "empty"


def run_model(m: dict, curve_points: list, timeout: int) -> dict:
    base, model = m["base_url"], m["id"]
    curve: dict[int, str] = {}
    ceiling = 0
    collapsed = False
    for n in curve_points:
        try:
            cell = _classify(_post(base, model, _padded_tools(n), timeout))
        except urllib.error.URLError as e:
            cell = f"error:{getattr(e, 'reason', e)}"
        except Exception as e:  # noqa: BLE001 — record, don't crash the whole run
            cell = f"error:{type(e).__name__}"
        curve[n] = cell
        print(f"     N={n:>3}: {cell}", flush=True)
        if cell == "ok" and not collapsed:
            ceiling = n
        elif cell != "ok":
            collapsed = True  # ceiling is the highest contiguous OK
    return {
        "model": model,
        "backend": m.get("backend"),
        "base_url": base,
        "role": m.get("role"),
        "curve": curve,
        "ceiling": ceiling,
        "expected_ceiling": m.get("expected_ceiling"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Shim-free tool-count curve (Layer 1).")
    ap.add_argument("--models", default=str(Path(__file__).parent / "models.toml"))
    ap.add_argument("--only", help="comma-separated model ids (default: all in the registry)")
    ap.add_argument("--out", help="write the JSON rows here")
    args = ap.parse_args()

    cfg = tomllib.loads(Path(args.models).read_text(encoding="utf-8"))
    defaults = cfg["defaults"]
    curve_points = defaults["curve_points"]
    only = set(args.only.split(",")) if args.only else None

    rows = []
    for m in cfg.get("model", []):
        if only and m["id"] not in only:
            continue
        print(f"== {m['id']} ({m.get('backend')}) — {m.get('role', '')} ==", flush=True)
        row = run_model(m, curve_points, m.get("timeout_secs", 120))
        exp = row["expected_ceiling"]
        flag = "" if exp is None else ("  ✓" if row["ceiling"] >= min(exp, 150) else "  ⚠ CHECK")
        print(f"   -> ceiling={row['ceiling']} (expected {exp}){flag}\n", flush=True)
        rows.append(row)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"wrote {len(rows)} row(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
