"""Drive the DEPLOYED shim with a high-tool-count prompt against the pod, in
isolation from nwiro, to see EXACTLY what the increment-1 overflow path does:
does it classify context_overflow, parse n_ctx, tail-trim, retry, and fire?

Usage: python scripts/drive-overflow.py [model] [n_tools]
Reads the shim's tracing file afterward to show the `context_overflow:
tail-trimming` warn (n_ctx / from / to) — the ground truth on the trim.
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _load(filename: str, mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod  # register BEFORE exec so @dataclass can resolve cls.__module__
    spec.loader.exec_module(mod)
    return mod


_rd = _load("run-day1.py", "run_day1")
_tc = _load("tool-curve.py", "tool_curve")
from _mcp_responder import drive_prompt_with_mcp, read_until, send, start_reader  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "hermes-3-llama-3.1-8b"
N_TOOLS = int(sys.argv[2]) if len(sys.argv) > 2 else 224
BASE_URL = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:1234/v1"
SHIM = HERE.parent / "target" / "release" / "local-llm-acp.exe"

GOLD_RESULT = {"content": [{"type": "text", "text": "Spawned PointLight at origin."}], "isError": False}


def spawn_actor_tool() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "spawn_actor",
            "description": "Spawn an actor of the given class in the current level.",
            "parameters": {"type": "object", "properties": {"class": {"type": "string"}}, "required": ["class"]},
        },
    }


tools = [spawn_actor_tool()] + [_tc._synth_tool(i) for i in range(1, N_TOOLS)]

trace_path = Path(tempfile.gettempdir()) / "shim-overflow-trace.log"
if trace_path.exists():
    trace_path.unlink()

env = {"NWIRO_LOCAL_LLM_TRACING_FILE": str(trace_path), "RUST_LOG": "local_llm_acp=debug,info"}
proc = _rd._spawn_shim(SHIM, BASE_URL, MODEL, env)
q = start_reader(proc)
nid = iter(range(1, 10_000))

send(proc, {"jsonrpc": "2.0", "id": next(nid), "method": "initialize", "params": {}})
read_until(q, 1, timeout_s=10.0)

wid = next(nid)
send(proc, {"jsonrpc": "2.0", "id": wid, "method": "session/warmup", "params": {"model": MODEL, "keepAlive": "15m"}})
wf = read_until(q, wid, timeout_s=120)
wres = next((f.get("result", {}) for f in wf if f.get("id") == wid), {})
print(f"WARMUP  toolTier={wres.get('toolTier')}  ceiling={wres.get('recommendedToolCeiling')}  errKind={wres.get('errorKind')}")

sid_n = next(nid)
send(proc, {"jsonrpc": "2.0", "id": sid_n, "method": "session/new", "params": {}})
sf = read_until(q, sid_n, timeout_s=10.0)
session_id = next((f["result"]["sessionId"] for f in sf if f.get("id") == sid_n and "result" in f), None)

cfg = next(nid)
send(proc, {"jsonrpc": "2.0", "id": cfg, "method": "session/set_config_option",
            "params": {"sessionId": session_id, "configId": "model", "value": MODEL}})
read_until(q, cfg, timeout_s=10.0)

pid = next(nid)
send(proc, {"jsonrpc": "2.0", "id": pid, "method": "session/prompt",
            "params": {"sessionId": session_id, "prompt": [{"type": "text", "text": "Spawn a point light actor now."}], "tools": tools}})
frames = drive_prompt_with_mcp(proc, q, pid, GOLD_RESULT, timeout_s=180)

final = next((f for f in reversed(frames) if "result" in f and f.get("id") == pid), None)
stop = (final or {}).get("result", {}).get("stopReason")


def updates(kind: str) -> list:
    return [f.get("params", {}).get("update", {}) for f in frames
            if f.get("method") == "session/update" and f.get("params", {}).get("update", {}).get("sessionUpdate") == kind]


pendings = updates("tool_call")
completeds = updates("tool_call_update")
print(f"RESULT  N_TOOLS={len(tools)}  stopReason={stop}  tool_call(pending)={len(pendings)}  tool_call_update={len(completeds)}")
if pendings:
    print("  fired:", [p.get("title") for p in pendings])

try:
    proc.terminate()
except Exception:
    pass

print("=== TRACE (overflow / trim / refusal lines) ===")
if trace_path.exists():
    for line in trace_path.read_text(encoding="utf-8", errors="replace").splitlines():
        low = line.lower()
        if any(k in low for k in ("overflow", "tail-trim", "n_ctx", "learned", "refus", "ceiling", "trim")):
            print(" ", line[:320])
else:
    print("  (no trace file written)")
