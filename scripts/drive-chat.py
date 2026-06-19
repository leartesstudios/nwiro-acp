"""Drive the shim with an arbitrary prompt + tool count, and capture what the USER
sees: the agent MESSAGE (visible), the agent THOUGHT (thinking indicator), fired
tool calls, errors, and the terminal stopReason. Used to reproduce the GLM
reasoning-leak and tool-call-error reports.

Usage: python scripts/drive-chat.py <model> "<prompt>" <n_tools> [base_url]
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _load(filename, mod_name):
    spec = importlib.util.spec_from_file_location(mod_name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_rd = _load("run-day1.py", "run_day1")
_tc = _load("tool-curve.py", "tool_curve")
from _mcp_responder import drive_prompt_with_mcp, read_until, send, start_reader  # noqa: E402

MODEL = sys.argv[1]
PROMPT = sys.argv[2]
N_TOOLS = int(sys.argv[3]) if len(sys.argv) > 3 else 40
BASE_URL = sys.argv[4] if len(sys.argv) > 4 else "http://127.0.0.1:1234/v1"
SHIM = HERE.parent / "target" / "release" / "local-llm-acp.exe"
GOLD_RESULT = {"content": [{"type": "text", "text": "Spawned PointLight at origin."}], "isError": False}


def spawn_actor_tool():
    return {"type": "function", "function": {"name": "spawn_actor",
            "description": "Spawn an actor of the given class at a location.",
            "parameters": {"type": "object", "properties": {
                "class": {"type": "string"},
                "location": {"type": "array", "items": {"type": "number"}}}, "required": ["class"]}}}


tools = [spawn_actor_tool()] + [_tc._synth_tool(i) for i in range(1, N_TOOLS)]

trace = Path(tempfile.gettempdir()) / "shim-chat-trace.log"
if trace.exists():
    trace.unlink()
env = {"NWIRO_LOCAL_LLM_TRACING_FILE": str(trace), "RUST_LOG": "local_llm_acp=info"}
proc = _rd._spawn_shim(SHIM, BASE_URL, MODEL, env)
q = start_reader(proc)
nid = iter(range(1, 10_000))
send(proc, {"jsonrpc": "2.0", "id": next(nid), "method": "initialize", "params": {}})
read_until(q, 1, timeout_s=10.0)
wid = next(nid)
send(proc, {"jsonrpc": "2.0", "id": wid, "method": "session/warmup", "params": {"model": MODEL, "keepAlive": "15m"}})
wf = read_until(q, wid, timeout_s=120)
wres = next((f.get("result", {}) for f in wf if f.get("id") == wid), {})
print("WARMUP tier=%s ceiling=%s" % (wres.get("toolTier"), wres.get("recommendedToolCeiling")))
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
            "params": {"sessionId": session_id, "prompt": [{"type": "text", "text": PROMPT}], "tools": tools}})
frames = drive_prompt_with_mcp(proc, q, pid, GOLD_RESULT, timeout_s=240)


def upd(kind):
    return [f.get("params", {}).get("update", {}) for f in frames
            if f.get("method") == "session/update" and f.get("params", {}).get("update", {}).get("sessionUpdate") == kind]


def txt(updates):
    out = ""
    for u in updates:
        c = u.get("content")
        if isinstance(c, dict):
            out += c.get("text", "")
        elif isinstance(c, str):
            out += c
    return out


final = next((f for f in reversed(frames) if "result" in f and f.get("id") == pid), None)
stop = (final or {}).get("result", {}).get("stopReason")
msg = txt(upd("agent_message_chunk"))
thought = txt(upd("agent_thought_chunk"))
tools_fired = [u.get("title") for u in upd("tool_call")]
errs = [f.get("error") for f in frames if "error" in f]
print("stopReason:", stop, "| tools_fired:", tools_fired)
print("AGENT MESSAGE (what the user sees) [:500]:")
print("  ", repr(msg[:500]))
print("AGENT THOUGHT (thinking indicator) [:200]:")
print("  ", repr(thought[:200]))
if errs:
    print("ERRORS:", str(errs)[:400])
try:
    proc.terminate()
except Exception:
    pass
