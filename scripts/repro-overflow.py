"""Reproduce the REAL LM Studio context-overflow 400 wording.

Sends a high-tool-count chat/completions request to the pod's LM Studio (via the
local :1234 tunnel) so we can see the EXACT error string the backend returns —
the thing the shim must classify as context_overflow AND parse n_ctx out of.
"""
import json
import sys
import urllib.request

BASE = "http://localhost:1234/v1/chat/completions"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "hermes-3-llama-3.1-8b"
N_TOOLS = int(sys.argv[2]) if len(sys.argv) > 2 else 224


def tool(i: int) -> dict:
    return {
        "type": "function",
        "function": {
            "name": f"nwiro_tool_{i:03d}",
            "description": (
                "Performs a Blueprint / actor / asset operation in the Unreal "
                "editor. Use this when the user asks to create, modify, query, "
                "or inspect a specific kind of game object or asset variant "
                f"number {i}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "asset or actor path"},
                    "location": {"type": "array", "items": {"type": "number"}},
                    "options": {"type": "object"},
                },
                "required": ["target"],
            },
        },
    }


tools = [
    {
        "type": "function",
        "function": {
            "name": "spawn_actor",
            "description": "Spawn an actor of the given class in the current level.",
            "parameters": {
                "type": "object",
                "properties": {"class": {"type": "string"}},
                "required": ["class"],
            },
        },
    }
] + [tool(i) for i in range(1, N_TOOLS)]

payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "Spawn a point light actor now."}],
    "tools": tools,
    "tool_choice": "auto",
    "stream": False,
    "max_tokens": 64,
}

body = json.dumps(payload).encode()
approx_tool_chars = len(json.dumps(tools))
print(f"MODEL={MODEL}  N_TOOLS={len(tools)}  tools_json_chars={approx_tool_chars} (~{approx_tool_chars//4} tok)")

req = urllib.request.Request(BASE, data=body, headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=60) as r:
        print(f"STATUS {r.status} — NO overflow (request fit). Body head:")
        print(r.read(600).decode("utf-8", "replace"))
except urllib.error.HTTPError as e:
    err_body = e.read().decode("utf-8", "replace")
    print(f"STATUS {e.code} (HTTPError)")
    print("=== RAW ERROR BODY ===")
    print(err_body)
    # Pull the message field if JSON
    try:
        j = json.loads(err_body)
        msg = j.get("error", {}).get("message") if isinstance(j.get("error"), dict) else j.get("error")
        print("=== error.message ===")
        print(msg)
        print("=== has 'n_ctx' token:", "n_ctx" in (msg or ""), "===")
    except Exception as ex:
        print(f"(body not JSON: {ex})")
except Exception as e:
    print(f"REQUEST FAILED: {type(e).__name__}: {e}")
