"""Reasoning-model ROBUSTNESS verification harness.

Drives the shim through the canonical reasoning-model failure modes for ANY model
+ backend and scores each — a reproducible companion to the automated goldens
(acp/golden.rs: reasoning_budget_exhausted_*, server_error_*, the empty/non-object
arg guards) and the prompt-architect mandate (bridge/mod.rs build_tool_invocation_mandate).

Usage:  python scripts/reasoning-robustness-check.py <model> [base_url]
Example: python scripts/reasoning-robustness-check.py thudm_glm-z1-9b-0414 \
             https://<pod>-1234.proxy.runpod.net/v1

Each scenario asserts the GENERAL property (model-agnostic):
  greeting  — a non-action turn must NOT fire a tool and must produce SOMETHING
              (a direct answer → end_turn, OR a clean reasoning_budget degrade →
              refusal with the helpful message). Never an empty/poisoned turn.
  action    — an action request must FIRE the gold tool.
"""
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = sys.argv[1] if len(sys.argv) > 1 else "thudm_glm-z1-9b-0414"
BASE = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:1234/v1"


def run(prompt, n_tools):
    p = subprocess.run(
        [sys.executable, str(HERE / "drive-chat.py"), MODEL, prompt, str(n_tools), BASE],
        capture_output=True, text=True, timeout=360, encoding="utf-8", errors="replace",
    )
    return p.stdout + p.stderr


def field(out, key):
    m = re.search(rf"{key}[:=]\s*(.+)", out)
    return m.group(1).strip() if m else ""


def check_greeting(out):
    fired = field(out, "tools_fired")
    stop = field(out, "stopReason")
    # robust = no tool fired AND a terminal stopReason AND the turn is not empty
    no_tool = fired.startswith("[]") or fired == "[]"
    terminal = ("end_turn" in stop) or ("refusal" in stop)
    not_empty = ("AGENT MESSAGE" in out) and ("''" not in out.split("AGENT MESSAGE")[1][:80])
    degraded = "full response budget thinking" in out  # the reasoning_budget message
    ok = no_tool and terminal and (not_empty or degraded)
    return ok, f"no_tool={no_tool} terminal={terminal} answered_or_degraded={not_empty or degraded}"


def check_action(out):
    fired = "spawn_actor" in out
    return fired, f"spawn_actor_fired={fired}"


SCENARIOS = [
    ("greeting (non-action)", "hey", 40, check_greeting),
    ("action (spawn)", "spawn a point light at 100,100,100", 40, check_action),
]

print(f"=== reasoning-robustness check: {MODEL} @ {BASE} ===")
passed = 0
for name, prompt, n, check in SCENARIOS:
    out = run(prompt, n)
    ok, why = check(out)
    passed += ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  ({why})")
    for line in out.splitlines():
        if any(k in line for k in ("stopReason:", "AGENT MESSAGE", "fired:")):
            print("      ", line.strip()[:140])
print(f"=== {passed}/{len(SCENARIOS)} passed ===")
sys.exit(0 if passed == len(SCENARIOS) else 1)
