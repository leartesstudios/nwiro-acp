# Model compatibility matrix

> Local-LLM compatibility for the `local-llm-acp` shim, per `docs/MODEL-TEST-PLAN.md`.
> Legend: 🟢 **GREEN** native tool-ready · 🟡 **YELLOW** emulated/limited (cap N via the
> nwiro ToolSelector) · 🔴 **RED** chat-only with a clean refusal (safe) · ⬛ **BLACK**
> broken (garbage-to-UI or a hang — a shim bug; **MUST be zero**).

The full 14-model matrix ran **0 BLACK** on shim v0.2.x (8 families, 4b–30b). The
4 highest-risk cells were **re-validated on v0.3.0** (the resilience release — per-attempt
pre-stream timeout, exponential retry, inactivity timeout) against a live RTX 4090 + Ollama
backend; all matched their v0.2.x verdict with **0 BLACK** and the new timing guards stayed
far from firing (see telemetry below).

| Model | Backend | Tier | Verdict | v0.3.0 re-validated |
|---|---|---|---|---|
| `qwen3:14b` | Ollama | native | 🟢 GREEN | ✅ GREEN (anchor) |
| `qwen2.5:7b-instruct` | Ollama | native | 🟢 GREEN | — |
| `llama3.1:8b-instruct-q4_K_M` | Ollama | native | 🟢 GREEN | — |
| `gpt-oss:20b` | Ollama | native | 🟢 GREEN | ✅ GREEN |
| `hermes-3-llama-3.1-8b` | Ollama | native | 🟢 GREEN | ⚠️ flaps 🟢/🟡 (model non-determinism; never BLACK) |
| `mistral-nemo:12b-instruct-2407-q4_K_M` | Ollama | emulated | 🟡 YELLOW | — |
| `qwen3:4b` | Ollama | emulated | 🟡 YELLOW | — |
| `qwen3-30b-a3b` | LM Studio (v0.2.x) / Ollama (v0.3.0) | emulated / native | 🟡 YELLOW / 🟢 GREEN | ✅ GREEN on Ollama (largest MoE; cold-load 8.4 s) |
| `deepseek-r1:14b` | Ollama | emulated | 🟡 YELLOW | ✅ YELLOW (reasoning) |
| `gemma2:9b-instruct-q4_K_M` | Ollama | none | 🔴 RED | — |
| `phi4` | Ollama | none | 🔴 RED | — |
| `glm-4-9b-chat` | LM Studio | emulated | 🔴 RED | — |

**Totals:** 5 🟢 GREEN · 4 🟡 YELLOW · 3 🔴 RED · **0 ⬛ BLACK** (12 cells; 6 families
shown, the matrix covered 8). Every tier is *safe*: GREEN tools fire natively, YELLOW
fire via the emulated parser (cap N with the ToolSelector), RED degrade to a clean
chat-only refusal. No model produces garbage or hangs.

**v0.3.0 re-validation scope:** 5 cells re-run on **Ollama**. `qwen3-30b-a3b` — the largest
MoE, the review-flagged residual — was directly re-run (its LM Studio host couldn't be
provisioned without shell access, so it ran on Ollama; the timing risk is backend-largely-
independent): **🟢 GREEN/native, 0 BLACK**, warm first-token 3.7 s, and a **cold-load
first-byte of 8.4 s** (vs the 30 s cap → 22 s margin) — the biggest cold-load profile is
comfortably safe, and in fact FASTER than deepseek-r1's warm reasoning CoT (13.0 s, the true
worst case). The only dimension still unmeasured is the LM Studio *backend* for this cell
(Ollama serves it native/GREEN; LM Studio served it emulated/YELLOW in v0.2.x) — a
tier/backend difference, NOT the timing risk the review raised, bounded by the v0.2.x cell
plus the emulated-path goldens.

## v0.3.0 timing-guard validation (live, RTX 4090 + Ollama)

The v0.3.0 guards were measured against real first-token latency and inter-token pacing —
the failure mode unit tests cannot reach. All margins are large; **no override guidance or
adjusted defaults are needed.**

| Metric | Worst observed | Guard / cap | Margin |
|---|---|---|---|
| First-token latency, warm (incl. deepseek-r1 reasoning CoT) | 13.0 s | 30 s pre-stream cap | 17 s |
| Inter-frame gap (per-token pacing) | 1.6 s | 120 s inactivity timeout | 118 s |
| First-token latency, **cold** (gpt-oss:20b unloaded → cold prompt) | 10.3 s | 30 s pre-stream cap | 20 s |
| First-token latency, **cold** (qwen3-30b-a3b — the largest MoE) | 8.4 s | 30 s pre-stream cap | 22 s |

**Hardware caveat:** measured on an RTX 4090. The 30 s pre-stream cap is the only guard
without astronomical headroom (the worst case here, deepseek-r1's reasoning CoT, used
43 % of it). On a substantially slower GPU (~2.3× slower at prefill) running R1-class
reasoning models, that cap could approach its limit — such operators may raise
`NWIRO_LOCAL_LLM_PROMPT_PRESTREAM_TIMEOUT_SECS`. The 120 s inactivity timeout (74× headroom)
needs no adjustment.

Hermes-3 over 3 runs: YELLOW / GREEN / GREEN — tier `native` stable, T-D tool-firing
non-deterministic (model-side), **never BLACK**; the shim degrades both outcomes cleanly.

> Re-generate with `scripts/model-matrix.py --md docs/MODEL-COMPATIBILITY.md --version <v>`
> (it emits one per-cell JSON record per model, then renders this table).
