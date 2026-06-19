# local-llm-acp — Model Test Plan

**Goal:** end-users run *arbitrary* local models without hitting problems.
**Companion:** `docs/MODEL-SETUP.md` · `docs/MODEL-COMPATIBILITY.md`.
**Derived from:** the repo's test harness (`scripts/run-day1.py`, `scripts/model-matrix.py`, `scripts/aggregate-day-results.py`).

---

## 0. Philosophy — what "test many models" actually means

We will **not** try to make every model work — model fitness is out of scope (a 9B model with a 29-tool ceiling cannot be prompted into competence). Instead:

> **Publish a generated compatibility matrix where every cell is GREEN / YELLOW / RED (all safe) or BLACK (a shim bug to fix).**

This converts an open-ended support burden into a **bounded, automated guarantee**: an end-user reads the matrix, picks a 🟢 model, and the shim's only job is to **never produce a ⬛**. A 🟡/🔴 is a *correct* outcome (capped tools / clean refusal), not a failure.

### Verdict tiers (the published legend)
| Tier | Meaning | Shim behaviour |
|---|---|---|
| 🟢 **GREEN** | Native tool-ready | tools fire natively, ceiling ≥ 150, clean |
| 🟡 **YELLOW** | Emulated / limited | tools fire via the emulated parser, but ceiling < 150 → usable **with ToolSelector capping N** |
| 🔴 **RED** | Chat-only | no tool tier, but the shim emits a **clean refusal** (safe) |
| ⬛ **BLACK** | Broken | garbage reaches the UI, or a hang — **the ONLY outcome that is a shim bug, not a model limitation** |

---

## 1. Models to cover (M1-M8)

| # | Model | Size · family · backend | Role |
|---|---|---|---|
| **M1** | `qwen3:14b` | 14B · Qwen · Ollama | 🟢 **reference PASS** — curl-proven Native, validated live at 220 tools; every run diffs against it |
| **M2** | `qwen2.5:7b-instruct` | 7B · Qwen · Ollama | smallest expected-to-work; finds the Qwen floor |
| **M3** | `qwen3-30b-a3b` (Q4_K_M) | 30B MoE · Qwen · LM Studio | MoE active-param behaviour; Native on a non-Ollama backend |
| **M4** | `GLM-4-9B-Chat` (Q4_K_S) | 9B · GLM · LM Studio | 🔴 **reference FAIL** — canonical schema-bleed + ceiling-29; guards the whole detection stack against regression |
| **M5** | `llama-3.1-8b-instruct` | 8B · Llama · Ollama | most-deployed OSS; Llama tool-call dialect differs from Qwen |
| **M6** | `mistral-nemo:12b-instruct` | 12B · Mistral · Ollama | a 3rd tool-call dialect |
| **M7** | `Hermes-3-Llama-3.1-8B` | 8B · Nous · LM Studio | `<tool_call>` XML fine-tune — exercises the **Emulated parser**, not Native |
| **M8** | `gemma-2-9b-it` *or* `phi-4` | 9-14B · Gemma/Phi · Ollama | **control** — no tool fine-tune; verifies graceful degrade + clean refusal, not garbage |

**Backend rule:** each family runs on its primary backend; the two anchors (M1, M4) **additionally** run on all three backends (Ollama / LM Studio / llama.cpp) to surface backend-specific bugs (LM Studio manual-template tool-dropping, llama.cpp `--jinja`) without a 3× blow-up.

**Out of scope (document, don't test):** > 32B and frontier MoE (Kimi-K2) — most end-users can't run them; covered by the separate Blackwell model setup, not this end-user matrix.

**Cloud backends (M9 — beta, chat-only):** `openrouter` runs a SEPARATE single-cell smoke, NOT part of the M1-M8 local curve. The Gap-5 mid-stream-error path IS automated (deterministic mockito coverage in `src/openai/`). The live chat smoke — one paid-slug streaming call (`openai/gpt-4o-mini` via `https://openrouter.ai/api/v1`) that should assert the response `model` echoes the requested slug — is a MANUAL/live check, not yet automated. Tools are **unsupported** over OpenRouter today (Gap-3 — the probe must classify the 404 "no endpoints found that support tool_choice" as tool-unsupported before tools graduate), so it runs T-A chat only, not T-B…T-D. Runnable gate: `scripts/verify-gap5.sh`.

---

## 2. Canonical tasks (T-A … T-D)

All tasks use **real Nwiro tool shapes** (`find_blueprints`, `spawn_actor`, `create_blueprint`) so the curve matches production.

| ID | Task | Tools | What it catches |
|---|---|---|---|
| **T-A** | "Spawn a point light at the origin." | 1 | baseline single-tool correctness |
| **T-B** | "Find the Cube blueprint, then spawn it at (100,0,0) and rename it Hero." | ~8 | sequential ReAct loop: ordering + history correctness |
| **T-C** | "List all blueprints." with the tool array **padded** to N synthetic-but-realistic schemas around the gold tool | **1 / 30 / 80 / 150 / 220** | **the per-model ceiling** — the curl curve automated. Fixed N points so historical results stay comparable; padding generator is deterministic (seeded by index) |
| **T-D** | "Create a blueprint called BP_Enemy." (action verb, tools present) | ~30 | **describer-bias trap** — does it INVOKE or DESCRIBE? Pass = a real `tool_call`, not prose |

---

## 3. Pass / fail criteria (C1 … C8)

**Warmup:**
- **C1 — tier classification:** `WarmupResult.toolTier ∈ {native, emulated}` for M1-M7; `none` is a FAIL for tool-expected models. (M8 control: `none`/`emulated` is PASS *iff* C4 holds.)
- **C2 — ceiling consistency:** if `recommendedToolCeiling` is present it must be ≤ the empirically-measured T-C collapse point (GLM = 29 today, `model_family.rs:208`).
- **C3 — no broken-template false-positive:** working models report `errorKind != broken_chat_template`; a deliberately-misconfigured GLM (no preset) MUST report `broken_chat_template` (gate fired correctly = PASS).

**Per task:**
- **C4 — no schema-bleed reaches the UI:** for every `agent_message_chunk`, `looks_like_schema_bleed(text) == false`. If the model collapses, the shim MUST instead emit `stopReason:"refusal"` (one clean line) — that is a **PASS** (`model_collapsed=true, shim_contained=true`). A raw `object/type/properties` wall streamed = FAIL.
- **C5 — tool actually executed:** for T-A/B/D the frame stream contains `tool_call(pending)` **then** `tool_call_update(completed)`, with the gold tool name and parsed `rawInput.arguments` (assertion exists in `smoke-test-tool-call-events.py`). Prose-only = FAIL.
- **C6 — correct class/args:** executed tool == gold tool; required args present + correctly typed (T-A `spawn_actor` has `class`/`location`). Wrong tool or malformed args = FAIL. *(Directly exercises the `spawn_actor`/`APointLight` failure.)*
- **C7 — no reasoning leak:** no `<think>`/`reasoning_content` in `agent_message_chunk` content (must be `agent_thought_chunk` or suppressed). Qwen3/R1-specific.
- **C8 — clean termination:** final `stopReason ∈ {end_turn, refusal}` within the per-model timeout; never a hang, never `-32000`.

**Verdict rollup → published tier:** 🟢 = C1 native + T-A/B/D pass C5/C6 + T-C ≥ 150 + C4/C7/C8 clean · 🟡 = tools fire (Emulated) but T-C ceiling < 150 · 🔴 = no tier but C4 holds · ⬛ = C4 fail (garbage) or C8 fail (hang).

---

## 3a. Worked example — why GLM-4-9B is chat-only at scale (Finding J)

The canonical illustration of the per-model ceiling (T-C) and of why the
schema-bleed guard exists. A controlled, **shim-free** tool-count curve (pure
backend, no shim, no MCP) shows `GLM-4-9B-Chat`:

- works at **1** and **30** tools,
- hits `finish_reason: length` at **80**,
- **schema-bleeds** at **150 / 220** — the model echoes the tool *schema* back as
  text (`"object"`, `"type"`, `"properties"`) instead of emitting a `tool_calls`
  envelope.

220 schemas (~25K tokens) saturate a 9B model's context and dilute attention; this
is a hard **capability** limit, not something the shim can format or prompt around.
`qwen3:14b` passes the same curve at 220. The measured collapse point is **29** (the
`recommendedToolCeiling` the C2 check asserts against); README guidance rounds this
to "≈30".

**What the shim does about it (so users never see garbage):** schema-bleed is
detected at the post-stream buffer-flush, the garbage is suppressed, and the turn
returns exactly one clean refusal line (`stopReason: "refusal"`) — that is a C4
**PASS** (`model_collapsed=true, shim_contained=true`), not a failure. Tool
invocation is **model-agnostic**: the shim keys behaviour on the MCP `isError`
signal and the probed tool tier, never on a model's name (the GLM-only special
cases were deleted in v0.1.35). A one-shot `tracing::warn!` fires when a non-Native
model is handed more tools than `NWIRO_LOCAL_LLM_HIGH_TOOL_WARN` (default 50).

The schema-bleed heuristic is shape-based, so a model that *legitimately* emits a
JSON-schema answer as content could be suppressed; every trip logs a `warn!`, and
`NWIRO_LOCAL_LLM_BLEED_GUARD=off` disables it for schema-output use cases. To keep a
small model in service, relevance-filter the registry down to its ceiling
(Nwiro-side ToolSelector) so it never reaches the bleed point.

---

## 4. Automation (two layers — failure lives at two levels)

**Layer 1 — `scripts/tool-curve.py` (NEW · shim-free · highest leverage).** *This is the curl curve the docs assume exists but that does not.* Pure-`requests` script hits `/v1/chat/completions` directly (no shim), walks T-C `1/30/80/150/220` with `tool_choice:auto`, records `finish_reason` + whether a native `tool_calls` envelope returned + `looks_like_schema_bleed(content)`. Output one JSON row per model: `{model, backend, curve:{1:ok,30:ok,80:length,150:bleed,220:bleed}, ceiling:N}`. **Isolates MODEL capability from shim logic** — ~5 prompts/model, seconds each.

**Layer 2 — `scripts/model-matrix.py` (NEW · E2E through the shim).** Extends `scripts/run-day1.py` (reuse its `TestCase`/`TestResult`/`_read_frames_timed` plumbing — the `tools` field already exists on `TestCase`); add the MCP-responder loop from the tool-call-events smoke test (the harness answers the shim's `mcp/connect`+`mcp/message` and asserts the round-trip → makes C5/C6 testable E2E); add C1-C8 as functions returning structured pass/fail; add `scripts/models.toml` (model id, base_url, backend, expected_tier, timeout) so **adding a model is one stanza, not code**.

**Shared oracle:** port `looks_like_schema_bleed`'s 3 gates (`client.rs:164`) to Python (~12 lines) — used by both layers, so the test definition of "bleed" stays identical to the shim's.

**Reproducibility:** the harness pins (in the manifest header) the existing env vars: `NWIRO_LOCAL_LLM_FORCE_TOOL_TIER`, `NWIRO_LOCAL_LLM_PROBE_TIMEOUT_SECS=30`, `NWIRO_LOCAL_LLM_BLEED_GUARD=off` (to capture a *raw* collapse for a fixture), `NWIRO_LOCAL_LLM_HIGH_TOOL_WARN`. Output JSON → the existing `scripts/aggregate-day-results.py`, unchanged.

**CI — two tiers** *(review finding: live local inference is slow — full matrix runs take hours, so it is NOT on every push):*
1. **Every push (existing CI, no GPU):** a `MockBackendScript` *replay* mode for the mock servers replays a recorded transcript per (model, task) — including a canned schema-bleed wall for the GLM cell — so the **C4/C5/C8 assertions run in GitHub Actions with no model present** (parallels the Rust-side `golden.rs::schema_bleed_guard_suppresses_garbage_and_refuses`). Guards harness + shim logic, zero new infra.
2. **Manual `workflow_dispatch` / self-hosted GPU box:** runs Layer 1 + Layer 2 live, regenerates the matrix JSON, opens a PR updating `docs/MODEL-COMPATIBILITY.md`. Run before a release / when adding a model.

---

## 5. The published compatibility matrix

A single **`docs/MODEL-COMPATIBILITY.md`**, **generated** from the matrix JSON (never hand-edited), linked from the README + `RUNNING.md`. Header: *"Generated by `scripts/model-matrix.py` on `<date>` · shim v`<ver>`"*.

**Columns:** Model | Size | Family | Backend | Tier | **Tool ceiling** | Spawn (T-A) | Chain (T-B) | Invokes? (T-D) | What-shim-does-on-failure | **Verdict**.

Each row links to a per-model section: exact id/quant, backend + required `n_ctx` (≥ 65536 for the full tool set), the measured curve (`1✅ 30✅ 80⚠️ 150❌`), and the **recommended ToolSelector cap `N = ceiling − safety-margin`**.

> The **cap-`N` column is the actionable output**: it tells the host's tool selector exactly how many tools to send per model — **the shim measures the ceiling and publishes it (`recommendedToolCeiling`); the host sizes `N` from it.** The published ceiling generalizes the single hard-coded GLM=29 into a measured, per-model table.

The "what-shim-does-on-failure" column makes explicit that 🟡/🔴 are **safe** (capped tools / clean refusal) and only ⬛ is a bug — aligning the published artifact with the "users don't hit problems" priority.

---

## 6. Build order

1. `scripts/tool-curve.py` + the Python `looks_like_schema_bleed` oracle → run M1 (🟢) + M4 (🔴) → the first two matrix rows.
2. `scripts/model-matrix.py` + `models.toml` + the C1-C8 assertions + the MCP-responder → E2E for M1/M4.
3. Generate `docs/MODEL-COMPATIBILITY.md`; wire the no-GPU mock-replay CI tier.
4. Fill M2/M3/M5/M6/M7/M8 as models are pulled; add the manual GPU `workflow_dispatch` job.
