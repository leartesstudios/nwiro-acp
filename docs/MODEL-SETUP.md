# Model Setup Runbook

This runbook walks through installing prerequisites, downloading the three models, and configuring each serving provider for the `local-llm-acp` × Kimi/Qwen3/GLM model setup.

This runbook is the "how do I actually do each step?" reference for setting up models against the shim.

---

## ⚠️ Context length (`n_ctx`) requirement — read first

Post-v0.2.0 the UE5 bridge attaches the full Nwiro tool registry on every
`session/prompt`. With 100+ tool definitions in OpenAI spec format, the
serialised tool array is **~28-30K tokens** before any user content. If
your backend is loaded with `n_ctx` smaller than that, the request will
be refused at admission control:

```
n_keep: 29667 >= n_ctx: 4096
```

(v0.1.21 surfaces this error cleanly via the SSE-fallback path, but the
prompt still fails — you have to fix the backend.)

**Recommended `n_ctx` values for Nwiro Pro full-tool-array users**:

| Backend | Default `n_ctx` | Recommended | Notes |
|---|---|---|---|
| **LM Studio** | 4096 | **≥ 65536** | Model load dialog → "Context Length". 32K is a thin floor (observed `n_keep` was 29.7K). |
| **Ollama** | 2048 | **≥ 65536** | Modelfile: `PARAMETER num_ctx 65536`, or env: `OLLAMA_NUM_CTX=65536` |
| **llama.cpp server** | from `--ctx-size` flag | `--ctx-size 65536` | Watch VRAM — bigger context = bigger KV cache |
| **vLLM** | from `--max-model-len` flag | `--max-model-len 65536` | Must be ≤ model architectural max |

Model architectural ceilings (most chat-tuned models support these natively):
- Qwen 2.5 / 3: 32K base, 128K with YaRN
- GLM-4-9B-chat: 32K native
- Llama 3.1: 128K native (use a smaller load for VRAM efficiency)
- Kimi-K2: 128K native

**If you only have a small VRAM budget** and can't fit 64K context: ask
the bridge to send a *subset* of the tool registry (Fix C — bridge tool
partitioning, future work). Until that ships, 64K is the practical floor.

### ⚠️ Chat-template compatibility (v0.1.27+)

For the shim to use OpenAI-style tool calls, your inference backend
MUST load the model with a **chat template that supports OpenAI
tool_calls**. The most common failure mode is loading a model whose
default template was trained without tool-call awareness; the backend
then passes the tool array as plain text and the model
autoregressively echoes the JSON schema back as content tokens
(`"object", "object", "type": "object"` repeating).

**Symptom**: your UE5 chat shows pages of `"object"`, braces, quotes
instead of normal responses. No tool badges. Same prompt with Qwen3
or Claude works fine.

**Per-backend fix**:
- **LM Studio**: open the model's settings → "Prompt Template" tab →
  ensure the template supports tool-calls for your model family.
  GLM-4.5-air requires the GLM-tool-calls Jinja template; the default
  may not include tool-call support.
- **Ollama**: most Ollama Modelfiles ship with the correct template.
  If you customised the Modelfile, ensure the `TEMPLATE` block
  includes tool-call handling for your model family.
- **llama.cpp server**: pass `--chat-template <name>` matching your
  model family (e.g. `--chat-template chatglm4` for GLM-4).

**v0.1.27 defensive handling** (no env var needed in most cases):
- The shim's warmup probe (`probe_tool_capability`) now runs the
  Emulated parser against the probe response. Schema-bleeding output
  doesn't pass the parser → classifies as `ToolTier::None`.
- When effective tier is `None`, the shim STRIPS the tools array
  from outbound requests. This prevents the model from receiving
  25K tokens of tool JSON it can't process.
- Net result: even a misconfigured GLM should produce coherent chat
  output (without working tools), not the schema-bleed garbage.

**Escape hatch**: if you need to force a specific tier (e.g. for
testing or to paper over a probe misclassification), set:

```
NWIRO_LOCAL_LLM_FORCE_TOOL_TIER=none      # strip tools, plain chat
NWIRO_LOCAL_LLM_FORCE_TOOL_TIER=emulated  # force prose-extraction path
NWIRO_LOCAL_LLM_FORCE_TOOL_TIER=native    # force OpenAI tool_calls path
```

Unset = use probe result (default).

### ⚠️ GLM family (v0.1.28+) {#glm-family}

GLM-4 family models (`glm-4.5-air`, `GLM-4-9B-Chat`, `chatglm3-6b`,
all derivatives matching `glm`/`chatglm` case-insensitive) ship with
a non-standard chat template. They require:

- **System prefix**: `[gMASK]<sop><|system|>\n`
- **User wrapper**: `<|user|>\n…<|assistant|>\n`
- **Stop strings**: `<|user|>`, `<|endoftext|>`, `<|assistant|>`,
  `<|observation|>`, `<|system|>`

Without these markers, GLM models fail autoregressively on every
turn — even plain chat — producing pages of `"object": "object"`
schema fragments instead of coherent responses (the symptom from
the v0.1.27 / v0.1.28 user reports).

**v0.1.28 detection gate**: when the shim detects this symptom at
warmup time on a GLM-family model (substring match + schema-bleed
content signature), it now REFUSES the warmup with a structured
ACP error:

- `WarmupResult.status = "failed"`
- `WarmupResult.error_kind = "broken_chat_template"`
- `WarmupResult.message` contains the marker hints + a link back
  to this section

The session does NOT load; the operator sees an actionable error
instead of garbage in the UE5 chat.

**Per-backend fix for GLM**:

- **LM Studio (CORRECTED v0.1.30)**: the v0.1.29 LMS log diagnostic
  revealed the v0.1.28-documented preset has `beforeSystem` /
  `afterSystem` **inverted** — system content lands OUTSIDE the
  `<|system|>` marker, where GLM treats it as untagged pre-context
  prose. The correct preset puts the marker BEFORE the system content:
  ```
  beforeSystem: "[gMASK]<sop>\n<|system|>\n"
  afterSystem:  ""
  beforeUser:   "<|user|>\n"
  afterUser:    "<|assistant|>\n"
  ```
  Stop strings: `<|user|>`, `<|endoftext|>`, `<|assistant|>`,
  `<|observation|>`, `<|system|>`. Enter these values directly in
  LM Studio's prompt-template editor (apply the corrected marker
  order shown above manually).

  **Critical caveat (v0.1.30)**: LM Studio's "manual" prompt-template
  mode (which the GLM-4-9B-Chat "(Legacy)" preset uses by default)
  **silently drops the OpenAI `tools` array** when sending the request
  to the model. The model never receives a structured callable schema.
  Even with the corrected marker order above, Native-mode tool calling
  via the OpenAI `tools` field will NOT work — LM Studio's manual mode
  is a low-level pass-through. The shim works around this by forcing
  GLM-family to Emulated tier (v0.1.30 Native→Emulated downgrade,
  see below). For true Native function-calling support with GLM,
  switch to `llama.cpp` with `--chat-template chatglm4` or use a
  later GLM variant like GLM-4.5-Air on LM Studio's jinja template
  mode.

- **llama.cpp server**: pass `--chat-template chatglm4` when
  launching the server. The built-in template covers the markers
  above.

- **Ollama**: the default GLM Modelfile templates ship with the
  correct markers. If you customised, ensure `TEMPLATE` includes
  the `[gMASK]<sop>` system prefix and role tags.

**Bypass** (not recommended; only for testing or paving over false
detection):

```
NWIRO_LOCAL_LLM_BYPASS_TEMPLATE_GATE=1
```

Distinct from `NWIRO_LOCAL_LLM_FORCE_TOOL_TIER` because that one
runs at `session/prompt` (post-warmup), too late to bypass a
warmup-level refusal. With `BYPASS_TEMPLATE_GATE=1`, the gate is
skipped but the underlying chat-template problem remains — expect
the same schema-bleed output you would have seen before v0.1.28.

#### v0.1.29 — GLM action mandate (Native tier directive)

Even after the chat-template fix, GLM-4 family models exhibit a
second failure mode: the model reads the tool catalog from the
OpenAI `tools` array (it cites tool names verbatim if asked) but
**refuses to invoke them under `tool_choice: auto`**, defaulting
to descriptive prose like "you would follow these steps to create
a blueprint..."

Root cause: GLM-4 RLHF alignment training favors cautious-describer
behavior. There's no shim-side fix for the model's training, but
v0.1.29 injects a Native-tier system directive when the model
family is enrolled in `requires_tool_invocation_mandate`:

> You have access to these registered tools in this session: …
> When the user requests an action (create, edit, delete, find,
> list, generate, …), you MUST emit a tool_calls envelope with
> the appropriate tool and arguments. Do NOT describe what the
> tool would do. INVOKE the tool directly.

This fires only when ALL of:
- effective tier is `Native`
- model family matches a "describer-bias" entry (currently GLM-4 only)
- at least one tool is registered

The directive is appended to the existing system message
(`_meta.systemPrompt.append` from the bridge) rather than added
as a separate message — some models weight only the first system
message, so a single concatenation is more reliable than two.

**Token budget**: the v0.1.29 directive overhead estimator
(`estimate_directive_overhead`) accounts for both EMIT-004
(Emulated, ~113 tokens) and the new Native mandate (~156 tokens
base + comma-joined tool names). No new env var needed.

**If your model refuses tool calls and ISN'T GLM family**, the
directive doesn't fire. File a report with `RUST_LOG=info` logs
+ a few representative declined-prompts so we can decide whether
to enroll the family.

#### v0.1.30 — GLM Native→Emulated tier downgrade

The v0.1.29 LMS log stream diagnostic revealed the actual mechanism
behind GLM's tool-refusal: LM Studio's manual prompt-template mode
**silently drops the OpenAI `tools` array** at runtime. The probe
gets a false-positive Native classification because the probe forces
`tool_choice: required`, which causes LM Studio to synthesize a
tool_calls envelope from the backend layer regardless of whether the
model received a callable schema. But at session/prompt time, without
forced tool_choice, the tools array is dropped — the model only sees
tool names as prose in the system prompt and treats them as
documentation.

v0.1.30 routes GLM-family probe-Native classifications through the
Emulated tier instead. The shim's existing Emulated-tier
infrastructure (EMIT-002 inline JSON parser, EMIT-004 system-prompt
directive teaching `{"tool": "<name>", "arguments": {...}}` envelope
shape) was built precisely for this scenario: backends that can't
deliver structured tools + models that can format calls as prose JSON.

**Activation conditions** (all required):
- `effective_tool_tier == ToolTier::Native` from probe
- `ModelFamily::detect(&model).forces_emulated_tier() == true`
  (today: GLM-4 only)

**Override path** (for operators running GLM on a backend that DOES
support Native tool calls — e.g. llama.cpp with `--chat-template
chatglm4`):

```
NWIRO_LOCAL_LLM_FORCE_TOOL_TIER=native
```

This existing env var (v0.1.27) supersedes the warmup-level downgrade
at session/prompt time, bypassing the Emulated routing for setups
that have verified Native works.

**Path-form model IDs** (v0.1.30 fix): LM Studio passes the full GGUF
path as the model id over OpenAI-compat
(e.g. `legraphista/glm-4-9b-chat-GGUF/glm-4-9b-chat.Q4_K_S.gguf`).
v0.1.30 normalizes path-form ids to the final segment before family
detection, closing a gap where the v0.1.28-v0.1.29 family detection
silently missed.

**If Emulated also doesn't work for your GLM variant**: GLM-4-9B-Chat
(Legacy) may not reliably emit inline JSON either — its training is
biased toward describer responses. In that case, the structural
options are:
- Switch to **GLM-4.5-Air** (later variant with explicit
  function-calling fine-tune)
- Switch to **llama.cpp server** with `--chat-template chatglm4`
  (bypasses LM Studio's manual-template layer entirely)
- Wait for **v0.2.0** which will include an optional raw-completion
  bypass for backends that drop the tools array

### Shim-side defaults (v0.1.25+)

The shim's token-budget thresholds were re-calibrated in v0.1.25 to
assume backends configured with the recommended `n_ctx ≥ 65536`:

| Constant | Default | Override env var |
|---|---|---|
| `DEFAULT_WARN_TOKEN_THRESHOLD` | **32768** | `NWIRO_LOCAL_LLM_WARN_TOKEN_THRESHOLD` |
| `DEFAULT_PRUNE_TOKEN_THRESHOLD` | **28000** | `NWIRO_LOCAL_LLM_PRUNE_TOKEN_THRESHOLD` |
| Backstop (hard abort) | **2× warn = 65536** | (raises with warn) |

**If you must run with a smaller backend context** (e.g. n_ctx = 8192
for a tight VRAM budget), set both env vars DOWN proportionally:

```
NWIRO_LOCAL_LLM_WARN_TOKEN_THRESHOLD=4096
NWIRO_LOCAL_LLM_PRUNE_TOKEN_THRESHOLD=3500
```

This restores the v0.1.22-era defaults. The backstop will then fire
at 2×4096 = 8192, which means the shim refuses to send requests larger
than your backend's context anyway — saving 5-10s of slow refusal time
on backends without fast admission control.

**Pre-v0.1.25 incident**: v0.1.22-v0.1.24 shipped with these defaults
set for the legacy LM Studio default `n_ctx = 4096`. With the v0.2.0
bridge attaching ~25K tokens of tools on every prompt, the 2× backstop
(8192) blocked even "hello" prompts before they reached any backend.
v0.1.25 fixes this by aligning the defaults with the recommended n_ctx.

---

## Table of contents

1. [Hardware confirmation](#1-hardware-confirmation)
2. [Tool prerequisites](#2-tool-prerequisites)
3. [Directory layout](#3-directory-layout)
4. [Windows registry — TDR fix](#4-windows-registry--tdr-fix)
5. [Build the shim with `PROBE_TIMEOUT_SECS = 30`](#5-build-the-shim)
6. [Install llama.cpp server for Blackwell](#6-install-llamacpp-server)
7. [Download the three models](#7-download-the-three-models)
   - 7.1 Qwen3-30B-A3B
   - 7.2 GLM-4.5-Air
   - 7.3 Kimi-K2-Instruct (overnight)
8. [Configure each provider](#8-configure-each-provider)
   - 8.1 Ollama (Qwen3)
   - 8.2 LM Studio (GLM)
   - 8.3 llama.cpp server (Kimi)
9. [Worked example: running Cell A](#9-worked-example-cell-a-qwen3--ollama)
10. [Troubleshooting](#10-troubleshooting)
11. [References](#11-references)

---

## 1. Hardware confirmation

Before any downloads, confirm the box meets the recommended specs.

```powershell
nvidia-smi
```

Recommended:
- **GPU:** a Blackwell-class GPU with 48GB+ VRAM
- **Driver:** ≥595.x
- **CUDA:** ≥12.4 (CUDA 13.x is fine — forward compatible)

```powershell
# System RAM
(Get-CimInstance Win32_PhysicalMemory | Measure-Object -Property Capacity -Sum).Sum / 1GB

# Disk space on D: (where models will live)
(Get-PSDrive D).Free / 1GB
```

Expected:
- **RAM:** ≥127 GB
- **D: free:** ≥350 GB (256 GB models + ~100 GB headroom)

If any of these don't match, re-evaluate the model selection before proceeding — the picks below assume comparable RAM/VRAM/disk constraints.

---

## 2. Tool prerequisites

### 2.1 — Rust toolchain

Needed to rebuild the shim with the `PROBE_TIMEOUT_SECS` patch.

```powershell
# If not already installed
winget install Rustlang.Rustup
rustup default stable
rustup target add x86_64-unknown-linux-gnu   # for cross-compile invariant check
```

Verify: `cargo --version` returns 1.7x.x or newer.

### 2.2 — huggingface-cli

Used for all model downloads. Avoids `ollama pull`'s blob-store duplication.

```powershell
pip install -U "huggingface_hub[cli]"
huggingface-cli --version
```

If any models are gated, authenticate:
```powershell
huggingface-cli login
# Paste HF token from https://huggingface.co/settings/tokens
```

### 2.3 — Ollama

Already installed and running (verified via `nvidia-smi` process list). If not:
- Download from https://ollama.com/download/windows
- Run installer — it sets up the `:11434` listener and starts at boot

Verify:
```powershell
Invoke-RestMethod http://127.0.0.1:11434/v1/models
# Returns a JSON object with "data" array (possibly empty)
```

### 2.4 — LM Studio

Already installed (verified via `nvidia-smi`). If not:
- Download from https://lmstudio.ai
- Install and launch

The local OpenAI-compat server is OFF by default. Enable it in **Developer → Local Server** tab (more details in §8.2).

### 2.5 — llama.cpp server

Not installed by default. Detailed install in [§6](#6-install-llamacpp-server).

### 2.6 — Python (for test scripts)

The shim's smoke tests are Python:
```powershell
python --version   # 3.10 or newer
pip install requests   # only dependency for smoke tests
```

---

## 3. Directory layout

Single source of truth for model GGUFs (substitute your own models directory for `<MODELS_DIR>`):

```
<MODELS_DIR>\gguf\
├── kimi\
│   ├── Kimi-K2-Instruct-Q2_K_XL-00001-of-NNNN.gguf
│   ├── Kimi-K2-Instruct-Q2_K_XL-00002-of-NNNN.gguf
│   └── ...
├── qwen3\
│   └── Qwen3-30B-A3B-Q4_K_M.gguf
└── glm\
    └── GLM-4.5-Air-Q4_K_M.gguf
```

Create the structure once:
```powershell
New-Item -ItemType Directory -Force <MODELS_DIR>\gguf\kimi
New-Item -ItemType Directory -Force <MODELS_DIR>\gguf\qwen3
New-Item -ItemType Directory -Force <MODELS_DIR>\gguf\glm
```

**Critical rule:** all three providers read GGUFs from these paths via configuration. No copies. No duplicates.

**Optional:** point the HuggingFace cache at a dedicated directory to avoid filling your system drive:
```powershell
[Environment]::SetEnvironmentVariable("HF_HOME", "%USERPROFILE%\.cache\huggingface", "User")
# Restart PowerShell for the new env var to apply
```

---

## 4. Windows registry — TDR fix

Windows WDDM has a watchdog (Timeout Detection and Recovery — TDR) that kills GPU kernels exceeding 2 seconds. For 70B+ inference and partial-offload Kimi, this triggers silently, resetting the driver mid-test.

**Run PowerShell as Administrator:**
```powershell
reg add "HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" /v TdrDelay /t REG_DWORD /d 60 /f
reg add "HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" /v TdrDdiDelay /t REG_DWORD /d 60 /f

# Reboot required for changes to take effect
Restart-Computer
```

After reboot, verify:
```powershell
(Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers").TdrDelay
# Expected: 60
```

**Before running the Kimi model:** bump `TdrDelay` to `300`. Partial-offload Kimi first-token latency can exceed 60s, which would otherwise trip the GPU timeout-detection reset.

```powershell
reg add "HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" /v TdrDelay /t REG_DWORD /d 300 /f
Restart-Computer
# After Kimi cell: restore to 60
reg add "HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" /v TdrDelay /t REG_DWORD /d 60 /f
Restart-Computer
```

---

## 5. Build the shim

The shim's `PROBE_TIMEOUT_SECS = 5` default is too short for any partial-offload model — the warmup probe times out before the model produces its first token, causing a false `ToolTier::None` classification.

### 5.1 — Apply the patch

Open `src/openai/client.rs` and change line 31:

```rust
// Before:
const PROBE_TIMEOUT_SECS: u64 = 5;

// After:
const PROBE_TIMEOUT_SECS: u64 = 30;
```

### 5.2 — Build

```powershell
cd path\to\local-llm-acp
cargo build --release
```

Verify: `target\release\local-llm-acp.exe` exists.

### 5.3 — Verify the rustls-only invariant

```powershell
cargo tree -e features | findstr "native-tls"
# Expected: zero matches
cargo tree -e features | findstr "rustls"
# Expected: at least one match
```

`native-tls` in the dep tree breaks cross-compile to `aarch64-pc-windows-msvc`. If it appears, see `STRUCTURE.md` for the dependency policy.

### 5.4 — Smoke tests against unreachable host (no models needed)

```powershell
python reports\smoke-test.py
python reports\smoke-test-model-switch.py
```

Both must print `=== ALL CHECKS PASSED ===`. If not, the shim binary is broken — don't proceed.

---

## 6. Install llama.cpp server

llama.cpp's `llama-server.exe` is the third provider. It needs CUDA support targeting Blackwell sm_120.

### 6.1 — Try pre-built (preferred)

1. Open https://github.com/ggml-org/llama.cpp/releases
2. Find the latest release with a Windows CUDA asset, named like:
   - `llama-<version>-bin-win-cuda-cu12.x-x64.zip`
   - `llama-<version>-bin-win-cuda-cu13.x-x64.zip` (if available)
3. Download and extract to `C:\llama.cpp\`
4. Verify Blackwell detection:
   ```powershell
   C:\llama.cpp\llama-server.exe --version
   # Should print "CUDA device: <your Blackwell-class GPU>"
   ```

If `llama-server.exe` runs but falls back to CPU (doesn't print the GPU name) — the pre-built binary doesn't have sm_120 kernels compiled in. Use the source build (§6.2).

### 6.2 — Build from source (fallback for Blackwell)

Prerequisites: Visual Studio Build Tools (C++ workload), CUDA toolkit 12.4+, CMake 3.20+, Git.

```powershell
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git C:\llama.cpp-src
cd C:\llama.cpp-src
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="120" -DLLAMA_CURL=OFF
cmake --build build --config Release --target llama-server -j 8
# Output: C:\llama.cpp-src\build\bin\Release\llama-server.exe
# Copy or alias to C:\llama.cpp\llama-server.exe
```

The `-DCMAKE_CUDA_ARCHITECTURES="120"` is critical — it compiles Blackwell-targeted kernels rather than relying on PTX JIT.

### 6.3 — Verify the install

```powershell
C:\llama.cpp\llama-server.exe --help | findstr "n-gpu-layers"
# Should show: -ngl, --gpu-layers, --n-gpu-layers
```

---

## 7. Download the three models

Run downloads sequentially, smallest first — gives you fast feedback if `huggingface-cli` has issues before committing to the 4-hour Kimi pull.

### 7.1 — Qwen3-30B-A3B (~19 GB, ~10-25 min)

```powershell
huggingface-cli download bartowski/Qwen3-30B-A3B-GGUF `
  --include "*Q4_K_M*.gguf" `
  --local-dir <MODELS_DIR>\gguf\qwen3 `
  --resume-download
```

After completion:
```powershell
Get-ChildItem <MODELS_DIR>\gguf\qwen3
# Expected: Qwen3-30B-A3B-Q4_K_M.gguf, ~18-20 GB
```

### 7.2 — GLM-4.5-Air (~66 GB, ~1-1.5 hr)

**Verify publisher first.** GLM-4.5-Air GGUF availability varies. Search https://huggingface.co/models?search=GLM-4.5-Air+GGUF for current options. Common publishers: `bartowski`, `THUDM`, `ddh0`. Update the repo in the command below before running.

```powershell
# Update <publisher> based on HF search:
huggingface-cli download <publisher>/GLM-4.5-Air-GGUF `
  --include "*Q4_K_M*.gguf" `
  --local-dir <MODELS_DIR>\gguf\glm `
  --resume-download
```

After completion:
```powershell
Get-ChildItem <MODELS_DIR>\gguf\glm
# Expected: GLM-4.5-Air-Q4_K_M.gguf (single file ~66 GB) OR
# GLM-4.5-Air-Q4_K_M-00001-of-N.gguf shards (sum to ~66 GB)
```

**Multi-shard verification (if applicable):**
```powershell
# Count vs. expected shard count from the *-of-NNNN suffix in filenames
(Get-ChildItem <MODELS_DIR>\gguf\glm\*.gguf).Count
```

### 7.3 — Kimi-K2-Instruct Q2_K_XL (~172 GB, ~3-4 hr — overnight)

```powershell
huggingface-cli download unsloth/Kimi-K2-Instruct-GGUF `
  --include "*Q2_K_XL*.gguf" `
  --local-dir <MODELS_DIR>\gguf\kimi `
  --resume-download
```

`--resume-download` is essential here — at 3-4 hours, a network blip is likely. Resume picks up where it left off and re-validates partial shards.

After completion:
```powershell
Get-ChildItem <MODELS_DIR>\gguf\kimi
# Expected: multiple shards, sum ~172 GB
# Filenames follow Kimi-K2-Instruct-Q2_K_XL-00001-of-NNNN.gguf pattern
```

**Verify shard count matches the filename suffix:**
```powershell
$files = Get-ChildItem <MODELS_DIR>\gguf\kimi\*.gguf
$files.Count
# Compare to the "-of-NNNN" in the filenames. If 8 files but filenames say "-of-00008", you're complete.
# If 7 files exist but suffix says "-of-00008", one is missing — re-run download.
```

**Disk check after Kimi:**
```powershell
(Get-PSDrive D).Free / 1GB
# Should still have ~100+ GB free (you started with ~1600, used ~257)
```

---

## 8. Configure each provider

### 8.1 — Ollama (for Qwen3)

Ollama reads GGUFs via Modelfile `FROM` directives. **Do NOT use `ollama pull`** — that copies the model into Ollama's blob store at `%USERPROFILE%\.ollama\models\blobs\` and doubles your disk usage.

**Create a Modelfile** at `<MODELS_DIR>\gguf\qwen3\Modelfile-qwen3-30b-a3b`:

```
FROM <MODELS_DIR>\gguf\qwen3\Qwen3-30B-A3B-Q4_K_M.gguf
PARAMETER num_ctx 65536
PARAMETER keep_alive 15m
```

> ⚠️ **n_ctx must be ≥ 65536 for the Nwiro Pro full tool array (v0.1.25+).**
> The earlier `num_ctx 8192` value documented here was sized for the
> pre-v0.2.0 bridge that didn't auto-attach tools on every prompt.
> If you're on a tight VRAM budget and can't fit 65536, see the
> "Shim-side defaults" section above — you can lower the shim's
> warn/prune env vars to match a smaller backend ceiling, but the
> backend itself must still accept whatever payload the bridge sends.

**Register the model with Ollama:**
```powershell
ollama create qwen3-30b-a3b -f <MODELS_DIR>\gguf\qwen3\Modelfile-qwen3-30b-a3b
```

Ollama symlinks the GGUF — no copy.

**Test load:**
```powershell
ollama run qwen3-30b-a3b "Say hi in one word"
# Should respond quickly. Tokens/sec is logged at exit if you Ctrl+C.
```

**Verify model is registered:**
```powershell
ollama list
# Should show qwen3-30b-a3b in the list
```

**Verify VRAM usage:**
```powershell
nvidia-smi --query-gpu=memory.used --format=csv,noheader
# Should show ~19 GB used (while model is loaded)
```

**Unload when done:**
```powershell
ollama stop qwen3-30b-a3b
# Or wait keep_alive=15m for auto-unload
```

### 8.2 — LM Studio (for GLM-4.5-Air)

LM Studio uses a GUI for model management.

**Add the model to LM Studio:**
1. Open LM Studio
2. **My Models** tab → "+" (Add) or menu → "Add Local Model"
3. Navigate to `<MODELS_DIR>\gguf\glm\`
4. Select the GLM-4.5-Air GGUF file (or the first shard if sharded)
5. LM Studio registers the model

**Load the model:**
1. Click the loaded model selector at the top
2. Choose GLM-4.5-Air
3. In the load settings dialog:
   - **n_gpu_layers:** set to a high number (e.g. 99) — model fits entirely in VRAM
   - **Context length:** **65536 minimum** for Nwiro Pro full-tool-array users
     (v0.1.25+). The earlier "16384 minimum" was sized for GLM testing
     before v0.2.0 bridge auto-attached tools.
4. Click **Load Model**
5. Watch the VRAM gauge — should land at ~66 GB

**Enable the OpenAI-compat server:**
1. **Developer** tab → **Local Server**
2. Confirm GLM-4.5-Air is the active model
3. Click **Start Server** — listens on `:1234` by default
4. Verify:
   ```powershell
   Invoke-RestMethod http://127.0.0.1:1234/v1/models
   # Returns JSON with GLM-4.5-Air in "data" array
   ```

**Manual chat-template sanity check (critical for GLM):**
```powershell
$body = @{
  model = "glm-4.5-air"
  messages = @(@{role="user"; content="Reply with exactly: hello"})
  stream = $false
} | ConvertTo-Json
Invoke-RestMethod -Method POST -Uri http://127.0.0.1:1234/v1/chat/completions -Body $body -ContentType "application/json"
```

The response should contain "hello" cleanly — no garbled tokens, no `<|user|>` literals leaking. If garbled, the chat template is wrong — see [§10.2 troubleshooting](#102--glm-chat-template-fallback).

**Eject when done:** click the loaded model name at top → "Eject". Or close LM Studio entirely.

### 8.3 — llama.cpp server (for Kimi)

llama.cpp is launched manually per cell.

**Launch command for Kimi:**

```powershell
C:\llama.cpp\llama-server.exe `
  --model "<MODELS_DIR>\gguf\kimi\Kimi-K2-Instruct-Q2_K_XL-00001-of-NNNN.gguf" `
  --n-gpu-layers 34 `
  --ctx-size 4096 `
  --jinja `
  --port 8080 `
  --host 127.0.0.1 `
  --threads 16 `
  --parallel 1
```

Replace `00001-of-NNNN` with the actual first-shard filename. llama.cpp auto-loads all shards in the same directory.

**Tuning `--n-gpu-layers`:**

The right value puts ~80 GB in VRAM (leaves headroom for KV cache). Start at 34, monitor with `nvidia-smi`, adjust:

| nvidia-smi shows | Action |
|---|---|
| `OOM` at load | Reduce by 4: `--n-gpu-layers 30`, retry |
| < 70 GB used | Increase by 4: `--n-gpu-layers 38`, retry |
| 80-94 GB used | Sweet spot — keep this value |

**Wait for the load:**
First load reads ~172 GB from disk and shards it across VRAM + RAM. Expect 5-20 minutes. Watch for log line:
```
all slots are idle
```
That means model is loaded and server is accepting requests.

**Verify:**
```powershell
Invoke-RestMethod http://127.0.0.1:8080/v1/models
# Returns JSON with model name
```

**Flags reference:**

| Flag | Purpose |
|---|---|
| `--jinja` | Required for OpenAI-spec tool-call streaming. Without it, tool calls use a non-spec format. |
| `--ctx-size 4096` | Conservative for Kimi partial offload — KV cache memory grows with context |
| `--threads 16` | CPU threads for the offloaded layers' compute |
| `--parallel 1` | One concurrent request — single-operator setup doesn't need parallelism |
| `--n-gpu-layers N` | Layers on GPU. The rest live in CPU/RAM. |
| `--chat-template <name>` | Override chat template if GGUF metadata is incorrect. Common: `chatml`, `llama3`, `kimi`. |
| `--chat-template-file <path>` | Use a custom Jinja2 template from disk |

**Stop:** Ctrl+C in the terminal where it's running.

---

## 9. Worked example: Cell A (Qwen3 × Ollama)

Once everything in §1-8 is done, this is what running the smallest, fastest cell looks like end-to-end.

### Step 1 — Verify pre-flight
```powershell
# Shim built with PROBE_TIMEOUT_SECS=30?
Select-String -Path src\openai\client.rs -Pattern "PROBE_TIMEOUT_SECS"
# Should show: const PROBE_TIMEOUT_SECS: u64 = 30;

# TDR delay set?
(Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers").TdrDelay
# Should show: 60

# Ollama responding?
Invoke-RestMethod http://127.0.0.1:11434/v1/models | ConvertTo-Json -Compress

# Qwen3 model registered?
ollama list | findstr qwen3-30b-a3b
```

### Step 2 — Load Qwen3 model
```powershell
# Pre-warm — first response takes 5-15 seconds while model loads into VRAM
ollama run qwen3-30b-a3b "test" --keepalive 15m
# Press Enter to exit the prompt; model stays loaded for 15 min
```

### Step 3 — Set shim env and run test
```powershell
$env:NWIRO_LOCAL_LLM_BASE_URL = "http://127.0.0.1:11434/v1"
$env:NWIRO_LOCAL_LLM_MODEL    = "qwen3-30b-a3b"
$env:RUST_LOG                 = "debug"

# Test script (to be written — extends smoke-test.py pattern)
python scripts\test-cell-qwen3.py 2> logs\qwen3-debug.log
```

The script drives `target\release\local-llm-acp.exe` via subprocess, sends ACP requests, captures frames, asserts:
- Warmup returns `toolTier == "native"`
- Tool call to `sum2(a=20, b=22)` produces correct `tool_calls` JSON
- Streaming chunks arrive incrementally
- No `<think>` tokens leak into ACP content
- Final answer contains `42`

### Step 4 — Teardown and review
```powershell
ollama stop qwen3-30b-a3b
nvidia-smi --query-gpu=memory.used --format=csv,noheader
# Should drop back to ~1-2 GB (just desktop apps)

# Aggregate the raw debug log into a per-cell summary
python scripts\aggregate-day-results.py `
  --raw logs\qwen3-debug.log `
  --output qwen3-summary.json `
  --cell A --family qwen3 --provider ollama
```

Review the summarized results. If the cell passed → proceed to GLM (§8.2). If a blocker shows up → diagnose before moving on.

---

## 10. Troubleshooting

### 10.1 — WDDM TDR fires mid-inference

**Symptom:** Inference crashes with "connection reset" or a CUDA error. Display momentarily blacks out. Ollama/LM Studio/llama.cpp loses its model state.

**Cause:** A single GPU kernel exceeded `TdrDelay` seconds.

**Fix:**
```powershell
# As Administrator
reg add "HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" /v TdrDelay /t REG_DWORD /d 300 /f
Restart-Computer
```

For Kimi cell specifically, you should already have `TdrDelay = 300`. After Kimi, restore to 60.

### 10.2 — GLM chat-template fallback

**Symptom:** GLM model responds with garbled output, or tool calls return prose with literal `<|assistant|>` tokens instead of valid `tool_calls` JSON.

**Cause:** The GGUF metadata's embedded chat template is missing or wrong; the provider falls back to generic ChatML which GLM doesn't speak.

**Fix for LM Studio:**
1. Open the GLM model in LM Studio's "My Models" view
2. Click the gear icon → "Prompt Template" tab
3. Manually set the template to GLM-4 format (paste from the model card on HuggingFace)
4. Reload the model

**Fix for llama.cpp (if you decide to host GLM there instead):**
```powershell
llama-server.exe --model ... --jinja --chat-template-file glm.jinja
```
Where `glm.jinja` is the template from the model card.

### 10.3 — Multi-shard partial download

**Symptom:** Model loads but produces gibberish. Or load fails with "failed to read tensor" mid-init.

**Cause:** One of the GGUF shards was truncated (network drop, disk full mid-write).

**Fix:**
```powershell
# Re-run the original huggingface-cli download command with --resume-download
# It re-validates partial files and only redownloads broken shards
huggingface-cli download <repo> --include "<pattern>" --local-dir <dir> --resume-download
```

**Prevention:** verify shard count after download:
```powershell
$expected = 8   # read from the *-of-NNNN suffix on filenames
$actual = (Get-ChildItem <MODELS_DIR>\gguf\kimi\*.gguf).Count
if ($actual -ne $expected) { Write-Warning "Missing shards: $actual of $expected" }
```

### 10.4 — Port collision

**Symptom:** A provider's server fails to start with "address already in use" or hangs.

**Diagnosis:**
```powershell
Get-NetTCPConnection -State Listen | Where-Object { $_.LocalPort -in 1234,8080,11434 } |
  Select-Object LocalAddress,LocalPort,OwningProcess
# Identify the PID holding the port
Get-Process -Id <PID>
```

**Resolution:**
- If a stale provider instance: kill it (`Stop-Process -Id <PID>`)
- If genuinely conflicting service: change the port for that provider and update the test scripts accordingly

### 10.5 — VRAM contention between cells

**Symptom:** Loading the second cell's model fails with CUDA OOM, even though `nvidia-smi` showed enough free VRAM before.

**Cause:** Previous provider's model is still resident. Ollama defaults to a 5-15 minute keep_alive; LM Studio holds models until manually ejected.

**Fix:**
```powershell
# For Ollama:
ollama stop <model-name>

# For LM Studio: click the loaded model name at top → "Eject"
# Or close LM Studio entirely

# For llama.cpp: Ctrl+C in its terminal

# Verify VRAM cleared:
nvidia-smi --query-gpu=memory.used --format=csv,noheader
# Should drop to ~1-2 GB (just desktop apps)
```

### 10.6 — Blackwell kernel fallback (silent slowness)

**Symptom:** Tokens/sec is 5-10x lower than expected. No error thrown.

**Cause:** llama.cpp / Ollama wasn't built with sm_120 kernels; falls back to PTX JIT compilation or generic code paths.

**Diagnosis:**
```powershell
# Quick benchmark — should be >50 tok/s on a small model
ollama run llama3.2:3b "Write one short sentence."
# Check Ollama logs (the run will display tokens/sec at exit)
```

**Fix:**
- Update Ollama: download fresh from https://ollama.com (Blackwell support landed in 0.7+)
- Rebuild llama.cpp from source with `-DCMAKE_CUDA_ARCHITECTURES=120` (§6.2)

### 10.7 — `huggingface-cli` rate limit

**Symptom:** Download stalls or returns 429. Common on large multi-shard pulls.

**Fix:** `huggingface-cli` auto-resumes via `--resume-download`. Wait a few minutes and re-run the command — it picks up where it stopped.

### 10.8 — `PROBE_TIMEOUT_SECS` not picked up

**Symptom:** Probe returns `ToolTier::None` even on capable models; no apparent reason.

**Cause:** Shim wasn't rebuilt after editing `client.rs:31`. Or PowerShell is running an older `.exe`.

**Fix:**
```powershell
# Confirm the source change is in
Select-String -Path src\openai\client.rs -Pattern "PROBE_TIMEOUT_SECS"
# Should show 30, not 5

# Rebuild
cargo build --release

# Verify the binary is fresh
(Get-Item target\release\local-llm-acp.exe).LastWriteTime
# Should be after your edit

# Use full path in test scripts to avoid PATH stale binary
```

---

## 11. References

### HuggingFace repos (verify current names before downloading)

| Model | Likely repo |
|---|---|
| Kimi-K2-Instruct Q2_K_XL | `unsloth/Kimi-K2-Instruct-GGUF` |
| Qwen3-30B-A3B Q4_K_M | `bartowski/Qwen3-30B-A3B-GGUF` |
| GLM-4.5-Air Q4_K_M | search `GLM-4.5-Air-GGUF` on HF — publisher varies |

### Tools

- llama.cpp releases: https://github.com/ggml-org/llama.cpp/releases
- Ollama: https://ollama.com/download/windows
- LM Studio: https://lmstudio.ai
- HuggingFace CLI: https://huggingface.co/docs/huggingface_hub/guides/cli

### llama.cpp docs

- Server: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- Function calling / `--jinja`: https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md
- GGUF Modelfile format: https://docs.ollama.com/modelfile

### Shim source landmarks

| Path | What lives there |
|---|---|
| `src/openai/client.rs:31` | `PROBE_TIMEOUT_SECS` constant (the one you patched) |
| `src/openai/client.rs:51-56` | `localhost → 127.0.0.1` normalization (Windows IPv6 fix) |
| `src/openai/client.rs:412-413` | Warmup probe success check (status + has_error_field, not finish_reason) |
| `src/openai/client.rs:605-633` | `accumulate_delta` — fragmented `tool_calls` assembly |
| `src/openai/messages.rs:189-207` | `Delta::reasoning_token()` — Qwen3 `reasoning_content` handling |
| `src/acp/server.rs` | `write_mcp_stub` — returns `-32601` for Phase 3 stub |
| `src/bridge/mod.rs:22-24` | `REFUSAL_MESSAGE` constant |

---

## 12. Quick reference (cheat sheet)

Compact commands for experienced operators. Copy/paste against your active shell. PowerShell-style.

### Verify everything before a test run

```powershell
# Shim patch in place?
Select-String -Path src\openai\client.rs -Pattern "PROBE_TIMEOUT_SECS" | Where-Object { $_.Line -notmatch "^\s*//" }
# TDR delay
(Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers").TdrDelay
# Provider health
Invoke-RestMethod http://127.0.0.1:11434/v1/models  # Ollama
Invoke-RestMethod http://127.0.0.1:1234/v1/models   # LM Studio (must be started in UI first)
Invoke-RestMethod http://127.0.0.1:8080/v1/models   # llama.cpp (must be launched manually)
# VRAM baseline
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
# Port collision
Get-NetTCPConnection -State Listen | Where-Object { $_.LocalPort -in 1234,8080,11434 } | Select LocalAddress,LocalPort,OwningProcess
```

### Download a model

```powershell
huggingface-cli download <repo> --include "<pattern>*.gguf" --local-dir <MODELS_DIR>\gguf\<family> --resume-download
# Verify shard count if multi-shard
(Get-ChildItem <MODELS_DIR>\gguf\<family>\*.gguf).Count
```

### Configure & load each provider

| Provider | Configure | Bring up | Health check | Tear down |
|---|---|---|---|---|
| **Ollama** | `ollama create <name> -f <Modelfile>` | `ollama run <name> "test" --keepalive 15m` | `ollama list` | `ollama stop <name>` |
| **LM Studio** | UI: Add Local Model → pick GGUF | UI: Local Server → Start | `Invoke-RestMethod http://127.0.0.1:1234/v1/models` | UI: Eject Model |
| **llama.cpp** | (none) | `llama-server.exe --model <file> --n-gpu-layers N --jinja --port 8080 --host 127.0.0.1 --threads 16 --parallel 1` | `Invoke-RestMethod http://127.0.0.1:8080/v1/models` | Ctrl+C |

### Set shim env for a cell

```powershell
$env:NWIRO_LOCAL_LLM_BASE_URL = "http://127.0.0.1:<port>/v1"
$env:NWIRO_LOCAL_LLM_MODEL    = "<model-tag>"
$env:RUST_LOG                 = "debug"
# Don't set NWIRO_LOCAL_LLM_API_KEY_localllm — local providers don't need auth
```

### Registry — TdrDelay (Administrator PowerShell)

```powershell
# Normal:
reg add "HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" /v TdrDelay /t REG_DWORD /d 60 /f
# Before Kimi:
reg add "HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" /v TdrDelay /t REG_DWORD /d 300 /f
# Reboot required after change
Restart-Computer
```

### Per-cell review

After each cell, aggregate its debug log into a per-cell summary and review the
results before moving to the next cell (the Phase 3 MCP stub returns `-32601`
intentionally — that is expected, not a failure).

### Cell-by-cell ports + commands at a glance

| Cell | Provider | Base URL | Key command |
|---|---|---|---|
| A — Qwen3 | Ollama | `http://127.0.0.1:11434/v1` | `ollama run qwen3-30b-a3b` |
| B — GLM | LM Studio | `http://127.0.0.1:1234/v1` | (load via GUI) |
| C — Kimi | llama.cpp | `http://127.0.0.1:8080/v1` | `llama-server.exe --n-gpu-layers 34 --jinja --port 8080 ...` |
