# Running `local-llm-acp`

Cross-platform guide for running and smoke-testing the shim binary against any OpenAI-compatible local LLM provider. Works the same way on Windows, macOS, and Linux.

> **You do NOT need the UE5 plugin, the C++ bridge, or any game engine to follow this doc.** The shim is a self-contained binary that reads ACP JSON-RPC over stdin and writes to stdout. In production it's spawned as a child process by the Nwiro Integration Kit bridge — this doc is for **manual / standalone** runs.

---

## Which document do I need?

| If you want to... | Read this |
|---|---|
| Understand what this crate is, build from source, or read security notes | [`README.md`](../README.md) |
| Run the binary against your local provider on your laptop / VM / CI runner | **this file** |
| Reproduce the Blackwell-specific model setup (Kimi/Qwen3/GLM, large-VRAM GPU, TDR registry tuning) | [`docs/MODEL-SETUP.md`](MODEL-SETUP.md) |
| Understand the release packaging contract or filename convention | [`RELEASING.md`](../RELEASING.md) |
| Understand the code layout, ACP message set, security boundary | [`STRUCTURE.md`](../STRUCTURE.md) |

End users of the Nwiro UE5 plugin never read this. The bridge auto-downloads and launches the shim for them; their docs live in the Nwiro Integration Kit repository, not here.

---

## Table of contents

1. [Quick-start checklist (30 seconds, all platforms)](#1-quick-start-checklist)
2. [Provider support matrix](#2-provider-support-matrix)
3. [Get the binary](#3-get-the-binary)
4. [Install a provider](#4-install-a-provider)
5. [Environment variables](#5-environment-variables)
6. [Run the smoke tests](#6-run-the-smoke-tests)
7. [OS-specific notes](#7-os-specific-notes)
8. [Common gotchas](#8-common-gotchas)
9. [When you actually need MODEL-SETUP.md](#9-when-you-actually-need-model-setupmd)

---

## 1. Quick-start checklist

The fastest possible "does the shim work on this box?" test. Ollama because it's one-click on every OS. ~30 seconds once Ollama is installed.

```
1. Install Ollama:              https://ollama.com/download
2. Pull a tiny model:           ollama pull llama3.2:3b
3. Download the shim binary:    see §3 for your OS
4. Set env vars:                see §5 for PowerShell / bash syntax
                                NWIRO_LOCAL_LLM_BASE_URL=http://127.0.0.1:11434/v1
                                NWIRO_LOCAL_LLM_MODEL=llama3.2:3b
5. Run a smoke test:            python scripts/smoke-test.py
6. Expected output:             === ALL CHECKS PASSED ===
```

If step 6 prints `=== ALL CHECKS PASSED ===`, the shim works on this machine. The rest of this doc is reference material — read what you need.

---

## 2. Provider support matrix

| Provider | Windows | macOS | Linux | Notes |
|---|:---:|:---:|:---:|---|
| **Ollama** | ✅ | ✅ | ✅ | Reference provider for this doc. Simplest install. |
| **LM Studio** | ✅ | ✅ | ✅ | GUI-driven. Has "Local Server" toggle for OpenAI-compat endpoint. |
| **llama.cpp server** | ✅ | ✅ | ✅ | Bare-metal `llama-server` binary. Most config flexibility. |
| **vLLM** | ❌ | ❌ | ✅ | Linux + NVIDIA only. Not available natively on Windows or macOS. |
| **Any OpenAI-compat HTTP server** | ✅ | ✅ | ✅ | If it speaks `/v1/chat/completions`, the shim talks to it. |

Pick whichever provider you already have running. The shim doesn't care which one as long as it speaks OpenAI-compatible HTTP.

---

## 3. Get the binary

Released binaries live on the [GitHub releases page](https://github.com/leartesstudios/nwiro-acp/releases). The filename convention (source-of-truth: [`RELEASING.md`](../RELEASING.md)) is `local-llm-acp-<VERSION>-<TARGET>.<EXT>` — `<VERSION>` is the release version with no `v` prefix, `<TARGET>` is the Rust target triple, and `<EXT>` is `zip` (Windows) or `tar.gz` (macOS/Linux).

To build from source instead, see [`README.md`](../README.md). This section assumes you're downloading a release asset. Substitute the latest release version for `<VERSION>` in the examples below.

### Windows

Pick the right archive:
- **x64 (most desktops/laptops):** `local-llm-acp-<VERSION>-x86_64-pc-windows-msvc.zip`
- **arm64 (Surface Pro X, ARM-based Windows):** `local-llm-acp-<VERSION>-aarch64-pc-windows-msvc.zip`

```powershell
# Example: extract to C:\tools\local-llm-acp\
Expand-Archive -Path .\local-llm-acp-<VERSION>-x86_64-pc-windows-msvc.zip -DestinationPath C:\tools\local-llm-acp
# The binary is at:
C:\tools\local-llm-acp\local-llm-acp.exe
```

### macOS

Apple Silicon only (M1/M2/M3/M4). The x86_64 (Intel) build was dropped in v0.1.13; Intel Macs can build from source — see [`README.md`](../README.md).
- **Apple Silicon:** `local-llm-acp-<VERSION>-aarch64-apple-darwin.tar.gz`

```bash
mkdir -p ~/tools/local-llm-acp
tar xzf local-llm-acp-<VERSION>-aarch64-apple-darwin.tar.gz -C ~/tools/local-llm-acp
# The binary is at:
~/tools/local-llm-acp/local-llm-acp
# Apple Gatekeeper may quarantine the binary on first run.
# If you see "cannot be opened because the developer cannot be verified":
xattr -d com.apple.quarantine ~/tools/local-llm-acp/local-llm-acp
```

### Linux

- **x64 (most servers/desktops):** `local-llm-acp-<VERSION>-x86_64-unknown-linux-gnu.tar.gz`
- **arm64 (Raspberry Pi 5, AWS Graviton, etc.):** `local-llm-acp-<VERSION>-aarch64-unknown-linux-gnu.tar.gz`

```bash
mkdir -p ~/tools/local-llm-acp
tar xzf local-llm-acp-<VERSION>-x86_64-unknown-linux-gnu.tar.gz -C ~/tools/local-llm-acp
chmod +x ~/tools/local-llm-acp/local-llm-acp
# The binary is at:
~/tools/local-llm-acp/local-llm-acp
```

---

## 4. Install a provider

You only need *one* provider to test the shim. Pick the easiest for your situation.

### Ollama (recommended for first-time setup, all OSes)

Install from https://ollama.com/download. It runs as a background service on `http://127.0.0.1:11434`. Use [`/v1/...`] paths to hit its OpenAI-compatible surface.

Pull a model:
```bash
ollama pull llama3.2:3b           # tiny, fast, good for smoke tests
ollama pull qwen2.5:14b           # bigger, tool-capable
```

Model lifecycle:
```bash
ollama list                       # what's downloaded
ollama run <name> "say hi"        # ad-hoc test
ollama stop <name>                # unload from VRAM
```

For loading custom GGUFs via Modelfile, see [`MODEL-SETUP.md §8.1`](MODEL-SETUP.md#81--ollama-for-qwen3).

### LM Studio (GUI-driven, all OSes)

Download from https://lmstudio.ai. The OpenAI-compat server is off by default — enable it in the **Developer** → **Local Server** tab. Default port `:1234`.

Use case: visual model loading, KV cache observation, easier troubleshooting when you're not sure if a model is actually loaded. See [`MODEL-SETUP.md §8.2`](MODEL-SETUP.md#82--lm-studio-for-glm-45-air) for the GLM-specific workflow.

### llama.cpp server (most control, all OSes)

Pre-built binaries: https://github.com/ggml-org/llama.cpp/releases — pick the latest with CUDA support for your platform.

Basic launch (substitute your model path and port):
```bash
llama-server --model /path/to/model.gguf --port 8080 --host 127.0.0.1 --jinja
```

**Critical flag:** `--jinja` is required for tool-call streaming. Without it, the shim's tool-tier probe will misclassify capable models.

For GPU partial offload tuning and Blackwell-specific source builds, see [`MODEL-SETUP.md §6`](MODEL-SETUP.md#6-install-llamacpp-server) and [`§8.3`](MODEL-SETUP.md#83--llamacpp-server-for-kimi).

### vLLM (Linux + NVIDIA only)

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server --model <hf-repo> --port 8000
```

Not supported on Windows or macOS natively. WSL2 + CUDA passthrough is possible but bleeding-edge — see [`MODEL-SETUP.md §1`](MODEL-SETUP.md#1-serving-stack-decision) for the trade-off analysis.

---

## 5. Environment variables

The shim reads its entire configuration from the environment. No CLI args. No config files.

| Variable | Type | Default | Required? | Purpose |
|---|---|---|---|---|
| `NWIRO_LOCAL_LLM_BASE_URL` | URL | — | **yes** | OpenAI-compatible endpoint base, e.g. `http://127.0.0.1:11434/v1` |
| `NWIRO_LOCAL_LLM_MODEL` | string | — | **yes** | Model identifier the provider exposes (e.g. `llama3.2:3b`, `qwen3-30b-a3b`) |
| `NWIRO_LOCAL_LLM_API_KEY_localllm` | string | empty | only if provider requires auth | API key; `Debug` impl prints `[REDACTED]`, never logged. **Cloud backends (RunPod etc.) require this** — see [Remote / cloud endpoints](#remote--cloud-endpoints-runpod--serverless-gpu) below. |
| `NWIRO_LOCAL_LLM_PROBE_TIMEOUT_SECS` | integer | `10` | recommended for partial-offload & cloud | Tool-capability probe HTTP timeout. Bump to `30`+ for large models that page weights from RAM on first token, or `120` for serverless cold starts. |
| `NWIRO_LOCAL_LLM_CONNECT_TIMEOUT_SECS` | integer | `10` | optional | TCP connect timeout for every backend request. Raise for high-latency cloud endpoints. (Caps *connect* only — the warmup load request is bounded separately by `WARMUP_TIMEOUT_SECS` below.) |
| `NWIRO_LOCAL_LLM_WARMUP_TIMEOUT_SECS` | integer | `300` | recommended for serverless | Total timeout (connect + headers + body) for the warmup model-load request. On expiry warmup fails fast with `errorKind=timeout` instead of hanging the UE5 spinner. `0` disables the cap (old unbounded behavior); invalid/negative values fall back to the default. Raise to `600`+ for uncached serverless cold loads. |
| `NWIRO_LOCAL_LLM_MAX_RESPONSE_BYTES` | integer | `8388608` (8 MiB) | optional | Hard cap on total accumulated response size (content + tool-call args). A runaway / repetition loop aborts cleanly instead of growing until the editor OOMs. `0` disables. |
| `NWIRO_LOCAL_LLM_MAX_TURN_DURATION_SECS` | integer | `1800` (30 min) | optional | Per-turn wall-clock deadline — the only bound that stops a *continuously-emitting* repetition loop (the HTTP read timeout is intentionally off). On expiry the turn aborts with `errorKind=turn_timeout`. `0` disables. Raise for legitimately very long generations on slow hardware (a 70B at 5 tok/s producing 4000 tokens ≈ 800 s). |
| `NWIRO_LOCAL_LLM_INACTIVITY_TIMEOUT_SECS` | integer | `120` (2 min) | optional | Per-token **inactivity** guard (SEC-DOS-1) — aborts if the backend emits **no** SSE token for this long (a silent *stall*, distinct from the wall-clock total above which bounds runaway *emission*). Resets on every received token. On expiry the turn aborts with `errorKind=stream_inactivity_timeout`. `0` disables. Raise if you run a very slow model that can legitimately go quiet for minutes mid-turn. |
| `NWIRO_LOCAL_LLM_PROMPT_PRESTREAM_TIMEOUT_SECS` | integer | `30` | optional | Per-attempt **pre-stream** cap (P0-C P1) — bounds the wait from sending a prompt to a usable streaming response (the request send **and** the admission-gate body reads), so a backend that accepts the connection then never sends response headers — or sends headers then stalls the body — fails fast with `errorKind=timeout` instead of hanging (`CONNECT_TIMEOUT_SECS` bounds only the TCP connect). Each transient failure is retried (up to 3 attempts) with exponential backoff; the attempt count is **auto-reduced** so total pre-stream time (`attempts × cap`) stays under nwiro's ~300 s first-token watchdog — so raising `CONNECT_TIMEOUT_SECS` (which raises this cap, since it is clamped above the connect timeout so a slow *connect* still surfaces as `unreachable`) trades retries for one longer attempt rather than blowing the watchdog. Never bounds the streamed SSE body (that is `INACTIVITY_TIMEOUT_SECS`). `0` disables. |
| `RUST_LOG` | tracing level | `info` | optional | `debug` or `trace` for diagnosis; never use `trace` with a real API key |

### PowerShell syntax

```powershell
$env:NWIRO_LOCAL_LLM_BASE_URL = "http://127.0.0.1:11434/v1"
$env:NWIRO_LOCAL_LLM_MODEL    = "llama3.2:3b"
$env:RUST_LOG                 = "debug"
# To clear:
Remove-Item Env:\NWIRO_LOCAL_LLM_BASE_URL -ErrorAction SilentlyContinue
```

### bash / zsh syntax

```bash
export NWIRO_LOCAL_LLM_BASE_URL="http://127.0.0.1:11434/v1"
export NWIRO_LOCAL_LLM_MODEL="llama3.2:3b"
export RUST_LOG="debug"
# To clear:
unset NWIRO_LOCAL_LLM_BASE_URL
```

### Persistent setup (optional)

If you'll run smoke tests repeatedly, save the env block to a script:

```powershell
# Save as set-shim-env.ps1, then dot-source: . .\set-shim-env.ps1
$env:NWIRO_LOCAL_LLM_BASE_URL = "http://127.0.0.1:11434/v1"
$env:NWIRO_LOCAL_LLM_MODEL    = "llama3.2:3b"
```

```bash
# Save as set-shim-env.sh, then source: source ./set-shim-env.sh
export NWIRO_LOCAL_LLM_BASE_URL="http://127.0.0.1:11434/v1"
export NWIRO_LOCAL_LLM_MODEL="llama3.2:3b"
```

### Remote / cloud endpoints (RunPod & serverless GPU)

The shim has no concept of "local" vs "cloud" — it speaks plain
OpenAI-compatible HTTP and attaches a `Bearer` key when one is set. So
any backend that exposes `/v1/chat/completions` works, including RunPod,
with **no code change**: just point the three env vars at the remote
endpoint. (The `localhost → 127.0.0.1` IPv4 rewrite only touches
`localhost`, so a remote `runpod.ai` host is left intact.)

> **HTTPS / corporate-proxy caveat.** The shim trusts the **bundled** Mozilla
> roots (rustls + `webpki-roots`), **not** the OS trust store. So HTTPS to a
> public endpoint with a publicly-trusted certificate works out of the box —
> but on a machine behind a **TLS-intercepting proxy** (Zscaler / Netskope),
> whose root lives in the OS store but not the bundled set, the handshake
> fails. That case is surfaced as `errorKind=tls_cert` (distinct from
> `unreachable`); use a direct / non-intercepted route to the endpoint, or an
> `http://` local backend (TLS is never engaged for plain HTTP).

**The API key MUST come from the env var.** The Nwiro UE5 bridge can
inject `baseUrl`/`model` through the ACP `initialize` context, but it
**cannot** pass the API key in-band — the shim always reads the key from
`NWIRO_LOCAL_LLM_API_KEY_localllm`. If the env var is unset, every request
omits the `Bearer` header and the cloud backend returns `401` →
`errorKind=auth`.

#### RunPod has two usable shapes (and one trap)

| Mode | `NWIRO_LOCAL_LLM_BASE_URL` | Loading behavior |
|---|---|---|
| **Pod** (dedicated GPU, always-on) | `https://{pod-id}-8000.proxy.runpod.net/v1` | No cold start — warmup returns in <1s, works as-is. You pay for idle GPU. |
| **Serverless vLLM** (autoscaling) | `https://api.runpod.ai/v2/{endpoint-id}/openai/v1` | Cold start on scale-from-zero (see caveat). |
| **Serverless job API** ❌ | `…/v2/{id}/run` · `/runsync` · `/status` | **NOT OpenAI-compatible** (`{"input":…}` body, no `/chat/completions`). The shim cannot speak it — never point at this. |

`base_url` must end at the `/v1` (Pod) or `/openai/v1` (serverless) root,
with **no** trailing `/chat/completions` — the shim appends that itself.

#### Config recipe (serverless vLLM)

```powershell
$env:NWIRO_LOCAL_LLM_BASE_URL       = "https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1"
$env:NWIRO_LOCAL_LLM_API_KEY_localllm = "<your RunPod API key — NOT an OpenAI key>"
$env:NWIRO_LOCAL_LLM_MODEL          = "<HF repo id, or your OPENAI_SERVED_MODEL_NAME_OVERRIDE>"
```

On the **RunPod endpoint** side, native `tool_calls` are **off by
default**. To get Native tier instead of Emulated, deploy the vLLM worker
with `ENABLE_AUTO_TOOL_CHOICE=true` and a model-matched
`TOOL_CALL_PARSER` (`hermes` · `llama3_json` · `mistral` · etc.). Without
these the shim still works — it falls back to the Emulated prose parser —
but you won't get backend-native tool calls.

#### ⚠️ Cold-start caveat (serverless scale-to-zero)

RunPod serverless defaults to **0 active workers** and a **5-second idle
timeout**. When the endpoint has scaled to zero, the first request
triggers a cold start (GPU allocation + container boot + model load) that
RunPod's own docs describe as *"a few seconds"* (model cached) to
*"several minutes"* (uncached).

On a cold start the TCP connect succeeds instantly, then the response
blocks for the **entire** cold load. The warmup load request is bounded
by `NWIRO_LOCAL_LLM_WARMUP_TIMEOUT_SECS` (default **300s**): within the
cap the UE5 "Load Model" / Save spinner simply waits; past it, warmup
fails fast with a diagnosable `errorKind=timeout` ("warmup timed out
after Ns…") instead of hanging indefinitely. Raise the cap for uncached
cold loads, or set `0` to restore the old unbounded wait. (A probe
timeout during this window is harmless: since the fail-open change the
probe degrades to Emulated tier and keeps the tools attached — it does
*not* strip them.)

The 5-second idle timeout adds a second hazard: a worker your warmup just
warmed can scale back down before the first real prompt arrives,
re-incurring the cold start mid-prompt — which warmup can't shield.

**Recommended (ranked):**

1. **Serverless with Active Workers ≥ 1** (a.k.a. min-workers=1) — no
   cold start, shim works unchanged, moderate cost. *Best fit.*
2. **Pod (always-on)** — zero config, highest reliability, higher idle cost.
3. **Serverless scale-to-zero** — cheapest, but accept the hang. If you
   use it, also raise the timeouts so the probe survives a still-loading
   worker, and bump RunPod's init ceiling for large models:
   ```powershell
   $env:NWIRO_LOCAL_LLM_CONNECT_TIMEOUT_SECS = "120"
   $env:NWIRO_LOCAL_LLM_PROBE_TIMEOUT_SECS   = "120"
   $env:NWIRO_LOCAL_LLM_WARMUP_TIMEOUT_SECS  = "600"   # uncached cold load
   # On the RunPod endpoint, if a large model's cold load exceeds the
   # ~7-min unhealthy ceiling:  RUNPOD_INIT_TIMEOUT=800
   ```

> **Closed gap (v0.2.1):** the warmup load request is now bounded by
> `NWIRO_LOCAL_LLM_WARMUP_TIMEOUT_SECS` (default 300s, `0` = unbounded) and
> a timed-out warmup fails fast with `errorKind=timeout` — the cold-start
> hang is cappable and diagnosable. Active Workers ≥ 1 or a Pod remain the
> smoothest options for serverless.

---

## 6. Run the smoke tests

The smoke tests are Python scripts in the [`scripts/`](../scripts/) directory. They drive the shim binary via subprocess (stdin/stdout) and assert on the ACP frame responses. Tests 1–3 stand up their own in-process mock backend, so **no real provider is required** for them.

**Prerequisite:** `python --version` returns 3.10+. The smoke tests are **stdlib-only** — there is nothing to `pip install`.

### Binary location

The smoke tests look for the binary at a path derived from the repo layout: `../target/release/local-llm-acp` (with a `.exe` suffix on Windows). That resolves correctly when you run them from a clone you've built with `cargo build --release`.

If you downloaded a release archive instead of building from source, either:

(a) Copy the downloaded binary into `target/release/`:
```bash
# Linux/macOS
mkdir -p target/release && cp ~/tools/local-llm-acp/local-llm-acp target/release/

# Windows PowerShell
New-Item -ItemType Directory -Force target\release; Copy-Item C:\tools\local-llm-acp\local-llm-acp.exe target\release\
```

(b) Or edit the `BIN` constant near the top of the script to point at the actual binary location.

### Test 1: refusal / error-mapping path (no provider needed)

```bash
python scripts/smoke-test.py
```

This test deliberately points the shim at an unreachable host (`127.0.0.1:1/v1`) to exercise the failed-warmup → `ToolTier::None` → backend-unreachable path. **No real provider needs to be running.** Validates ACP framing and JSON-RPC error mapping.

Expected output ends with `=== ALL CHECKS PASSED ===`.

### Test 2: mid-session model switch (mock provider built-in)

```bash
python scripts/smoke-test-model-switch.py
```

This test stands up an in-process mock OpenAI server so it doesn't need Ollama either. Validates tool-tier reclassification across mid-session model switches.

Expected: a series of `OK ...` lines, then `=== ALL CHECKS PASSED ===`.

### Test 3: tool-call event ordering & streaming (mock providers built-in)

Two more no-provider tests cover the streaming surface:

```bash
python scripts/smoke-test-tool-call-events.py   # tool_call / tool_call_update event ordering
python scripts/smoke-test-streaming.py          # real-time SSE streaming + mid-stream cancel
```

Each stands up its own in-process mock backend and reports a PASS result.

### Test 4: real provider validation (Ollama running)

With env vars set per §5 and Ollama running, the minimal end-to-end check is a single `initialize` round-trip:

```powershell
# Windows PowerShell
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | C:\tools\local-llm-acp\local-llm-acp.exe
```

```bash
# Linux/macOS
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | ~/tools/local-llm-acp/local-llm-acp
```

You should see one JSON-RPC response line with `serverInfo`. If it hangs, see [§8 — Common gotchas](#8-common-gotchas).

---

## 7. OS-specific notes

### Windows

- **WDDM TDR watchdog kills long GPU kernels** (default 2s). If you'll run large models (>30B parameters or partial offload), bump `TdrDelay` in the registry. Full procedure: [`MODEL-SETUP.md §4`](MODEL-SETUP.md#4-windows-registry--tdr-fix).
- **IPv6 first** — Windows resolves `localhost` to `::1` before `127.0.0.1`, which trips some providers. The shim normalizes `localhost` → `127.0.0.1` internally, but if you connect from another tool (curl, smoke tests), prefer `127.0.0.1` in your URLs.
- **PowerShell vs cmd.exe** — env-var syntax differs. PowerShell uses `$env:VAR = "x"`; cmd uses `set VAR=x`. The smoke tests don't care which you use.
- **Defender / antivirus** — first run of the downloaded binary may trigger a SmartScreen warning. Right-click → Properties → Unblock if needed.

### macOS

- **Apple Silicon = unified memory.** No VRAM/RAM split. A 192 GB M3 Ultra can run models that wouldn't fit on a 96 GB discrete GPU. Provider-side, llama.cpp / Ollama / LM Studio use Metal acceleration.
- **No vLLM.** Use Ollama or llama.cpp instead.
- **Gatekeeper quarantine** on first run (see §3). Once cleared, subsequent runs work normally.
- **Network firewall** — System Settings → Network → Firewall may prompt on first run of the shim. Allow.

### Linux

- **vLLM is back in scope.** If you have an NVIDIA GPU, vLLM is the most production-realistic provider. Install with `pip install vllm`.
- **No TDR watchdog.** Long inference doesn't risk a kernel reset like Windows.
- **systemd Ollama** — on most distros, Ollama installs as a systemd service. `systemctl status ollama` to check.
- **CUDA driver mismatches** — `nvidia-smi` should show your card and a working driver. If `vllm` complains about CUDA, your driver version is probably stale.
- **Executable bit** — `chmod +x` the binary after extracting (some tarball tools strip it).

---

## 8. Common gotchas

| Gotcha | Affects | Symptom | Fix / Reference |
|---|---|---|---|
| Forgot `/v1` suffix on `NWIRO_LOCAL_LLM_BASE_URL` | All | 404 from provider, "model not found" | Always include `/v1` — e.g., `http://127.0.0.1:11434/v1` |
| `PROBE_TIMEOUT_SECS` default (5s) too short | All, partial-offload models | Warmup returns `toolTier: none` on a capable model | `NWIRO_LOCAL_LLM_PROBE_TIMEOUT_SECS=30` |
| Treating the binary like a CLI tool | All | Process hangs, no output | The shim reads ACP JSON-RPC from stdin; use smoke tests or pipe input |
| WDDM TDR fires during large-model inference | Windows only | CUDA crash, display blink, provider loses state | [`MODEL-SETUP.md §4`](MODEL-SETUP.md#4-windows-registry--tdr-fix) |
| vLLM "doesn't install" on macOS / Windows | macOS, Windows | `pip install vllm` errors or runtime CUDA import failure | Use Ollama, LM Studio, or llama.cpp instead |
| Shell env var syntax cross-contamination | All | `command not found` or `export: command not found` | PowerShell: `$env:VAR=` · bash: `export VAR=` (see §5) |
| Smoke test can't find binary | All | `FileNotFoundError` before any frame is sent | Copy binary into `target/release/` or edit `BIN` constant (see §6) |
| Provider running but model not loaded | All | Connection succeeds, warmup times out | Pull / load the model in provider FIRST: `ollama pull <name>` |
| VRAM contention between two providers | All with GPU | OOM when loading a model | Stop the previous provider before loading the next: `ollama stop`, LM Studio Eject, llama.cpp Ctrl+C |
| Port collision between providers | All | "address already in use" | `netstat -ano \| findstr :11434` (Win) / `lsof -i :11434` (Unix) — kill stale process or pick a different port |
| llama.cpp tool calls return prose | All | Model says "I would call..." instead of emitting `tool_calls` JSON | Add `--jinja` to the launch command |
| API key sentinel leaks via `RUST_LOG=trace` | All | API key value appears in trace output | Never use `RUST_LOG=trace` with a real key; `debug` is safe |

---

## 9. When you actually need MODEL-SETUP.md

`RUNNING.md` covers running the shim against any provider on any OS. It deliberately stops short of:

- Specific GGUF model recommendations (Kimi K2 IQ1_M, Qwen3-30B-A3B Q4_K_M, GLM-4.5-Air Q4_K_M)
- The `<MODELS_DIR>\gguf\` cache directory layout
- `--n-gpu-layers` tuning math for partial offload
- TDR registry edit commands (Windows)
- Source builds with `-DCMAKE_CUDA_ARCHITECTURES=120` for Blackwell
- The three-cell test matrix (Cell A / B / C)
- The per-cell review workflow
- HuggingFace download commands for specific frontier models

If you need any of the above, go to **[`docs/MODEL-SETUP.md`](MODEL-SETUP.md)**. That doc owns the Blackwell-specific procedures and the three-cell model setup.

`MODEL-SETUP.md` also has a [§12 Quick Reference](MODEL-SETUP.md#12-quick-reference-cheat-sheet) cheat-sheet for experienced operators returning to the setup after a break.

---

## References

- [`README.md`](../README.md) — what the crate is, build instructions, security notes
- [`STRUCTURE.md`](../STRUCTURE.md) — code layout, ACP message inventory
- [`RELEASING.md`](../RELEASING.md) — release asset filename convention
- [`docs/MODEL-SETUP.md`](MODEL-SETUP.md) — Blackwell-specific model setup runbook
- [GitHub releases](https://github.com/leartesstudios/nwiro-acp/releases) — binary downloads
- [Ollama](https://ollama.com) · [LM Studio](https://lmstudio.ai) · [llama.cpp](https://github.com/ggml-org/llama.cpp) · [vLLM](https://docs.vllm.ai)
