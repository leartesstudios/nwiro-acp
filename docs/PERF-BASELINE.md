# Performance baseline (F-PERF-01..07)

Recorded measurement for the project's performance budgets. The bar is
**"budgets measured + `perf-harness.py` committed"**; wiring the harness to
fail CI on a budget violation is a post-release regression guard.

**Reproduce:**
```
cargo build --release
python scripts/perf-harness.py            # full run (incl. the 500-prompt RSS soak)
python scripts/perf-harness.py --quick    # skip F-PERF-06
python scripts/perf-harness.py --json reports/perf-baseline.json
```

`scripts/perf-harness.py` drives the **real release binary** as a subprocess
against a **deterministic in-process mock OpenAI backend** (no real model, no
network), timestamps every ACP frame at the shim's stdout, and checks each metric
against its budget (exits non-zero on a hard-budget violation). Stdlib-only.

## Baseline (x86_64-pc-windows-msvc, release build, mock backend)

_**Measured against v0.3.0** (commit `10a302d`), 2026-06-16, via
`scripts/perf-harness.py --json reports/perf-baseline.json`. Re-recorded on each release
— the new per-attempt pre-stream timeout + retry add only a wrapper, no hot-path cost
(first-token latency unchanged at 3.2 ms); prior-version figures live in git history._

| ID | Metric | Measured | Budget | Verdict |
|---|---|---|---|---|
| **F-PERF-07** | Startup (`Popen` → first `initialize`, p95 of 20) | **8.1 ms** | ≤ 250 ms | ✅ PASS |
| **F-PERF-02** | First-token latency, warm (prompt → first content frame, p95 of 20) | **3.2 ms** | ≤ 50 ms | ✅ PASS |
| **F-PERF-01** | Frame-rate @ `coalesce_ms=25` under an 800-token / 3 ms flood | **37.7 frames/s** (102 frames, content byte-identical) | ≤ 45 (target 36–40) | ✅ PASS |
| **F-PERF-04** | Throughput + content integrity (as-fast-as-possible stream) | **78,945 tok/s** (informational) + integrity holds | integrity must hold | ✅ PASS |
| **F-PERF-05** | Tool-dispatch overhead, MCP (`tool_call` → `tool_call_update`, median of 3) | **0.26 ms** | ≤ 100 ms | ✅ PASS |
| **F-PERF-03** | Warmup timeout cap (`WARMUP_TIMEOUT_SECS=2`, backend hangs 10 s) | **2.01 s** | 1.9–3.5 s | ✅ PASS |
| **F-PERF-06** | RSS soak slope over 500 single-session prompts (least-squares over 10 samples) | **0.135 MB / 100 prompts** | < 0.5 | ✅ PASS |

**7 / 7 pass.** The headline is **F-PERF-01**: at the production `coalesce_ms=25`
(the goldens run at `coalesce_ms=0`, so this is the only coverage of the live
coalescer), the shim emits **37 frames/s** under a fast token flood — squarely in
the 36–40 target band and far below the > 100/s "coalescer dead" failure mode —
while delivering every input token byte-identical. **F-PERF-06** shows no memory
leak (0.251 MB/100 prompts over a 500-prompt soak).

## Notes & caveats

- These are **mock-backend** numbers (deterministic, hardware-dependent in
  absolute terms). F-PERF-04's tokens/sec is **informational** — for a local mock
  the meaningful assertion is content integrity (every streamed token survives);
  an absolute passthrough ratio is only comparable against a real backend baseline.
- F-PERF-02's "first-token latency" reflects the coalescer's flush-on-first
  behaviour (the first frame is emitted promptly, not after a full 25 ms window).
- **F-PERF-05** is the shim-side **MCP-dispatch** overhead (the `tool_call` →
  `tool_call_update` span), NOT the full end-to-end tool latency. Reported as a
  median of 3 fresh-session runs. Sub-millisecond values are bounded by the
  stdout-read granularity, i.e. "within one read" ≈ instant dispatch.
- **F-PERF-06** uses a least-squares slope over all samples from prompt 100 on
  (not a 2-point delta), so one transient RSS reading can't skew the leak signal.
- RSS is read via `psutil` if installed, else PowerShell `Get-Process WorkingSet64`
  (Windows) or `/proc/<pid>/status` (Linux) — no third-party dependency required.

> **Review round (request_changes → addressed):** fixed a silent-pass
> bug in F-PERF-03 (gate now matches the displayed budget), removed a dead
> `rounds` param that made F-PERF-05 an n=1 measurement (now median-of-3),
> corrected the `SO_REUSEADDR` setup (class attr + `server_close()`), and replaced
> the F-PERF-06 2-point slope with least-squares regression.
