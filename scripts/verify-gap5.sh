#!/usr/bin/env bash
# verify-gap5.sh — runnable gate for the OpenRouter Gap-5 mid-stream-error
# contract + the chat-only graduation. Consolidates the deterministic
# verification layers into one CI-gateable command.
#
#   Usage:  bash scripts/verify-gap5.sh
#   Exit:   0 = all deterministic layers green; non-zero = a gate failed.
#
# Layers 3-4 (live OpenRouter smoke, nwiro UE5 end-to-end) and the source
# mutation check stay MANUAL by design.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== Gap-5 / OpenRouter graduation gate =="
echo "Load-bearing tests (run as part of the full suite below):"
echo "  Layer 1 contract : streaming_response_captures_openrouter_midstream_error"
echo "                     openrouter_midstream_error_chunk_surfaces_as_error_not_clean_finish"
echo "                     server_error_on_prompt_round_refuses_cleanly_not_minus_32000"
echo "  Layer 1 hardening: bare_finish_reason_error_maps_to_refusal_with_error_kind          (MAJOR-A)"
echo "                     http_status_from_error_object_reads_numeric_string_and_type       (MAJOR-B/C)"
echo "                     midstream_error_with_string_code_classifies_rate_limited_not_unknown"
echo
echo "-- Layer 2: full regression suite (the authoritative gate) --"
cargo test
echo
echo "== PASS: all deterministic layers green =="
cat <<'EOF'

Graduation status:
  [x] Gap 5            mid-stream errors surface as a clean refusal (not a silent finish)
  [x] Gap-5 hardening  bare finish_reason:"error" + string/absent error-code tag
  [x] Docs/UI relabel  nwiro settings card + integrations.mdx OpenRouter row
  [x] Test-matrix row  MODEL-TEST-PLAN.md M9 cloud-beta chat smoke
  [x] Cost disclosure  settings-card amber note + docs callout
  [ ] Gap 3 (DEFERRED) tools: the probe must classify OpenRouter's 404
                       "no endpoints found that support tool_choice" as tool-unsupported.
                       Capture a live 404 body + a pinning mockito test FIRST.

Supported scope today: BETA, CHAT-ONLY (tools unsupported).
Still required per release: Layer 3 (live OpenRouter) + Layer 4 (nwiro UE5) — manual.
EOF
