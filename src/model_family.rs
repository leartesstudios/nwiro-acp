//! Model-family detection registry for v0.1.28+ defensive warmup gates.
//!
//! ## Why this module exists
//!
//! v0.1.27 introduced schema-bleed detection at probe time
//! (`looks_like_schema_bleed`) and tools-stripping when the probe classifies
//! the model as `ToolTier::None`. That fixed the cases where the failure
//! was caused by the 25 KB tool array being passed as plain text to a
//! model whose chat template couldn't process `tool_calls`.
//!
//! It did NOT fix the underlying BASE chat-template problem: some model
//! families (GLM-4 family being the v0.1.27 user-reported case) require
//! non-standard prompt markers like `[gMASK]<sop><|system|>\n`,
//! `<|user|>\n`, `<|assistant|>\n`. When the backend's loaded chat
//! template is missing these markers, the model's input doesn't even
//! match its conversation training distribution — every turn fails
//! autoregressively, producing garbage regardless of whether tools
//! are present.
//!
//! ACP cannot install LM Studio chat templates (that's the backend
//! operator's job). What ACP CAN do is detect the symptom + recognize
//! the model family + refuse warmup cleanly with actionable setup
//! guidance, instead of letting the session load into a broken state
//! that streams garbage to the UE5 chat.
//!
//! ## v0.1.28 design decision (Option F)
//!
//! - Hard refusal at warmup, not graceful degrade. The streaming filter
//!   (Option D) would mask the real problem.
//! - ACP should diagnose and contain, not become a chat-template engine
//!   for OpenAI-compat backends. Raw-completion bypass (Option B) is a
//!   v0.2.0+ project.
//! - Hard refusal is the correct engineering choice. Surface the LM
//!   Studio preset path the user needs.
//!
//! ## Detection contract
//!
//! Family detection is a pure function of the configured model name.
//! Case-insensitive substring match. The gate (refusal) fires ONLY when
//! BOTH a family is detected AND `schema_bleed_detected == true` in the
//! probe assessment — protecting working GLM setups from false refusal.
//!
//! ## Adding a new family
//!
//! 1. Add a variant to `ModelFamily`.
//! 2. Extend `detect()` with the substring patterns. Validate against
//!    actual model names users run (quants, suffixes, version aliases).
//! 3. Add the family's `template_guidance()` string with the specific
//!    LM Studio Prompt Template values + a `docs/MODEL-SETUP.md` anchor.
//! 4. Add a unit test covering at least 3 model-name variants per family
//!    (base, quant suffix, version variant).
//! 5. Document in `docs/MODEL-SETUP.md` under a new `## <Family> family`
//!    section.
//!
//! Start small. v0.1.28 ships GLM-4 only because that's the only family
//! with a confirmed user report. Other families ship when we have
//! evidence — not pre-emptively.

/// A recognized model family that has a documented LM Studio chat-template
/// requirement. Used by `warmup()` to gate the session with actionable
/// guidance when schema-bleed is also detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModelFamily {
    /// GLM-4 family: ChatGLM4, GLM-4-9B-Chat, GLM-4.5-air, etc. Requires
    /// `[gMASK]<sop>` system prefix and `<|user|>` / `<|assistant|>`
    /// role markers in the chat template.
    Glm4,
}

impl ModelFamily {
    /// Detect a known model family from the configured model name.
    /// Returns `None` for unknown / well-behaved families (Qwen, Llama,
    /// Mistral, Gemma, Phi, etc. — all of which ship working templates
    /// in LM Studio's default registry).
    ///
    /// Match contract is **word-boundary anchored**, case-insensitive,
    /// NOT a raw substring match. v0.1.28 critic codex DEFECT 1
    /// flagged the prior `contains("glm")` as over-broad: it would
    /// misclassify real HuggingFace models like `facebook/xglm-7.5B`
    /// (cross-lingual generative LM) as GLM-4, then refuse warmup
    /// with GLM-4-specific remediation that doesn't apply. The
    /// boundary-aware matcher below covers the user-reported variants
    /// without sweeping in other `*glm*` suffix families.
    ///
    /// Accept conditions (lowercased):
    ///   - Name starts with `glm` AND next char is one of:
    ///     end-of-string, separator (`-_:/. `), or digit (version
    ///     suffix like `glm4`).
    ///   - Name contains `chatglm` anywhere — this token is
    ///     GLM-family-exclusive (no known collision with other model
    ///     families).
    ///
    /// Reject (true negatives we now correctly handle):
    ///   - `xglm-7.5b` — starts with `xglm`, not `glm`.
    ///   - `glmedge-1.0` — `glm` followed by letter `e`, not separator
    ///     or digit; presumed unrelated derivative.
    ///   - `glmnet-3` — `glm` followed by letter `n`; ditto.
    ///
    /// False-positive safety net: the warmup gate ALSO requires
    /// schema-bleed signature, so even if the matcher overshoots for
    /// some hypothetical model name, a well-configured backend that
    /// emits coherent probe output won't trigger refusal.
    pub(crate) fn detect(model: &str) -> Option<Self> {
        let path_lower = model.to_lowercase();
        // v0.1.30: normalize path-form model IDs to the final path component
        // before pattern matching. LM Studio passes the FULL GGUF path as
        // the model id over OpenAI-compat (e.g.
        // `legraphista/glm-4-9b-chat-GGUF/glm-4-9b-chat.Q4_K_S.gguf`), and
        // the v0.1.28 boundary check was anchored to the start of the
        // string — those path-prefixed names silently fell through to None.
        // The v0.1.29 LMS log diagnostic confirmed this caused the GLM
        // family directive to not fire even when the user was running a GLM
        // model. Splitting by both `/` and `\` covers POSIX + Windows
        // backslash separators that some backends use internally.
        let lower: &str = path_lower
            .rsplit(['/', '\\'])
            .next()
            .unwrap_or(&path_lower);
        // Check 1 (v0.2.6): `glm<digit|end>` as a delimiter-bounded TOKEN
        // anywhere in the normalized id — covers `glm-4-9b`, `glm-z1-9b`, AND
        // org/uploader-prefixed ids LM Studio emits verbatim, e.g.
        // `thudm_glm-z1-9b-0414`, `zai-org_glm-4.5`. The old start-anchored
        // `strip_prefix("glm")` missed those, so a Native GLM variant got
        // NEITHER the tool ceiling NOR the v0.2.6 bleed buffer — a real BLACK
        // (live: GLM-Z1 `thudm_glm-z1-9b-0414` collapsed T-D to stopReason=null
        // at 30 uncapped tools). Splitting on the same delimiters keeps the
        // boundary check (a token is `glm`, `glm4`, `glm-z1`→`glm`, etc.) so a
        // substring like `biglmodel` never matches.
        for token in lower.split(['-', '_', ':', '/', '.', ' ']) {
            if let Some(rest) = token.strip_prefix("glm") {
                if rest.is_empty() || rest.as_bytes().first().is_some_and(u8::is_ascii_digit) {
                    return Some(Self::Glm4);
                }
            }
        }
        // Check 2: `chatglm` anywhere. Unique enough to identify the
        // ChatGLM line without colliding with other families. (No
        // separator check needed — `chatglm` is not a substring of
        // any other published model family this shim supports.)
        if lower.contains("chatglm") {
            return Some(Self::Glm4);
        }
        None
    }

    /// Human-readable, actionable guidance for the operator. Embedded
    /// verbatim in the `WarmupResult.message` when the gate refuses
    /// the session. Keep these strings short — they render in the UE5
    /// chat as a single notification line. Detail belongs in
    /// `docs/MODEL-SETUP.md`.
    pub(crate) fn template_guidance(&self) -> &'static str {
        match self {
            Self::Glm4 => {
                // v0.1.30 review MUST_FIX 1:
                // the v0.1.28-original guidance had `afterSystem` and
                // `beforeSystem` INVERTED — system content lands OUTSIDE
                // the GLM `<|system|>` envelope where the model treats it
                // as untagged pre-context prose. The v0.1.29 LMS log
                // diagnostic confirmed this. Marker order corrected here
                // to match the v0.1.30 docs update in docs/MODEL-SETUP.md
                // — the in-product guidance and the docs guidance must
                // agree, since operators hitting the schema-bleed refusal
                // gate get this exact string in WarmupResult.message.
                "Detected GLM-family model with broken backend chat template. \
                 ACP cannot continue safely. Required: load the GLM-4 prompt \
                 template preset in LM Studio (beforeSystem='[gMASK]<sop>\\n<|system|>\\n', \
                 afterSystem='', beforeUser='<|user|>\\n', afterUser='<|assistant|>\\n'), \
                 or in llama.cpp pass --chat-template chatglm4. \
                 See docs/MODEL-SETUP.md#glm-family for the full preset. \
                 Set NWIRO_LOCAL_LLM_BYPASS_TEMPLATE_GATE=1 to override \
                 (not recommended; you'll see garbage output)."
            }
        }
    }

    /// Short identifier embedded in `WarmupResult.error_kind` for
    /// machine-parseable upstream handling. The bridge / UE5 client can
    /// switch on this without parsing the message string.
    pub(crate) fn error_kind(&self) -> &'static str {
        match self {
            Self::Glm4 => "broken_chat_template",
        }
    }

    // v0.1.35 — model-agnostic tool coverage. The per-family tool-POLICY
    // predicates `requires_tool_invocation_mandate()` and
    // `forces_emulated_tier()` were DELETED here (both were `Glm4 => true`
    // allow-lists of one). Their jobs are now done family-independently:
    //   - the "invoke, don't describe" mandate fires for EVERY Native/Emulated
    //     session that registers tools (see `build_tool_invocation_mandate` in
    //     bridge/mod.rs) — describer-over-actor is a property of any RLHF chat
    //     model under tool_choice:auto, not a GLM quirk;
    //   - an inconclusive/transient probe now fails OPEN to Emulated instead of
    //     None (see `ProbeAssessment::failed` + the terminal probe return in
    //     client.rs), so tools are never silently stripped for any model.
    // No per-family tool-policy gate remains. `detect()`, `template_guidance()`,
    // `error_kind()`, and `recommended_tool_ceiling()` stay — those are genuine
    // broken-template / capacity diagnostics, not tool-policy guesses.

    /// Empirical per-family ceiling on tool count. Returns `None` when
    /// the family has no documented ceiling. The shim publishes the
    /// EMPIRICAL hardware limit; the consuming app (Nwiro) is expected
    /// to apply its own safety margin on top of this value.
    pub(crate) fn recommended_tool_ceiling(&self) -> Option<u32> {
        match self {
            // GLM-4: spec §1 documents schema-bleed onset at N=30 (gold
            // tool present, N=24 OK). Return 29 = threshold-1 (highest
            // empirically safe N). Apps may apply additional safety
            // margin on top.
            Self::Glm4 => Some(29),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // v0.1.28 — ModelFamily detection coverage.
    //
    // Per the planner's IMPL-002 contract: substring match must catch the
    // variants users actually run. The verbatim model names below are
    // sourced from:
    //   - The v0.1.27 user report (GLM-4.5-air on LM Studio)
    //   - The v0.1.28 user preset file (GLM-4-9B-Chat Legacy)
    //   - HuggingFace ChatGLM3 model ID (chatglm3-6b)
    //   - Realistic quant suffixes (q4_k_m, q5_0)
    // Each positive test pins a real-world name. Negative tests pin the
    // well-behaved families that MUST NOT misclassify.

    #[test]
    fn detects_glm_4_5_air_lm_studio_form() {
        // v0.1.27 user report verbatim
        assert_eq!(ModelFamily::detect("glm-4.5-air"), Some(ModelFamily::Glm4));
    }

    #[test]
    fn detects_glm_4_9b_chat_legacy() {
        // v0.1.28 user report verbatim
        assert_eq!(
            ModelFamily::detect("GLM-4-9B-Chat (Legacy)"),
            Some(ModelFamily::Glm4)
        );
    }

    #[test]
    fn detects_glm_with_quant_suffix() {
        // Realistic LM Studio loaded-model name
        assert_eq!(
            ModelFamily::detect("glm-4.5-air-q4_k_m"),
            Some(ModelFamily::Glm4)
        );
    }

    #[test]
    fn detects_chatglm_lowercase_variant() {
        // HuggingFace ChatGLM3 id
        assert_eq!(ModelFamily::detect("chatglm3-6b"), Some(ModelFamily::Glm4));
    }

    #[test]
    fn detects_glm_uppercase() {
        // Some backends echo the model id in original case
        assert_eq!(ModelFamily::detect("GLM-4"), Some(ModelFamily::Glm4));
    }

    #[test]
    fn rejects_qwen3() {
        // User confirmed Qwen3:14b on Ollama works flawlessly with v0.1.26
        // → must NOT match a family gate
        assert_eq!(ModelFamily::detect("qwen3:14b"), None);
        assert_eq!(ModelFamily::detect("qwen2-72b-instruct"), None);
    }

    #[test]
    fn rejects_llama_family() {
        // Llama family ships working templates in LM Studio defaults
        assert_eq!(ModelFamily::detect("llama-3.1-8b-instruct"), None);
        assert_eq!(ModelFamily::detect("Meta-Llama-3-70B"), None);
    }

    #[test]
    fn rejects_mistral_family() {
        assert_eq!(ModelFamily::detect("mistral-7b-instruct-v0.3"), None);
        assert_eq!(ModelFamily::detect("Mixtral-8x7B"), None);
    }

    #[test]
    fn rejects_gemma_family() {
        assert_eq!(ModelFamily::detect("gemma-2-9b-it"), None);
    }

    #[test]
    fn rejects_phi_family() {
        assert_eq!(ModelFamily::detect("phi-3-medium-128k-instruct"), None);
    }

    #[test]
    fn rejects_empty_string() {
        // Defensive: empty model name (shouldn't happen at runtime, but
        // pin the contract anyway)
        assert_eq!(ModelFamily::detect(""), None);
    }

    // v0.1.28 critic round-1 codex DEFECT 1: word-boundary regression
    // anchors. These names contain the substring "glm" but are NOT
    // GLM-4 family members. Pre-fix `contains("glm")` would have
    // misclassified them, leading to a hard-refusal warmup with
    // GLM-4-specific remediation that doesn't help the operator.

    #[test]
    fn rejects_xglm_cross_lingual_lm() {
        // facebook/xglm-7.5B — real HuggingFace model, cross-lingual
        // generative LM, entirely unrelated to GLM-4. Pre-fix would
        // false-match because "xglm" contains "glm".
        assert_eq!(ModelFamily::detect("xglm-7.5b"), None);
        assert_eq!(ModelFamily::detect("facebook/xglm-7.5B"), None);
    }

    #[test]
    fn rejects_glmedge_hypothetical_derivative() {
        // Hypothetical name where "glm" prefix is followed by letter.
        // We don't know if such a model exists, but the boundary check
        // must reject it because we have no evidence its remediation
        // is GLM-4's.
        assert_eq!(ModelFamily::detect("glmedge-1.0"), None);
        assert_eq!(ModelFamily::detect("glmnet-3-b"), None);
    }

    #[test]
    fn detects_org_underscore_prefixed_glm_artifacts() {
        // v0.2.6: LM Studio emits org/uploader-prefixed ids with an UNDERSCORE
        // separator (no path slash for rsplit to strip), e.g.
        // `thudm_glm-z1-9b-0414`. The old start-anchored `strip_prefix("glm")`
        // returned None → no tool ceiling, no Native bleed buffer → a live BLACK
        // (GLM-Z1 T-D collapsed to stopReason=null at 30 uncapped tools). `glm`
        // as a delimiter-bounded token must match these.
        assert_eq!(ModelFamily::detect("thudm_glm-z1-9b-0414"), Some(ModelFamily::Glm4));
        assert_eq!(ModelFamily::detect("thudm_glm-4-32b-0414"), Some(ModelFamily::Glm4));
        assert_eq!(ModelFamily::detect("zai-org_glm-4.5-air"), Some(ModelFamily::Glm4));
        // ...without regressing the xglm / glmedge false-positive guards even
        // when org-prefixed (the matched token must still be exactly `glm<digit|end>`).
        assert_eq!(ModelFamily::detect("thudm_xglm-7.5b"), None);
        assert_eq!(ModelFamily::detect("someorg_glmedge-1.0"), None);
    }

    #[test]
    fn detects_glm_with_path_prefix() {
        // v0.1.30: path-prefixed model IDs are now correctly detected via
        // the rsplit normalization in `detect()`. The v0.1.28 contract
        // (path-prefixed → None) was the documented limitation that
        // caused the v0.1.29 LMS-log diagnostic finding: LM Studio
        // passes the full GGUF path as model id, and family detection
        // silently missed.
        //
        // After v0.1.30, the final path component is what matters.
        assert_eq!(
            ModelFamily::detect("models/glm-4-9b"),
            Some(ModelFamily::Glm4)
        );
        assert_eq!(
            ModelFamily::detect("models/chatglm3-6b"),
            Some(ModelFamily::Glm4)
        );
    }

    #[test]
    fn detects_glm_with_lm_studio_full_gguf_path() {
        // Verbatim path from the v0.1.29 user diagnostic lms log stream.
        // This is what LM Studio actually sends to the shim as the model
        // id over OpenAI-compat. Pre-v0.1.30 returned None and the GLM
        // family directive silently never fired.
        assert_eq!(
            ModelFamily::detect("legraphista/glm-4-9b-chat-GGUF/glm-4-9b-chat.Q4_K_S.gguf"),
            Some(ModelFamily::Glm4)
        );
    }

    #[test]
    fn detects_glm_with_windows_backslash_path() {
        // Some backends use backslash separators internally on Windows
        // (e.g. when listing local files). Cover both POSIX and Windows.
        assert_eq!(
            ModelFamily::detect("C:\\models\\glm-4-9b-chat.gguf"),
            Some(ModelFamily::Glm4)
        );
    }

    #[test]
    fn path_normalization_does_not_match_unrelated_basenames() {
        // The rsplit normalization extracts the FINAL segment. If the
        // final segment doesn't match the GLM pattern, we still get
        // None — the path prefix doesn't accidentally upgrade a
        // non-GLM model. Negative guard.
        assert_eq!(
            ModelFamily::detect("models/glm-collection/qwen3-14b.gguf"),
            None
        );
        assert_eq!(
            ModelFamily::detect("models/glmedge-collection/some-other-model"),
            None
        );
    }

    #[test]
    fn detects_bare_glm_with_no_suffix() {
        // Edge case: model literally named "glm" or "GLM" — match.
        assert_eq!(ModelFamily::detect("glm"), Some(ModelFamily::Glm4));
        assert_eq!(ModelFamily::detect("GLM"), Some(ModelFamily::Glm4));
    }

    #[test]
    fn template_guidance_mentions_glm_specific_markers() {
        // The guidance string must include the markers operators need
        // to recognize and apply. Anti-regression: if someone shortens
        // the message to "fix your template", the user is stuck.
        let g = ModelFamily::Glm4.template_guidance();
        assert!(g.contains("[gMASK]"), "must mention [gMASK] prefix");
        assert!(g.contains("<|user|>"), "must mention <|user|> marker");
        assert!(
            g.contains("<|assistant|>"),
            "must mention <|assistant|> marker"
        );
        assert!(
            g.contains("docs/MODEL-SETUP.md"),
            "must point at the docs anchor"
        );
        assert!(
            g.contains("NWIRO_LOCAL_LLM_BYPASS_TEMPLATE_GATE"),
            "must document the bypass env var"
        );
    }

    #[test]
    fn template_guidance_uses_corrected_marker_order_v0_1_30() {
        // v0.1.30 critic round-1 MUST_FIX 1: ensure the in-product
        // guidance text matches the corrected preset layout (markers
        // in `beforeSystem`, NOT in the inverted `afterSystem` form
        // that was shipped in v0.1.28-v0.1.29). This test fails if
        // a future edit accidentally reverts the marker order — the
        // user-facing remediation must stay synchronized with
        // docs/MODEL-SETUP.md.
        let g = ModelFamily::Glm4.template_guidance();
        assert!(
            g.contains("beforeSystem='[gMASK]<sop>\\n<|system|>\\n'"),
            "must specify markers in beforeSystem (corrected v0.1.30 order); \
             got: {g}"
        );
        assert!(
            g.contains("afterSystem=''"),
            "must specify afterSystem='' (corrected v0.1.30 order); got: {g}"
        );
        // Negative anti-regression: the v0.1.28 inverted form must NOT
        // reappear. If someone re-introduces `afterSystem='[gMASK]...`
        // they'll trip this assertion.
        assert!(
            !g.contains("afterSystem='[gMASK]"),
            "must NOT contain the v0.1.28 inverted marker order; got: {g}"
        );
    }

    #[test]
    fn error_kind_is_machine_parseable() {
        // The bridge / UE5 client switches on this value. If it changes
        // the upstream parser breaks silently. Pin the contract.
        assert_eq!(ModelFamily::Glm4.error_kind(), "broken_chat_template");
    }

    // v0.1.35: the `requires_tool_invocation_mandate` / `forces_emulated_tier`
    // predicate tests (and their compile-time "match is exhaustive" enrollment
    // guards) were deleted with the predicates themselves — there is no longer a
    // per-family tool-policy gate to enrol families into. Model-agnostic coverage
    // is exercised by the bridge/client tests instead.
}
