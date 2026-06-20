//! Vision-capability detection for local models.
//!
//! Decides whether the configured model should receive image content (OpenAI
//! `image_url` parts) or clean-degrade to text. The decision is a pure function
//! of the model id plus an env override; the actual data-URL passthrough is the
//! same across Ollama, LM Studio, llama.cpp, and vLLM — all accept the OpenAI
//! multimodal `content` array — so backend transport support is assumed for the
//! OpenAI-compatible `/v1/chat/completions` surface and the env override is the
//! escape hatch if a specific backend rejects images.
//!
//! Detection order:
//!   1. `NWIRO_LOCAL_LLM_FORCE_VISION` env override (`on`/`off`) — operator escape
//!      hatch to force-enable a model the registry doesn't recognise, or to
//!      force-disable on a backend that 400s on `image_url`.
//!   2. Static model-family registry (substring / token match on the model id).
//!
//! Conservative by design: an UNRECOGNISED model is treated as TEXT-ONLY so the
//! shim never sends `image_url` to a model/backend that can't decode it (a false
//! negative degrades gracefully with a note; a false positive would 400 the turn).
//! A vision model the registry misses can be force-enabled with the env override.

/// Substring markers that unambiguously indicate a vision-capable family.
/// Matched against the lowercased final path component of the model id, so
/// org/quant/path prefixes (e.g. `library/llava:13b-v1.6-q4_0`) still match.
const VISION_MARKERS: &[&str] = &[
    "llava",
    "bakllava",
    "moondream",
    "minicpm-v",
    "pixtral",
    "internvl",
    "granite-vision",
    "vision", // llama3.2-vision, *-vision-*
    "glm-4v",
    "glm4v",
    "mllama", // llama 3.2 vision arch id
    "qwen-vl",
    "qwen2-vl",
    "qwen2.5-vl",
    "qwen3-vl",
    "gemma3", // Gemma 3 is multimodal
    "gemma-3",
    "llama4", // Llama 4 (Scout/Maverick) is multimodal
    "llama-4",
];

/// Decide whether `model` should receive image content.
pub fn model_supports_vision(model: &str) -> bool {
    if let Some(forced) = force_vision_override() {
        return forced;
    }
    let lower = model.to_lowercase();
    let base = lower.rsplit(['/', '\\']).next().unwrap_or(&lower);

    if VISION_MARKERS.iter().any(|m| base.contains(m)) {
        return true;
    }
    // Qwen-VL across ALL tag formats — Ollama glues the version and drops the
    // hyphen (`qwen2.5vl`), HF/LM Studio use `qwen2-vl` / `qwen2.5-vl`, etc.
    // A `qwen*` id that ALSO contains `vl` is the vision line (text Qwen ids like
    // `qwen3:14b` have no `vl`). This is what a delimiter-split token check misses
    // because `qwen2.5vl` splits to `5vl`, not `vl`. Strip the `vllm` runtime
    // token first so a text Qwen served by vLLM isn't misread as vision.
    if base.contains("qwen") && base.replace("vllm", "").contains("vl") {
        return true;
    }
    // Delimiter-bounded `vl` token (covers `...-vl:7b`, `...-vl-instruct`)
    // without matching substrings like "vllm" or "available".
    base.split(['-', '_', ':', '/', '.', ' ', '@'])
        .any(|t| t == "vl")
}

/// Read `NWIRO_LOCAL_LLM_FORCE_VISION`. `Some(true/false)` forces the decision;
/// `None` (unset or unrecognised value) falls through to the registry.
fn force_vision_override() -> Option<bool> {
    let v = std::env::var("NWIRO_LOCAL_LLM_FORCE_VISION").ok()?;
    match v.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "on" | "yes" => Some(true),
        "0" | "false" | "off" | "no" => Some(false),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_common_vision_families() {
        for m in [
            "llava",
            "llava:13b",
            "library/llava:13b-v1.6-q4_0",
            "moondream",
            "minicpm-v:8b",
            "pixtral-12b",
            "qwen2.5-vl:7b",
            "qwen2-vl-7b-instruct",
            "qwen3-vl:30b",
            "llama3.2-vision:11b",
            "llama-3.2-90b-vision-instruct",
            "gemma3:4b",
            "llama4:scout",
            "granite-vision-3.2",
        ] {
            assert!(model_supports_vision(m), "expected vision for {m}");
        }
    }

    #[test]
    fn rejects_text_only_models() {
        for m in [
            "qwen3:14b",
            "qwen2-72b-instruct",
            "llama3.2:3b",
            "mistral-nemo:12b-instruct-2407-q4_K_M",
            "glm-4-9b-chat",
            "gemma2:9b-instruct-q4_K_M",
            "phi4",
            "deepseek-r1:14b",
            "", // defensive
        ] {
            assert!(!model_supports_vision(m), "expected text-only for {m}");
        }
    }

    #[test]
    fn detects_ollama_glued_qwen_vl_tags() {
        // Regression (caught by the live RunPod smoke): Ollama's tag glues the
        // version and drops the hyphen — `qwen2.5vl:7b` — which a delimiter-split
        // `vl`-token check misses (it splits to `5vl`).
        assert!(model_supports_vision("qwen2.5vl:7b"));
        assert!(model_supports_vision("qwen2vl"));
        assert!(model_supports_vision("qwen3vl:30b"));
        assert!(model_supports_vision("Qwen2.5-VL-7B-Instruct"));
        // ...without matching text Qwen ids.
        assert!(!model_supports_vision("qwen3:14b"));
        assert!(!model_supports_vision("qwen2.5:7b-instruct"));
    }

    #[test]
    fn vl_token_is_boundary_bounded_not_substring() {
        // `vl` must be a delimiter-bounded token, not a raw substring, so
        // unrelated ids that merely CONTAIN "vl" do not falsely match.
        assert!(!model_supports_vision("vllm-served-qwen3"));
        assert!(!model_supports_vision("available-model"));
        assert!(model_supports_vision("qwen2.5-vl"));
    }
}
