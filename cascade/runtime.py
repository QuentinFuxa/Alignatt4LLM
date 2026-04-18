from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence
import os
import string
import subprocess
import wave

import numpy as np

from cascade.alignment.base import AlignmentBackend, WordAlignment
from cascade.artifacts import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WAV_PATH,
    InferenceArtifacts,
    StreamUpdate,
    write_inference_artifacts,
)
from cascade.emission import (
    RAW_PASSTHROUGH,
    apply_emission_policy,
    register_translation_timestamps,
    register_translation_words,
)
from cascade.mt.base import MTBackendResult, PromptCacheState, build_mt_backend
from cascade.source_frontier import (
    SourceAccessibilityFrontier,
    build_source_accessibility_frontier,
    normalize_word_timestamps_ms,
)
from cascade.source_text import NormalizedSourceText, normalize_source_text_for_mt
from cascade.text_surface import normalize_incremental_target_text
from cascade.translation_variants import (
    FOUNDATIONAL_TRANSLATION_VARIANT_ID,
    RenderedTranslationPrompt,
    TRANSLATION_VARIANTS,
    TranslationVariant,
)
from context_injection import (
    CONTEXT_MODE_OFF,
    VALID_CONTEXT_MODES,
    PaperArtifact,
    PaperContextBlock,
    PaperContextSelector,
    build_retrieval_query,
)
from simulstream.server.speech_processors import SAMPLE_RATE


# Avoid repeated HF HEAD requests for optional files that are already cached as absent.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


LANGUAGE_NAME_TO_CODE = {
    "Czech": "cs",
    "English": "en",
    "German": "de",
    "Italian": "it",
    "Simplified Chinese": "zh",
}
LANGUAGE_CODE_TO_NAME = {
    code: name for name, code in LANGUAGE_NAME_TO_CODE.items()
}
VALID_ALIGNMENT_BACKEND_NAMES = ("qwen_forced", "gemma_onepass_qk_fast", "gemma_vllm_qk_fast")
# The stable set is used by default in comparison scripts; the experimental
# vLLM backend is opt-in until validated under the full SimulStream loop.
STABLE_ALIGNMENT_BACKEND_NAMES = ("qwen_forced", "gemma_onepass_qk_fast")

# MT is now a single shipped path: Gemma AlignAtt through vLLM.
VALID_MT_BACKEND_NAMES = ("gemma_vllm_alignatt",)
STABLE_MT_BACKEND_NAMES = ("gemma_vllm_alignatt",)

# Sentinel returned by the ASR commit helpers when the current chunk must not
# produce a cascade-visible update (e.g. the word-count invariant is broken
# and find_end_time refused to return an end time).
_COMMIT_ABORT = object()


def _resolve_hf_snapshot(repo_subpath: str, env_var: str | None = None) -> str:
    if env_var:
        override = os.environ.get(env_var)
        if override:
            return override

    hub_roots: list[str] = []
    for candidate in (
        os.environ.get("HF_HUB_CACHE"),
        os.path.join(os.environ.get("HF_HOME", ""), "hub") if os.environ.get("HF_HOME") else None,
        "/home/.cache/huggingface/hub",
        os.path.join(os.path.expanduser("~/.cache/huggingface/hub")),
    ):
        if candidate and candidate not in hub_roots:
            hub_roots.append(candidate)

    candidates = [os.path.join(root, repo_subpath) for root in hub_roots]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


asr_model_name = _resolve_hf_snapshot(
    "models--Qwen--Qwen3-ASR-1.7B/snapshots/7278e1e70fe206f11671096ffdd38061171dd6e5",
    env_var="CASCADE_QWEN_ASR_SNAPSHOT",
)
forced_aligner_model_name = _resolve_hf_snapshot(
    "models--Qwen--Qwen3-ForcedAligner-0.6B/snapshots/c7cbfc2048c462b0d63a45797104fc9db3ad62b7",
    env_var="CASCADE_QWEN_ALIGNER_SNAPSHOT",
)
gemma_model_name = _resolve_hf_snapshot(
    "models--google--gemma-4-E4B-it/snapshots/83df0a889143b1dbfc61b591bbc639540fd9ce4c",
    env_var="CASCADE_GEMMA_SNAPSHOT",
)


def alignatt_heads_path_for(source_lang: str, target_lang: str) -> str:
    source_code = LANGUAGE_NAME_TO_CODE.get(source_lang, source_lang.lower())
    target_code = LANGUAGE_NAME_TO_CODE.get(target_lang, target_lang.lower())
    return (
        "assets/attention_heads/"
        f"translation_heads_google_gemma-4-E4B-it_{source_code}-{target_code}.json"
    )


def target_lang_code_for(target_lang: str) -> str:
    return LANGUAGE_NAME_TO_CODE.get(target_lang, target_lang.lower())


@dataclass
class CascadeRuntimeConfig:
    source_lang: str = "English"
    target_lang: str = "German"
    latency_unit: str = "word"
    min_start_seconds: float = 5.0
    translation_variant_id: str = FOUNDATIONAL_TRANSLATION_VARIANT_ID
    max_history_utterances: int = 0
    translation_alignatt_heads_path: str | None = None
    translation_alignatt_top_k_heads: int = 8
    translation_alignatt_filter_width: int = 7
    translation_alignatt_probe_mode: str = "qk_fast"
    translation_alignatt_inaccessible_ms: float = 0.0
    translation_alignatt_rewind_threshold: int = 8
    translation_alignatt_border_margin: int = 0
    translation_alignatt_min_source_mass: float = 0.0
    # Confidence-gated acceptance on top of QK-reconstruction AlignAtt.
    # For each drafted token the reconstructed source-row softmax peaks at a
    # local source position ``p``; this is the raw per-head-averaged mass at
    # ``p`` (∈ [0, 1]). When ``threshold > 0`` and the mass falls below it,
    # the policy stops with reason ``alignatt:argmax_mass_weak`` — the token's
    # attention is too diffuse to trust it as an anchored translation of an
    # accessible source position. Default 0.0 disables the check and
    # preserves the argmax-only policy.
    translation_alignatt_argmax_mass_threshold: float = 0.0
    max_new_tokens: int = 160
    partial_max_new_tokens: int = 16
    translation_min_new_tokens: int = 32
    translation_token_budget_ratio: float = 3.0
    translation_token_budget_buffer: int = 24
    partial_translation_min_new_tokens: int = 4
    partial_translation_token_budget_ratio: float = 1.0
    partial_translation_token_budget_buffer: int = 8
    translation_generation_margin: int = 8
    translation_emit_policy: str = RAW_PASSTHROUGH
    translation_max_tail_rewrite_words: int = 14
    temperature: float = 0.0
    repetition_penalty: float = 1.05
    asr_gpu_memory_utilization: float = 0.2
    gemma_max_model_len: int = 1024
    gemma_enable_prefix_caching: bool = True
    gemma_transformers_device: str = "cuda:0"
    gemma_transformers_dtype: str = "bfloat16"
    gemma_transformers_fast_attention: str = "sdpa"
    gemma_transformers_prompt_kv_reuse: bool = True
    alignment_backend_name: str = "qwen_forced"
    mt_backend_name: str = "gemma_vllm_alignatt"
    gemma_audio_alignment_heads_path: str | None = (
        "assets/attention_heads/audio_alignment_heads_google_gemma-4-E4B-it_en_forced.json"
    )
    gemma_audio_align_probe_mode: str = "qk_fast"
    gemma_audio_alignment_top_k_heads: int = 8
    gemma_audio_alignment_filter_width: int = 7
    gemma_audio_alignment_max_new_tokens: int = 256
    # vLLM-specific config for the experimental gemma_vllm_qk_fast backend.
    # Defaults reflect the validated cudagraph=full seam (PLAN.md section 6).
    gemma_vllm_enforce_eager: bool = False
    gemma_vllm_enable_prefix_caching: bool = False
    gemma_vllm_cudagraph_mode: str | None = "full"
    gemma_vllm_gpu_memory_utilization: float = 0.5
    # Ablation knob for the ASR-side same-SHA A/B control. When True
    # (gemma_vllm_qk_fast only) and no streaming prefix is requested, the
    # backend still invokes llm.generate(prompt_token_ids=..., multi_modal_data=...)
    # with an empty prefix instead of llm.chat(). This isolates the input-path
    # change from the prefix-prefill decode-delta effect.
    gemma_vllm_force_generate_api: bool = False
    # vLLM-specific config for the experimental gemma_vllm_alignatt MT backend.
    # Prefix caching stays off by default (observer-safety requirement); the MT
    # backend reuses the Gemma max_model_len so prompt budgeting is identical
    # to the Transformers MT path. 0.5 utilization matches the ASR vLLM backend:
    # Gemma 4 E4B weights (audio tower bundled) take ~15 GiB, so anything below
    # ~0.45 leaves no budget for the KV cache on a 40 GiB A100.
    mt_vllm_enforce_eager: bool = False
    mt_vllm_enable_prefix_caching: bool = False
    mt_vllm_cudagraph_mode: str | None = "full"
    mt_vllm_gpu_memory_utilization: float = 0.5
    # Qwen-style prompt-prefix streaming for the Gemma ASR backend.
    # When enabled (gemma_vllm_qk_fast only), the previously decoded text
    # (minus asr_streaming_rollback_words tail words) is injected as an
    # assistant-turn prefix so the model only decodes the text delta.
    # The state is per-utterance and resets on sentence commit.
    asr_streaming_prefix_enabled: bool = False
    # Word-level rollback is used (not token-level) so the rolled-back
    # prefix always ends at a clean word boundary. This preserves the
    # `len(words) == len(remove_punctuation(text).split())` invariant
    # that `find_end_time` relies on for sentence commits.
    asr_streaming_rollback_words: int = 2
    asr_streaming_unfixed_chunks: int = 2
    # Extra-context injection (IWSLT 2026 Speech-to-Text with Extra Context
    # sub-track). Default off so every non-context path keeps current
    # behaviour exactly. When a paper artifact is configured, the session
    # prepends a `[Paper context]` block to the Gemma MT prompt outside the
    # current-source span. See docs/CONTEXT_INJECTION.md.
    paper_context_path: str | None = None
    paper_context_mode: str = CONTEXT_MODE_OFF
    paper_context_top_k: int = 3
    paper_context_max_chars: int = 1200
    paper_context_history_window_words: int = 60

    def __post_init__(self) -> None:
        if self.translation_alignatt_heads_path is None:
            self.translation_alignatt_heads_path = alignatt_heads_path_for(
                self.source_lang, self.target_lang
            )
        self._validate()

    def _validate(self) -> None:
        if self.alignment_backend_name not in VALID_ALIGNMENT_BACKEND_NAMES:
            raise ValueError(
                f"Unknown alignment_backend_name: {self.alignment_backend_name!r}"
            )
        if self.mt_backend_name not in VALID_MT_BACKEND_NAMES:
            raise ValueError(
                f"Unknown mt_backend_name: {self.mt_backend_name!r}"
            )
        if (
            self.asr_streaming_prefix_enabled
            and self.alignment_backend_name != "gemma_vllm_qk_fast"
        ):
            raise ValueError(
                "asr_streaming_prefix_enabled=True is only supported with "
                "alignment_backend_name='gemma_vllm_qk_fast' "
                f"(got {self.alignment_backend_name!r})."
            )
        if (
            self.gemma_vllm_force_generate_api
            and self.alignment_backend_name != "gemma_vllm_qk_fast"
        ):
            raise ValueError(
                "gemma_vllm_force_generate_api=True is only meaningful with "
                "alignment_backend_name='gemma_vllm_qk_fast' "
                f"(got {self.alignment_backend_name!r})."
            )
        if self.paper_context_mode not in VALID_CONTEXT_MODES:
            raise ValueError(
                f"paper_context_mode must be one of {VALID_CONTEXT_MODES}, "
                f"got {self.paper_context_mode!r}."
            )
        if (
            self.paper_context_mode != CONTEXT_MODE_OFF
            and self.paper_context_path is None
        ):
            raise ValueError(
                "paper_context_mode is enabled but paper_context_path is None; "
                "either set paper_context_path to a PaperArtifact JSON file or "
                "set paper_context_mode='off'."
            )

    def apply_overrides(self, **overrides) -> None:
        for key, value in overrides.items():
            if not hasattr(self, key):
                raise AttributeError(f"Unknown runtime config override: {key}")
            setattr(self, key, value)
        lang_changed = "source_lang" in overrides or "target_lang" in overrides
        if lang_changed and "translation_alignatt_heads_path" not in overrides:
            self.translation_alignatt_heads_path = alignatt_heads_path_for(
                self.source_lang, self.target_lang
            )
        self._validate()

    def alignment_backend_fingerprint(self) -> tuple:
        """Identity under which a loaded ASR backend is safe to reuse.

        Contains engine-construction knobs only. Live policy knobs
        (commit mode, thresholds, heads path) must not participate —
        they're read per-session, not baked into the engine.
        """
        name = self.alignment_backend_name
        if name == "qwen_forced":
            return (name, float(self.asr_gpu_memory_utilization))
        if name == "gemma_onepass_qk_fast":
            return (
                name,
                self.gemma_transformers_device,
                self.gemma_transformers_dtype,
                self.gemma_transformers_fast_attention,
            )
        if name == "gemma_vllm_qk_fast":
            return (
                name,
                bool(self.gemma_vllm_enforce_eager),
                bool(self.gemma_vllm_enable_prefix_caching),
                self.gemma_vllm_cudagraph_mode,
                float(self.asr_gpu_memory_utilization),
                int(self.gemma_max_model_len),
                bool(self.asr_streaming_prefix_enabled),
            )
        return (name,)

    def mt_backend_fingerprint(self) -> tuple:
        """Identity under which a loaded MT backend is safe to reuse.

        See alignment_backend_fingerprint for the engine-vs-policy split.
        """
        name = self.mt_backend_name
        if name == "gemma_vllm_alignatt":
            return (
                name,
                bool(self.mt_vllm_enforce_eager),
                bool(self.mt_vllm_enable_prefix_caching),
                self.mt_vllm_cudagraph_mode,
                float(self.mt_vllm_gpu_memory_utilization),
                int(self.gemma_max_model_len),
            )
        return (name,)


@dataclass
class PartialTranslationState:
    source_prefix: str = ""
    draft_target: str = ""
    draft_token_ids: tuple[int, ...] = ()
    accepted_target: str = ""
    accepted_token_ids: tuple[int, ...] = ()
    source_accessible_unit_count: int = 0
    source_total_unit_count: int = 0
    last_num_cached_tokens: int | None = None
    last_prompt_num_tokens: int | None = None
    last_accept_audio_seconds: float = 0.0
    last_mt_audio_seconds: float = 0.0
    last_alignatt_metadata: dict[str, Any] | None = None
    blocked_source_local_position: int | None = None
    blocked_source_unit_index: int | None = None


@dataclass
class CascadeState:
    speech_id: int = 0
    source: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    utt_timestamps: list[int] = field(default_factory=lambda: [0])
    utt_sources: list[str] = field(default_factory=lambda: [""])
    utt_translations: list[str] = field(default_factory=lambda: [""])
    asr_hypotheses: list[str] = field(default_factory=lambda: [""])
    partial_word_timestamps_ms: list[tuple[float | None, float | None]] = field(
        default_factory=list
    )
    partial_translation: PartialTranslationState = field(
        default_factory=PartialTranslationState
    )


@dataclass
class SessionProcessingResult:
    asr_text: str
    raw_translation_text: str
    translation_result: MTBackendResult | None
    committed_segments: int


def longest_common_prefix(s1: str, s2: str) -> str:
    for i in range(min(len(s1), len(s2))):
        if s1[i] != s2[i]:
            return s1[:i]
    return s1[: min(len(s1), len(s2))]


def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))


def find_end_time(word_alignments, position: int, text: str):
    if len(word_alignments) != len(remove_punctuation(text).split()):
        return None
    n_words_right = len(remove_punctuation(text[position + 1 :]).strip().split())
    return word_alignments[-n_words_right - 1].end_time


def n_utterances(text: str) -> int:
    n_utt = text.count(". ") + text.count("! ") + text.count("? ")
    if text.endswith((".", "!", "?")):
        n_utt += 1
    return n_utt


def normalize_partial_asr_hypothesis(text: str) -> str:
    """Expose the live ASR tail the way the shipped MT path expects it.

    Sentence-finalized history is managed separately by ``punctuation_lcp +
    EOS flush``. For the still-live tail we keep the full ASR prefix, but we
    strip unstable trailing sentence-final punctuation before building the MT
    prompt. MT-side AlignAtt then decides how much target text is safe to emit.
    """
    text = text.rstrip()
    while text.endswith((".", "!", "?")):
        text = text[:-1].rstrip()
    return text


def translation_history_window(
    items: list[str],
    end_exclusive: int,
    *,
    max_history_utterances: int,
) -> list[str]:
    if max_history_utterances <= 0:
        return []

    start = max(1, end_exclusive - max_history_utterances)
    return [item.strip() for item in items[start:end_exclusive] if item.strip()]


def normalized_source_history_window(
    items: list[str],
    end_exclusive: int,
    *,
    max_history_utterances: int,
) -> list[str]:
    return [
        normalize_source_text_for_mt(item.strip()).text
        for item in translation_history_window(
            items,
            end_exclusive,
            max_history_utterances=max_history_utterances,
        )
        if item.strip()
    ]


def should_run_partial_mt_update(
    *,
    previous_state: PartialTranslationState,
    source_prefix: str,
    accessible_unit_count: int,
) -> tuple[bool, str]:
    source_prefix = source_prefix.strip()
    if not source_prefix:
        return False, "empty_source"
    if not previous_state.source_prefix:
        return True, "initial_partial"
    if not source_prefix.startswith(previous_state.source_prefix):
        return True, "source_rebased"
    if accessible_unit_count > previous_state.source_accessible_unit_count:
        return True, "accessible_frontier_advanced"
    if source_prefix == previous_state.source_prefix:
        return False, "source_prefix_unchanged"
    return True, "source_prefix_extended"


def derive_monotone_partial_acceptance(
    *,
    previous_state: PartialTranslationState,
    source_prefix: str,
    result: MTBackendResult,
) -> tuple[str, tuple[int, ...]]:
    candidate_text = result.acceptance_text.strip()
    candidate_ids = tuple(int(token_id) for token_id in result.accepted_token_ids)
    source_prefix = source_prefix.strip()
    if not source_prefix:
        return "", ()
    if not candidate_text:
        return previous_state.accepted_target, previous_state.accepted_token_ids
    if not previous_state.source_prefix:
        return candidate_text, candidate_ids

    previous_accepted_ids = previous_state.accepted_token_ids
    if not previous_accepted_ids:
        return candidate_text, candidate_ids
    if candidate_ids[: len(previous_accepted_ids)] != previous_accepted_ids:
        return previous_state.accepted_target, previous_accepted_ids
    return candidate_text, candidate_ids


def load_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        num_channels = wav_file.getnchannels()
        raw = wav_file.readframes(wav_file.getnframes())

    if sample_width != 2:
        raise ValueError("Only 16-bit PCM WAV is supported in this simple example.")

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if num_channels > 1:
        audio = audio.reshape(-1, num_channels).mean(axis=1)

    if sample_rate != SAMPLE_RATE:
        duration = len(audio) / sample_rate
        old_times = np.linspace(0.0, duration, num=len(audio), endpoint=False)
        new_length = int(duration * SAMPLE_RATE)
        new_times = np.linspace(0.0, duration, num=new_length, endpoint=False)
        audio = np.interp(new_times, old_times, audio).astype(np.float32)

    return audio


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def _enrich_provenance(
    config: CascadeRuntimeConfig,
    run_provenance: dict[str, Any] | None,
) -> dict[str, Any]:
    provenance = dict(run_provenance or {})
    provenance.setdefault("git_sha", _git_sha())
    provenance.setdefault("framework_mode", "research_harness")
    provenance.setdefault("source_lang", config.source_lang)
    provenance.setdefault("target_lang", config.target_lang)
    provenance.setdefault("alignment_backend_name", config.alignment_backend_name)
    return provenance


def get_translation_variant(config: CascadeRuntimeConfig) -> TranslationVariant:
    return TRANSLATION_VARIANTS[config.translation_variant_id]


def build_alignment_backend(
    config: CascadeRuntimeConfig,
    *,
    qwen_model_path: str = asr_model_name,
    qwen_forced_aligner_model_path: str = forced_aligner_model_name,
    gemma_path: str = gemma_model_name,
) -> AlignmentBackend:
    if config.alignment_backend_name == "qwen_forced":
        from cascade.alignment.qwen_forced_backend import QwenAlignmentBackend

        return QwenAlignmentBackend(
            asr_model_path=qwen_model_path,
            forced_aligner_model_path=qwen_forced_aligner_model_path,
            runtime_config=config,
        )
    if config.alignment_backend_name == "gemma_onepass_qk_fast":
        from cascade.alignment.gemma_transformers_asr_backend import GemmaTransformersASRBackend

        return GemmaTransformersASRBackend(
            model_name=gemma_path,
            runtime_config=config,
            audio_heads_path=config.gemma_audio_alignment_heads_path,
            audio_heads_top_k=int(config.gemma_audio_alignment_top_k_heads),
            filter_width=int(config.gemma_audio_alignment_filter_width),
            max_new_tokens=int(config.gemma_audio_alignment_max_new_tokens),
            audio_align_probe_mode=config.gemma_audio_align_probe_mode,
        )
    if config.alignment_backend_name == "gemma_vllm_qk_fast":
        from cascade.alignment.gemma_vllm_asr_backend import GemmaVLLMASRBackend

        return GemmaVLLMASRBackend(
            model_name=gemma_path,
            runtime_config=config,
            audio_heads_path=config.gemma_audio_alignment_heads_path,
            audio_heads_top_k=int(config.gemma_audio_alignment_top_k_heads),
            filter_width=int(config.gemma_audio_alignment_filter_width),
            max_new_tokens=int(config.gemma_audio_alignment_max_new_tokens),
        )
    raise ValueError(f"Unknown alignment_backend_name: {config.alignment_backend_name!r}")


class LoadedModelBundle:
    def __init__(self, config: CascadeRuntimeConfig):
        self.config = config
        self.qwen_model_path = asr_model_name
        self.qwen_forced_aligner_model_path = forced_aligner_model_name
        self.gemma_path = gemma_model_name
        self.alignment_backend: AlignmentBackend | None = None
        self.mt_backend = None
        self._alignment_backend_fp: tuple | None = None
        self._mt_backend_fp: tuple | None = None
        self._mt_heads_path: str | None = None
        self._paper_context_selector: PaperContextSelector | None = None
        self._paper_context_path: str | None = None

    def ensure_alignment_backend(self) -> AlignmentBackend:
        current_fp = self.config.alignment_backend_fingerprint()
        if (
            self.alignment_backend is None
            or self._alignment_backend_fp != current_fp
        ):
            self.alignment_backend = build_alignment_backend(
                self.config,
                qwen_model_path=self.qwen_model_path,
                qwen_forced_aligner_model_path=self.qwen_forced_aligner_model_path,
                gemma_path=self.gemma_path,
            )
            self.alignment_backend.load()
            self._alignment_backend_fp = current_fp
        else:
            runtime_config = getattr(self.alignment_backend, "runtime_config", None)
            if runtime_config is not None:
                self.alignment_backend.runtime_config = self.config
        return self.alignment_backend

    def ensure_mt_backend(self):
        current_fp = self.config.mt_backend_fingerprint()
        if self.mt_backend is None or self._mt_backend_fp != current_fp:
            self.mt_backend = build_mt_backend(
                model_name=self.gemma_path,
                runtime_config=self.config,
            )
            self.mt_backend.load()
            self._mt_heads_path = self.config.translation_alignatt_heads_path
            self._mt_backend_fp = current_fp
        else:
            self.mt_backend.runtime_config = self.config
            current_heads_path = self.config.translation_alignatt_heads_path
            if current_heads_path != self._mt_heads_path:
                self.mt_backend.refresh_alignatt_artifacts()
                self._mt_heads_path = current_heads_path
        return self.mt_backend

    def ensure_paper_context_selector(self) -> PaperContextSelector | None:
        """Load / reload the PaperContextSelector iff the config points at a PDF artifact.

        The selector is immutable per artifact path: we only rebuild when the
        path changes (flipping modes on the same artifact does *not* trigger a
        rebuild). Returns ``None`` when extra-context injection is disabled.
        """
        path = self.config.paper_context_path
        if path is None:
            self._paper_context_selector = None
            self._paper_context_path = None
            return None
        if self._paper_context_selector is None or self._paper_context_path != path:
            artifact = PaperArtifact.read_json(path)
            self._paper_context_selector = PaperContextSelector.from_artifact(artifact)
            self._paper_context_path = path
        return self._paper_context_selector

    def load(self) -> None:
        self.ensure_alignment_backend()
        self.ensure_mt_backend()
        self.ensure_paper_context_selector()

    def new_session(self) -> "CascadeSession":
        return CascadeSession(self)


class TranslationUnitManager:
    def __init__(self, session: "CascadeSession"):
        self.session = session

    @property
    def config(self) -> CascadeRuntimeConfig:
        return self.session.config

    @property
    def state(self) -> CascadeState:
        return self.session.state

    def reset_partial_state(self) -> None:
        self.state.partial_translation = PartialTranslationState()

    def normalize_source_text(
        self,
        source_text: str,
        *,
        is_final: bool,
    ) -> NormalizedSourceText:
        return normalize_source_text_for_mt(
            source_text.strip(),
            word_timestamps_ms=(None if is_final else self.state.partial_word_timestamps_ms),
        )

    def build_source_frontier(
        self,
        source_text: NormalizedSourceText,
        *,
        is_final: bool,
    ) -> SourceAccessibilityFrontier:
        return build_source_accessibility_frontier(
            source_text.text,
            word_timestamps_ms=source_text.word_timestamps_ms,
            current_audio_ms=self.session.current_audio_seconds() * 1000.0,
            inaccessible_ms=float(self.config.translation_alignatt_inaccessible_ms),
            is_final=is_final,
        )

    def current_accepted_prefill(self, source_text: str) -> str:
        source_text = source_text.strip()
        if not source_text:
            self.reset_partial_state()
            return ""
        return self.state.partial_translation.accepted_target

    def should_run_partial_mt(
        self,
        *,
        source_text: str,
        source_frontier: SourceAccessibilityFrontier,
    ) -> tuple[bool, str]:
        return should_run_partial_mt_update(
            previous_state=self.state.partial_translation,
            source_prefix=source_text,
            accessible_unit_count=source_frontier.accessible_unit_count,
        )

    def snapshot_skipped_partial_result(
        self,
        *,
        source_frontier: SourceAccessibilityFrontier,
        scheduler_reason: str,
    ) -> MTBackendResult:
        previous_state = self.state.partial_translation
        alignatt_metadata = dict(previous_state.last_alignatt_metadata or {})
        alignatt_metadata.update(
            {
                "scheduler_skipped": True,
                "scheduler_reason": scheduler_reason,
                "accessible_source_unit_count": source_frontier.accessible_unit_count,
                "source_unit_count": len(source_frontier.units),
                "current_audio_ms": source_frontier.current_audio_ms,
                "inaccessible_ms": source_frontier.inaccessible_ms,
            }
        )
        return MTBackendResult(
            draft_text=previous_state.draft_target,
            acceptance_text=previous_state.accepted_target,
            draft_token_ids=previous_state.draft_token_ids,
            accepted_token_ids=previous_state.accepted_token_ids,
            num_cached_tokens=previous_state.last_num_cached_tokens,
            prompt_num_tokens=previous_state.last_prompt_num_tokens,
            stop_reason=f"scheduler:{scheduler_reason}",
            alignatt_metadata=alignatt_metadata,
            timings_ms={"scheduler_skip": 0.0},
        )

    def derive_monotone_acceptance(
        self,
        *,
        source_prefix: str,
        result: MTBackendResult,
    ) -> tuple[str, tuple[int, ...]]:
        return derive_monotone_partial_acceptance(
            previous_state=self.state.partial_translation,
            source_prefix=source_prefix,
            result=result,
        )

    def update_partial_state(
        self,
        source_text: str,
        result: MTBackendResult,
        source_frontier: SourceAccessibilityFrontier,
    ) -> None:
        previous_accepted = self.state.partial_translation.accepted_target
        accepted_target, accepted_token_ids = self.derive_monotone_acceptance(
            source_prefix=source_text.strip(),
            result=result,
        )
        last_accept_audio_seconds = self.state.partial_translation.last_accept_audio_seconds
        if accepted_target and accepted_target != previous_accepted:
            last_accept_audio_seconds = self.session.current_audio_seconds()
        self.state.partial_translation = PartialTranslationState(
            source_prefix=source_text.strip(),
            draft_target=result.draft_text,
            draft_token_ids=tuple(int(token_id) for token_id in result.draft_token_ids),
            accepted_target=accepted_target,
            accepted_token_ids=accepted_token_ids,
            source_accessible_unit_count=source_frontier.accessible_unit_count,
            source_total_unit_count=len(source_frontier.units),
            last_num_cached_tokens=result.num_cached_tokens,
            last_prompt_num_tokens=result.prompt_num_tokens,
            last_accept_audio_seconds=last_accept_audio_seconds,
            last_mt_audio_seconds=self.session.current_audio_seconds(),
            last_alignatt_metadata=result.alignatt_metadata,
            blocked_source_local_position=(
                None
                if result.alignatt_metadata is None
                else result.alignatt_metadata.get("blocked_source_local_position")
            ),
            blocked_source_unit_index=(
                None
                if result.alignatt_metadata is None
                else result.alignatt_metadata.get("blocked_source_unit_index")
            ),
        )

    def sync_committed_translations(self) -> MTBackendResult | None:
        last_result: MTBackendResult | None = None
        while len(self.state.utt_translations) < len(self.state.utt_sources):
            segment_idx = len(self.state.utt_translations)
            segment_source = self.state.utt_sources[segment_idx].strip()
            normalized_segment_source = self.normalize_source_text(
                segment_source,
                is_final=True,
            )
            assistant_prefill = ""
            if (
                self.state.partial_translation.accepted_target
                and segment_idx == len(self.state.utt_sources) - 1
            ):
                assistant_prefill = self.state.partial_translation.accepted_target
            last_result = self.session.translate_with_mt(
                normalized_segment_source.text,
                source_frontier=self.build_source_frontier(
                    normalized_segment_source,
                    is_final=True,
                ),
                source_history=normalized_source_history_window(
                    self.state.utt_sources,
                    segment_idx,
                    max_history_utterances=self.config.max_history_utterances,
                ),
                translation_history=translation_history_window(
                    self.state.utt_translations,
                    segment_idx,
                    max_history_utterances=self.config.max_history_utterances,
                ),
                is_partial=False,
                assistant_prefill=assistant_prefill,
            )
            self.state.utt_translations.append(last_result.acceptance_text)
            if assistant_prefill:
                self.reset_partial_state()
        return last_result

    def render_translation(self) -> tuple[str, MTBackendResult | None]:
        latest_result = self.sync_committed_translations()

        translation_segments = [
            segment for segment in self.state.utt_translations[1:] if segment.strip()
        ]
        # Partial MT conditions on the full live ASR tail; only unstable
        # trailing sentence-final punctuation is stripped beforehand.
        partial_source = self.session.current_live_asr_tail_text()
        normalized_partial_source = self.normalize_source_text(
            partial_source,
            is_final=False,
        )
        if normalized_partial_source.text:
            partial_frontier = self.build_source_frontier(
                normalized_partial_source,
                is_final=False,
            )
            should_run_partial, scheduler_reason = self.should_run_partial_mt(
                source_text=normalized_partial_source.text,
                source_frontier=partial_frontier,
            )
            if should_run_partial:
                partial_result = self.session.translate_with_mt(
                    normalized_partial_source.text,
                    source_frontier=partial_frontier,
                    source_history=normalized_source_history_window(
                        self.state.utt_sources,
                        len(self.state.utt_sources),
                        max_history_utterances=self.config.max_history_utterances,
                    ),
                    translation_history=translation_history_window(
                        self.state.utt_translations,
                        len(self.state.utt_translations),
                        max_history_utterances=self.config.max_history_utterances,
                    ),
                    is_partial=True,
                    assistant_prefill=self.current_accepted_prefill(
                        normalized_partial_source.text
                    ),
                )
                self.update_partial_state(
                    normalized_partial_source.text,
                    partial_result,
                    partial_frontier,
                )
            else:
                partial_result = self.snapshot_skipped_partial_result(
                    source_frontier=partial_frontier,
                    scheduler_reason=scheduler_reason,
                )
            if self.state.partial_translation.accepted_target.strip():
                translation_segments.append(self.state.partial_translation.accepted_target)
            latest_result = partial_result
        else:
            self.reset_partial_state()

        return (
            normalize_incremental_target_text(
                " ".join(
                    segment.strip() for segment in translation_segments if segment.strip()
                )
            ),
            latest_result,
        )


class CascadeSession:
    def __init__(self, bundle: LoadedModelBundle):
        self.bundle = bundle
        self.config = bundle.config
        self.state = CascadeState()
        self.mt_prompt_cache = PromptCacheState()
        self.translation_units = TranslationUnitManager(self)
        # Step 5: Qwen-style prompt-prefix streaming state. The session
        # carries the previous *full* ASR hypothesis (text + words) across
        # chunks so the next backend call can reuse it as an assistant-side
        # prefix, decoding only the text delta. Resets on utterance commit
        # and on clear().
        self._asr_streaming_last_text: str = ""
        self._asr_streaming_last_words: tuple[WordAlignment, ...] = ()
        self._asr_streaming_chunk_count: int = 0
        self._asr_streaming_committed_segment_count: int = len(self.state.utt_sources)

    def load_models(self) -> None:
        self.bundle.load()

    def clear(self) -> None:
        speech_id = self.state.speech_id
        self.state = CascadeState(speech_id=speech_id)
        self.mt_prompt_cache = PromptCacheState()
        self.translation_units = TranslationUnitManager(self)
        self._reset_asr_streaming_state(reset_backend=True)

    def _reset_asr_streaming_state(self, *, reset_backend: bool) -> None:
        self._asr_streaming_last_text = ""
        self._asr_streaming_last_words = ()
        self._asr_streaming_chunk_count = 0
        self._asr_streaming_committed_segment_count = len(self.state.utt_sources)
        if reset_backend:
            alignment_backend = getattr(self.bundle, "alignment_backend", None)
            if alignment_backend is not None:
                alignment_backend.reset_streaming_state()

    def current_audio_seconds(self) -> float:
        return len(self.state.source) / SAMPLE_RATE

    def current_live_asr_tail_text(self) -> str:
        if not self.state.asr_hypotheses:
            return ""
        return normalize_partial_asr_hypothesis(self.state.asr_hypotheses[-1])

    def render_public_asr_text(self) -> str:
        committed_segments = [
            segment.strip() for segment in self.state.utt_sources[1:] if segment.strip()
        ]
        partial_segment = self.current_live_asr_tail_text()
        if partial_segment:
            committed_segments.append(partial_segment)
        return " ".join(committed_segments).strip()

    def build_translation_messages(
        self,
        text: str,
        *,
        source_frontier: SourceAccessibilityFrontier | None,
        source_history: list[str],
        translation_history: list[str],
        is_partial: bool,
        assistant_prefill: str = "",
    ) -> RenderedTranslationPrompt:
        variant = get_translation_variant(self.config)
        paper_context_block = self._render_paper_context_block(
            current_source_prefix=text,
            source_history=source_history,
        )
        return variant.render_messages(
            source_lang=self.config.source_lang,
            target_lang=self.config.target_lang,
            text=text,
            source_frontier=source_frontier,
            source_history=source_history,
            translation_history=translation_history,
            is_partial=is_partial,
            assistant_prefill=assistant_prefill,
            paper_context_block=paper_context_block.render(),
        )

    def _render_paper_context_block(
        self,
        *,
        current_source_prefix: str,
        source_history: Sequence[str],
    ) -> PaperContextBlock:
        mode = self.config.paper_context_mode
        if mode == CONTEXT_MODE_OFF:
            return PaperContextBlock(text="", mode=CONTEXT_MODE_OFF)
        selector = self.bundle.ensure_paper_context_selector()
        if selector is None:
            return PaperContextBlock(text="", mode=CONTEXT_MODE_OFF)
        history_words: list[str] = []
        for item in source_history:
            if not item:
                continue
            history_words.extend(item.split())
        query = build_retrieval_query(
            current_source_prefix=current_source_prefix,
            history_words=history_words,
            history_window_words=int(self.config.paper_context_history_window_words),
        )
        return selector.select(
            mode=mode,
            query=query,
            top_k=int(self.config.paper_context_top_k),
            max_chars=int(self.config.paper_context_max_chars),
        )

    def translate_with_mt(
        self,
        text: str,
        *,
        source_frontier: SourceAccessibilityFrontier | None = None,
        source_history: list[str] | None = None,
        translation_history: list[str] | None = None,
        is_partial: bool = False,
        assistant_prefill: str = "",
    ) -> MTBackendResult:
        mt_backend = self.bundle.ensure_mt_backend()
        text = text.strip()
        if not text:
            prefixed_text = assistant_prefill.rstrip(" \n")
            semantic_token_ids: tuple[int, ...] = ()
            if hasattr(mt_backend, "encode_semantic_target_token_ids"):
                semantic_token_ids = tuple(
                    mt_backend.encode_semantic_target_token_ids(prefixed_text)
                )
            return MTBackendResult(
                draft_text=prefixed_text,
                acceptance_text=prefixed_text,
                draft_token_ids=semantic_token_ids,
                accepted_token_ids=semantic_token_ids,
            )

        variant = get_translation_variant(self.config)
        rendered_prompt = self.build_translation_messages(
            text,
            source_frontier=source_frontier,
            source_history=source_history or [],
            translation_history=translation_history or [],
            is_partial=is_partial,
            assistant_prefill=assistant_prefill,
        )
        return mt_backend.translate(
            rendered_prompt=rendered_prompt,
            variant=variant,
            is_partial=is_partial,
            prompt_cache_state=self.mt_prompt_cache,
        )

    def transcribe_audio(self, *, is_final_chunk: bool = False) -> str | None:
        """Run ASR on the currently-uncommitted audio tail and apply the commit rule.

        ``is_final_chunk=True`` (used by ``finalize_stream``) switches the
        punctuation-LCP commit rule to an **EOS flush** that emits the whole
        current hypothesis even without a sentence-final punctuation cue.
        Without it, the last 1–2 trailing words of every clip stay "partial"
        and never make it into the public hypothesis.
        """
        alignment_backend = self.bundle.ensure_alignment_backend()
        audio = np.array(
            self.state.source[self.state.utt_timestamps[-1] :], dtype=np.float32
        )

        if (
            self.config.asr_streaming_prefix_enabled
            and len(self.state.utt_sources) != self._asr_streaming_committed_segment_count
        ):
            self._reset_asr_streaming_state(reset_backend=True)

        prefix_text, prefix_words = self._compute_streaming_prefix(alignment_backend)

        result = alignment_backend.transcribe_and_align(
            audio,
            sample_rate=SAMPLE_RATE,
            language=self.config.source_lang,
            streaming_prefix_text=prefix_text,
            streaming_prefix_words=prefix_words,
        )
        if result is None:
            return None

        asr_hypo = result.text
        self.state.asr_hypotheses.append(asr_hypo)
        self.state.partial_word_timestamps_ms = normalize_word_timestamps_ms(result.words)

        if self.config.asr_streaming_prefix_enabled:
            self._asr_streaming_last_text = asr_hypo
            self._asr_streaming_last_words = tuple(result.words)
            self._asr_streaming_chunk_count += 1

        asr_segment = longest_common_prefix(
            self.state.asr_hypotheses[-2],
            self.state.asr_hypotheses[-1],
        )
        if os.environ.get("CASCADE_ASR_STREAMING_DEBUG"):
            audio_s = (len(self.state.source) - self.state.utt_timestamps[-1]) / SAMPLE_RATE
            print(
                f"[asr-stream] chunk={self._asr_streaming_chunk_count} "
                f"audio_slice_s={audio_s:.2f} "
                f"prefix_words={len(prefix_words)} "
                f"hypo_words={len(result.words)} n_utt={n_utterances(asr_segment)} "
                f"hypo[-60:]={asr_hypo[-60:]!r} "
                f"lcp[-60:]={asr_segment[-60:]!r}",
                flush=True,
            )

        committed = self._try_commit_punctuation_lcp(
            asr_hypo=asr_hypo,
            result=result,
            lcp_text=asr_segment,
            is_final_chunk=is_final_chunk,
        )
        if committed is _COMMIT_ABORT:
            return asr_hypo.strip()

        if self.state.utt_sources[1:]:
            return self.render_public_asr_text()
        return self.current_live_asr_tail_text()

    def _try_commit_punctuation_lcp(
        self,
        *,
        asr_hypo: str,
        result: "AlignmentResult",
        lcp_text: str,
        is_final_chunk: bool = False,
    ) -> "object":
        if is_final_chunk:
            # EOS flush: commit the whole current hypothesis, even without
            # sentence-final punctuation. No more audio is coming, so waiting
            # for a punctuation signal would lose the trailing words.
            words = result.words or ()
            if not asr_hypo.strip() or not words:
                return None
            last_word_end = float(words[-1].end_time) if words else 0.0
            self._apply_commit(
                committed_text=asr_hypo.strip(),
                remainder_text="",
                end_time_s=last_word_end,
            )
            return None

        if n_utterances(lcp_text) < 1:
            return None
        rightest_punct_idx = max(
            lcp_text.rfind(". "),
            lcp_text.rfind("! "),
            lcp_text.rfind("? "),
        )
        if rightest_punct_idx == -1 and lcp_text.endswith((".", "!", "?")):
            rightest_punct_idx = len(lcp_text) - 1

        end_time = find_end_time(result.words, rightest_punct_idx, asr_hypo)
        if end_time is None:
            return _COMMIT_ABORT

        self._apply_commit(
            committed_text=lcp_text[: rightest_punct_idx + 1],
            remainder_text=asr_hypo[rightest_punct_idx + 1 :].strip(),
            end_time_s=float(end_time),
        )
        return None

    def _apply_commit(
        self, *, committed_text: str, remainder_text: str, end_time_s: float
    ) -> None:
        utt_end_time = int(end_time_s * SAMPLE_RATE) + self.state.utt_timestamps[-1]
        utt_end_time = min(utt_end_time, len(self.state.source))
        self.state.utt_timestamps.append(utt_end_time)
        self.state.utt_sources.append(committed_text)
        n_words_right = len(remove_punctuation(remainder_text).strip().split())
        self.state.asr_hypotheses = [remainder_text]
        self.state.partial_word_timestamps_ms = (
            self.state.partial_word_timestamps_ms[-n_words_right:]
            if n_words_right > 0
            else []
        )
        if self.config.asr_streaming_prefix_enabled:
            self._reset_asr_streaming_state(reset_backend=True)

    def _compute_streaming_prefix(
        self,
        alignment_backend: AlignmentBackend,
    ) -> tuple[str, tuple[WordAlignment, ...]]:
        """Build the Qwen-style rolled-back assistant prefix for streaming.

        Uses word-level rollback so the prefix always ends at a clean word
        boundary. This preserves the word-count invariant that
        :func:`find_end_time` relies on for sentence commits, which
        token-level rollback broke when the rolled-back boundary landed
        mid-word and the continuation's word aggregation produced a
        fragment word that did not match the full text's split.
        """
        del alignment_backend
        if not self.config.asr_streaming_prefix_enabled:
            return "", ()
        if self._asr_streaming_chunk_count < self.config.asr_streaming_unfixed_chunks:
            return "", ()
        last_text = self._asr_streaming_last_text
        last_words = self._asr_streaming_last_words
        if not last_text or not last_words:
            return "", ()
        rollback = max(0, int(self.config.asr_streaming_rollback_words))
        keep_count = len(last_words) - rollback
        if keep_count <= 0:
            return "", ()
        kept_words = last_words[:keep_count]
        cursor = 0
        last_end_pos = 0
        for word in kept_words:
            candidate = word.text
            if not candidate:
                continue
            idx = last_text.find(candidate, cursor)
            if idx < 0:
                return "", ()
            cursor = idx + len(candidate)
            last_end_pos = cursor
        # Include trailing punctuation immediately after the last kept
        # word so sentence-terminal markers are preserved in the prefix.
        while last_end_pos < len(last_text) and last_text[last_end_pos] in ".,!?;:":
            last_end_pos += 1
        prefix_text = last_text[:last_end_pos]
        if not prefix_text:
            return "", ()
        return prefix_text, tuple(kept_words)

    def render_translation(self) -> tuple[str, MTBackendResult | None]:
        return self.translation_units.render_translation()

    def apply_translation_emit_policy(
        self,
        previous_translation: str,
        raw_translation: str,
        *,
        is_final: bool,
    ) -> tuple[str, str]:
        return apply_emission_policy(
            self.config.translation_emit_policy,
            previous_translation,
            raw_translation,
            max_tail_rewrite_words=self.config.translation_max_tail_rewrite_words,
            is_final=is_final,
            target_lang_code=target_lang_code_for(self.config.target_lang),
        )

    def process_audio_chunk(self, chunk: np.ndarray) -> SessionProcessingResult | None:
        self.state.source = np.concatenate(
            [self.state.source, np.asarray(chunk, dtype=np.float32)]
        )
        if self.current_audio_seconds() < self.config.min_start_seconds:
            return None

        current_asr = self.transcribe_audio()
        if not current_asr:
            return None

        raw_translation, translation_result = self.render_translation()
        return SessionProcessingResult(
            asr_text=current_asr,
            raw_translation_text=raw_translation,
            translation_result=translation_result,
            committed_segments=len(self.state.utt_sources),
        )

    def finalize_stream(self) -> SessionProcessingResult:
        # is_final_chunk=True lets the punctuation-LCP commit rule flush the
        # tail even without a sentence-final punctuation cue. Without it the
        # last 1–2 trailing words of every clip stay "partial" and never make
        # it into the hypothesis. Env flag CASCADE_DISABLE_EOS_FLUSH=1 lets
        # the A/B compare the fix.
        eos_flush = os.environ.get("CASCADE_DISABLE_EOS_FLUSH", "") != "1"
        final_asr = (
            self.transcribe_audio(is_final_chunk=eos_flush)
            or self.render_public_asr_text()
        )
        final_raw_translation, final_translation_result = self.render_translation()
        return SessionProcessingResult(
            asr_text=final_asr,
            raw_translation_text=final_raw_translation,
            translation_result=final_translation_result,
            committed_segments=len(self.state.utt_sources),
        )

    def run_stream_to_artifacts(
        self,
        wav_path: str,
        chunk_ms: int = 960,
        *,
        run_provenance: dict[str, Any] | None = None,
    ) -> InferenceArtifacts:
        self.load_models()
        self.clear()

        variant = get_translation_variant(self.config)
        target_lang_code = target_lang_code_for(self.config.target_lang)
        audio = load_wav(wav_path)
        chunk_size = int(SAMPLE_RATE * chunk_ms / 1000)
        last_asr = ""
        last_translation = ""
        last_raw_translation = ""
        last_committed_segments = len(self.state.utt_sources)
        word_delays_ms: list[float] = []
        word_elapsed_ms: list[float] = []
        updates: list[StreamUpdate] = []
        start_time = perf_counter()

        for start in range(0, len(audio), chunk_size):
            chunk = audio[start : start + chunk_size]
            session_result = self.process_audio_chunk(chunk)
            committed_segments = len(self.state.utt_sources)
            if session_result is None:
                continue
            if (
                session_result.asr_text == last_asr
                and committed_segments == last_committed_segments
            ):
                continue

            last_asr = session_result.asr_text
            last_committed_segments = committed_segments
            translation, emission_policy_action = self.apply_translation_emit_policy(
                last_translation,
                session_result.raw_translation_text,
                is_final=False,
            )
            audio_processed_ms = len(self.state.source) * 1000.0 / SAMPLE_RATE
            wallclock_elapsed_ms = (perf_counter() - start_time) * 1000.0
            register_translation_timestamps(
                last_raw_translation,
                session_result.raw_translation_text,
                wallclock_elapsed_ms,
                word_elapsed_ms,
                target_lang_code=target_lang_code,
            )
            new_words = register_translation_words(
                last_translation,
                translation,
                audio_processed_ms,
                word_delays_ms,
                target_lang_code=target_lang_code,
            )
            updates.append(
                StreamUpdate(
                    update_idx=len(updates),
                    audio_processed_ms=audio_processed_ms,
                    wallclock_elapsed_ms=wallclock_elapsed_ms,
                    asr_text=session_result.asr_text,
                    translation_text=translation,
                    new_words=new_words,
                    raw_translation_text=session_result.raw_translation_text,
                    emission_policy_action=emission_policy_action,
                    translation_prompt_num_cached_tokens=(
                        None
                        if session_result.translation_result is None
                        else session_result.translation_result.num_cached_tokens
                    ),
                    translation_prompt_num_tokens=(
                        None
                        if session_result.translation_result is None
                        else session_result.translation_result.prompt_num_tokens
                    ),
                    partial_accepted_target=(
                        self.state.partial_translation.accepted_target or None
                    ),
                    partial_draft_target=(
                        self.state.partial_translation.draft_target or None
                    ),
                    partial_accepted_token_count=(
                        len(self.state.partial_translation.accepted_token_ids) or None
                    ),
                    alignatt_metadata=(
                        None
                        if session_result.translation_result is None
                        else session_result.translation_result.alignatt_metadata
                    ),
                    translation_timings_ms=(
                        None
                        if session_result.translation_result is None
                        else session_result.translation_result.timings_ms
                    ),
                )
            )
            current_time = audio_processed_ms / 1000.0
            print(f"[{current_time:6.2f}s] ASR: {session_result.asr_text}")
            print(f"[{current_time:6.2f}s] {target_lang_code.upper():<3}: {translation}")
            last_translation = translation
            last_raw_translation = session_result.raw_translation_text

        final_result = self.finalize_stream()
        final_translation, final_emission_policy_action = self.apply_translation_emit_policy(
            last_translation,
            final_result.raw_translation_text,
            is_final=True,
        )
        audio_duration_ms = len(audio) * 1000.0 / SAMPLE_RATE
        final_elapsed_ms = (perf_counter() - start_time) * 1000.0
        register_translation_timestamps(
            last_raw_translation,
            final_result.raw_translation_text,
            final_elapsed_ms,
            word_elapsed_ms,
            target_lang_code=target_lang_code,
        )
        final_new_words = register_translation_words(
            last_translation,
            final_translation,
            audio_duration_ms,
            word_delays_ms,
            target_lang_code=target_lang_code,
        )
        if (
            final_result.asr_text != last_asr
            or final_translation != last_translation
        ):
            updates.append(
                StreamUpdate(
                    update_idx=len(updates),
                    audio_processed_ms=audio_duration_ms,
                    wallclock_elapsed_ms=final_elapsed_ms,
                    asr_text=final_result.asr_text,
                    translation_text=final_translation,
                    new_words=final_new_words,
                    is_eos=True,
                    raw_translation_text=final_result.raw_translation_text,
                    emission_policy_action=final_emission_policy_action,
                    translation_prompt_num_cached_tokens=(
                        None
                        if final_result.translation_result is None
                        else final_result.translation_result.num_cached_tokens
                    ),
                    translation_prompt_num_tokens=(
                        None
                        if final_result.translation_result is None
                        else final_result.translation_result.prompt_num_tokens
                    ),
                    partial_accepted_target=(
                        self.state.partial_translation.accepted_target or None
                    ),
                    partial_draft_target=(
                        self.state.partial_translation.draft_target or None
                    ),
                    partial_accepted_token_count=(
                        len(self.state.partial_translation.accepted_token_ids) or None
                    ),
                    alignatt_metadata=(
                        None
                        if final_result.translation_result is None
                        else final_result.translation_result.alignatt_metadata
                    ),
                    translation_timings_ms=(
                        None
                        if final_result.translation_result is None
                        else final_result.translation_result.timings_ms
                    ),
                )
            )

        print("\nFinal ASR:")
        print(final_result.asr_text)
        print("\nFinal translation:")
        print(final_translation)

        return InferenceArtifacts(
            wav_path=wav_path,
            chunk_ms=chunk_ms,
            translation_variant=variant.variant_id,
            source_language=self.config.source_lang,
            target_language=self.config.target_lang,
            source_language_code=LANGUAGE_NAME_TO_CODE.get(
                self.config.source_lang,
                self.config.source_lang.lower(),
            ),
            target_language_code=target_lang_code_for(self.config.target_lang),
            latency_unit=self.config.latency_unit,
            audio_duration_ms=audio_duration_ms,
            final_asr_text=final_result.asr_text,
            final_translation_text=final_translation,
            translation_word_delays_ms=word_delays_ms,
            translation_word_elapsed_ms=word_elapsed_ms,
            updates=updates,
            runtime_config={
                "translation_variant_id": variant.variant_id,
                "translation_variant_description": variant.description,
                "alignment_backend_name": self.config.alignment_backend_name,
                "translation_alignatt_heads_path": self.config.translation_alignatt_heads_path,
                "translation_alignatt_top_k_heads": self.config.translation_alignatt_top_k_heads,
                "translation_alignatt_filter_width": self.config.translation_alignatt_filter_width,
                "translation_alignatt_probe_mode": self.config.translation_alignatt_probe_mode,
                "translation_alignatt_inaccessible_ms": self.config.translation_alignatt_inaccessible_ms,
                "translation_alignatt_rewind_threshold": self.config.translation_alignatt_rewind_threshold,
                "translation_alignatt_min_source_mass": self.config.translation_alignatt_min_source_mass,
                "min_start_seconds": self.config.min_start_seconds,
                "max_history_utterances": self.config.max_history_utterances,
                "max_new_tokens": self.config.max_new_tokens,
                "partial_max_new_tokens": self.config.partial_max_new_tokens,
                "translation_min_new_tokens": self.config.translation_min_new_tokens,
                "translation_token_budget_ratio": self.config.translation_token_budget_ratio,
                "translation_token_budget_buffer": self.config.translation_token_budget_buffer,
                "partial_translation_min_new_tokens": self.config.partial_translation_min_new_tokens,
                "partial_translation_token_budget_ratio": self.config.partial_translation_token_budget_ratio,
                "partial_translation_token_budget_buffer": self.config.partial_translation_token_budget_buffer,
                "translation_generation_margin": self.config.translation_generation_margin,
                "translation_emit_policy": self.config.translation_emit_policy,
                "translation_max_tail_rewrite_words": self.config.translation_max_tail_rewrite_words,
                "temperature": self.config.temperature,
                "repetition_penalty": self.config.repetition_penalty,
                "asr_gpu_memory_utilization": self.config.asr_gpu_memory_utilization,
                "gemma_max_model_len": self.config.gemma_max_model_len,
                "gemma_enable_prefix_caching": self.config.gemma_enable_prefix_caching,
                "gemma_transformers_device": self.config.gemma_transformers_device,
                "gemma_transformers_dtype": self.config.gemma_transformers_dtype,
                "gemma_transformers_fast_attention": self.config.gemma_transformers_fast_attention,
                "gemma_transformers_prompt_kv_reuse": self.config.gemma_transformers_prompt_kv_reuse,
                "gemma_audio_align_probe_mode": self.config.gemma_audio_align_probe_mode,
                "gemma_audio_alignment_heads_path": self.config.gemma_audio_alignment_heads_path,
            },
            run_provenance=_enrich_provenance(self.config, run_provenance),
        )


@contextmanager
def temporary_runtime_config(
    config: CascadeRuntimeConfig,
    **overrides,
):
    if not overrides:
        yield config
        return

    original_values: dict[str, Any] = {}
    for key, value in overrides.items():
        if not hasattr(config, key):
            raise AttributeError(f"Unknown runtime config override: {key}")
        original_values[key] = getattr(config, key)
        setattr(config, key, value)

    lang_changed = "source_lang" in overrides or "target_lang" in overrides
    if lang_changed and "translation_alignatt_heads_path" not in overrides:
        original_values.setdefault(
            "translation_alignatt_heads_path",
            getattr(config, "translation_alignatt_heads_path"),
        )
        config.translation_alignatt_heads_path = alignatt_heads_path_for(
            config.source_lang,
            config.target_lang,
        )

    try:
        yield config
    finally:
        for key, value in original_values.items():
            setattr(config, key, value)


def run_stream_to_artifacts(
    wav_path: str,
    chunk_ms: int = 960,
    *,
    config: CascadeRuntimeConfig | None = None,
    bundle: LoadedModelBundle | None = None,
    run_provenance: dict[str, Any] | None = None,
) -> InferenceArtifacts:
    runtime_config = config or CascadeRuntimeConfig()
    runtime_bundle = bundle or LoadedModelBundle(runtime_config)
    session = runtime_bundle.new_session()
    return session.run_stream_to_artifacts(
        wav_path,
        chunk_ms=chunk_ms,
        run_provenance=run_provenance,
    )


def run_stream(
    wav_path: str,
    chunk_ms: int = 960,
    output_dir: str | None = None,
    runtime_overrides: dict[str, Any] | None = None,
    run_provenance: dict[str, Any] | None = None,
    *,
    config: CascadeRuntimeConfig | None = None,
    bundle: LoadedModelBundle | None = None,
):
    runtime_config = config or CascadeRuntimeConfig()
    with temporary_runtime_config(runtime_config, **(runtime_overrides or {})):
        artifacts = run_stream_to_artifacts(
            wav_path,
            chunk_ms=chunk_ms,
            config=runtime_config,
            bundle=bundle,
            run_provenance=run_provenance,
        )
        if output_dir is not None:
            write_inference_artifacts(artifacts, output_dir)

    return artifacts.final_asr_text, artifacts.final_translation_text


def run_baseline(
    wav_path: str = DEFAULT_WAV_PATH,
    *,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    chunk_ms: int = 960,
    runtime_overrides: dict[str, Any] | None = None,
    run_provenance: dict[str, Any] | None = None,
    config: CascadeRuntimeConfig | None = None,
    bundle: LoadedModelBundle | None = None,
):
    runtime_config = config or CascadeRuntimeConfig()
    with temporary_runtime_config(runtime_config, **(runtime_overrides or {})):
        artifacts = run_stream_to_artifacts(
            wav_path,
            chunk_ms=chunk_ms,
            config=runtime_config,
            bundle=bundle,
            run_provenance=run_provenance,
        )
        written_files = write_inference_artifacts(artifacts, output_dir)
        print(f"\nWrote baseline artifacts to {output_dir}")
        for label, path in written_files.items():
            print(f"- {label}: {path}")

    return written_files
