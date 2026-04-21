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

from cascade.alignment.base import AlignmentBackend, AlignmentResult, WordAlignment
from cascade.alignment.gemma_alignatt_stream import (
    GemmaAlignAttStream,
    StreamStepDelta,
)
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
from cascade.paper_context import (
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
VALID_ALIGNMENT_BACKEND_NAMES = ("qwen_forced", "gemma_vllm_qk_fast")
# The stable set is used by default in comparison scripts.
STABLE_ALIGNMENT_BACKEND_NAMES = ("qwen_forced", "gemma_vllm_qk_fast")

# MT is now a single shipped path: Gemma AlignAtt through vLLM.
VALID_MT_BACKEND_NAMES = ("gemma_vllm_alignatt",)
STABLE_MT_BACKEND_NAMES = ("gemma_vllm_alignatt",)
VALID_ASR_ALIGNATT_COMMIT_POLICIES = ("frontier_flush", "rewind_abort")

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
        "data/alignatt_heads/"
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
    # The Qwen forced-aligner is used both online (short live chunks) and
    # offline for full-audio diagnostic alignment. The latter can exceed the
    # historical 1024-token cap, so we expose a larger prompt budget here
    # instead of hardcoding it in the backend. ``2560`` fits the default
    # Qwen GPU-memory slice on this A100 while covering the tracked longform
    # diagnostics that overflowed at 1024.
    qwen_asr_max_model_len: int = 2560
    gemma_max_model_len: int = 1024
    gemma_enable_prefix_caching: bool = True
    alignment_backend_name: str = "qwen_forced"
    mt_backend_name: str = "gemma_vllm_alignatt"
    gemma_audio_alignment_heads_path: str | None = (
        "data/alignatt_heads/audio_alignment_heads_google_gemma-4-E4B-it_en_forced.json"
    )
    # The head file is not just a ranking of interesting heads: it also
    # ships the calibrated word-end offset for that exact head set. Treat it
    # as one timestamp policy artifact.
    gemma_audio_alignment_top_k_heads: int = 8
    gemma_audio_alignment_filter_width: int = 7
    gemma_audio_alignment_max_new_tokens: int = 256
    # vLLM-specific config for the experimental gemma_vllm_qk_fast backend.
    # Defaults reflect the validated cudagraph=full seam (PLAN.md section 6).
    gemma_vllm_enforce_eager: bool = False
    gemma_vllm_enable_prefix_caching: bool = False
    gemma_vllm_cudagraph_mode: str | None = "full"
    gemma_vllm_gpu_memory_utilization: float = 0.5
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
    # AlignAtt streaming ASR knobs (gemma_vllm_qk_fast only). See
    # ``cascade/alignment/gemma_alignatt_stream.py``. ``frontier_flush``
    # commits the maximal monotone prefix on every chunk and only keeps
    # the trailing ``asr_alignatt_frame_threshold`` audio-frame band
    # uncommitted. ``rewind_abort`` preserves the historical behaviour:
    # a large rewind before the previous commit frontier aborts the
    # current chunk instead of projecting the path back to monotonicity.
    asr_alignatt_commit_policy: str = "frontier_flush"
    asr_alignatt_frame_threshold: int = 4
    asr_alignatt_rewind_threshold: int = 200
    # Experimental empty-stop rescue. On some chunks Gemma emits only a single
    # special stop token; this knob enables one tiny retry with special tokens
    # suppressed at the first step. Kept off by default because the smoke clip
    # still shows prefix corruption when the retry hallucinates a plausible but
    # wrong continuation.
    gemma_audio_eos_only_rescue_enabled: bool = False
    gemma_audio_eos_only_rescue_max_new_tokens: int = 3
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
        if self.paper_context_mode not in VALID_CONTEXT_MODES:
            raise ValueError(
                f"paper_context_mode must be one of {VALID_CONTEXT_MODES}, "
                f"got {self.paper_context_mode!r}."
            )
        if int(self.asr_alignatt_frame_threshold) < 1:
            raise ValueError(
                "asr_alignatt_frame_threshold must be >= 1, got "
                f"{self.asr_alignatt_frame_threshold!r}."
            )
        if self.asr_alignatt_commit_policy not in VALID_ASR_ALIGNATT_COMMIT_POLICIES:
            raise ValueError(
                "asr_alignatt_commit_policy must be one of "
                f"{VALID_ASR_ALIGNATT_COMMIT_POLICIES}, got "
                f"{self.asr_alignatt_commit_policy!r}."
            )
        if int(self.asr_alignatt_rewind_threshold) < 1:
            raise ValueError(
                "asr_alignatt_rewind_threshold must be >= 1, got "
                f"{self.asr_alignatt_rewind_threshold!r}."
            )
        if int(self.gemma_audio_eos_only_rescue_max_new_tokens) < 1:
            raise ValueError(
                "gemma_audio_eos_only_rescue_max_new_tokens must be >= 1, got "
                f"{self.gemma_audio_eos_only_rescue_max_new_tokens!r}."
            )
        if int(self.qwen_asr_max_model_len) < 1024:
            raise ValueError(
                "qwen_asr_max_model_len must be >= 1024, got "
                f"{self.qwen_asr_max_model_len!r}."
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
        if name == "gemma_vllm_qk_fast":
            return (
                name,
                bool(self.gemma_vllm_enforce_eager),
                bool(self.gemma_vllm_enable_prefix_caching),
                self.gemma_vllm_cudagraph_mode,
                float(self.asr_gpu_memory_utilization),
                int(self.gemma_max_model_len),
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


def split_text_at_word_boundary(text: str, n_words: int) -> tuple[str, str]:
    """Split ``text`` after the ``n_words``-th lexical word."""
    if n_words <= 0:
        return "", text.strip()
    tokens = text.split()
    committed_tokens: list[str] = []
    committed_word_count = 0
    last_committed_idx = -1
    for idx, token in enumerate(tokens):
        if remove_punctuation(token).strip():
            committed_word_count += 1
        committed_tokens.append(token)
        last_committed_idx = idx
        if committed_word_count == n_words:
            break
    if committed_word_count < n_words:
        return text, ""
    return " ".join(committed_tokens), " ".join(tokens[last_committed_idx + 1 :])


def n_utterances(text: str) -> int:
    n_utt = text.count(". ") + text.count("! ") + text.count("? ")
    if text.endswith((".", "!", "?")):
        n_utt += 1
    return n_utt


def normalize_partial_asr_hypothesis(text: str) -> str:
    """Expose the still-live ASR tail the way the MT path expects it."""
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
        # The Gemma AlignAtt stream owns the committed-token list and the
        # audio-window policy for the gemma_vllm_qk_fast backend. The
        # qwen_forced backend uses the punctuation-LCP commit path and
        # never touches this field.
        self._gemma_align_att_stream: "GemmaAlignAttStream | None" = None
        self._asr_stream_trace: list[dict[str, Any]] = []
        # Per-token commit log: persists across window slides so the
        # evaluator can use alignatt's per-token `end_frame_abs` (the
        # acoustic end the aligner picked) rather than the coarser
        # segment-level `audio_processed_s`. Without this, LongYAAL
        # inflates by ~5x because every word in a segment gets the same
        # chunk-boundary delay.
        self._per_token_commits: list[dict[str, Any]] = []

    def load_models(self) -> None:
        self.bundle.load()

    def clear(self) -> None:
        speech_id = self.state.speech_id
        self.state = CascadeState(speech_id=speech_id)
        self.mt_prompt_cache = PromptCacheState()
        self.translation_units = TranslationUnitManager(self)
        self._asr_stream_trace = []
        self._per_token_commits = []
        self._reset_alignatt_stream()

    def _reset_alignatt_stream(self) -> None:
        """Drop the Gemma AlignAtt streaming state (new utterance).

        The backend's alignment caches are reset too so that per-chunk
        generation does not inherit prefix-cached state from a previous
        utterance.
        """
        self._gemma_align_att_stream = None
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

    def asr_stream_trace(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._asr_stream_trace]

    def per_token_commits(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._per_token_commits]

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
        """Run ASR on the current live audio tail and apply the commit rule."""
        alignment_backend = self.bundle.ensure_alignment_backend()
        audio = np.array(
            self.state.source[self.state.utt_timestamps[-1] :], dtype=np.float32
        )
        if self.config.alignment_backend_name == "gemma_vllm_qk_fast":
            return self._drive_alignatt_stream(
                alignment_backend=alignment_backend,
                audio=audio,
                is_final_chunk=is_final_chunk,
            )
        return self._transcribe_non_streaming(
            alignment_backend=alignment_backend,
            audio=audio,
            is_final_chunk=is_final_chunk,
        )

    def _transcribe_non_streaming(
        self,
        *,
        alignment_backend: AlignmentBackend,
        audio: np.ndarray,
        is_final_chunk: bool,
    ) -> str | None:
        """Non-streaming ASR path for qwen_forced.

        The full utterance tail is re-transcribed from scratch on every
        chunk and committed with the legacy punctuation-LCP rule. This
        path does not use a streaming prefix or a sliding audio window;
        utterances longer than the backend's audio cap will raise.
        """
        try:
            result = alignment_backend.transcribe_and_align(
                audio,
                sample_rate=SAMPLE_RATE,
                language=self.config.source_lang,
            )
        except Exception as exc:
            raise type(exc)(
                f"{exc} | cascade_context="
                f"audio_tail_s={len(audio) / SAMPLE_RATE:.3f}"
            ) from exc
        if result is None:
            return None

        asr_hypo = result.text
        self.state.asr_hypotheses.append(asr_hypo)
        self.state.partial_word_timestamps_ms = normalize_word_timestamps_ms(result.words)

        asr_segment = longest_common_prefix(
            self.state.asr_hypotheses[-2],
            self.state.asr_hypotheses[-1],
        )
        commit_info = self._try_commit_punctuation_lcp(
            asr_hypo=asr_hypo,
            result=result,
            lcp_text=asr_segment,
            is_final_chunk=is_final_chunk,
        )
        self._append_trace_row(
            audio_tail_s=float(len(audio)) / float(SAMPLE_RATE),
            audio_window_s=float(len(audio)) / float(SAMPLE_RATE),
            hypothesis_text=asr_hypo,
            lcp_text=asr_segment,
            commit_mode="punct_lcp",
            commit_info=commit_info,
            is_final_chunk=is_final_chunk,
        )
        if commit_info is _COMMIT_ABORT:
            return asr_hypo.strip()
        if self.state.utt_sources[1:]:
            return self.render_public_asr_text()
        return self.current_live_asr_tail_text()

    def _drive_alignatt_stream(
        self,
        *,
        alignment_backend: AlignmentBackend,
        audio: np.ndarray,
        is_final_chunk: bool,
    ) -> str | None:
        """Adapter from ``GemmaAlignAttStream`` to the runtime's state model.

        The stream owns the token-level commit policy and the sliding
        audio window. Its committed-token list is the single source of
        truth for what the decoder has seen; we only translate that into
        ``state.asr_hypotheses`` / ``state.utt_sources`` so the rest of
        the runtime (MT, emission, artifacts) keeps working unchanged.
        """
        if self._gemma_align_att_stream is None:
            self._gemma_align_att_stream = GemmaAlignAttStream(
                backend=alignment_backend,
                language=self.config.source_lang,
                commit_policy=str(self.config.asr_alignatt_commit_policy),
                frame_threshold=int(self.config.asr_alignatt_frame_threshold),
                rewind_threshold=int(self.config.asr_alignatt_rewind_threshold),
            )
        stream = self._gemma_align_att_stream

        delta = stream.step(audio, is_final_chunk=is_final_chunk)

        # Persist per-token commit records across window slides, with the
        # alignatt-estimated acoustic end (end_frame_abs) and the chunk
        # boundary at which the commit was made. Once the stream's window
        # slides past these tokens they are dropped from
        # stream.committed_tokens, so we capture them here.
        #
        # ``end_time_s`` and ``committed_at_audio_processed_s`` serve
        # different purposes and are intentionally both logged:
        #   - ``end_time_s`` is AlignAtt's best acoustic word-end proxy
        #   - ``committed_at_audio_processed_s`` is the true emission time
        #     seen by the downstream evaluator
        # Confusing the two makes LongYAAL look impossibly low.
        audio_ms_per_token = float(alignment_backend.audio_ms_per_token)
        chunk_audio_processed_s = float(len(audio)) / float(SAMPLE_RATE)
        for tok in delta.new_committed_tokens:
            self._per_token_commits.append(
                {
                    "text": str(tok.text),
                    "token_id": int(tok.token_id),
                    "end_frame_abs": int(tok.end_frame_abs),
                    "end_time_s": float(tok.end_frame_abs) * audio_ms_per_token / 1000.0,
                    "committed_at_audio_processed_s": chunk_audio_processed_s,
                }
            )

        committed_text = stream.committed_text
        partial_text = committed_text + delta.partial_tail_text
        asr_hypo = partial_text.strip()
        self.state.asr_hypotheses.append(asr_hypo)
        # AlignAtt commits at the token level; we do not expose per-word
        # timestamps for the partial region. The source frontier falls
        # back to the "all but last word accessible" default, which is
        # correct for a live tail that has not yet cleared the frontier
        # gate. This means the runtime's public partial tail is only a UI
        # convenience; the real latency/accounting path is the committed
        # token log above.
        self.state.partial_word_timestamps_ms = []

        lcp_text = longest_common_prefix(
            self.state.asr_hypotheses[-2],
            self.state.asr_hypotheses[-1],
        )

        commit_info = self._emit_alignatt_segment(
            stream=stream,
            delta=delta,
            is_final_chunk=is_final_chunk,
        )

        self._append_trace_row(
            audio_tail_s=float(len(audio)) / float(SAMPLE_RATE),
            audio_window_s=float(delta.audio_window_content_frames)
            * float(alignment_backend.audio_ms_per_token)
            / 1000.0,
            hypothesis_text=asr_hypo,
            lcp_text=lcp_text,
            commit_mode="alignatt_token_frontier",
            commit_info=commit_info,
            is_final_chunk=is_final_chunk,
            extra={
                "alignatt_backend_finish_reason": (
                    None
                    if not isinstance(delta.diagnostics.get("backend"), dict)
                    else delta.diagnostics["backend"].get("finish_reason")
                ),
                "alignatt_backend_empty_completion": bool(
                    isinstance(delta.diagnostics.get("backend"), dict)
                    and delta.diagnostics["backend"].get("empty_completion", False)
                ),
                "alignatt_backend_generated_token_count": (
                    0
                    if not isinstance(delta.diagnostics.get("backend"), dict)
                    else int(
                        delta.diagnostics["backend"].get("generated_token_count", 0)
                    )
                ),
                "alignatt_backend_raw_generated_token_count": (
                    0
                    if not isinstance(delta.diagnostics.get("backend"), dict)
                    else int(
                        delta.diagnostics["backend"].get(
                            "raw_generated_token_count",
                            0,
                        )
                    )
                ),
                "alignatt_backend_raw_completion_text": (
                    ""
                    if not isinstance(delta.diagnostics.get("backend"), dict)
                    else str(
                        delta.diagnostics["backend"].get("raw_completion_text", "")
                    )
                ),
                "alignatt_backend_eos_only_rescue": (
                    None
                    if not isinstance(delta.diagnostics.get("backend"), dict)
                    else delta.diagnostics["backend"].get("eos_only_rescue")
                ),
                "alignatt_generated_count": int(
                    delta.diagnostics.get("generated_count", 0)
                ),
                "alignatt_accepted_count": int(
                    delta.diagnostics.get("accepted_count", 0)
                ),
                "alignatt_commit_policy": str(
                    delta.diagnostics.get("commit_policy", "")
                ),
                "alignatt_aborted_by_rewind": bool(
                    delta.diagnostics.get("aborted_by_rewind", False)
                ),
                "alignatt_raw_rewind_count": int(
                    delta.diagnostics.get("raw_rewind_count", 0)
                ),
                "alignatt_raw_max_rewind_frames": int(
                    delta.diagnostics.get("raw_max_rewind_frames", 0)
                ),
                "alignatt_projected_repair_count": int(
                    delta.diagnostics.get("projected_repair_count", 0)
                ),
                "alignatt_projected_max_repair_frames": int(
                    delta.diagnostics.get("projected_max_repair_frames", 0)
                ),
                "alignatt_forced_prefix_token_count": int(
                    delta.diagnostics.get("forced_prefix_token_count", 0)
                ),
                "alignatt_out_of_window_cumulative": int(
                    delta.diagnostics.get("out_of_window_cumulative", 0)
                ),
                "alignatt_window_start_frame_abs": int(
                    delta.audio_window_start_frame_abs
                ),
                "alignatt_stream_committed_text": stream.committed_text,
                "alignatt_stream_committed_word_count": len(
                    remove_punctuation(stream.committed_text).split()
                ),
            },
        )

        if self.state.utt_sources[1:]:
            return self.render_public_asr_text()
        return self.current_live_asr_tail_text()

    def _emit_alignatt_segment(
        self,
        *,
        stream: GemmaAlignAttStream,
        delta: StreamStepDelta,
        is_final_chunk: bool,
    ) -> dict[str, Any] | None:
        """Close an utterance segment when a sentence boundary lands, or at EOS.

        The token-level frontier decides *what* has been committed; this
        helper decides *when* to expose that commit as a cascade-visible
        utterance boundary. The rule is deliberately minimal:

        - at EOS: flush whatever has been committed;
        - otherwise: only emit when the most-recent committed token ends
          in sentence-final punctuation (``.``, ``!``, ``?``). This is
          the same boundary signal the MT pipeline has always used.

        No LCP, no gap search, no word-aggregation invariant. The decision
        is a pure function of the stream's own committed tokens.
        """
        committed_text = stream.committed_text
        if not committed_text.strip():
            return None

        if not is_final_chunk:
            last_char = committed_text.rstrip()[-1:] if committed_text.strip() else ""
            if last_char not in {".", "!", "?"}:
                return None

        end_time_s = stream.last_committed_end_seconds()
        segment_text = committed_text.strip()
        return self._apply_commit(
            committed_text=segment_text,
            remainder_text="",
            end_time_s=end_time_s,
        )

    def _try_commit_punctuation_lcp(
        self,
        *,
        asr_hypo: str,
        result: "AlignmentResult",
        lcp_text: str,
        is_final_chunk: bool = False,
    ) -> dict[str, Any] | object | None:
        if is_final_chunk:
            # EOS flush: commit the whole current hypothesis, even without
            # sentence-final punctuation. No more audio is coming, so waiting
            # for a punctuation signal would lose the trailing words.
            words = result.words or ()
            if not asr_hypo.strip() or not words:
                return None
            last_word_end = float(words[-1].end_time) if words else 0.0
            return self._apply_commit(
                committed_text=asr_hypo.strip(),
                remainder_text="",
                end_time_s=last_word_end,
            )

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

        remainder_text = asr_hypo[rightest_punct_idx + 1 :].strip()
        return self._apply_commit(
            committed_text=lcp_text[: rightest_punct_idx + 1],
            remainder_text=remainder_text,
            end_time_s=float(end_time),
        )

    def _apply_commit(
        self,
        *,
        committed_text: str,
        remainder_text: str,
        end_time_s: float,
    ) -> dict[str, Any]:
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
        if self.config.alignment_backend_name == "gemma_vllm_qk_fast":
            # AlignAtt stream state is utterance-scoped; reset on commit.
            self._reset_alignatt_stream()
        return {
            "committed_text": committed_text,
            "committed_word_count": len(remove_punctuation(committed_text).split()),
            "remainder_text": remainder_text,
            "remainder_word_count": n_words_right,
            "end_time_s": float(end_time_s),
        }

    def _append_trace_row(
        self,
        *,
        audio_tail_s: float,
        audio_window_s: float,
        hypothesis_text: str,
        lcp_text: str,
        commit_mode: str,
        commit_info: dict[str, Any] | object | None,
        is_final_chunk: bool,
        extra: dict[str, Any] | None = None,
    ) -> None:
        committed_payload = (
            commit_info
            if isinstance(commit_info, dict)
            else {"committed_text": "", "committed_word_count": 0, "end_time_s": None}
        )
        public_asr_text = self.render_public_asr_text()
        row: dict[str, Any] = {
            "update_idx": len(self._asr_stream_trace),
            "chunk_idx": len(self._asr_stream_trace),
            "audio_processed_s": self.current_audio_seconds(),
            "audio_tail_s": float(audio_tail_s),
            "audio_window_s": float(audio_window_s),
            "hypothesis_text": hypothesis_text,
            "hypothesis_word_count": len(remove_punctuation(hypothesis_text).split()),
            "lcp_text": lcp_text,
            "lcp_word_count": len(remove_punctuation(lcp_text).split()),
            "commit_mode": commit_mode,
            "committed_text": committed_payload["committed_text"],
            "committed_word_count": int(committed_payload["committed_word_count"]),
            "commit_end_time_s": (
                None
                if committed_payload["end_time_s"] is None
                else float(committed_payload["end_time_s"])
            ),
            "predicted_boundary_count_so_far": len(self.state.utt_sources) - 1,
            "committed_segment_count": len(self.state.utt_sources) - 1,
            "public_asr_text": public_asr_text,
            "public_asr_word_count": len(remove_punctuation(public_asr_text).split()),
            "public_asr_char_count": len(public_asr_text),
            "is_final_chunk": bool(is_final_chunk),
        }
        if extra:
            row.update(extra)
        self._asr_stream_trace.append(row)

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
        # is_final_chunk=True lets the active ASR commit policy flush the
        # remaining live tail at EOS. Without it the last punctuation-less or
        # frontier-gated words would never make it into the public hypothesis.
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
                "gemma_audio_alignment_heads_path": self.config.gemma_audio_alignment_heads_path,
                # Aggregation is hard-coded in the maintained path on purpose:
                # the median-over-head-argmaxes proved robust, and keeping the
                # value explicit in provenance is more useful than keeping a
                # live runtime knob that callers can silently mis-set.
                "gemma_audio_alignment_aggregation": "median_argmax",
                "gemma_audio_eos_only_rescue_enabled": self.config.gemma_audio_eos_only_rescue_enabled,
                "gemma_audio_eos_only_rescue_max_new_tokens": self.config.gemma_audio_eos_only_rescue_max_new_tokens,
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
