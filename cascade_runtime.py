from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any
import os
import string
import subprocess
import wave

import numpy as np

from alignment_backend import AlignmentBackend
from cascade_artifacts import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WAV_PATH,
    InferenceArtifacts,
    StreamUpdate,
    write_inference_artifacts,
)
from cascade_emission import (
    RAW_PASSTHROUGH,
    apply_emission_policy,
    register_translation_timestamps,
    register_translation_words,
)
from cascade_mt_backend import MTBackendResult, PromptCacheState, build_mt_backend
from cascade_source_frontier import (
    SourceAccessibilityFrontier,
    build_source_accessibility_frontier,
    normalize_word_timestamps_ms,
)
from cascade_source_text import NormalizedSourceText, normalize_source_text_for_mt
from cascade_text_surface import normalize_incremental_target_text
from cascade_translation_variants import (
    FOUNDATIONAL_TRANSLATION_VARIANT_ID,
    RenderedTranslationPrompt,
    TRANSLATION_VARIANTS,
    TranslationVariant,
)
from simulstream.server.speech_processors import SAMPLE_RATE


# Avoid repeated HF HEAD requests for optional files that are already cached as absent.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


LANGUAGE_NAME_TO_CODE = {
    "English": "en",
    "German": "de",
    "Italian": "it",
    "Chinese": "zh",
}
LANGUAGE_CODE_TO_NAME = {
    code: name for name, code in LANGUAGE_NAME_TO_CODE.items()
}
VALID_ALIGNMENT_BACKEND_NAMES = ("qwen_forced", "gemma_onepass_qk_fast", "gemma_vllm_qk_fast")
# The stable set is used by default in comparison scripts; the experimental
# vLLM backend is opt-in until validated under the full SimulStream loop.
STABLE_ALIGNMENT_BACKEND_NAMES = ("qwen_forced", "gemma_onepass_qk_fast")


def _resolve_hf_snapshot(repo_subpath: str) -> str:
    candidates = [
        os.path.join("/home/.cache/huggingface/hub", repo_subpath),
        os.path.join(os.path.expanduser("~/.cache/huggingface/hub"), repo_subpath),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


asr_model_name = _resolve_hf_snapshot(
    "models--Qwen--Qwen3-ASR-1.7B/snapshots/7278e1e70fe206f11671096ffdd38061171dd6e5"
)
forced_aligner_model_name = _resolve_hf_snapshot(
    "models--Qwen--Qwen3-ForcedAligner-0.6B/snapshots/c7cbfc2048c462b0d63a45797104fc9db3ad62b7"
)
gemma_model_name = _resolve_hf_snapshot(
    "models--google--gemma-4-E4B-it/snapshots/83df0a889143b1dbfc61b591bbc639540fd9ce4c"
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
    translation_alignatt_min_source_mass: float = 0.0
    max_new_tokens: int = 160
    partial_max_new_tokens: int = 48
    partial_followup_max_new_tokens: int = 16
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
    translation_scheduler_stall_seconds: float = 1.2
    alignment_backend_name: str = "qwen_forced"
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

    def __post_init__(self) -> None:
        if self.translation_alignatt_heads_path is None:
            self.translation_alignatt_heads_path = alignatt_heads_path_for(
                self.source_lang, self.target_lang
            )
        if self.alignment_backend_name not in VALID_ALIGNMENT_BACKEND_NAMES:
            raise ValueError(
                f"Unknown alignment_backend_name: {self.alignment_backend_name!r}"
            )

    def apply_overrides(self, **overrides) -> None:
        for key, value in overrides.items():
            if not hasattr(self, key):
                raise AttributeError(f"Unknown runtime config override: {key}")
            setattr(self, key, value)
        if "target_lang" in overrides and "translation_alignatt_heads_path" not in overrides:
            self.translation_alignatt_heads_path = alignatt_heads_path_for(
                self.source_lang, self.target_lang
            )


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
    current_audio_seconds_value: float,
    stall_seconds: float,
) -> tuple[bool, str]:
    source_prefix = source_prefix.strip()
    if not source_prefix:
        return False, "empty_source"
    if not previous_state.source_prefix:
        return True, "initial_partial"
    if not source_prefix.startswith(previous_state.source_prefix):
        return True, "source_rebased"
    blocked_source_unit_index = previous_state.blocked_source_unit_index
    if (
        blocked_source_unit_index is not None
        and accessible_unit_count <= blocked_source_unit_index
    ):
        if source_prefix == previous_state.source_prefix:
            return False, "source_prefix_unchanged"
        stalled_seconds = max(
            0.0,
            current_audio_seconds_value - previous_state.last_mt_audio_seconds,
        )
        if stalled_seconds >= float(stall_seconds):
            return True, "stall_probe"
        return False, "blocked_frontier_not_reached"
    if accessible_unit_count > previous_state.source_accessible_unit_count:
        return True, "accessible_frontier_advanced"
    if source_prefix == previous_state.source_prefix:
        return False, "source_prefix_unchanged"

    stalled_seconds = max(
        0.0,
        current_audio_seconds_value - previous_state.last_mt_audio_seconds,
    )
    if stalled_seconds >= float(stall_seconds):
        return True, "stall_probe"
    return False, "frontier_not_advanced"


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
    if not previous_state.source_prefix or not source_prefix.startswith(previous_state.source_prefix):
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
        from qwen_alignment_backend import QwenAlignmentBackend

        return QwenAlignmentBackend(
            asr_model_path=qwen_model_path,
            forced_aligner_model_path=qwen_forced_aligner_model_path,
            runtime_config=config,
        )
    if config.alignment_backend_name == "gemma_onepass_qk_fast":
        from gemma_alignment_probe import GemmaAttentionAlignmentBackend

        return GemmaAttentionAlignmentBackend(
            model_name=gemma_path,
            runtime_config=config,
            audio_heads_path=config.gemma_audio_alignment_heads_path,
            audio_heads_top_k=int(config.gemma_audio_alignment_top_k_heads),
            filter_width=int(config.gemma_audio_alignment_filter_width),
            max_new_tokens=int(config.gemma_audio_alignment_max_new_tokens),
            audio_align_probe_mode=config.gemma_audio_align_probe_mode,
        )
    if config.alignment_backend_name == "gemma_vllm_qk_fast":
        from gemma_vllm_alignment_backend import GemmaVLLMAttentionAlignmentBackend

        return GemmaVLLMAttentionAlignmentBackend(
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
        self._alignment_backend_id: str | None = None
        self._mt_heads_path: str | None = None

    def ensure_alignment_backend(self) -> AlignmentBackend:
        if (
            self.alignment_backend is None
            or self._alignment_backend_id != self.config.alignment_backend_name
        ):
            self.alignment_backend = build_alignment_backend(
                self.config,
                qwen_model_path=self.qwen_model_path,
                qwen_forced_aligner_model_path=self.qwen_forced_aligner_model_path,
                gemma_path=self.gemma_path,
            )
            self.alignment_backend.load()
            self._alignment_backend_id = self.config.alignment_backend_name
        else:
            runtime_config = getattr(self.alignment_backend, "runtime_config", None)
            if runtime_config is not None:
                self.alignment_backend.runtime_config = self.config
        return self.alignment_backend

    def ensure_mt_backend(self):
        if self.mt_backend is None:
            self.mt_backend = build_mt_backend(
                model_name=self.gemma_path,
                runtime_config=self.config,
            )
            self.mt_backend.load()
            self._mt_heads_path = self.config.translation_alignatt_heads_path
        else:
            self.mt_backend.runtime_config = self.config
            current_heads_path = self.config.translation_alignatt_heads_path
            if current_heads_path != self._mt_heads_path:
                self.mt_backend.refresh_alignatt_artifacts()
                self._mt_heads_path = current_heads_path
        return self.mt_backend

    def load(self) -> None:
        self.ensure_alignment_backend()
        self.ensure_mt_backend()

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
        partial_state = self.state.partial_translation
        if not source_text:
            self.reset_partial_state()
            return ""
        if partial_state.source_prefix and not source_text.startswith(partial_state.source_prefix):
            self.reset_partial_state()
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
            current_audio_seconds_value=self.session.current_audio_seconds(),
            stall_seconds=float(self.config.translation_scheduler_stall_seconds),
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
                self.state.partial_translation.source_prefix
                and self.state.partial_translation.accepted_target
                and segment_idx == len(self.state.utt_sources) - 1
                and normalized_segment_source.text.startswith(
                    self.state.partial_translation.source_prefix
                )
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
        partial_source = normalize_partial_asr_hypothesis(self.state.asr_hypotheses[-1])
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

    def load_models(self) -> None:
        self.bundle.load()

    def clear(self) -> None:
        speech_id = self.state.speech_id
        self.state = CascadeState(speech_id=speech_id)
        self.mt_prompt_cache = PromptCacheState()
        self.translation_units = TranslationUnitManager(self)

    def current_audio_seconds(self) -> float:
        return len(self.state.source) / SAMPLE_RATE

    def render_public_asr_text(self) -> str:
        committed_segments = [
            segment.strip() for segment in self.state.utt_sources[1:] if segment.strip()
        ]
        partial_segment = ""
        if self.state.asr_hypotheses:
            partial_segment = normalize_partial_asr_hypothesis(
                self.state.asr_hypotheses[-1]
            )
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
        return variant.render_messages(
            source_lang=self.config.source_lang,
            target_lang=self.config.target_lang,
            text=text,
            source_frontier=source_frontier,
            source_history=source_history,
            translation_history=translation_history,
            is_partial=is_partial,
            assistant_prefill=assistant_prefill,
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

    def transcribe_audio(self) -> str | None:
        alignment_backend = self.bundle.ensure_alignment_backend()
        audio = np.array(
            self.state.source[self.state.utt_timestamps[-1] :], dtype=np.float32
        )
        result = alignment_backend.transcribe_and_align(
            audio,
            sample_rate=SAMPLE_RATE,
            language=self.config.source_lang,
        )
        if result is None:
            return None

        asr_hypo = result.text
        self.state.asr_hypotheses.append(asr_hypo)
        self.state.partial_word_timestamps_ms = normalize_word_timestamps_ms(result.words)

        asr_segment = longest_common_prefix(
            self.state.asr_hypotheses[-2],
            self.state.asr_hypotheses[-1],
        )
        if n_utterances(asr_segment) >= 1:
            rightest_punct_idx = max(
                asr_segment.rfind(". "),
                asr_segment.rfind("! "),
                asr_segment.rfind("? "),
            )
            if rightest_punct_idx == -1 and asr_segment.endswith((".", "!", "?")):
                rightest_punct_idx = len(asr_segment) - 1

            end_time = find_end_time(result.words, rightest_punct_idx, asr_hypo)
            if end_time is None:
                return asr_hypo.strip()

            utt_end_time = int(end_time * SAMPLE_RATE) + self.state.utt_timestamps[-1]
            utt_end_time = min(utt_end_time, len(self.state.source))
            self.state.utt_timestamps.append(utt_end_time)
            self.state.utt_sources.append(asr_segment[: rightest_punct_idx + 1])
            remainder = asr_hypo[rightest_punct_idx + 1 :].strip()
            n_words_right = len(remove_punctuation(remainder).strip().split())
            self.state.asr_hypotheses = [remainder]
            self.state.partial_word_timestamps_ms = (
                self.state.partial_word_timestamps_ms[-n_words_right:]
                if n_words_right > 0
                else []
            )

        if self.state.utt_sources[1:]:
            return self.render_public_asr_text()
        return normalize_partial_asr_hypothesis(self.state.asr_hypotheses[-1])

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
        final_asr = self.transcribe_audio() or self.render_public_asr_text()
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
                "partial_followup_max_new_tokens": self.config.partial_followup_max_new_tokens,
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
                "translation_scheduler_stall_seconds": self.config.translation_scheduler_stall_seconds,
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

    if "target_lang" in overrides and "translation_alignatt_heads_path" not in overrides:
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
