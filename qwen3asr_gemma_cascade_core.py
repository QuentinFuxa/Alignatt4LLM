from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
import os
from types import SimpleNamespace
from typing import Any, List
import string
import wave

import numpy as np
import torch
import patch_qwen_asr_for_transformers5
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

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
from cascade_mt_backend import MTBackendResult, build_mt_backend
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


# Avoid repeated HF HEAD requests for optional files that are already cached as absent.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


patch_qwen_asr_for_transformers5.main()

from qwen_asr import Qwen3ASRModel
from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRConfig
from qwen_asr.core.transformers_backend import modeling_qwen3_asr
from simulstream.server.speech_processors import SAMPLE_RATE


def _qwen3_asr_default_rope_init(config, device=None, seq_len=None, layer_type=None):
    standardize = getattr(config, "standardize_rope_params", None)
    if callable(standardize):
        standardize()

    rope_parameters = getattr(config, "rope_parameters", None) or {}
    if layer_type is not None and isinstance(rope_parameters, dict) and layer_type in rope_parameters:
        rope_parameters = rope_parameters[layer_type]

    base = rope_parameters.get("rope_theta", getattr(config, "rope_theta", 10000.0))
    partial_rotary_factor = rope_parameters.get("partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float)
            / dim
        )
    )
    return inv_freq, 1.0


modeling_qwen3_asr._qwen3_asr_default_rope_init = _qwen3_asr_default_rope_init
if "default" not in ROPE_INIT_FUNCTIONS:
    ROPE_INIT_FUNCTIONS["default"] = _qwen3_asr_default_rope_init


def _patched_qwen3_asr_get_text_config(self, decoder=False):
    thinker_config = getattr(self, "thinker_config", None)
    if thinker_config is not None:
        return thinker_config.get_text_config()

    text_config = getattr(self, "text_config", None)
    if text_config is not None:
        return text_config

    return self


Qwen3ASRConfig.get_text_config = _patched_qwen3_asr_get_text_config


def longest_common_prefix(s1: str, s2: str) -> str:
    for i in range(min(len(s1), len(s2))):
        if s1[i] != s2[i]:
            return s1[:i]
    return s1[: min(len(s1), len(s2))]


def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))


def find_end_time(time_stamps, position: int, text: str):
    if len(time_stamps) != len(remove_punctuation(text).split()):
        return None
    n_words_right = len(remove_punctuation(text[position + 1 :]).strip().split())
    return time_stamps[-n_words_right - 1].end_time


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


def render_public_asr_text() -> str:
    committed_segments = [segment.strip() for segment in state.utt_sources[1:] if segment.strip()]
    partial_segment = ""
    if state.asr_hypotheses:
        partial_segment = normalize_partial_asr_hypothesis(state.asr_hypotheses[-1])
    if partial_segment:
        committed_segments.append(partial_segment)
    return " ".join(committed_segments).strip()


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
    utt_timestamps: List[int] = field(default_factory=lambda: [0])
    utt_sources: List[str] = field(default_factory=lambda: [""])
    utt_translations: List[str] = field(default_factory=lambda: [""])
    asr_hypotheses: List[str] = field(default_factory=lambda: [""])
    partial_word_timestamps_ms: List[tuple[float | None, float | None]] = field(default_factory=list)
    partial_translation: PartialTranslationState = field(default_factory=PartialTranslationState)


asr_model_name = "/home/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/7278e1e70fe206f11671096ffdd38061171dd6e5"
forced_aligner_model_name = "/home/.cache/huggingface/hub/models--Qwen--Qwen3-ForcedAligner-0.6B/snapshots/c7cbfc2048c462b0d63a45797104fc9db3ad62b7"
gemma_model_name = "/home/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/83df0a889143b1dbfc61b591bbc639540fd9ce4c"

config = SimpleNamespace(
    source_lang="English",
    target_lang="German",
    latency_unit="word",
    min_start_seconds=5.0,
    translation_variant_id=FOUNDATIONAL_TRANSLATION_VARIANT_ID,
    max_history_utterances=0,
    translation_alignatt_heads_path="assets/attention_heads/translation_heads_google_gemma-4-E4B-it_en-de.json",
    translation_alignatt_top_k_heads=8,
    translation_alignatt_filter_width=7,
    translation_alignatt_probe_mode="qk_fast",
    # Calibrated for the current latency-first AlignAtt cascade: make newly
    # timestamped source units accessible immediately, then rely on a more
    # permissive inline rewind guard to preserve German reordering freedom.
    translation_alignatt_inaccessible_ms=0.0,
    translation_alignatt_rewind_threshold=8,
    max_new_tokens=160,
    partial_max_new_tokens=48,
    partial_followup_max_new_tokens=16,
    translation_min_new_tokens=32,
    translation_token_budget_ratio=3.0,
    translation_token_budget_buffer=24,
    partial_translation_min_new_tokens=4,
    partial_translation_token_budget_ratio=1.0,
    partial_translation_token_budget_buffer=8,
    translation_generation_margin=8,
    translation_emit_policy=RAW_PASSTHROUGH,
    translation_max_tail_rewrite_words=14,
    temperature=0.0,
    repetition_penalty=1.05,
    asr_gpu_memory_utilization=0.2,
    gemma_max_model_len=1024,
    gemma_enable_prefix_caching=True,
    gemma_transformers_device="cuda:0",
    gemma_transformers_dtype="bfloat16",
    gemma_transformers_fast_attention="sdpa",
    gemma_transformers_prompt_kv_reuse=True,
    translation_scheduler_stall_seconds=1.2,
)

asr = None
gemma_tokenizer = None
gemma_llm = None
mt_backend = None
state = CascadeState()


def get_translation_variant() -> TranslationVariant:
    return TRANSLATION_VARIANTS[config.translation_variant_id]


@contextmanager
def temporary_runtime_config(**overrides):
    if not overrides:
        yield
        return

    original_values: dict[str, Any] = {}
    for key, value in overrides.items():
        if not hasattr(config, key):
            raise AttributeError(f"Unknown runtime config override: {key}")
        original_values[key] = getattr(config, key)
        setattr(config, key, value)

    try:
        yield
    finally:
        for key, value in original_values.items():
            setattr(config, key, value)


# %%
def load_models():
    global asr, gemma_tokenizer, gemma_llm, mt_backend, state

    if asr is None:
        asr = Qwen3ASRModel.LLM(
            model=asr_model_name,
            gpu_memory_utilization=config.asr_gpu_memory_utilization,
            max_inference_batch_size=1,
            max_model_len=1024,
            max_new_tokens=1024,
            forced_aligner=forced_aligner_model_name,
            forced_aligner_kwargs={
                "dtype": torch.bfloat16,
                "device_map": "cuda",
            },
        )

    if mt_backend is None:
        mt_backend = build_mt_backend(
            model_name=gemma_model_name,
            runtime_config=config,
        )
        mt_backend.load()
        gemma_tokenizer = mt_backend.tokenizer
        gemma_llm = getattr(mt_backend, "llm", None) or getattr(mt_backend, "model", None)

    state = CascadeState(speech_id=state.speech_id)

# %%
def clear_state():
    global state
    state = CascadeState(speech_id=state.speech_id)

def transcribe_audio():
    if asr is None:
        raise RuntimeError("Models are not loaded. Run load_models() first.")

    audio = np.array(state.source[state.utt_timestamps[-1] :], dtype=np.float32)
    asr_outputs = asr.transcribe(
        (audio, SAMPLE_RATE),
        language=config.source_lang,
        context="",
        return_time_stamps=True,
    )

    if (
        asr_outputs[0].time_stamps is not None
        and asr_outputs[0].time_stamps[-1].end_time > len(audio) / SAMPLE_RATE
    ):
        return None

    asr_hypo = asr_outputs[0].text
    state.asr_hypotheses.append(asr_hypo)
    state.partial_word_timestamps_ms = normalize_word_timestamps_ms(asr_outputs[0].time_stamps)

    asr_segment = longest_common_prefix(state.asr_hypotheses[-2], state.asr_hypotheses[-1])
    if n_utterances(asr_segment) >= 1:
        rightest_punct_idx = max(
            asr_segment.rfind(". "),
            asr_segment.rfind("! "),
            asr_segment.rfind("? "),
        )
        if rightest_punct_idx == -1 and asr_segment.endswith((".", "!", "?")):
            rightest_punct_idx = len(asr_segment) - 1

        end_time = find_end_time(asr_outputs[0].time_stamps, rightest_punct_idx, asr_hypo)
        if end_time is None:
            return asr_hypo.strip()

        utt_end_time = int(end_time * SAMPLE_RATE) + state.utt_timestamps[-1]
        utt_end_time = min(utt_end_time, len(state.source))
        state.utt_timestamps.append(utt_end_time)
        state.utt_sources.append(asr_segment[: rightest_punct_idx + 1])
        remainder = asr_hypo[rightest_punct_idx + 1 :].strip()
        n_words_right = len(remove_punctuation(remainder).strip().split())
        state.asr_hypotheses = [remainder]
        state.partial_word_timestamps_ms = (
            state.partial_word_timestamps_ms[-n_words_right:] if n_words_right > 0 else []
        )

    if state.utt_sources[1:]:
        return render_public_asr_text()
    return normalize_partial_asr_hypothesis(state.asr_hypotheses[-1])


def translation_history_window(items: List[str], end_exclusive: int) -> List[str]:
    if config.max_history_utterances <= 0:
        return []

    start = max(1, end_exclusive - config.max_history_utterances)
    return [item.strip() for item in items[start:end_exclusive] if item.strip()]


def normalized_source_history_window(items: List[str], end_exclusive: int) -> List[str]:
    return [
        normalize_source_text_for_mt(item.strip()).text
        for item in translation_history_window(items, end_exclusive)
        if item.strip()
    ]


def build_translation_messages(
    text: str,
    *,
    source_frontier: SourceAccessibilityFrontier | None,
    source_history: List[str],
    translation_history: List[str],
    is_partial: bool,
    assistant_prefill: str = "",
) -> RenderedTranslationPrompt:
    variant = get_translation_variant()
    return variant.render_messages(
        source_lang=config.source_lang,
        target_lang=config.target_lang,
        text=text,
        source_frontier=source_frontier,
        source_history=source_history,
        translation_history=translation_history,
        is_partial=is_partial,
        assistant_prefill=assistant_prefill,
    )


def current_audio_seconds() -> float:
    return len(state.source) / SAMPLE_RATE


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
        stalled_seconds = max(0.0, current_audio_seconds_value - previous_state.last_mt_audio_seconds)
        if stalled_seconds >= float(stall_seconds):
            return True, "stall_probe"
        return False, "blocked_frontier_not_reached"
    if accessible_unit_count > previous_state.source_accessible_unit_count:
        return True, "accessible_frontier_advanced"
    if source_prefix == previous_state.source_prefix:
        return False, "source_prefix_unchanged"

    stalled_seconds = max(0.0, current_audio_seconds_value - previous_state.last_mt_audio_seconds)
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


def translate_with_mt(
    text: str,
    *,
    source_frontier: SourceAccessibilityFrontier | None = None,
    source_history: List[str] | None = None,
    translation_history: List[str] | None = None,
    is_partial: bool = False,
    assistant_prefill: str = "",
) -> MTBackendResult:
    if mt_backend is None:
        raise RuntimeError("MT backend is not loaded. Run load_models() first.")

    text = text.strip()
    if not text:
        prefixed_text = assistant_prefill.rstrip(" \n")
        semantic_token_ids: tuple[int, ...] = ()
        if mt_backend is not None and hasattr(mt_backend, "encode_semantic_target_token_ids"):
            semantic_token_ids = tuple(mt_backend.encode_semantic_target_token_ids(prefixed_text))
        return MTBackendResult(
            draft_text=prefixed_text,
            acceptance_text=prefixed_text,
            draft_token_ids=semantic_token_ids,
            accepted_token_ids=semantic_token_ids,
        )

    variant = get_translation_variant()
    rendered_prompt = build_translation_messages(
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
    )


class TranslationUnitManager:
    def __init__(self, runtime_config: SimpleNamespace):
        self.runtime_config = runtime_config

    def reset_partial_state(self) -> None:
        state.partial_translation = PartialTranslationState()

    def normalize_source_text(
        self,
        source_text: str,
        *,
        is_final: bool,
    ) -> NormalizedSourceText:
        return normalize_source_text_for_mt(
            source_text.strip(),
            word_timestamps_ms=(None if is_final else state.partial_word_timestamps_ms),
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
            current_audio_ms=current_audio_seconds() * 1000.0,
            inaccessible_ms=float(self.runtime_config.translation_alignatt_inaccessible_ms),
            is_final=is_final,
        )

    def current_accepted_prefill(self, source_text: str) -> str:
        source_text = source_text.strip()
        partial_state = state.partial_translation
        if not source_text:
            self.reset_partial_state()
            return ""
        if partial_state.source_prefix and not source_text.startswith(partial_state.source_prefix):
            self.reset_partial_state()
        return state.partial_translation.accepted_target

    def should_run_partial_mt(
        self,
        *,
        source_text: str,
        source_frontier: SourceAccessibilityFrontier,
    ) -> tuple[bool, str]:
        return should_run_partial_mt_update(
            previous_state=state.partial_translation,
            source_prefix=source_text,
            accessible_unit_count=source_frontier.accessible_unit_count,
            current_audio_seconds_value=current_audio_seconds(),
            stall_seconds=float(self.runtime_config.translation_scheduler_stall_seconds),
        )

    def snapshot_skipped_partial_result(
        self,
        *,
        source_frontier: SourceAccessibilityFrontier,
        scheduler_reason: str,
    ) -> MTBackendResult:
        previous_state = state.partial_translation
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
            previous_state=state.partial_translation,
            source_prefix=source_prefix,
            result=result,
        )

    def update_partial_state(
        self,
        source_text: str,
        result: MTBackendResult,
        source_frontier: SourceAccessibilityFrontier,
    ) -> None:
        previous_accepted = state.partial_translation.accepted_target
        accepted_target, accepted_token_ids = self.derive_monotone_acceptance(
            source_prefix=source_text.strip(),
            result=result,
        )
        last_accept_audio_seconds = state.partial_translation.last_accept_audio_seconds
        if accepted_target and accepted_target != previous_accepted:
            last_accept_audio_seconds = current_audio_seconds()
        state.partial_translation = PartialTranslationState(
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
            last_mt_audio_seconds=current_audio_seconds(),
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
        while len(state.utt_translations) < len(state.utt_sources):
            segment_idx = len(state.utt_translations)
            segment_source = state.utt_sources[segment_idx].strip()
            normalized_segment_source = self.normalize_source_text(
                segment_source,
                is_final=True,
            )
            assistant_prefill = ""
            if (
                state.partial_translation.source_prefix
                and state.partial_translation.accepted_target
                and segment_idx == len(state.utt_sources) - 1
                and normalized_segment_source.text.startswith(state.partial_translation.source_prefix)
            ):
                assistant_prefill = state.partial_translation.accepted_target
            last_result = translate_with_mt(
                normalized_segment_source.text,
                source_frontier=self.build_source_frontier(
                    normalized_segment_source,
                    is_final=True,
                ),
                source_history=normalized_source_history_window(state.utt_sources, segment_idx),
                translation_history=translation_history_window(state.utt_translations, segment_idx),
                is_partial=False,
                assistant_prefill=assistant_prefill,
            )
            state.utt_translations.append(last_result.acceptance_text)
            if assistant_prefill:
                self.reset_partial_state()
        return last_result

    def render_translation(self) -> tuple[str, MTBackendResult | None]:
        latest_result = self.sync_committed_translations()

        translation_segments = [segment for segment in state.utt_translations[1:] if segment.strip()]
        partial_source = normalize_partial_asr_hypothesis(state.asr_hypotheses[-1])
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
                partial_result = translate_with_mt(
                    normalized_partial_source.text,
                    source_frontier=partial_frontier,
                    source_history=normalized_source_history_window(
                        state.utt_sources,
                        len(state.utt_sources),
                    ),
                    translation_history=translation_history_window(
                        state.utt_translations,
                        len(state.utt_translations),
                    ),
                    is_partial=True,
                    assistant_prefill=self.current_accepted_prefill(normalized_partial_source.text),
                )
                self.update_partial_state(normalized_partial_source.text, partial_result, partial_frontier)
            else:
                partial_result = self.snapshot_skipped_partial_result(
                    source_frontier=partial_frontier,
                    scheduler_reason=scheduler_reason,
                )
            if state.partial_translation.accepted_target.strip():
                translation_segments.append(state.partial_translation.accepted_target)
            latest_result = partial_result
        else:
            self.reset_partial_state()

        return (
            normalize_incremental_target_text(
                " ".join(segment.strip() for segment in translation_segments if segment.strip())
            ),
            latest_result,
        )


translation_units = TranslationUnitManager(config)


def apply_translation_emit_policy(
    previous_translation: str,
    raw_translation: str,
    *,
    is_final: bool,
) -> tuple[str, str]:
    return apply_emission_policy(
        config.translation_emit_policy,
        previous_translation,
        raw_translation,
        max_tail_rewrite_words=config.translation_max_tail_rewrite_words,
        is_final=is_final,
    )


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


def run_stream_to_artifacts(
    wav_path: str,
    chunk_ms: int = 960,
) -> InferenceArtifacts:
    variant = get_translation_variant()
    load_models()
    clear_state()

    audio = load_wav(wav_path)
    chunk_size = int(SAMPLE_RATE * chunk_ms / 1000)
    last_asr = ""
    last_translation = ""
    last_raw_translation = ""
    last_committed_segments = len(state.utt_sources)
    word_delays_ms: List[float] = []
    word_elapsed_ms: List[float] = []
    updates: List[StreamUpdate] = []
    start_time = perf_counter()

    for start in range(0, len(audio), chunk_size):
        chunk = audio[start : start + chunk_size]
        state.source = np.concatenate([state.source, chunk])

        if len(state.source) / SAMPLE_RATE < config.min_start_seconds:
            continue

        current_asr = transcribe_audio()
        committed_segments = len(state.utt_sources)
        if not current_asr:
            continue
        if current_asr == last_asr and committed_segments == last_committed_segments:
            continue

        last_asr = current_asr
        last_committed_segments = committed_segments
        raw_translation, translation_result = translation_units.render_translation()
        translation, emission_policy_action = apply_translation_emit_policy(
            last_translation,
            raw_translation,
            is_final=False,
        )
        audio_processed_ms = len(state.source) * 1000.0 / SAMPLE_RATE
        wallclock_elapsed_ms = (perf_counter() - start_time) * 1000.0
        register_translation_timestamps(
            last_raw_translation,
            raw_translation,
            wallclock_elapsed_ms,
            word_elapsed_ms,
        )
        new_words = register_translation_words(
            last_translation,
            translation,
            audio_processed_ms,
            word_delays_ms,
        )
        updates.append(
            StreamUpdate(
                update_idx=len(updates),
                audio_processed_ms=audio_processed_ms,
                wallclock_elapsed_ms=wallclock_elapsed_ms,
                asr_text=current_asr,
                translation_text=translation,
                new_words=new_words,
                raw_translation_text=raw_translation,
                emission_policy_action=emission_policy_action,
                translation_prompt_num_cached_tokens=(
                    None if translation_result is None else translation_result.num_cached_tokens
                ),
                translation_prompt_num_tokens=(
                    None if translation_result is None else translation_result.prompt_num_tokens
                ),
                partial_accepted_target=(
                    state.partial_translation.accepted_target or None
                ),
                partial_draft_target=(
                    state.partial_translation.draft_target or None
                ),
                partial_accepted_token_count=(
                    len(state.partial_translation.accepted_token_ids) or None
                ),
                alignatt_metadata=(
                    None if translation_result is None else translation_result.alignatt_metadata
                ),
                translation_timings_ms=(
                    None if translation_result is None else translation_result.timings_ms
                ),
            )
        )
        current_time = audio_processed_ms / 1000.0
        print(f"[{current_time:6.2f}s] ASR: {current_asr}")
        print(f"[{current_time:6.2f}s] DE : {translation}")
        last_translation = translation
        last_raw_translation = raw_translation

    final_asr = transcribe_audio() or last_asr
    final_raw_translation, final_translation_result = translation_units.render_translation()
    final_translation, final_emission_policy_action = apply_translation_emit_policy(
        last_translation,
        final_raw_translation,
        is_final=True,
    )
    audio_duration_ms = len(audio) * 1000.0 / SAMPLE_RATE
    final_elapsed_ms = (perf_counter() - start_time) * 1000.0
    register_translation_timestamps(
        last_raw_translation,
        final_raw_translation,
        final_elapsed_ms,
        word_elapsed_ms,
    )
    final_new_words = register_translation_words(
        last_translation,
        final_translation,
        audio_duration_ms,
        word_delays_ms,
    )
    if final_asr != last_asr or final_translation != last_translation:
        updates.append(
            StreamUpdate(
                update_idx=len(updates),
                audio_processed_ms=audio_duration_ms,
                wallclock_elapsed_ms=final_elapsed_ms,
                asr_text=final_asr,
                translation_text=final_translation,
                new_words=final_new_words,
                raw_translation_text=final_raw_translation,
                emission_policy_action=final_emission_policy_action,
                translation_prompt_num_cached_tokens=(
                    None
                    if final_translation_result is None
                    else final_translation_result.num_cached_tokens
                ),
                translation_prompt_num_tokens=(
                    None
                    if final_translation_result is None
                    else final_translation_result.prompt_num_tokens
                ),
                partial_accepted_target=(
                    state.partial_translation.accepted_target or None
                ),
                partial_draft_target=(
                    state.partial_translation.draft_target or None
                ),
                partial_accepted_token_count=(
                    len(state.partial_translation.accepted_token_ids) or None
                ),
                alignatt_metadata=(
                    None
                    if final_translation_result is None
                    else final_translation_result.alignatt_metadata
                ),
                translation_timings_ms=(
                    None
                    if final_translation_result is None
                    else final_translation_result.timings_ms
                ),
            )
        )

    print("\nFinal ASR:")
    print(final_asr)
    print("\nFinal translation:")
    print(final_translation)

    return InferenceArtifacts(
        wav_path=wav_path,
        chunk_ms=chunk_ms,
        translation_variant=variant.variant_id,
        source_language=config.source_lang,
        target_language=config.target_lang,
        latency_unit=config.latency_unit,
        audio_duration_ms=audio_duration_ms,
        final_asr_text=final_asr,
        final_translation_text=final_translation,
        translation_word_delays_ms=word_delays_ms,
        translation_word_elapsed_ms=word_elapsed_ms,
        updates=updates,
        runtime_config={
            "translation_variant_id": variant.variant_id,
            "translation_variant_description": variant.description,
            "translation_alignatt_heads_path": config.translation_alignatt_heads_path,
            "translation_alignatt_top_k_heads": config.translation_alignatt_top_k_heads,
            "translation_alignatt_filter_width": config.translation_alignatt_filter_width,
            "translation_alignatt_probe_mode": config.translation_alignatt_probe_mode,
            "translation_alignatt_inaccessible_ms": config.translation_alignatt_inaccessible_ms,
            "translation_alignatt_rewind_threshold": config.translation_alignatt_rewind_threshold,
            "min_start_seconds": config.min_start_seconds,
            "max_history_utterances": config.max_history_utterances,
            "max_new_tokens": config.max_new_tokens,
            "partial_max_new_tokens": config.partial_max_new_tokens,
            "partial_followup_max_new_tokens": config.partial_followup_max_new_tokens,
            "translation_min_new_tokens": config.translation_min_new_tokens,
            "translation_token_budget_ratio": config.translation_token_budget_ratio,
            "translation_token_budget_buffer": config.translation_token_budget_buffer,
            "partial_translation_min_new_tokens": config.partial_translation_min_new_tokens,
            "partial_translation_token_budget_ratio": config.partial_translation_token_budget_ratio,
            "partial_translation_token_budget_buffer": config.partial_translation_token_budget_buffer,
            "translation_generation_margin": config.translation_generation_margin,
            "translation_emit_policy": config.translation_emit_policy,
            "translation_max_tail_rewrite_words": config.translation_max_tail_rewrite_words,
            "temperature": config.temperature,
            "repetition_penalty": config.repetition_penalty,
            "asr_gpu_memory_utilization": config.asr_gpu_memory_utilization,
            "gemma_max_model_len": config.gemma_max_model_len,
            "gemma_enable_prefix_caching": config.gemma_enable_prefix_caching,
            "gemma_transformers_device": config.gemma_transformers_device,
            "gemma_transformers_dtype": config.gemma_transformers_dtype,
            "gemma_transformers_fast_attention": config.gemma_transformers_fast_attention,
            "gemma_transformers_prompt_kv_reuse": config.gemma_transformers_prompt_kv_reuse,
            "translation_scheduler_stall_seconds": config.translation_scheduler_stall_seconds,
        },
    )


def run_stream(
    wav_path: str,
    chunk_ms: int = 960,
    output_dir: str | None = None,
    runtime_overrides: dict[str, Any] | None = None,
):
    with temporary_runtime_config(**(runtime_overrides or {})):
        artifacts = run_stream_to_artifacts(
            wav_path,
            chunk_ms=chunk_ms,
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
):
    with temporary_runtime_config(**(runtime_overrides or {})):
        artifacts = run_stream_to_artifacts(
            wav_path,
            chunk_ms=chunk_ms,
        )
        written_files = write_inference_artifacts(artifacts, output_dir)
        print(f"\nWrote baseline artifacts to {output_dir}")
        for label, path in written_files.items():
            print(f"- {label}: {path}")

    return written_files


if __name__ == "__main__":
    run_baseline()
