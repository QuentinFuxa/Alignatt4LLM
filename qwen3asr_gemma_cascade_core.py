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
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

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
from cascade_translation_variants import (
    DEFAULT_TRANSLATION_VARIANT_ID,
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


@dataclass
class PartialTranslationState:
    source_text: str = ""
    assistant_prefill: str = ""
    last_num_cached_tokens: int | None = None
    last_prompt_num_tokens: int | None = None
    last_boundary_emitted: bool = False
    last_commit_audio_seconds: float = 0.0


@dataclass
class TranslationResult:
    text: str
    num_cached_tokens: int | None = None
    prompt_num_tokens: int | None = None
    uncertainty_boundary_emitted: bool = False
    stop_reason: str | int | None = None


@dataclass
class CascadeState:
    speech_id: int = 0
    source: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    utt_timestamps: List[int] = field(default_factory=lambda: [0])
    utt_sources: List[str] = field(default_factory=lambda: [""])
    utt_translations: List[str] = field(default_factory=lambda: [""])
    asr_hypotheses: List[str] = field(default_factory=lambda: [""])
    partial_translation: PartialTranslationState = field(default_factory=PartialTranslationState)


asr_model_name = "/home/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/7278e1e70fe206f11671096ffdd38061171dd6e5"
forced_aligner_model_name = "/home/.cache/huggingface/hub/models--Qwen--Qwen3-ForcedAligner-0.6B/snapshots/c7cbfc2048c462b0d63a45797104fc9db3ad62b7"
gemma_model_name = "/home/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/83df0a889143b1dbfc61b591bbc639540fd9ce4c"

config = SimpleNamespace(
    source_lang="English",
    target_lang="German",
    latency_unit="word",
    min_start_seconds=5.0,
    translation_variant_id=DEFAULT_TRANSLATION_VARIANT_ID,
    max_history_utterances=0,
    translation_mode="utterance_incremental",
    max_new_tokens=160,
    translation_min_new_tokens=32,
    translation_token_budget_ratio=3.0,
    translation_token_budget_buffer=24,
    translation_generation_margin=8,
    translation_emit_policy=RAW_PASSTHROUGH,
    translation_max_tail_rewrite_words=14,
    temperature=0.0,
    repetition_penalty=1.05,
    asr_gpu_memory_utilization=0.2,
    gemma_gpu_memory_utilization=0.44,
    gemma_max_model_len=1024,
    gemma_enforce_eager=True,
    gemma_enable_prefix_caching=True,
    uncertainty_marker_logit_bias_max=20.0,
    uncertainty_marker_logit_bias_min=4.0,
    uncertainty_marker_decay_audio_seconds=6.0,
    uncertainty_marker_followup_bias_cap=6.0,
)

asr = None
gemma_tokenizer = None
gemma_llm = None
gemma_uncertainty_token_id: int | None = None
state = CascadeState()


def get_translation_variant(variant_id: str) -> TranslationVariant:
    try:
        return TRANSLATION_VARIANTS[variant_id]
    except KeyError as exc:
        known = ", ".join(sorted(TRANSLATION_VARIANTS))
        raise ValueError(f"Unknown translation variant '{variant_id}'. Known variants: {known}") from exc


def available_translation_variants() -> list[TranslationVariant]:
    return [TRANSLATION_VARIANTS[variant_id] for variant_id in sorted(TRANSLATION_VARIANTS)]


def set_translation_variant(variant_id: str) -> TranslationVariant:
    variant = get_translation_variant(variant_id)
    config.translation_variant_id = variant.variant_id
    config.max_history_utterances = variant.max_history_utterances
    return variant


# %%
def load_models():
    global asr, gemma_tokenizer, gemma_llm, gemma_uncertainty_token_id, state

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

    if gemma_tokenizer is None:
        gemma_tokenizer = AutoTokenizer.from_pretrained(
            gemma_model_name,
            trust_remote_code=True,
            local_files_only=True,
        )
    gemma_uncertainty_token_id = ensure_registered_special_token(
        gemma_tokenizer,
        "<unused0>",
    )

    if gemma_llm is None:
        gemma_llm = LLM(
            model=gemma_model_name,
            tensor_parallel_size=max(1, torch.cuda.device_count()),
            max_model_len=config.gemma_max_model_len,
            gpu_memory_utilization=config.gemma_gpu_memory_utilization,
            enforce_eager=config.gemma_enforce_eager,
            enable_prefix_caching=config.gemma_enable_prefix_caching,
            trust_remote_code=True,
        )

    state = CascadeState(speech_id=state.speech_id)

# %%
def clear_state():
    global state
    state = CascadeState(speech_id=state.speech_id)


def ensure_registered_special_token(tokenizer, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id == getattr(tokenizer, "unk_token_id", None):
        raise RuntimeError(f"Gemma tokenizer does not expose {token!r} as a vocab token.")

    encoded = tokenizer.encode(token, add_special_tokens=False)
    if encoded != [int(token_id)]:
        tokenizer.add_special_tokens({"additional_special_tokens": [token]})
        encoded = tokenizer.encode(token, add_special_tokens=False)

    if encoded != [int(token_id)]:
        raise RuntimeError(
            f"Gemma tokenizer failed to encode {token!r} as a single special token: {encoded}"
        )
    if token not in tokenizer.all_special_tokens:
        raise RuntimeError(f"Gemma tokenizer did not register {token!r} as a special token.")
    return int(token_id)


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
        state.asr_hypotheses = [asr_hypo[rightest_punct_idx + 1 :].strip()]

    if state.utt_sources[1:]:
        return " ".join(state.utt_sources[1:] + [state.asr_hypotheses[-1]]).strip()
    return state.asr_hypotheses[-1].strip()


def translation_history_window(items: List[str], end_exclusive: int) -> List[str]:
    if config.max_history_utterances <= 0:
        return []

    start = max(1, end_exclusive - config.max_history_utterances)
    return [item.strip() for item in items[start:end_exclusive] if item.strip()]


def build_translation_messages(
    text: str,
    *,
    source_history: List[str],
    translation_history: List[str],
    is_partial: bool,
    assistant_prefill: str = "",
) -> RenderedTranslationPrompt:
    variant = get_translation_variant(config.translation_variant_id)
    return variant.render_messages(
        source_lang=config.source_lang,
        target_lang=config.target_lang,
        text=text,
        source_history=source_history,
        translation_history=translation_history,
        is_partial=is_partial,
        assistant_prefill=assistant_prefill,
    )


def apply_chat_template_token_ids(rendered_prompt: RenderedTranslationPrompt) -> list[int]:
    template_kwargs: dict[str, Any] = {
        "tokenize": True,
        "return_dict": True,
    }
    if rendered_prompt.continue_final_message:
        template_kwargs["continue_final_message"] = True
    else:
        template_kwargs["add_generation_prompt"] = True
    prompt_token_ids = gemma_tokenizer.apply_chat_template(
        rendered_prompt.messages,
        **template_kwargs,
    )
    if hasattr(prompt_token_ids, "keys") and "input_ids" in prompt_token_ids:
        prompt_token_ids = prompt_token_ids["input_ids"]
    elif hasattr(prompt_token_ids, "ids"):
        prompt_token_ids = prompt_token_ids.ids
    return list(prompt_token_ids)


def build_translation_request(
    rendered_prompt: RenderedTranslationPrompt,
    source_text: str,
    *,
    variant: TranslationVariant,
    is_partial: bool,
    assistant_prefill: str,
) -> tuple[dict[str, Any], SamplingParams, int]:
    prompt_token_ids = apply_chat_template_token_ids(rendered_prompt)
    prompt_tokens = len(prompt_token_ids)
    source_tokens = len(gemma_tokenizer(source_text, add_special_tokens=False)["input_ids"])
    desired_max_tokens = max(
        config.translation_min_new_tokens,
        int(source_tokens * config.translation_token_budget_ratio) + config.translation_token_budget_buffer,
    )
    available_max_tokens = config.gemma_max_model_len - prompt_tokens - config.translation_generation_margin
    if available_max_tokens < 1:
        raise RuntimeError(
            f"Gemma prompt exhausted the context window: prompt_tokens={prompt_tokens} "
            f"gemma_max_model_len={config.gemma_max_model_len}"
        )

    sampling_kwargs: dict[str, Any] = {
        "temperature": config.temperature,
        "max_tokens": min(config.max_new_tokens, desired_max_tokens, available_max_tokens),
        "repetition_penalty": config.repetition_penalty,
        "skip_reading_prefix_cache": False,
    }
    uncertainty_logit_bias = compute_uncertainty_marker_logit_bias(
        variant=variant,
        is_partial=is_partial,
        assistant_prefill=assistant_prefill,
    )
    if (
        is_partial
        and variant.uses_uncertainty_boundary
        and variant.uncertainty_marker
        and gemma_uncertainty_token_id is not None
    ):
        sampling_kwargs["stop_token_ids"] = [gemma_uncertainty_token_id]
        if uncertainty_logit_bias is not None:
            sampling_kwargs["logit_bias"] = {gemma_uncertainty_token_id: uncertainty_logit_bias}

    return {"prompt_token_ids": prompt_token_ids}, SamplingParams(**sampling_kwargs), prompt_tokens


def current_audio_seconds() -> float:
    return len(state.source) / SAMPLE_RATE


def compute_uncertainty_marker_logit_bias(
    *,
    variant: TranslationVariant,
    is_partial: bool,
    assistant_prefill: str,
) -> float | None:
    if (
        not is_partial
        or not variant.uses_uncertainty_boundary
        or gemma_uncertainty_token_id is None
    ):
        return None

    max_bias = float(config.uncertainty_marker_logit_bias_max)
    min_bias = float(config.uncertainty_marker_logit_bias_min)
    decay_seconds = max(1e-6, float(config.uncertainty_marker_decay_audio_seconds))

    new_audio_seconds = max(
        0.0,
        current_audio_seconds() - state.partial_translation.last_commit_audio_seconds,
    )
    if new_audio_seconds >= decay_seconds:
        scheduled_bias = min_bias
    else:
        ratio = 1.0 - (new_audio_seconds / decay_seconds)
        scheduled_bias = min_bias + ratio * (max_bias - min_bias)

    if assistant_prefill.strip():
        scheduled_bias = min(
            scheduled_bias,
            float(config.uncertainty_marker_followup_bias_cap),
        )
    return scheduled_bias


def completion_emitted_uncertainty_boundary(
    *,
    variant: TranslationVariant,
    completion,
) -> bool:
    if not variant.uses_uncertainty_boundary or not variant.uncertainty_marker:
        return False
    if variant.uncertainty_marker in completion.text:
        return True
    if (
        gemma_uncertainty_token_id is not None
        and gemma_uncertainty_token_id in completion.token_ids
    ):
        return True
    return completion.stop_reason in {variant.uncertainty_marker, gemma_uncertainty_token_id}


def translate_with_gemma(
    text: str,
    *,
    source_history: List[str] | None = None,
    translation_history: List[str] | None = None,
    is_partial: bool = False,
    assistant_prefill: str = "",
) -> TranslationResult:
    if gemma_tokenizer is None or gemma_llm is None:
        raise RuntimeError("Models are not loaded. Run load_models() first.")

    text = text.strip()
    if not text:
        return TranslationResult(text=assistant_prefill.rstrip(" \n"))

    variant = get_translation_variant(config.translation_variant_id)
    rendered_prompt = build_translation_messages(
        text,
        source_history=source_history or [],
        translation_history=translation_history or [],
        is_partial=is_partial,
        assistant_prefill=assistant_prefill,
    )
    prompt, sampling_params, prompt_num_tokens = build_translation_request(
        rendered_prompt,
        text,
        variant=variant,
        is_partial=is_partial,
        assistant_prefill=assistant_prefill,
    )
    outputs = gemma_llm.generate(prompt, sampling_params)
    request_output = outputs[0]
    completion = request_output.outputs[0]
    boundary_seen = completion_emitted_uncertainty_boundary(
        variant=variant,
        completion=completion,
    )
    normalized_text, boundary_seen = variant.normalize_output(
        generated_text=completion.text,
        assistant_prefill=assistant_prefill,
        is_partial=is_partial,
    )
    boundary_seen = boundary_seen or completion_emitted_uncertainty_boundary(
        variant=variant,
        completion=completion,
    )
    return TranslationResult(
        text=normalized_text,
        num_cached_tokens=request_output.num_cached_tokens,
        prompt_num_tokens=prompt_num_tokens,
        uncertainty_boundary_emitted=boundary_seen,
        stop_reason=completion.stop_reason,
    )


def reset_partial_translation_state() -> None:
    state.partial_translation = PartialTranslationState()


def current_assistant_prefill(source_text: str) -> str:
    source_text = source_text.strip()
    partial_state = state.partial_translation
    if not source_text:
        reset_partial_translation_state()
        return ""
    if partial_state.source_text and not source_text.startswith(partial_state.source_text):
        reset_partial_translation_state()
    return state.partial_translation.assistant_prefill


def update_partial_translation_state(source_text: str, result: TranslationResult) -> None:
    previous_prefill = state.partial_translation.assistant_prefill
    last_commit_audio_seconds = state.partial_translation.last_commit_audio_seconds
    if result.text and result.text != previous_prefill:
        last_commit_audio_seconds = current_audio_seconds()
    state.partial_translation = PartialTranslationState(
        source_text=source_text.strip(),
        assistant_prefill=result.text,
        last_num_cached_tokens=result.num_cached_tokens,
        last_prompt_num_tokens=result.prompt_num_tokens,
        last_boundary_emitted=result.uncertainty_boundary_emitted,
        last_commit_audio_seconds=last_commit_audio_seconds,
    )


def sync_committed_translations() -> TranslationResult | None:
    last_result: TranslationResult | None = None
    while len(state.utt_translations) < len(state.utt_sources):
        segment_idx = len(state.utt_translations)
        segment_source = state.utt_sources[segment_idx].strip()
        assistant_prefill = ""
        if (
            state.partial_translation.source_text
            and state.partial_translation.assistant_prefill
            and segment_idx == len(state.utt_sources) - 1
            and segment_source.startswith(state.partial_translation.source_text)
        ):
            assistant_prefill = state.partial_translation.assistant_prefill
        last_result = translate_with_gemma(
            segment_source,
            source_history=translation_history_window(state.utt_sources, segment_idx),
            translation_history=translation_history_window(state.utt_translations, segment_idx),
            is_partial=False,
            assistant_prefill=assistant_prefill,
        )
        state.utt_translations.append(last_result.text)
        if assistant_prefill:
            reset_partial_translation_state()
    return last_result


def render_translation_from_state() -> tuple[str, TranslationResult | None]:
    latest_result = sync_committed_translations()

    translation_segments = [segment for segment in state.utt_translations[1:] if segment.strip()]
    partial_source = state.asr_hypotheses[-1].strip()
    if partial_source:
        partial_result = translate_with_gemma(
            partial_source,
            source_history=translation_history_window(state.utt_sources, len(state.utt_sources)),
            translation_history=translation_history_window(
                state.utt_translations,
                len(state.utt_translations),
            ),
            is_partial=True,
            assistant_prefill=current_assistant_prefill(partial_source),
        )
        update_partial_translation_state(partial_source, partial_result)
        translation_segments.append(partial_result.text)
        latest_result = partial_result
    else:
        reset_partial_translation_state()

    return " ".join(segment.strip() for segment in translation_segments if segment.strip()).strip(), latest_result


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
    *,
    translation_variant: str | None = None,
) -> InferenceArtifacts:
    variant = set_translation_variant(translation_variant or config.translation_variant_id)
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
        raw_translation, translation_result = render_translation_from_state()
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
                partial_committed_prefix=(
                    state.partial_translation.assistant_prefill or None
                ),
                uncertainty_boundary_emitted=(
                    None
                    if translation_result is None
                    else translation_result.uncertainty_boundary_emitted
                ),
            )
        )
        current_time = audio_processed_ms / 1000.0
        print(f"[{current_time:6.2f}s] ASR: {current_asr}")
        print(f"[{current_time:6.2f}s] DE : {translation}")
        last_translation = translation
        last_raw_translation = raw_translation

    final_asr = transcribe_audio() or last_asr
    final_raw_translation, final_translation_result = render_translation_from_state()
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
                partial_committed_prefix=(
                    state.partial_translation.assistant_prefill or None
                ),
                uncertainty_boundary_emitted=(
                    None
                    if final_translation_result is None
                    else final_translation_result.uncertainty_boundary_emitted
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
            "translation_mode": config.translation_mode,
            "min_start_seconds": config.min_start_seconds,
            "max_history_utterances": config.max_history_utterances,
            "max_new_tokens": config.max_new_tokens,
            "translation_min_new_tokens": config.translation_min_new_tokens,
            "translation_token_budget_ratio": config.translation_token_budget_ratio,
            "translation_token_budget_buffer": config.translation_token_budget_buffer,
            "translation_generation_margin": config.translation_generation_margin,
            "translation_emit_policy": config.translation_emit_policy,
            "translation_max_tail_rewrite_words": config.translation_max_tail_rewrite_words,
            "temperature": config.temperature,
            "repetition_penalty": config.repetition_penalty,
            "asr_gpu_memory_utilization": config.asr_gpu_memory_utilization,
            "gemma_gpu_memory_utilization": config.gemma_gpu_memory_utilization,
            "gemma_max_model_len": config.gemma_max_model_len,
            "gemma_enforce_eager": config.gemma_enforce_eager,
            "gemma_enable_prefix_caching": config.gemma_enable_prefix_caching,
            "translation_uses_uncertainty_boundary": variant.uses_uncertainty_boundary,
            "translation_uncertainty_marker": variant.uncertainty_marker,
            "translation_uncertainty_token_id": gemma_uncertainty_token_id,
            "translation_uncertainty_marker_logit_bias_max": config.uncertainty_marker_logit_bias_max,
            "translation_uncertainty_marker_logit_bias_min": config.uncertainty_marker_logit_bias_min,
            "translation_uncertainty_marker_decay_audio_seconds": config.uncertainty_marker_decay_audio_seconds,
            "translation_uncertainty_marker_followup_bias_cap": config.uncertainty_marker_followup_bias_cap,
        },
    )


def run_stream(
    wav_path: str,
    chunk_ms: int = 960,
    output_dir: str | None = None,
    *,
    translation_variant: str | None = None,
):
    artifacts = run_stream_to_artifacts(
        wav_path,
        chunk_ms=chunk_ms,
        translation_variant=translation_variant,
    )
    if output_dir is not None:
        write_inference_artifacts(artifacts, output_dir)

    return artifacts.final_asr_text, artifacts.final_translation_text


def run_baseline(
    wav_path: str = DEFAULT_WAV_PATH,
    *,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    chunk_ms: int = 960,
    translation_variant: str | None = None,
):
    artifacts = run_stream_to_artifacts(
        wav_path,
        chunk_ms=chunk_ms,
        translation_variant=translation_variant,
    )
    written_files = write_inference_artifacts(artifacts, output_dir)
    print(f"\nWrote baseline artifacts to {output_dir}")
    for label, path in written_files.items():
        print(f"- {label}: {path}")

    return written_files


if __name__ == "__main__":
    run_baseline()
