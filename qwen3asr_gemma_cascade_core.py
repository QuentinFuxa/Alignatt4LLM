from dataclasses import dataclass, field
from time import perf_counter
import os
from types import SimpleNamespace
from typing import List
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
class CascadeState:
    speech_id: int = 0
    source: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    utt_timestamps: List[int] = field(default_factory=lambda: [0])
    utt_sources: List[str] = field(default_factory=lambda: [""])
    utt_translations: List[str] = field(default_factory=lambda: [""])
    asr_hypotheses: List[str] = field(default_factory=lambda: [""])


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
)

asr = None
gemma_tokenizer = None
gemma_llm = None
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
    global asr, gemma_tokenizer, gemma_llm, state

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

    if gemma_llm is None:
        gemma_llm = LLM(
            model=gemma_model_name,
            tensor_parallel_size=max(1, torch.cuda.device_count()),
            max_model_len=config.gemma_max_model_len,
            gpu_memory_utilization=config.gemma_gpu_memory_utilization,
            enforce_eager=config.gemma_enforce_eager,
            trust_remote_code=True,
        )

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


def build_translation_prompt(
    text: str,
    *,
    source_history: List[str],
    translation_history: List[str],
    is_partial: bool,
) -> str:
    variant = get_translation_variant(config.translation_variant_id)
    return variant.render_prompt(
        source_lang=config.source_lang,
        target_lang=config.target_lang,
        text=text,
        source_history=source_history,
        translation_history=translation_history,
        is_partial=is_partial,
    )


def build_translation_request(prompt: str, source_text: str) -> tuple[str, int]:
    messages = [{"role": "user", "content": prompt}]
    final_prompt = gemma_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    prompt_tokens = len(gemma_tokenizer(final_prompt, add_special_tokens=False)["input_ids"])
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

    return final_prompt, min(config.max_new_tokens, desired_max_tokens, available_max_tokens)


def translate_with_gemma(
    text: str,
    *,
    source_history: List[str] | None = None,
    translation_history: List[str] | None = None,
    is_partial: bool = False,
) -> str:
    if gemma_tokenizer is None or gemma_llm is None:
        raise RuntimeError("Models are not loaded. Run load_models() first.")

    text = text.strip()
    if not text:
        return ""

    prompt = build_translation_prompt(
        text,
        source_history=source_history or [],
        translation_history=translation_history or [],
        is_partial=is_partial,
    )
    final_prompt, max_tokens = build_translation_request(prompt, text)
    outputs = gemma_llm.generate(
        final_prompt,
        SamplingParams(
            temperature=config.temperature,
            max_tokens=max_tokens,
            repetition_penalty=config.repetition_penalty,
        ),
    )
    return outputs[0].outputs[0].text.strip()


def sync_committed_translations() -> None:
    while len(state.utt_translations) < len(state.utt_sources):
        segment_idx = len(state.utt_translations)
        segment_source = state.utt_sources[segment_idx].strip()
        state.utt_translations.append(
            translate_with_gemma(
                segment_source,
                source_history=translation_history_window(state.utt_sources, segment_idx),
                translation_history=translation_history_window(state.utt_translations, segment_idx),
                is_partial=False,
            )
        )


def render_translation_from_state() -> str:
    sync_committed_translations()

    translation_segments = [segment for segment in state.utt_translations[1:] if segment.strip()]
    partial_source = state.asr_hypotheses[-1].strip()
    if partial_source:
        translation_segments.append(
            translate_with_gemma(
                partial_source,
                source_history=translation_history_window(state.utt_sources, len(state.utt_sources)),
                translation_history=translation_history_window(
                    state.utt_translations,
                    len(state.utt_translations),
                ),
                is_partial=True,
            )
        )

    return " ".join(segment.strip() for segment in translation_segments if segment.strip()).strip()


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
        raw_translation = render_translation_from_state()
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
            )
        )
        current_time = audio_processed_ms / 1000.0
        print(f"[{current_time:6.2f}s] ASR: {current_asr}")
        print(f"[{current_time:6.2f}s] DE : {translation}")
        last_translation = translation
        last_raw_translation = raw_translation

    final_asr = transcribe_audio() or last_asr
    final_raw_translation = render_translation_from_state()
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
