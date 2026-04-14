from dataclasses import dataclass, field
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
    asr_hypotheses: List[str] = field(default_factory=lambda: [""])


asr_model_name = "/home/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/7278e1e70fe206f11671096ffdd38061171dd6e5"
forced_aligner_model_name = "/home/.cache/huggingface/hub/models--Qwen--Qwen3-ForcedAligner-0.6B/snapshots/c7cbfc2048c462b0d63a45797104fc9db3ad62b7"
gemma_model_name = "/home/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/83df0a889143b1dbfc61b591bbc639540fd9ce4c"

config = SimpleNamespace(
    source_lang="English",
    target_lang="German",
    latency_unit="word",
    min_start_seconds=5.0,
    max_history_utterances=0,
    max_new_tokens=100,
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


def translate_with_gemma(text: str) -> str:
    if gemma_tokenizer is None or gemma_llm is None:
        raise RuntimeError("Models are not loaded. Run load_models() first.")

    text = text.strip()
    if not text:
        return ""

    prompt = (
        f"You are a professional translator from {config.source_lang} to {config.target_lang}. "
        "The input may be incomplete because it comes from streaming ASR. "
        "Translate only what is clear. Return only the translation.\n\n"
        f"Source:\n{text}"
    )
    messages = [{"role": "user", "content": prompt}]
    final_prompt = gemma_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    outputs = gemma_llm.generate(
        final_prompt,
        SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_new_tokens,
            repetition_penalty=config.repetition_penalty,
        ),
    )
    return outputs[0].outputs[0].text.strip()


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


def run_stream(wav_path: str, chunk_ms: int = 960):
    load_models()
    clear_state()

    audio = load_wav(wav_path)
    chunk_size = int(SAMPLE_RATE * chunk_ms / 1000)
    last_asr = ""

    for start in range(0, len(audio), chunk_size):
        chunk = audio[start : start + chunk_size]
        state.source = np.concatenate([state.source, chunk])

        if len(state.source) / SAMPLE_RATE < config.min_start_seconds:
            continue

        current_asr = transcribe_audio()
        if not current_asr or current_asr == last_asr:
            continue

        last_asr = current_asr
        translation = translate_with_gemma(current_asr)
        current_time = start / SAMPLE_RATE
        print(f"[{current_time:6.2f}s] ASR: {current_asr}")
        print(f"[{current_time:6.2f}s] DE : {translation}")

    final_asr = transcribe_audio() or last_asr
    final_translation = translate_with_gemma(final_asr)

    print("\nFinal ASR:")
    print(final_asr)
    print("\nFinal translation:")
    print(final_translation)

    return final_asr, final_translation
if __name__ == "__main__":
    load_models()
    run_stream("test-set/audio/wJAPXMIoIG.wav")
