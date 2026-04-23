from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace


@dataclass(frozen=True)
class SubmissionPreset:
    name: str
    track: str
    latency_regime: str
    chunk_ms: int
    description: str
    paper_context_mode: str = "off"
    translation_alignatt_min_source_mass: float = 0.0
    alignment_backend_name: str = "qwen_forced"
    mt_backend_name: str = "gemma_vllm_alignatt"
    min_start_seconds: float = 2.0
    max_history_utterances: int = 0
    partial_max_new_tokens: int = 16
    translation_alignatt_border_margin: int = 0
    translation_alignatt_inaccessible_ms: float = 0.0
    translation_alignatt_argmax_mass_threshold: float = 0.0
    mt_vllm_enforce_eager: bool = False
    mt_vllm_cudagraph_mode: str = "full"
    mt_vllm_enable_prefix_caching: bool = False
    mt_vllm_gpu_memory_utilization: float = 0.5
    paper_context_top_k: int = 3
    paper_context_max_chars: int = 1200
    paper_context_history_window_words: int = 60

    def build_speech_processor_config(
        self,
        *,
        source_lang_code: str,
        target_lang_code: str,
        paper_context_path: str | None = None,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            type="cascade.simulstream_processor.CascadeAlignAttProcessor",
            source_lang_code=source_lang_code,
            target_lang_code=target_lang_code,
            chunk_ms=self.chunk_ms,
            speech_chunk_size=self.chunk_ms / 1000.0,
            alignment_backend_name=self.alignment_backend_name,
            mt_backend_name=self.mt_backend_name,
            min_start_seconds=self.min_start_seconds,
            max_history_utterances=self.max_history_utterances,
            partial_max_new_tokens=self.partial_max_new_tokens,
            translation_alignatt_min_source_mass=self.translation_alignatt_min_source_mass,
            translation_alignatt_border_margin=self.translation_alignatt_border_margin,
            translation_alignatt_inaccessible_ms=self.translation_alignatt_inaccessible_ms,
            translation_alignatt_argmax_mass_threshold=self.translation_alignatt_argmax_mass_threshold,
            mt_vllm_enforce_eager=self.mt_vllm_enforce_eager,
            mt_vllm_cudagraph_mode=self.mt_vllm_cudagraph_mode,
            mt_vllm_enable_prefix_caching=self.mt_vllm_enable_prefix_caching,
            mt_vllm_gpu_memory_utilization=self.mt_vllm_gpu_memory_utilization,
            paper_context_path=paper_context_path,
            paper_context_mode=self.paper_context_mode,
            paper_context_top_k=self.paper_context_top_k,
            paper_context_max_chars=self.paper_context_max_chars,
            paper_context_history_window_words=self.paper_context_history_window_words,
        )


SUBMISSION_PRESETS = {
    "main_low_latency": SubmissionPreset(
        name="main_low_latency",
        track="main",
        latency_regime="low",
        chunk_ms=1100,
        translation_alignatt_border_margin=1,
        description=(
            "Main-track low-latency preset: qwen_forced ASR + gemma_vllm_alignatt "
            "MT with chunk_ms=1100 and AlignAtt border margin=1. Validated on "
            "the dev-set inside the LOW LongYAAL regime."
        ),
    ),
    "main_high_latency": SubmissionPreset(
        name="main_high_latency",
        track="main",
        latency_regime="high",
        chunk_ms=1500,
        translation_alignatt_border_margin=1,
        description=(
            "Main-track high-latency preset: same mechanism as main_low_latency "
            "with larger chunk_ms=1500, validated inside the HIGH LongYAAL regime."
        ),
    ),
    "context_low_latency": SubmissionPreset(
        name="context_low_latency",
        track="extra_context",
        latency_regime="low",
        chunk_ms=1100,
        translation_alignatt_border_margin=1,
        paper_context_mode="title_abstract",
        translation_alignatt_min_source_mass=0.3,
        description=(
            "Extra-context low-latency preset: main_low_latency plus the "
            "title+abstract prompt with min_source_mass=0.3 guardrail."
        ),
    ),
    "context_high_latency": SubmissionPreset(
        name="context_high_latency",
        track="extra_context",
        latency_regime="high",
        chunk_ms=1500,
        translation_alignatt_border_margin=1,
        paper_context_mode="title_abstract",
        translation_alignatt_min_source_mass=0.3,
        description=(
            "Extra-context high-latency preset: main_high_latency plus the "
            "title+abstract prompt with min_source_mass=0.3 guardrail."
        ),
    ),
}

VALID_SUBMISSION_PRESET_NAMES = tuple(SUBMISSION_PRESETS)


def get_submission_preset(name: str) -> SubmissionPreset:
    try:
        return SUBMISSION_PRESETS[name]
    except KeyError as exc:
        valid = ", ".join(VALID_SUBMISSION_PRESET_NAMES)
        raise ValueError(f"Unknown submission preset {name!r}. Valid presets: {valid}") from exc
