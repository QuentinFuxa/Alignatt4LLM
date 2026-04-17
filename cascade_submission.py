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
    max_history_utterances: int = 1
    partial_max_new_tokens: int = 16
    partial_followup_max_new_tokens: int = 8
    translation_alignatt_rewind_threshold: int = 8
    translation_alignatt_inaccessible_ms: float = 0.0
    asr_commit_mode: str = "punctuation_lcp"
    asr_alignatt_frontier_margin_ms: float = 500.0
    asr_stability_k: int = 3
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
            type="cascade_simulstream_processor.CascadeAlignAttProcessor",
            source_lang_code=source_lang_code,
            target_lang_code=target_lang_code,
            chunk_ms=self.chunk_ms,
            speech_chunk_size=self.chunk_ms / 1000.0,
            alignment_backend_name=self.alignment_backend_name,
            mt_backend_name=self.mt_backend_name,
            min_start_seconds=self.min_start_seconds,
            max_history_utterances=self.max_history_utterances,
            partial_max_new_tokens=self.partial_max_new_tokens,
            partial_followup_max_new_tokens=self.partial_followup_max_new_tokens,
            translation_alignatt_min_source_mass=self.translation_alignatt_min_source_mass,
            translation_alignatt_rewind_threshold=self.translation_alignatt_rewind_threshold,
            translation_alignatt_inaccessible_ms=self.translation_alignatt_inaccessible_ms,
            asr_commit_mode=self.asr_commit_mode,
            asr_alignatt_frontier_margin_ms=self.asr_alignatt_frontier_margin_ms,
            asr_stability_k=self.asr_stability_k,
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
        chunk_ms=450,
        description="Main-track low-latency preset (~0-2 s LongYAAL on current calibration).",
    ),
    "main_high_latency": SubmissionPreset(
        name="main_high_latency",
        track="main",
        latency_regime="high",
        chunk_ms=700,
        description="Main-track high-latency preset (~2-4 s LongYAAL on current calibration).",
    ),
    "context_low_latency": SubmissionPreset(
        name="context_low_latency",
        track="extra_context",
        latency_regime="low",
        chunk_ms=450,
        paper_context_mode="title_abstract",
        translation_alignatt_min_source_mass=0.3,
        description=(
            "Extra-context low-latency preset using the current paper-ready "
            "title+abstract guarded setting."
        ),
    ),
    "context_high_latency": SubmissionPreset(
        name="context_high_latency",
        track="extra_context",
        latency_regime="high",
        chunk_ms=700,
        paper_context_mode="title_abstract",
        translation_alignatt_min_source_mass=0.3,
        description=(
            "Extra-context high-latency preset using the current paper-ready "
            "title+abstract guarded setting."
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
