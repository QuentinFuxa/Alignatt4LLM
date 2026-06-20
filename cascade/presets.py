from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


SUPPORTED_SOURCE_LANG_CODES = ("en",)
SUPPORTED_TARGET_LANG_CODES = ("de", "it", "zh")
SUPPORTED_DIRECTION_PAIRS = frozenset(
    ("en", target) for target in SUPPORTED_TARGET_LANG_CODES
)


def translation_heads_path_for_direction(
    source_lang_code: str,
    target_lang_code: str,
) -> Path:
    return Path(
        "data/alignatt_heads/"
        f"translation_heads_google_gemma-4-E4B-it_{source_lang_code}-{target_lang_code}.json"
    )


def validate_runtime_direction(
    *,
    source_lang_code: str,
    target_lang_code: str,
    repo_root: Path | None = None,
) -> None:
    source_lang_code = source_lang_code.strip().lower()
    target_lang_code = target_lang_code.strip().lower()
    if (source_lang_code, target_lang_code) not in SUPPORTED_DIRECTION_PAIRS:
        valid = ", ".join(
            f"{source}-{target}"
            for source, target in sorted(SUPPORTED_DIRECTION_PAIRS)
        )
        raise ValueError(
            f"Unsupported runtime direction {source_lang_code}-{target_lang_code}. "
            f"The maintained development presets support only: {valid}."
        )
    if repo_root is None:
        return
    heads_path = repo_root / translation_heads_path_for_direction(
        source_lang_code,
        target_lang_code,
    )
    if not heads_path.is_file():
        raise FileNotFoundError(
            f"Missing Gemma AlignAtt translation heads for "
            f"{source_lang_code}-{target_lang_code}: {heads_path}"
        )


@dataclass(frozen=True)
class RuntimePreset:
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
    translation_alignatt_top_k_heads: int = 4
    translation_alignatt_border_margin: int = 1
    translation_alignatt_inaccessible_ms: float = 0.0
    translation_alignatt_argmax_mass_threshold: float = 0.0
    translation_alignatt_frontier_min_inaccessible_mass: float = 0.03
    translation_alignatt_max_inaccessible_source_mass: float = 1.0
    translation_alignatt_min_accessible_inaccessible_margin: float = -1.0
    translation_alignatt_source_bearing_min_source_mass: float = 0.005
    translation_alignatt_source_bearing_hard_inaccessible_cap: float = 1.0
    translation_acceptance_policy: str = "alignatt"
    translation_static_cutoff_units: int = 0
    # Capture-safety default: cudagraph replay corrupts the MT attention
    # observer's q/k payload (docs/status.md, 2026-06-09), so presets run the
    # MT engine eagerly. The cudagraph mode only applies when eager is
    # explicitly disabled.
    mt_vllm_enforce_eager: bool = True
    mt_vllm_cudagraph_mode: str = "full"
    mt_vllm_enable_prefix_caching: bool = False
    mt_vllm_gpu_memory_utilization: float = 0.5
    mt_vllm_enable_speculative_decoding: bool = False
    mt_vllm_speculative_assistant_model: str | None = None
    mt_vllm_num_speculative_tokens: int = 4
    paper_context_top_k: int = 3
    paper_context_max_chars: int = 1200
    paper_context_history_window_words: int = 60
    asr_punctuation_min_commit_words: int = 0

    def build_speech_processor_config(
        self,
        *,
        source_lang_code: str,
        target_lang_code: str,
        paper_context_path: str | None = None,
        repo_root: Path | None = None,
    ) -> SimpleNamespace:
        source_lang_code = source_lang_code.strip().lower()
        target_lang_code = target_lang_code.strip().lower()
        validate_runtime_direction(
            source_lang_code=source_lang_code,
            target_lang_code=target_lang_code,
            repo_root=repo_root,
        )
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
            translation_alignatt_top_k_heads=self.translation_alignatt_top_k_heads,
            translation_alignatt_min_source_mass=self.translation_alignatt_min_source_mass,
            translation_alignatt_border_margin=self.translation_alignatt_border_margin,
            translation_alignatt_inaccessible_ms=self.translation_alignatt_inaccessible_ms,
            translation_alignatt_argmax_mass_threshold=self.translation_alignatt_argmax_mass_threshold,
            translation_alignatt_frontier_min_inaccessible_mass=self.translation_alignatt_frontier_min_inaccessible_mass,
            translation_alignatt_max_inaccessible_source_mass=self.translation_alignatt_max_inaccessible_source_mass,
            translation_alignatt_min_accessible_inaccessible_margin=self.translation_alignatt_min_accessible_inaccessible_margin,
            translation_alignatt_source_bearing_min_source_mass=self.translation_alignatt_source_bearing_min_source_mass,
            translation_alignatt_source_bearing_hard_inaccessible_cap=self.translation_alignatt_source_bearing_hard_inaccessible_cap,
            translation_acceptance_policy=self.translation_acceptance_policy,
            translation_static_cutoff_units=self.translation_static_cutoff_units,
            mt_vllm_enforce_eager=self.mt_vllm_enforce_eager,
            mt_vllm_cudagraph_mode=self.mt_vllm_cudagraph_mode,
            mt_vllm_enable_prefix_caching=self.mt_vllm_enable_prefix_caching,
            mt_vllm_gpu_memory_utilization=self.mt_vllm_gpu_memory_utilization,
            mt_vllm_enable_speculative_decoding=self.mt_vllm_enable_speculative_decoding,
            mt_vllm_speculative_assistant_model=self.mt_vllm_speculative_assistant_model,
            mt_vllm_num_speculative_tokens=self.mt_vllm_num_speculative_tokens,
            asr_punctuation_min_commit_words=self.asr_punctuation_min_commit_words,
            paper_context_path=paper_context_path,
            paper_context_mode=self.paper_context_mode,
            paper_context_top_k=self.paper_context_top_k,
            paper_context_max_chars=self.paper_context_max_chars,
            paper_context_history_window_words=self.paper_context_history_window_words,
        )


RUNTIME_PRESETS = {
    "gemma_low_latency": RuntimePreset(
        name="gemma_low_latency",
        track="research",
        latency_regime="low",
        chunk_ms=850,
        translation_alignatt_top_k_heads=4,
        translation_alignatt_border_margin=1,
        translation_alignatt_min_source_mass=0.0,
        translation_alignatt_frontier_min_inaccessible_mass=0.03,
        translation_alignatt_max_inaccessible_source_mass=1.0,
        description=(
            "Gemma development low-latency preset: qwen_forced ASR + "
            "gemma_vllm_alignatt MT with chunk_ms=850 and permissive AlignAtt "
            "defaults."
        ),
    ),
    "gemma_high_latency": RuntimePreset(
        name="gemma_high_latency",
        track="research",
        latency_regime="high",
        chunk_ms=1500,
        translation_alignatt_top_k_heads=4,
        translation_alignatt_border_margin=1,
        translation_alignatt_min_source_mass=0.0,
        translation_alignatt_frontier_min_inaccessible_mass=0.03,
        translation_alignatt_max_inaccessible_source_mass=1.0,
        description=(
            "Gemma development high-latency preset: same ASR, MT, and "
            "permissive AlignAtt defaults as gemma_low_latency with "
            "chunk_ms=1500."
        ),
    ),
}

LEGACY_RUNTIME_PRESET_ALIASES = {
    "main_low_latency": "gemma_low_latency",
    "main_high_latency": "gemma_high_latency",
}

VALID_RUNTIME_PRESET_NAMES = tuple(RUNTIME_PRESETS)
_VALID_PRESET_NAMES_WITH_LEGACY = tuple(
    list(RUNTIME_PRESETS) + list(LEGACY_RUNTIME_PRESET_ALIASES)
)


def get_runtime_preset(name: str) -> RuntimePreset:
    resolved_name = LEGACY_RUNTIME_PRESET_ALIASES.get(name, name)
    try:
        return RUNTIME_PRESETS[resolved_name]
    except KeyError as exc:
        valid = ", ".join(_VALID_PRESET_NAMES_WITH_LEGACY)
        raise ValueError(f"Unknown runtime preset {name!r}. Valid presets: {valid}") from exc
