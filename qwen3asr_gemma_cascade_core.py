"""Compatibility shim over the instantiable cascade runtime.

New code should import from ``cascade_runtime`` directly. This module keeps a
single default runtime alive so older notebooks and harnesses continue to
function during the transition.
"""

from __future__ import annotations

from typing import Any

from cascade_mt_backend import build_mt_backend
from cascade_artifacts import DEFAULT_OUTPUT_DIR, DEFAULT_WAV_PATH
from cascade_runtime import (
    LANGUAGE_CODE_TO_NAME,
    LANGUAGE_NAME_TO_CODE,
    CascadeRuntimeConfig,
    CascadeSession,
    CascadeState,
    LoadedModelBundle,
    PartialTranslationState,
    alignatt_heads_path_for,
    asr_model_name,
    forced_aligner_model_name,
    gemma_model_name,
    get_translation_variant,
    load_wav,
    normalize_partial_asr_hypothesis,
    run_baseline as _run_baseline,
    run_stream as _run_stream,
    run_stream_to_artifacts as _run_stream_to_artifacts,
    should_run_partial_mt_update,
    derive_monotone_partial_acceptance,
    target_lang_code_for,
    temporary_runtime_config as _temporary_runtime_config,
)


config = CascadeRuntimeConfig()
_default_bundle = LoadedModelBundle(config)
_default_session = _default_bundle.new_session()

asr = None
gemma_tokenizer = None
gemma_llm = None
mt_backend = None
alignment_backend = None
state = _default_session.state
translation_units = _default_session.translation_units


def _sync_compat_handles() -> None:
    global asr, gemma_tokenizer, gemma_llm, mt_backend, alignment_backend, state, translation_units
    state = _default_session.state
    translation_units = _default_session.translation_units
    mt_backend = _default_bundle.mt_backend
    alignment_backend = _default_bundle.alignment_backend
    gemma_tokenizer = None if mt_backend is None else mt_backend.tokenizer
    gemma_llm = None if mt_backend is None else getattr(mt_backend, "model", None)
    asr = None if alignment_backend is None else getattr(alignment_backend, "asr", None)


def load_models() -> None:
    _default_session.load_models()
    _sync_compat_handles()


def clear_state() -> None:
    _default_session.clear()
    _sync_compat_handles()


def rebuild_mt_backend_preserving_weights(existing_backend=None) -> None:
    global mt_backend
    source_backend = existing_backend if existing_backend is not None else _default_bundle.mt_backend
    saved_model = getattr(source_backend, "model", None) if source_backend is not None else None
    saved_tokenizer = getattr(source_backend, "tokenizer", None) if source_backend is not None else None
    _default_bundle.mt_backend = build_mt_backend(
        model_name=gemma_model_name,
        runtime_config=config,
    )
    _default_bundle.mt_backend.model = saved_model
    _default_bundle.mt_backend.tokenizer = saved_tokenizer
    _default_bundle.mt_backend.load()
    _sync_compat_handles()


def render_public_asr_text() -> str:
    return _default_session.render_public_asr_text()


def transcribe_audio():
    result = _default_session.transcribe_audio()
    _sync_compat_handles()
    return result


def run_stream_to_artifacts(
    wav_path: str,
    chunk_ms: int = 960,
    *,
    run_provenance: dict[str, Any] | None = None,
):
    artifacts = _run_stream_to_artifacts(
        wav_path,
        chunk_ms=chunk_ms,
        config=config,
        bundle=_default_bundle,
        run_provenance=run_provenance,
    )
    _sync_compat_handles()
    return artifacts


def run_stream(
    wav_path: str,
    chunk_ms: int = 960,
    output_dir: str | None = None,
    runtime_overrides: dict[str, Any] | None = None,
    run_provenance: dict[str, Any] | None = None,
):
    result = _run_stream(
        wav_path,
        chunk_ms=chunk_ms,
        output_dir=output_dir,
        runtime_overrides=runtime_overrides,
        run_provenance=run_provenance,
        config=config,
        bundle=_default_bundle,
    )
    _sync_compat_handles()
    return result


def run_baseline(
    wav_path: str = DEFAULT_WAV_PATH,
    *,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    chunk_ms: int = 960,
    runtime_overrides: dict[str, Any] | None = None,
    run_provenance: dict[str, Any] | None = None,
):
    written = _run_baseline(
        wav_path=wav_path,
        output_dir=output_dir,
        chunk_ms=chunk_ms,
        runtime_overrides=runtime_overrides,
        run_provenance=run_provenance,
        config=config,
        bundle=_default_bundle,
    )
    _sync_compat_handles()
    return written


def temporary_runtime_config(**overrides):
    return _temporary_runtime_config(config, **overrides)


_sync_compat_handles()


__all__ = [
    "LANGUAGE_CODE_TO_NAME",
    "LANGUAGE_NAME_TO_CODE",
    "CascadeRuntimeConfig",
    "CascadeSession",
    "CascadeState",
    "LoadedModelBundle",
    "PartialTranslationState",
    "alignatt_heads_path_for",
    "asr",
    "asr_model_name",
    "alignment_backend",
    "clear_state",
    "config",
    "derive_monotone_partial_acceptance",
    "forced_aligner_model_name",
    "gemma_llm",
    "gemma_model_name",
    "gemma_tokenizer",
    "get_translation_variant",
    "load_models",
    "load_wav",
    "mt_backend",
    "normalize_partial_asr_hypothesis",
    "rebuild_mt_backend_preserving_weights",
    "render_public_asr_text",
    "run_baseline",
    "run_stream",
    "run_stream_to_artifacts",
    "should_run_partial_mt_update",
    "state",
    "target_lang_code_for",
    "temporary_runtime_config",
    "transcribe_audio",
    "translation_units",
]
