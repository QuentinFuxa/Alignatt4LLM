from __future__ import annotations

from typing import Any

from vllm.v1.worker.worker_base import CompilationTimes


def ensure_compilation_times(value: Any) -> CompilationTimes:
    """Normalize vLLM warmup timings across API revisions.

    Older local workers returned a bare float for the language-model compile
    time, while newer vLLM releases expect a ``CompilationTimes`` pair.
    """
    if isinstance(value, CompilationTimes):
        return value
    if hasattr(value, "language_model") and hasattr(value, "encoder"):
        return CompilationTimes(
            language_model=float(value.language_model),
            encoder=float(value.encoder),
        )
    return CompilationTimes(language_model=float(value or 0.0), encoder=0.0)


def compilation_time_seconds(value: Any) -> float:
    return float(ensure_compilation_times(value).language_model)
