# %%
"""Notebook-safe facade for the cascade demo.

Use this module from `.venv-inference`:

    from qwen3asr_gemma_cascade_notebook import load_models, run_baseline, set_translation_variant
    load_models()
    set_translation_variant("context1_terminology_guard")
    run_baseline()
"""

from qwen3asr_gemma_cascade_core import (
    available_translation_variants,
    clear_state,
    config,
    load_models,
    run_baseline,
    run_stream,
    set_translation_variant,
)

__all__ = [
    "available_translation_variants",
    "clear_state",
    "config",
    "load_models",
    "run_baseline",
    "run_stream",
    "set_translation_variant",
]

# %%
# Run this in a later notebook cell:
# load_models()
# run_baseline()
