# %%
"""Notebook-safe facade for the cascade demo.

Use this module from `.venv-inference`:

    from qwen3asr_gemma_cascade_notebook import clear_state, load_models, run_baseline
    load_models()
    run_baseline()
"""

from qwen3asr_gemma_cascade_core import (
    clear_state,
    config,
    load_models,
    run_baseline,
    run_stream,
)

__all__ = [
    "clear_state",
    "config",
    "load_models",
    "run_baseline",
    "run_stream",
]

# %%
# Run this in a later notebook cell:
# load_models()
# run_baseline()
