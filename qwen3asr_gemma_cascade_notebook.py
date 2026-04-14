# %%
"""Notebook-safe facade for the cascade demo.

Use this module from `.venv-inference`:

    from qwen3asr_gemma_cascade_notebook import load_models, run_stream
    load_models()
    run_stream("path/to/file.wav")
"""

from qwen3asr_gemma_cascade_core import clear_state, config, load_models, run_stream

__all__ = ["clear_state", "config", "load_models", "run_stream"]

# %%
# Run this in a later notebook cell:
# load_models()
# run_stream("path/to/file.wav")

# %%

run_stream("test-set/audio/ccpXHNfaoy.wav")
# %%
