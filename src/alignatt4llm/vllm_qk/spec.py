"""The model plug-in point for the generic vLLM Q/K observer.

A :class:`VLLMAttentionSpec` is the only thing a new decoder-only LLM must
supply (besides a thin backend subclass and calibrated heads): which vLLM
attention class to patch, which attributes its ``forward`` exposes, and how to
build the patched ``forward``. This module is intentionally torch-free so the
spec and its resolver can be imported and unit-tested without torch/vLLM.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class VLLMAttentionSpec:
    """Describes how to capture Q/K from one model family's vLLM attention.

    Attributes:
        family: short identifier (e.g. ``"qwen2"``), used for log lines and to
            namespace the saved-original-forward attribute.
        attention_import_paths: ``(module_path, class_name)`` pairs to try, in
            order; the first importable class(es) get patched. Multiple entries
            cover families whose attention class moved across vLLM versions.
        required_attrs: attribute names the patched ``forward`` reads off the
            attention module. Asserted at install time so a vLLM bump that
            renames an internal fails loudly instead of mid-generation.
        make_patched_forward: callable that, given this spec, returns the
            replacement ``forward(self, positions, hidden_states, **kwargs)``.
            Use ``make_standard_decoder_patched_forward`` for the common
            no-QK-norm shape (Llama/Qwen2); norm-heavy families pass their own.
    """

    family: str
    attention_import_paths: tuple[tuple[str, str], ...]
    required_attrs: tuple[str, ...]
    make_patched_forward: Callable[["VLLMAttentionSpec"], Callable[..., Any]]

    def original_forward_attr(self) -> str:
        """Attribute name under which the unpatched ``forward`` is preserved."""
        return f"_alignatt_{self.family}_mt_qk_original_forward"


def resolve_attention_classes(
    import_paths: Sequence[tuple[str, str]],
) -> tuple[type, ...]:
    """Import the attention classes named in ``import_paths``, skipping misses.

    Swallows ``ImportError``/``AttributeError`` so a spec can list class paths
    for several vLLM versions and simply use whichever exist in the installed
    build. Returns an empty tuple if none resolve (the caller decides whether
    that is fatal). Torch/vLLM are imported lazily here, never at module load.
    """
    classes: list[type] = []
    for module_path, class_name in import_paths:
        try:
            module = __import__(module_path, fromlist=[class_name])
        except Exception:
            continue
        cls = getattr(module, class_name, None)
        if isinstance(cls, type):
            classes.append(cls)
    return tuple(classes)
