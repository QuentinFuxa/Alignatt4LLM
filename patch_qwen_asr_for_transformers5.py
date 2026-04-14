#!/usr/bin/env python3
"""Patch qwen_asr 0.0.6 for the repo's validated transformers 5.x stack.

This script is intentionally idempotent so it can be run during Docker builds,
bootstrap scripts, and local env setup without caring whether the patch has
already been applied.


The environement that should be patched is .venv-inference.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def find_target_dir() -> Path:
    spec = importlib.util.find_spec("qwen_asr")
    if spec is None or spec.origin is None:
        raise RuntimeError("qwen_asr is not installed in the current Python environment")

    target_dir = Path(spec.origin).resolve().parent / "core" / "transformers_backend"
    if not target_dir.is_dir():
        raise RuntimeError(f"Could not find qwen_asr transformers backend at {target_dir}")
    return target_dir


def remove_forward_auto_docstring_blocks(text: str) -> tuple[str, int]:
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    removed = 0
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        if not stripped.startswith("@auto_docstring"):
            out.append(line)
            i += 1
            continue

        indent = line[: len(line) - len(stripped)]
        block_end = i + 1
        paren_depth = line.count("(") - line.count(")")
        while paren_depth > 0 and block_end < len(lines):
            paren_depth += lines[block_end].count("(") - lines[block_end].count(")")
            block_end += 1

        lookahead = block_end
        while lookahead < len(lines):
            candidate = lines[lookahead]
            candidate_stripped = candidate.strip()
            if not candidate_stripped:
                lookahead += 1
                continue
            if candidate.startswith(f"{indent}@"):
                lookahead += 1
                continue
            if candidate.startswith(f"{indent}def forward("):
                removed += 1
                i = block_end
                break
            out.extend(lines[i:block_end])
            i = block_end
            break
        else:
            out.extend(lines[i:block_end])
            i = block_end

    return "".join(out), removed


def patch_modeling_text(text: str) -> tuple[str, list[str]]:
    updated = text
    changes: list[str] = []

    old_import = "from transformers.utils.generic import TransformersKwargs, check_model_inputs"
    new_import = "from transformers.utils.generic import TransformersKwargs, merge_with_config_defaults"
    if old_import in updated:
        updated = updated.replace(old_import, new_import)
        changes.append("replaced check_model_inputs import with merge_with_config_defaults")

    if "@check_model_inputs()\n" in updated:
        updated = updated.replace("@check_model_inputs()\n", "@merge_with_config_defaults\n")
        changes.append("replaced @check_model_inputs() with @merge_with_config_defaults")

    if "@check_model_inputs\n" in updated:
        updated = updated.replace("@check_model_inputs\n", "@merge_with_config_defaults\n")
        changes.append("replaced @check_model_inputs with @merge_with_config_defaults")

    rope_import = "from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update"
    rope_patch = """from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update

if "default" not in ROPE_INIT_FUNCTIONS:
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

    ROPE_INIT_FUNCTIONS["default"] = _qwen3_asr_default_rope_init"""
    if rope_import in updated and "_qwen3_asr_default_rope_init" not in updated:
        updated = updated.replace(rope_import, rope_patch)
        changes.append("added default RoPE initializer fallback for transformers 5.x")

    rope_init_line = "        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]\n"
    rope_init_replacement = """        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        self.compute_default_rope_parameters = _qwen3_asr_default_rope_init
"""
    if rope_init_line in updated and "self.compute_default_rope_parameters = _qwen3_asr_default_rope_init" not in updated:
        updated = updated.replace(rope_init_line, rope_init_replacement)
        changes.append("attached compute_default_rope_parameters hook for transformers 5.x")

    updated, removed = remove_forward_auto_docstring_blocks(updated)
    if removed:
        changes.append(f"removed {removed} forward auto_docstring decorator block(s)")

    if "merge_with_config_defaults" not in updated:
        raise RuntimeError("Expected merge_with_config_defaults to be present after patching")
    if "class Qwen3ASRThinkerTextModel" not in updated:
        raise RuntimeError("Unexpected qwen_asr file layout; refusing to patch blindly")

    return updated, changes


def patch_configuration_text(text: str) -> tuple[str, list[str]]:
    updated = text
    changes: list[str] = []

    old_thinker_init = """    def __init__(
        self,
        audio_config=None,
        text_config=None,
        audio_token_id=151646,
        audio_start_token_id=151647,
        user_token_id=872,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.user_token_id = user_token_id
        self.audio_start_token_id = audio_start_token_id
        self.initializer_range = initializer_range

        if isinstance(audio_config, dict):
            audio_config = Qwen3ASRAudioEncoderConfig(**audio_config)
        elif audio_config is None:
            audio_config = Qwen3ASRAudioEncoderConfig()
        self.audio_config = audio_config

        if isinstance(text_config, dict):
            text_config = Qwen3ASRTextConfig(**text_config)
        elif text_config is None:
            text_config = Qwen3ASRTextConfig()
        self.text_config = text_config
        self.audio_token_id = audio_token_id
"""
    new_thinker_init = """    def __init__(
        self,
        audio_config=None,
        text_config=None,
        audio_token_id=151646,
        audio_start_token_id=151647,
        user_token_id=872,
        initializer_range=0.02,
        **kwargs,
    ):
        if isinstance(audio_config, dict):
            audio_config = Qwen3ASRAudioEncoderConfig(**audio_config)
        elif audio_config is None:
            audio_config = Qwen3ASRAudioEncoderConfig()

        if isinstance(text_config, dict):
            text_config = Qwen3ASRTextConfig(**text_config)
        elif text_config is None:
            text_config = Qwen3ASRTextConfig()

        # Transformers 5.x validates token ids during PretrainedConfig init, so
        # nested text config and mirrored token metadata need to exist first.
        self.user_token_id = user_token_id
        self.audio_start_token_id = audio_start_token_id
        self.initializer_range = initializer_range
        self.audio_config = audio_config
        self.text_config = text_config
        self.audio_token_id = audio_token_id
        self.pad_token_id = getattr(text_config, "pad_token_id", None)
        self.bos_token_id = getattr(text_config, "bos_token_id", None)
        self.eos_token_id = getattr(text_config, "eos_token_id", None)
        self.decoder_start_token_id = getattr(text_config, "decoder_start_token_id", None)
        self.vocab_size = getattr(text_config, "vocab_size", None)

        super().__init__(**kwargs)
        self.user_token_id = user_token_id
        self.audio_start_token_id = audio_start_token_id
        self.initializer_range = initializer_range
        self.audio_config = audio_config
        self.text_config = text_config
        self.audio_token_id = audio_token_id
        self.pad_token_id = getattr(text_config, "pad_token_id", None)
        self.bos_token_id = getattr(text_config, "bos_token_id", None)
        self.eos_token_id = getattr(text_config, "eos_token_id", None)
        self.decoder_start_token_id = getattr(text_config, "decoder_start_token_id", None)
        self.vocab_size = getattr(text_config, "vocab_size", None)
"""
    if old_thinker_init in updated and "nested text config and mirrored token metadata need to exist first" not in updated:
        updated = updated.replace(old_thinker_init, new_thinker_init)
        changes.append("initialized thinker text config before PretrainedConfig validation")

    thinker_get_text = """    def get_text_config(self, decoder=False) -> "PretrainedConfig":
        text_config = getattr(self, "text_config", None)
        if text_config is not None:
            return text_config
        return self


class Qwen3ASRConfig(PretrainedConfig):
"""
    if "def get_text_config(self, decoder=False) -> \"PretrainedConfig\":" not in updated.split("class Qwen3ASRConfig(PretrainedConfig):")[0]:
        updated = updated.replace("\n\nclass Qwen3ASRConfig(PretrainedConfig):\n", "\n\n" + thinker_get_text)
        changes.append("added thinker get_text_config fallback for transformers 5.x")

    old_line = "        return self.thinker_config.get_text_config()\n"
    new_block = """        thinker_config = getattr(self, "thinker_config", None)
        if thinker_config is not None:
            return thinker_config.get_text_config()
        text_config = getattr(self, "text_config", None)
        if text_config is not None:
            return text_config
        return self
"""
    if old_line in updated and "text_config = getattr(self, \"text_config\", None)" not in updated:
        updated = updated.replace(old_line, new_block)
        changes.append("made get_text_config resilient before thinker_config is attached")

    return updated, changes


def patch_file(target: Path, patcher) -> list[str]:
    original = target.read_text(encoding="utf-8")
    updated, changes = patcher(original)

    if updated != original:
        target.write_text(updated, encoding="utf-8")
        print(f"patched {target}")
        for change in changes:
            print(f"- {change}")
    else:
        print(f"already compatible: {target}")

    return changes


def main() -> int:
    target_dir = find_target_dir()
    modeling_target = target_dir / "modeling_qwen3_asr.py"
    configuration_target = target_dir / "configuration_qwen3_asr.py"

    if not modeling_target.is_file():
        raise RuntimeError(f"Could not find qwen_asr modeling file at {modeling_target}")
    if not configuration_target.is_file():
        raise RuntimeError(f"Could not find qwen_asr config file at {configuration_target}")

    patch_file(modeling_target, patch_modeling_text)
    patch_file(configuration_target, patch_configuration_text)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - setup helper
        print(f"patch_qwen_asr_for_transformers5.py: {exc}", file=sys.stderr)
        raise SystemExit(1)
