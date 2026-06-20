"""Pin the capture-safe MT vLLM engine defaults.

docs/status.md (2026-06-09) records that CUDA graph replay — full AND
piecewise — NaN-corrupts the MT attention observer's captured q/k payload on
the vLLM 0.22.1rc cu129 stack (58% of all chunks, garbage argmax positions),
while enforce-eager eliminates the corruption completely. The artifact index
quarantines any run with more than 1% nonfinite provenance stops
(provenance_nonfinite_capture_corruption), so a run launched with the
corrupting configuration is unusable for claims. This module pins the safe
default (mt_vllm_enforce_eager=True) on every surface that can define it: the
runtime config dataclass, the maintained presets, the vLLM MT backends'
fallback, and both canonical runners — whose flags must defer to the runtime
default when absent while keeping --no-mt-vllm-enforce-eager as an explicit
debugging opt-out.
"""

from __future__ import annotations

import dataclasses
import sys
from types import SimpleNamespace

from cascade.mt.gemma_vllm_backend import GemmaVLLMMTBackend, MiLMMTVLLMMTBackend
from cascade.presets import RUNTIME_PRESETS, RuntimePreset
from cascade.runtime import CascadeRuntimeConfig
from cascade.simulstream_processor import CascadeAlignAttProcessor
import run_simulstream_batch as batch
import run_simulstream_compare as compare


BATCH_ARGV = [
    "run_simulstream_batch.py",
    "--inputs",
    "data/smoke/alignatt_smoke18.wav",
    "--output-dir",
    "outputs/tmp",
]


def test_runtime_config_defaults_to_eager_mt_engine():
    assert CascadeRuntimeConfig().mt_vllm_enforce_eager is True, (
        "CascadeRuntimeConfig must default mt_vllm_enforce_eager=True: "
        "cudagraph replay corrupts MT observer capture and the artifact "
        "index quarantines such runs (docs/status.md, 2026-06-09)."
    )


def test_presets_default_to_eager_mt_engine():
    field_defaults = {
        field.name: field.default for field in dataclasses.fields(RuntimePreset)
    }
    assert field_defaults["mt_vllm_enforce_eager"] is True, (
        "RuntimePreset must default mt_vllm_enforce_eager=True "
        "(capture-safe MT engine)."
    )
    for name, preset in RUNTIME_PRESETS.items():
        assert preset.mt_vllm_enforce_eager is True, (
            f"preset {name}: mt_vllm_enforce_eager must stay True; a preset "
            "run without eager produces quarantined artifacts."
        )


def test_mt_backends_fall_back_to_eager_capture_safe_engine():
    for backend_cls in (GemmaVLLMMTBackend, MiLMMTVLLMMTBackend):
        backend = backend_cls(
            model_name="capture-contract-probe", runtime_config=SimpleNamespace()
        )
        assert backend.enforce_eager is True, (
            f"{backend_cls.__name__} must fall back to enforce_eager=True "
            "when the runtime config does not set mt_vllm_enforce_eager."
        )
        assert backend._build_compilation_config() is None, (
            f"{backend_cls.__name__}: the eager default must hand no "
            "cudagraph compilation config to vLLM."
        )


def test_batch_runner_defers_to_the_safe_runtime_default(monkeypatch):
    monkeypatch.setattr(sys, "argv", BATCH_ARGV)
    args = batch.parse_args()
    assert args.mt_vllm_enforce_eager is None, (
        "run_simulstream_batch must not override the capture-safe runtime "
        "default when --mt-vllm-enforce-eager is absent (None defers to "
        "CascadeRuntimeConfig, which defaults to True)."
    )

    monkeypatch.setattr(sys, "argv", BATCH_ARGV + ["--no-mt-vllm-enforce-eager"])
    assert batch.parse_args().mt_vllm_enforce_eager is False, (
        "--no-mt-vllm-enforce-eager must stay available as the explicit "
        "cudagraph debugging opt-out."
    )

    monkeypatch.setattr(sys, "argv", BATCH_ARGV + ["--mt-vllm-enforce-eager"])
    assert batch.parse_args().mt_vllm_enforce_eager is True


def test_compare_runner_resolves_to_eager_mt_engine(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_simulstream_compare.py"])
    args = compare.parse_args()
    assert args.mt_vllm_enforce_eager is None, (
        "run_simulstream_compare must not override the capture-safe runtime "
        "default when --mt-vllm-enforce-eager is absent."
    )

    config = compare.build_processor_config(args, backend_name="qwen_forced")
    runtime_config = CascadeAlignAttProcessor._build_runtime_config(config)
    assert runtime_config.mt_vllm_enforce_eager is True, (
        "A flagless compare run must resolve to the eager (capture-safe) "
        "MT engine end to end."
    )

    monkeypatch.setattr(
        sys, "argv", ["run_simulstream_compare.py", "--no-mt-vllm-enforce-eager"]
    )
    args = compare.parse_args()
    config = compare.build_processor_config(args, backend_name="qwen_forced")
    runtime_config = CascadeAlignAttProcessor._build_runtime_config(config)
    assert runtime_config.mt_vllm_enforce_eager is False, (
        "--no-mt-vllm-enforce-eager must reach the runtime config as an "
        "explicit override."
    )
