"""Pin the maintained permissive-default contract for clean AlignAtt runs.

AGENTS.md requires that a default run is a clean token soft-frontier AlignAtt
point: the only active in-loop stop is the AlignAtt source frontier itself.
Every auxiliary guard (source-mass floors, provenance caps, source regression,
token-argmax frontier gate, min source units, source LCP, lookback/holdback,
ASR word-count commit delay) must stay disabled unless explicitly requested,
because any of them can silently turn AlignAtt local-agreement-like and break
the same-chunk no-context latency comparison. This module asserts that contract
on every surface that can define defaults: the runtime config dataclass, the
maintained presets, and both canonical runners.
"""

from __future__ import annotations

import dataclasses
import sys

from alignatt4llm.presets import RUNTIME_PRESETS, RuntimePreset
from alignatt4llm.runtime import CascadeRuntimeConfig
from alignatt4llm.cli import batch as batch
from alignatt4llm.cli import compare as compare


PERMISSIVE_DEFAULT_CONTRACT = {
    # The default policy is the clean token soft-frontier AlignAtt point.
    "translation_alignatt_acceptance_variant": "token",
    "translation_alignatt_source_frontier_action": "stop",
    "translation_alignatt_frontier_min_inaccessible_mass": 0.03,
    "translation_alignatt_argmax_mass_threshold": 0.0,
    # Source-mass floors are explicit experiment axes, not defaults.
    "translation_alignatt_min_source_mass": 0.0,
    "translation_alignatt_min_accepted_accessible_source_mass": 0.0,
    # Provenance caps are guarded diagnostics and must stay disabled.
    "translation_alignatt_max_inaccessible_source_mass": 1.0,
    "translation_alignatt_max_non_source_prompt_mass": 1.0,
    "translation_alignatt_min_accessible_inaccessible_margin": -1.0,
    "translation_alignatt_source_bearing_hard_inaccessible_cap": 1.0,
    # Auxiliary guards that can act like local agreement stay off.
    "translation_alignatt_min_accessible_source_units": 0,
    "translation_alignatt_max_source_regression": -1,
    "translation_alignatt_source_regression_action": "stop",
    "translation_alignatt_token_argmax_frontier_gate": False,
    "translation_alignatt_source_lcp_stability": False,
    "translation_alignatt_source_lookback_holdback": False,
    "translation_alignatt_defer_low_source_terminal_punctuation": False,
    "translation_alignatt_hold_back_target_units": 0,
    "translation_alignatt_min_emit_target_units": 0,
    # The unit_conf alignment-confidence gate stays disabled.
    "translation_alignatt_min_alignment_confidence": 0.0,
    # The qwen_forced punctuation-LCP commit path adds no word-count delay.
    "asr_punctuation_min_commit_words": 0,
    # ASR self-conditioning on the committed transcript stays off by default.
    "asr_context_committed_words": 0,
}

# Subset of the contract that RuntimePreset carries as fields.
PRESET_CONTRACT_KEYS = (
    "translation_alignatt_min_source_mass",
    "translation_alignatt_argmax_mass_threshold",
    "translation_alignatt_frontier_min_inaccessible_mass",
    "translation_alignatt_max_inaccessible_source_mass",
    "translation_alignatt_min_accessible_inaccessible_margin",
    "translation_alignatt_source_bearing_hard_inaccessible_cap",
    "asr_punctuation_min_commit_words",
)


def _assert_contract(obj, *, keys=None, surface: str) -> None:
    for key in keys if keys is not None else PERMISSIVE_DEFAULT_CONTRACT:
        expected = PERMISSIVE_DEFAULT_CONTRACT[key]
        actual = getattr(obj, key)
        assert actual == expected, (
            f"{surface}: {key} defaults to {actual!r}, expected {expected!r}; "
            "maintained defaults must stay clean-AlignAtt permissive "
            "(see AGENTS.md)."
        )


def test_runtime_config_defaults_are_clean_alignatt():
    _assert_contract(CascadeRuntimeConfig(), surface="CascadeRuntimeConfig")


def test_preset_fields_cover_and_match_the_contract():
    preset_field_names = {field.name for field in dataclasses.fields(RuntimePreset)}
    missing = [key for key in PRESET_CONTRACT_KEYS if key not in preset_field_names]
    assert not missing, f"RuntimePreset lost contract fields: {missing}"
    for name, preset in RUNTIME_PRESETS.items():
        _assert_contract(preset, keys=PRESET_CONTRACT_KEYS, surface=f"preset {name}")


def test_compare_runner_parses_clean_alignatt_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["alignatt-compare"])
    args = compare.parse_args()
    _assert_contract(args, surface="alignatt-compare args")

    config = compare.build_processor_config(args, backend_name="qwen_forced")
    _assert_contract(config, surface="alignatt-compare processor config")


def test_batch_runner_parses_clean_alignatt_defaults(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "alignatt-batch",
            "--inputs",
            "data/smoke/alignatt_smoke18.wav",
            "--output-dir",
            "outputs/tmp",
        ],
    )
    args = batch.parse_args()
    _assert_contract(args, surface="alignatt-batch args")
