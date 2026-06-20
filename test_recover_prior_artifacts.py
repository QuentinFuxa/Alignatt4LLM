from __future__ import annotations

import json
from pathlib import Path

from scripts.recover_prior_artifacts import index_existing_artifacts, recover_artifacts


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _manifest() -> dict:
    return {
        "schema_version": "cascade_v1",
        "num_inputs": 1,
        "source_language_code": "en",
        "target_language_code": "zh",
        "runtime_config": {
            "chunk_ms": 640,
            "mt_backend_name": "milmmt_vllm_alignatt",
            "translation_acceptance_policy": "alignatt",
            "translation_static_cutoff_units": 0,
        },
        "run_provenance": {"script": "run_simulstream_batch.py"},
    }


def _manifest_with_runtime_config(**runtime_overrides) -> dict:
    manifest = _manifest()
    manifest["runtime_config"].update(runtime_overrides)
    return manifest


def _evaluation(*, longyaal_cu: float) -> dict:
    return {
        "contract_scores": {
            "BLEU": 40.0,
            "CHRF": 35.0,
            "XCOMETXL": 0.8,
            "LongYAAL CU": longyaal_cu,
            "LongYAAL CA": longyaal_cu - 100.0,
        }
    }


def test_recovery_index_marks_only_claimable_artifacts_valid(tmp_path: Path):
    source_root = tmp_path / "old_outputs"
    output_root = tmp_path / "recovered"

    _write_json(source_root / "good" / "manifest.json", _manifest())
    _write_json(source_root / "good" / "evaluation.json", _evaluation(longyaal_cu=2000.0))
    (source_root / "good" / "hypothesis.jsonl").write_text("{}\n", encoding="utf-8")

    _write_json(source_root / "negative_latency" / "manifest.json", _manifest())
    _write_json(
        source_root / "negative_latency" / "evaluation.json",
        _evaluation(longyaal_cu=-1.0),
    )

    _write_json(
        source_root / "missing_manifest" / "evaluation.json",
        _evaluation(longyaal_cu=2000.0),
    )

    _write_json(source_root / "replay_sr_case" / "manifest.json", _manifest())
    _write_json(
        source_root / "replay_sr_case" / "evaluation.json",
        _evaluation(longyaal_cu=2000.0),
    )
    _write_json(source_root / "surface_dedup_case" / "manifest.json", _manifest())
    _write_json(
        source_root / "surface_dedup_case" / "evaluation.json",
        _evaluation(longyaal_cu=2000.0),
    )
    _write_json(source_root / "shift_diagnostic_case" / "manifest.json", _manifest())
    _write_json(
        source_root / "shift_diagnostic_case" / "evaluation.json",
        _evaluation(longyaal_cu=2000.0),
    )

    rows = recover_artifacts(source_root=source_root, output_root=output_root)
    by_dir = {row["relative_dir"]: row for row in rows}

    assert by_dir["good"]["valid_for_claims"] is True
    assert by_dir["negative_latency"]["valid_for_claims"] is False
    assert "negative_longyaal_cu" in by_dir["negative_latency"]["invalid_reasons"]
    assert by_dir["missing_manifest"]["valid_for_claims"] is False
    assert "missing_manifest" in by_dir["missing_manifest"]["invalid_reasons"]
    assert by_dir["replay_sr_case"]["valid_for_claims"] is False
    assert "replay_sr_diagnostic" in by_dir["replay_sr_case"]["invalid_reasons"]
    assert by_dir["surface_dedup_case"]["valid_for_claims"] is False
    assert "surface_dedup_diagnostic" in by_dir["surface_dedup_case"]["invalid_reasons"]
    assert by_dir["shift_diagnostic_case"]["valid_for_claims"] is False
    assert "diagnostic_artifact" in by_dir["shift_diagnostic_case"]["invalid_reasons"]
    assert (output_root / "recovered_artifact_index.tsv").is_file()


def test_index_only_does_not_copy_artifact_files(tmp_path: Path):
    source_root = tmp_path / "downloaded_outputs"
    output_root = tmp_path / "downloaded_outputs"

    _write_json(source_root / "good" / "manifest.json", _manifest())
    _write_json(source_root / "good" / "evaluation.json", _evaluation(longyaal_cu=2000.0))

    rows = index_existing_artifacts(source_root=source_root, output_root=output_root)

    assert rows[0]["relative_dir"] == "good"
    assert rows[0]["copied_dir"] == str((source_root / "good").resolve())
    assert (output_root / "recovered_artifact_index.tsv").is_file()
    assert not (output_root / "good" / "recovered_artifact_index.tsv").exists()


def test_index_excludes_offline_replay_diagnostics(tmp_path: Path):
    source_root = tmp_path / "outputs"
    output_root = tmp_path / "indexed"
    _write_json(
        source_root / "offline_replay_unit_mass" / "manifest.json",
        _manifest() | {"offline_replay": {"diagnostic_only": True}},
    )
    _write_json(
        source_root / "offline_replay_unit_mass" / "evaluation.json",
        _evaluation(longyaal_cu=1200.0),
    )

    rows = index_existing_artifacts(
        source_root=source_root,
        output_root=output_root,
    )

    assert rows[0]["valid_for_claims"] is False
    assert rows[0]["invalid_reasons"].split(",") == [
        "offline_replay_diagnostic",
        "diagnostic_only_manifest",
    ]


def test_index_classifies_pure_and_guarded_alignatt_runs(tmp_path: Path):
    source_root = tmp_path / "outputs"
    output_root = tmp_path / "outputs"

    _write_json(
        source_root / "pure_soft" / "manifest.json",
        _manifest_with_runtime_config(
            translation_alignatt_acceptance_variant="token",
            translation_alignatt_min_source_mass=0.02,
            translation_alignatt_inaccessible_ms=160,
            translation_alignatt_frontier_min_inaccessible_mass=0.03,
            translation_alignatt_max_inaccessible_source_mass=1.0,
            translation_alignatt_min_accessible_inaccessible_margin=-1.0,
            translation_alignatt_min_accessible_source_units=0,
            translation_alignatt_max_source_regression=-1,
            translation_alignatt_token_argmax_frontier_gate=False,
            translation_alignatt_top_k_heads=8,
            translation_alignatt_filter_width=7,
            translation_alignatt_online_normalization="zscore",
        ),
    )
    _write_json(
        source_root / "pure_soft" / "evaluation.json",
        _evaluation(longyaal_cu=1200.0),
    )
    _write_json(
        source_root / "accepted_prefix_soft" / "manifest.json",
        _manifest_with_runtime_config(
            translation_alignatt_acceptance_variant="token",
            translation_alignatt_min_source_mass=0.0,
            translation_alignatt_min_accepted_accessible_source_mass=0.001,
            translation_alignatt_accepted_accessible_source_mass_recent_units=2,
            translation_alignatt_inaccessible_ms=0,
            translation_alignatt_frontier_min_inaccessible_mass=0.03,
            translation_alignatt_max_inaccessible_source_mass=1.0,
            translation_alignatt_min_accessible_inaccessible_margin=-1.0,
            translation_alignatt_min_accessible_source_units=0,
            translation_alignatt_max_source_regression=-1,
            translation_alignatt_token_argmax_frontier_gate=False,
            translation_alignatt_top_k_heads=8,
            translation_alignatt_filter_width=7,
            translation_alignatt_online_normalization="zscore",
        ),
    )
    _write_json(
        source_root / "accepted_prefix_soft" / "evaluation.json",
        _evaluation(longyaal_cu=1180.0),
    )
    _write_json(
        source_root / "recoverable_soft" / "manifest.json",
        _manifest_with_runtime_config(
            translation_alignatt_acceptance_variant="token",
            translation_alignatt_min_source_mass=0.0,
            translation_alignatt_source_frontier_action="trim_unrecovered",
            translation_alignatt_inaccessible_ms=0,
            translation_alignatt_frontier_min_inaccessible_mass=0.03,
            translation_alignatt_max_inaccessible_source_mass=1.0,
            translation_alignatt_min_accessible_inaccessible_margin=-1.0,
            translation_alignatt_min_accessible_source_units=0,
            translation_alignatt_max_source_regression=-1,
            translation_alignatt_token_argmax_frontier_gate=False,
        ),
    )
    _write_json(
        source_root / "recoverable_soft" / "evaluation.json",
        _evaluation(longyaal_cu=1170.0),
    )
    _write_json(
        source_root / "guarded" / "manifest.json",
        _manifest_with_runtime_config(
            translation_alignatt_frontier_min_inaccessible_mass=0.03,
            translation_alignatt_max_inaccessible_source_mass=0.15,
            translation_alignatt_min_accessible_inaccessible_margin=0.0,
            translation_alignatt_min_accessible_source_units=6,
            translation_alignatt_source_lcp_stability=True,
            translation_alignatt_source_lcp_append_slack_units=2,
            translation_alignatt_max_source_regression=1,
            translation_alignatt_token_argmax_frontier_gate=True,
        ),
    )
    _write_json(
        source_root / "guarded" / "evaluation.json",
        _evaluation(longyaal_cu=2000.0),
    )
    _write_json(
        source_root / "non_source_guarded" / "manifest.json",
        _manifest_with_runtime_config(
            translation_alignatt_acceptance_variant="token",
            translation_alignatt_min_source_mass=0.0,
            translation_alignatt_max_non_source_prompt_mass=0.80,
            translation_alignatt_max_source_regression=-1,
            translation_alignatt_token_argmax_frontier_gate=False,
        ),
    )
    _write_json(
        source_root / "non_source_guarded" / "evaluation.json",
        _evaluation(longyaal_cu=1500.0),
    )
    _write_json(
        source_root / "unit_mass" / "manifest.json",
        _manifest_with_runtime_config(
            translation_alignatt_acceptance_variant="unit_mass",
            translation_alignatt_min_source_mass=0.001,
            translation_alignatt_frontier_min_inaccessible_mass=0.0,
            translation_alignatt_max_inaccessible_source_mass=1.0,
            translation_alignatt_min_accessible_inaccessible_margin=-1.0,
            translation_alignatt_min_accessible_source_units=0,
            translation_alignatt_max_source_regression=-1,
            translation_alignatt_token_argmax_frontier_gate=False,
        ),
    )
    _write_json(
        source_root / "unit_mass" / "evaluation.json",
        _evaluation(longyaal_cu=1400.0),
    )
    _write_json(
        source_root / "unit_mass_zero_floor" / "manifest.json",
        _manifest_with_runtime_config(
            translation_alignatt_acceptance_variant="unit_mass",
            translation_alignatt_min_source_mass=0.0,
            translation_alignatt_max_source_regression=-1,
            translation_alignatt_token_argmax_frontier_gate=False,
        ),
    )
    _write_json(
        source_root / "unit_mass_zero_floor" / "evaluation.json",
        _evaluation(longyaal_cu=1000.0),
    )
    _write_json(
        source_root / "unit_argmax" / "manifest.json",
        _manifest_with_runtime_config(
            translation_alignatt_acceptance_variant="unit_argmax",
            translation_alignatt_min_source_mass=0.0,
            translation_alignatt_max_source_regression=-1,
            translation_alignatt_token_argmax_frontier_gate=False,
        ),
    )
    _write_json(
        source_root / "unit_argmax" / "evaluation.json",
        _evaluation(longyaal_cu=1100.0),
    )
    _write_json(
        source_root / "unit_consensus" / "manifest.json",
        _manifest_with_runtime_config(
            translation_alignatt_acceptance_variant="unit_consensus",
            translation_alignatt_min_source_mass=0.0,
            translation_alignatt_unit_consensus_min_head_ratio=0.45,
            translation_alignatt_max_source_regression=-1,
            translation_alignatt_token_argmax_frontier_gate=False,
        ),
    )
    _write_json(
        source_root / "unit_consensus" / "evaluation.json",
        _evaluation(longyaal_cu=1120.0),
    )
    _write_json(
        source_root / "unit_source_bearing" / "manifest.json",
        _manifest_with_runtime_config(
            translation_alignatt_acceptance_variant="unit_mass_source_bearing",
            translation_alignatt_min_source_mass=0.0,
            translation_alignatt_source_bearing_min_source_mass=0.04,
            translation_alignatt_source_bearing_hard_inaccessible_cap=1.0,
            translation_alignatt_max_source_regression=-1,
            translation_alignatt_token_argmax_frontier_gate=False,
        ),
    )
    _write_json(
        source_root / "unit_source_bearing" / "evaluation.json",
        _evaluation(longyaal_cu=1080.0),
    )
    _write_json(
        source_root / "unit_source_bearing_guarded_cap" / "manifest.json",
        _manifest_with_runtime_config(
            translation_alignatt_acceptance_variant="unit_mass_source_bearing",
            translation_alignatt_min_source_mass=0.0,
            translation_alignatt_source_bearing_min_source_mass=0.04,
            translation_alignatt_source_bearing_hard_inaccessible_cap=0.75,
            translation_alignatt_max_source_regression=-1,
            translation_alignatt_token_argmax_frontier_gate=False,
        ),
    )
    _write_json(
        source_root / "unit_source_bearing_guarded_cap" / "evaluation.json",
        _evaluation(longyaal_cu=1090.0),
    )
    _write_json(
        source_root / "unit_argmax_floor" / "manifest.json",
        _manifest_with_runtime_config(
            translation_alignatt_acceptance_variant="unit_argmax",
            translation_alignatt_min_source_mass=0.001,
            translation_alignatt_max_source_regression=-1,
            translation_alignatt_token_argmax_frontier_gate=False,
        ),
    )
    _write_json(
        source_root / "unit_argmax_floor" / "evaluation.json",
        _evaluation(longyaal_cu=1300.0),
    )
    _write_json(
        source_root / "cutoff" / "manifest.json",
        _manifest_with_runtime_config(
            translation_acceptance_policy="cut_last_target_units",
            translation_static_cutoff_units=3,
        ),
    )
    _write_json(
        source_root / "cutoff" / "evaluation.json",
        _evaluation(longyaal_cu=1800.0),
    )

    rows = index_existing_artifacts(source_root=source_root, output_root=output_root)
    by_dir = {row["relative_dir"]: row for row in rows}

    assert by_dir["pure_soft"]["alignatt_policy_family"] == (
        "clean_soft_frontier_source_mass_floor"
    )
    assert by_dir["pure_soft"]["alignatt_guard_flags"] == ""
    assert by_dir["pure_soft"]["translation_alignatt_acceptance_variant"] == "token"
    assert by_dir["pure_soft"]["translation_alignatt_min_source_mass"] == 0.02
    assert by_dir["pure_soft"]["translation_alignatt_inaccessible_ms"] == 160
    assert by_dir["pure_soft"]["translation_alignatt_frontier_min_inaccessible_mass"] == 0.03
    assert by_dir["pure_soft"]["translation_alignatt_top_k_heads"] == 8
    assert by_dir["pure_soft"]["translation_alignatt_filter_width"] == 7
    assert by_dir["pure_soft"]["translation_alignatt_online_normalization"] == "zscore"
    assert by_dir["accepted_prefix_soft"]["alignatt_policy_family"] == (
        "clean_soft_frontier_source_mass_floor"
    )
    assert by_dir["accepted_prefix_soft"]["alignatt_guard_flags"] == ""
    assert (
        by_dir["accepted_prefix_soft"][
            "translation_alignatt_min_accepted_accessible_source_mass"
        ]
        == 0.001
    )
    assert (
        by_dir["accepted_prefix_soft"][
            "translation_alignatt_accepted_accessible_source_mass_recent_units"
        ]
        == 2
    )
    assert by_dir["recoverable_soft"]["alignatt_policy_family"] == (
        "clean_recoverable_soft_frontier"
    )
    assert by_dir["recoverable_soft"]["alignatt_guard_flags"] == ""
    assert by_dir["recoverable_soft"]["translation_alignatt_source_frontier_action"] == (
        "trim_unrecovered"
    )
    assert by_dir["guarded"]["alignatt_policy_family"] == "guarded_alignatt"
    assert by_dir["guarded"]["alignatt_guard_flags"].split(",") == [
        "source_regression",
        "token_argmax_frontier",
        "min_accessible_source_units",
        "source_lcp_stability",
        "source_lcp_append_slack",
        "max_inaccessible_source_mass",
        "accessible_inaccessible_margin",
    ]
    assert by_dir["non_source_guarded"]["alignatt_policy_family"] == "guarded_alignatt"
    assert by_dir["non_source_guarded"]["alignatt_guard_flags"] == (
        "max_non_source_prompt_mass"
    )
    assert (
        by_dir["non_source_guarded"][
            "translation_alignatt_max_non_source_prompt_mass"
        ]
        == 0.80
    )
    assert by_dir["unit_mass"]["alignatt_policy_family"] == (
        "clean_unit_source_mass_floor"
    )
    assert by_dir["unit_mass"]["alignatt_guard_flags"] == ""
    assert by_dir["unit_mass"]["translation_alignatt_acceptance_variant"] == "unit_mass"
    assert by_dir["unit_mass"]["translation_alignatt_min_source_mass"] == 0.001
    assert by_dir["unit_mass_zero_floor"]["alignatt_policy_family"] == "guarded_alignatt"
    assert by_dir["unit_mass_zero_floor"]["alignatt_guard_flags"] == (
        "acceptance_variant:unit_mass_without_source_mass_floor"
    )
    assert by_dir["unit_argmax"]["alignatt_policy_family"] == (
        "clean_unit_argmax_frontier"
    )
    assert by_dir["unit_argmax"]["alignatt_guard_flags"] == ""
    assert by_dir["unit_consensus"]["alignatt_policy_family"] == (
        "clean_unit_consensus_frontier"
    )
    assert by_dir["unit_consensus"]["alignatt_guard_flags"] == ""
    assert by_dir["unit_consensus"]["translation_alignatt_unit_consensus_min_head_ratio"] == 0.45
    assert by_dir["unit_source_bearing"]["alignatt_policy_family"] == (
        "clean_unit_source_bearing"
    )
    assert by_dir["unit_source_bearing"]["alignatt_guard_flags"] == ""
    assert (
        by_dir["unit_source_bearing"][
            "translation_alignatt_source_bearing_min_source_mass"
        ]
        == 0.04
    )
    assert (
        by_dir["unit_source_bearing"][
            "translation_alignatt_source_bearing_hard_inaccessible_cap"
        ]
        == 1.0
    )
    assert by_dir["unit_source_bearing_guarded_cap"]["alignatt_policy_family"] == (
        "guarded_alignatt"
    )
    assert by_dir["unit_source_bearing_guarded_cap"]["alignatt_guard_flags"] == (
        "source_bearing_hard_inaccessible_cap:0.75"
    )
    assert by_dir["unit_argmax_floor"]["alignatt_policy_family"] == "guarded_alignatt"
    assert by_dir["unit_argmax_floor"]["alignatt_guard_flags"] == "unused_source_mass_floor"
    assert by_dir["cutoff"]["alignatt_policy_family"] == "cut_last_target_units"


def test_index_invalidates_provenance_nonfinite_capture_corruption(tmp_path: Path):
    source_root = tmp_path / "old_outputs"

    corrupt = _manifest()
    corrupt["speed"] = {
        "per_input": [
            {
                "input": "a.wav",
                "chunk_decision_summary": {
                    "stop_reason_counts": {
                        "alignatt:provenance_nonfinite": 60,
                        "stop": 40,
                    }
                },
            }
        ]
    }
    _write_json(source_root / "corrupt_capture_case" / "manifest.json", corrupt)
    _write_json(
        source_root / "corrupt_capture_case" / "evaluation.json",
        _evaluation(longyaal_cu=2000.0),
    )

    healthy = _manifest()
    healthy["speed"] = {
        "per_input": [
            {
                "input": "a.wav",
                "chunk_decision_summary": {
                    "stop_reason_counts": {"stop": 90, "alignatt:observer_empty": 10}
                },
            }
        ]
    }
    _write_json(source_root / "healthy_capture_case" / "manifest.json", healthy)
    _write_json(
        source_root / "healthy_capture_case" / "evaluation.json",
        _evaluation(longyaal_cu=900.0),
    )

    rows = index_existing_artifacts(source_root=source_root, output_root=source_root)
    by_dir = {row["relative_dir"]: row for row in rows}

    corrupt_row = by_dir["corrupt_capture_case"]
    assert corrupt_row["valid_for_claims"] == "False" or corrupt_row["valid_for_claims"] is False
    assert "provenance_nonfinite_capture_corruption" in corrupt_row["invalid_reasons"]

    healthy_row = by_dir["healthy_capture_case"]
    assert healthy_row["valid_for_claims"] is True or healthy_row["valid_for_claims"] == "True"
    assert "provenance_nonfinite" not in healthy_row["invalid_reasons"]


def test_index_classifies_unit_conf_clean_and_guarded(tmp_path: Path):
    source_root = tmp_path / "outputs"

    clean = _manifest_with_runtime_config(
        translation_alignatt_acceptance_variant="unit_conf",
        translation_alignatt_min_alignment_confidence=0.75,
        translation_alignatt_min_source_mass=0.0,
    )
    _write_json(source_root / "unit_conf_clean" / "manifest.json", clean)
    _write_json(
        source_root / "unit_conf_clean" / "evaluation.json",
        _evaluation(longyaal_cu=1500.0),
    )

    guarded = _manifest_with_runtime_config(
        translation_alignatt_acceptance_variant="unit_conf",
        translation_alignatt_min_alignment_confidence=0.75,
        translation_alignatt_min_source_mass=0.003,
    )
    _write_json(source_root / "unit_conf_floor" / "manifest.json", guarded)
    _write_json(
        source_root / "unit_conf_floor" / "evaluation.json",
        _evaluation(longyaal_cu=1600.0),
    )

    rows = index_existing_artifacts(source_root=source_root, output_root=source_root)
    by_dir = {row["relative_dir"]: row for row in rows}

    assert by_dir["unit_conf_clean"]["alignatt_policy_family"] == (
        "clean_unit_confidence_frontier"
    )
    assert by_dir["unit_conf_clean"]["translation_alignatt_min_alignment_confidence"] == 0.75
    assert by_dir["unit_conf_floor"]["alignatt_policy_family"] == "guarded_alignatt"
    assert by_dir["unit_conf_floor"]["alignatt_guard_flags"] == "unused_source_mass_floor"
