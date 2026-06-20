#!/usr/bin/env python3
"""Replay recovered EN->ZH source-regression stops as trim diagnostics.

This is diagnostic only. It uses recorded drafts and AlignAtt provenance to
estimate whether a source-regression hard stop behaved like a conservative wait:
if a regressive source-position suffix recovers later in the same draft, the
replay emits the recovered prefix at the original update timestamp.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cascade.artifacts import (  # noqa: E402
    HYPOTHESIS_ELAPSED_SEMANTICS_CA_COMPATIBLE,
    normalize_computation_aware_timestamps,
    utc_now_isoformat,
)
from cascade.emission import register_translation_timestamps, register_translation_words  # noqa: E402
from cascade.text_surface import (  # noqa: E402
    join_public_emission_units,
    prediction_text_from_target_surface,
    split_public_emission_units,
)
from scripts.replay_enzh_source_mass_thresholds import (  # noqa: E402
    append_only_accepts,
    load_jsonl,
    original_prefix_for_update,
    source_order,
    write_jsonl,
)
from scripts.report_enzh_source_regression_diagnostics import (  # noqa: E402
    finite_int,
    gate_aware_trim_unrecovered_accept_count,
    int_positions,
    provenance_rows,
    source_regression_trim_unrecovered_accept_count,
)


DEFAULT_ARTIFACT_DIR = (
    REPO_ROOT
    / "outputs"
    / "diagnostics_jarvislab_20260606"
    / "enzh_milmmt_chunk640_maxreg1_recent1_full21_20260606"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "outputs"
    / "diagnostics_jarvislab_20260606"
    / "offline_replay_srtrimunrecovered_full21_20260607"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-lang-code", default="zh")
    parser.add_argument(
        "--gate-aware",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Replay other available AlignAtt gates after bypassing the "
            "source-regression hard stop. Enabled by default."
        ),
    )
    return parser.parse_args()


def source_regression_trim_accept_count(
    metadata: dict[str, Any],
    runtime_config: dict[str, Any],
    *,
    gate_aware: bool,
) -> tuple[int, bool]:
    positions = int_positions(metadata.get("aligned_source_local_positions"))
    accepted_count = finite_int(
        metadata.get("accepted_candidate_token_count"),
        default=finite_int(metadata.get("accepted_token_count"), default=0),
    )
    max_regression = finite_int(
        runtime_config.get("translation_alignatt_max_source_regression"),
        default=-1,
    )
    if max_regression < 0 or not positions:
        return accepted_count, False
    recent_tokens = finite_int(
        runtime_config.get("translation_alignatt_source_regression_recent_tokens"),
        default=0,
    )
    reference_mode = str(
        runtime_config.get("translation_alignatt_source_regression_reference_mode")
        or "max"
    )
    patience_tokens = finite_int(
        runtime_config.get("translation_alignatt_source_regression_patience_tokens"),
        default=1,
    )
    if gate_aware:
        return gate_aware_trim_unrecovered_accept_count(
            positions,
            provenance=provenance_rows(metadata.get("provenance_per_draft_token")),
            metadata=metadata,
            runtime_config=runtime_config,
            accepted_count=accepted_count,
            max_regression=max_regression,
            recent_tokens=recent_tokens,
            reference_mode=reference_mode,
            patience_tokens=patience_tokens,
        )
    return (
        source_regression_trim_unrecovered_accept_count(
            positions,
            accepted_count=accepted_count,
            max_regression=max_regression,
            recent_tokens=recent_tokens,
            reference_mode=reference_mode,
            patience_tokens=patience_tokens,
        ),
        False,
    )


def source_regression_trim_partial_target(
    update: dict[str, Any],
    *,
    previous_partial_target: str,
    runtime_config: dict[str, Any],
    target_lang_code: str,
    gate_aware: bool,
) -> tuple[str, dict[str, Any]]:
    original_partial = str(update.get("partial_accepted_target") or "")
    metadata = update.get("alignatt_metadata") or {}
    if not isinstance(metadata, dict):
        return original_partial, {"changed": False}
    if metadata.get("stop_reason") != "alignatt:source_regression":
        return original_partial, {"changed": False}
    if bool(metadata.get("final_source_completed_full_accept")):
        return original_partial, {"changed": False}

    accepted_count = finite_int(
        metadata.get("accepted_candidate_token_count"),
        default=finite_int(metadata.get("accepted_token_count"), default=0),
    )
    token_prefix, blocked_by_other_gate = source_regression_trim_accept_count(
        metadata,
        runtime_config,
        gate_aware=gate_aware,
    )
    token_prefix = max(0, int(token_prefix))
    if token_prefix <= accepted_count:
        return original_partial, {
            "changed": False,
            "blocked_by_other_gate": blocked_by_other_gate,
            "accepted_count": accepted_count,
            "simulated_token_prefix": token_prefix,
        }

    draft = str(update.get("partial_draft_target") or "")
    if not draft:
        return original_partial, {
            "changed": False,
            "blocked_by_other_gate": blocked_by_other_gate,
            "accepted_count": accepted_count,
            "simulated_token_prefix": token_prefix,
        }
    base = previous_partial_target if draft.startswith(previous_partial_target) else ""
    continuation = draft[len(base) :]
    continuation_units = split_public_emission_units(
        continuation,
        target_lang_code=target_lang_code,
    )
    keep_units = min(token_prefix, len(continuation_units))
    simulated_partial = base + join_public_emission_units(
        continuation_units[:keep_units],
        target_lang_code=target_lang_code,
    )
    original_units = split_public_emission_units(
        original_partial,
        target_lang_code=target_lang_code,
    )
    simulated_units = split_public_emission_units(
        simulated_partial,
        target_lang_code=target_lang_code,
    )
    if (
        len(simulated_units) <= len(original_units)
        or simulated_units[: len(original_units)] != original_units
    ):
        return original_partial, {
            "changed": False,
            "blocked_by_other_gate": blocked_by_other_gate,
            "accepted_count": accepted_count,
            "simulated_token_prefix": token_prefix,
            "kept_surface_unit_count": keep_units,
            "non_monotone_surface": True,
        }
    return simulated_partial, {
        "changed": simulated_partial != original_partial,
        "blocked_by_other_gate": blocked_by_other_gate,
        "accepted_count": accepted_count,
        "simulated_token_prefix": token_prefix,
        "kept_surface_unit_count": keep_units,
    }


def replay_updates(
    *,
    stream_updates: list[dict[str, Any]],
    original_hypothesis: list[dict[str, Any]],
    runtime_config: dict[str, Any],
    target_lang_code: str,
    gate_aware: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_input: dict[str, list[dict[str, Any]]] = {}
    for update in stream_updates:
        by_input.setdefault(str(update.get("input_name") or ""), []).append(update)

    original_by_input = {
        str(row["source"][0] if isinstance(row.get("source"), list) else row.get("source")): row
        for row in original_hypothesis
    }
    hypothesis_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "kind": "source_regression_trim_unrecovered",
        "gate_aware": bool(gate_aware),
        "input_count": 0,
        "accepted_update_count": 0,
        "rejected_update_count": 0,
        "source_regression_update_count": 0,
        "changed_partial_update_count": 0,
        "expanded_partial_update_count": 0,
        "trimmed_partial_update_count": 0,
        "blocked_by_other_gate_count": 0,
        "non_monotone_surface_count": 0,
        "added_unit_count": 0,
    }

    for input_name in source_order(original_hypothesis):
        updates = by_input.get(input_name, [])
        emitted_translation = ""
        previous_raw_translation = ""
        previous_partial_target = ""
        word_delays_ms: list[float] = []
        word_elapsed_ms: list[float] = []
        accepted_updates: list[dict[str, Any]] = []
        for update in updates:
            metadata = update.get("alignatt_metadata") or {}
            if isinstance(metadata, dict) and metadata.get("stop_reason") == "alignatt:source_regression":
                summary["source_regression_update_count"] += 1
            simulated_partial, decision = source_regression_trim_partial_target(
                update,
                previous_partial_target=previous_partial_target,
                runtime_config=runtime_config,
                target_lang_code=target_lang_code,
                gate_aware=gate_aware,
            )
            original_partial = str(update.get("partial_accepted_target") or "")
            if decision.get("blocked_by_other_gate"):
                summary["blocked_by_other_gate_count"] += 1
            if decision.get("non_monotone_surface"):
                summary["non_monotone_surface_count"] += 1
            if simulated_partial != original_partial:
                summary["changed_partial_update_count"] += 1
                original_units = split_public_emission_units(
                    original_partial,
                    target_lang_code=target_lang_code,
                )
                simulated_units = split_public_emission_units(
                    simulated_partial,
                    target_lang_code=target_lang_code,
                )
                if len(simulated_units) > len(original_units):
                    summary["expanded_partial_update_count"] += 1
                    summary["added_unit_count"] += len(simulated_units) - len(original_units)
                elif len(simulated_units) < len(original_units):
                    summary["trimmed_partial_update_count"] += 1

            previous_partial_target = simulated_partial
            candidate_translation = original_prefix_for_update(update) + simulated_partial
            accepts, added_units = append_only_accepts(
                previous_translation=emitted_translation,
                candidate_translation=candidate_translation,
                target_lang_code=target_lang_code,
            )
            if not accepts:
                summary["rejected_update_count"] += 1
                continue
            register_translation_timestamps(
                previous_raw_translation,
                candidate_translation,
                float(update["wallclock_elapsed_ms"]),
                word_elapsed_ms,
                target_lang_code=target_lang_code,
            )
            new_words = register_translation_words(
                emitted_translation,
                candidate_translation,
                float(update["audio_processed_ms"]),
                word_delays_ms,
                target_lang_code=target_lang_code,
            )
            emitted_translation = candidate_translation
            previous_raw_translation = candidate_translation
            accepted = dict(update)
            accepted["translation_text"] = candidate_translation
            accepted["new_words"] = new_words
            accepted["offline_replay_variant"] = "source_regression_trim_unrecovered"
            accepted["offline_replay_added_units"] = added_units
            accepted_updates.append(accepted)
            summary["accepted_update_count"] += 1

        original = original_by_input[input_name]
        normalized_elapsed_ms = normalize_computation_aware_timestamps(
            word_delays_ms,
            word_elapsed_ms,
        )
        hypothesis_rows.append(
            {
                "source": [input_name],
                "source_length": original["source_length"],
                "prediction": prediction_text_from_target_surface(
                    emitted_translation,
                    target_lang_code=target_lang_code,
                ),
                "delays": word_delays_ms,
                "elapsed": normalized_elapsed_ms,
                "elapsed_wallclock_ms": word_elapsed_ms,
                "elapsed_semantics": HYPOTHESIS_ELAPSED_SEMANTICS_CA_COMPATIBLE,
                "offline_replay_update_count": len(accepted_updates),
            }
        )
        summary["input_count"] += 1
    return hypothesis_rows, summary


def write_replay_artifacts(
    *,
    output_dir: Path,
    artifact_dir: Path,
    target_lang_code: str,
    gate_aware: bool,
    hypothesis_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "hypothesis.jsonl", hypothesis_rows)
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    runtime_config = dict(manifest.get("runtime_config") or {})
    runtime_config["translation_alignatt_source_regression_action"] = "trim_unrecovered"
    runtime_config["offline_replay_from"] = str(artifact_dir)
    manifest.update(
        {
            "generated_at_utc": utc_now_isoformat(),
            "num_inputs": len(hypothesis_rows),
            "target_language_code": target_lang_code,
            "runtime_config": runtime_config,
            "offline_replay": {
                "kind": "source_regression_trim_unrecovered",
                "gate_aware": bool(gate_aware),
                "source_artifact_dir": str(artifact_dir),
                "diagnostic_only": True,
            },
        }
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "offline_replay_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    stream_updates = load_jsonl(artifact_dir / "stream_updates.jsonl")
    original_hypothesis = load_jsonl(artifact_dir / "hypothesis.jsonl")
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    runtime_config = dict(manifest.get("runtime_config") or {})
    hypothesis_rows, summary = replay_updates(
        stream_updates=stream_updates,
        original_hypothesis=original_hypothesis,
        runtime_config=runtime_config,
        target_lang_code=args.target_lang_code,
        gate_aware=bool(args.gate_aware),
    )
    write_replay_artifacts(
        output_dir=output_dir,
        artifact_dir=artifact_dir,
        target_lang_code=args.target_lang_code,
        gate_aware=bool(args.gate_aware),
        hypothesis_rows=hypothesis_rows,
        summary=summary,
    )
    print(output_dir)
    print(output_dir / "offline_replay_summary.json")


if __name__ == "__main__":
    main()
