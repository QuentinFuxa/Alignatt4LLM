#!/usr/bin/env python3
"""Replay recorded EN->ZH stream updates under agreement-based commit policies.

Offline diagnostic, not a substitute for a real GPU run. From the recorded
per-chunk drafts (``partial_draft_target``) it simulates target-unit-level
commit policies that do not exist in the runtime:

- ``control``: re-commits the recorded ``translation_text`` per update; must
  reproduce the original ``hypothesis.jsonl`` exactly (validity gate).
- ``la``: pure local agreement — commit the longest common unit prefix of the
  last n cumulative draft surfaces.
- ``hybrid_recorded``: the longer of the recorded acceptance and the agreement
  prefix (can the agreement rescue latency that unit_conf deferred?).
- ``conf_sim``: unit_conf simulated from the recorded per-token consensus
  ratios, trimming the recorded acceptance (theta sweep without GPU).
- ``hybrid_conf``: the longer of ``conf_sim`` and the agreement prefix.

Structural caveats (documented per artifact): drafts were generated with the
ORIGINAL policy's accepted prefix as prefill, so simulated trajectories are
latency bounds, not counterfactual reruns. A free-running agreement commit can
leave the recorded trajectory entirely (later drafts contradict it and the
append-only gate then rejects every subsequent candidate), so agreement
candidates are CLIPPED to the final recorded surface: only text the original
run eventually emitted may be committed early, and the clipped-away units are
counted in ``agreement_offtrajectory_unit_count``. Every update's candidate is
therefore a unit prefix of the recorded final surface and the final update
flushes it, so final surfaces — and quality scores — coincide with the
recorded run by construction. The measured axis is latency (LongYAAL CU/CA).
Artifacts are quarantined from claims via the ``offline_replay`` manifest
block and directory naming.
"""

from __future__ import annotations

import argparse
from collections import deque
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
from cascade.emission import (  # noqa: E402
    register_translation_timestamps,
    register_translation_words,
)
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
    threshold_tag,
    write_jsonl,
)
from scripts.report_unit_concentration_separability import (  # noqa: E402
    accepted_unit_feature_spans,
    unit_feature_minima,
)

POLICIES = ("control", "la", "hybrid_recorded", "conf_sim", "hybrid_conf")
AGREEMENT_POLICIES = ("la", "hybrid_recorded", "hybrid_conf")
CONFIDENCE_POLICIES = ("conf_sim", "hybrid_conf")


def unit_longest_common_prefix(unit_lists: list[list[str]]) -> list[str]:
    if not unit_lists:
        return []
    prefix: list[str] = []
    for index in range(min(len(units) for units in unit_lists)):
        first = unit_lists[0][index]
        if any(units[index] != first for units in unit_lists[1:]):
            break
        prefix.append(first)
    return prefix


def _is_ascii_alnum_unit(unit: str) -> bool:
    return bool(unit) and unit.isascii() and unit.isalnum()


def snap_back_incomplete_ascii_word(
    kept_units: list[str], full_units: list[str]
) -> list[str]:
    """Back a unit cut out of the middle of an ASCII alphanumeric run."""
    kept = len(kept_units)
    if kept == 0 or kept >= len(full_units):
        return list(kept_units)
    if not (
        _is_ascii_alnum_unit(full_units[kept - 1])
        and _is_ascii_alnum_unit(full_units[kept])
    ):
        return list(kept_units)
    cut = kept
    while cut > 0 and _is_ascii_alnum_unit(full_units[cut - 1]):
        cut -= 1
    return list(full_units[:cut])


def cumulative_draft_surface(update: dict[str, Any]) -> str:
    return original_prefix_for_update(update) + str(
        update.get("partial_draft_target") or ""
    )


def conf_sim_candidate(
    update: dict[str, Any],
    *,
    previous_partial_accepted: str,
    confidence_threshold: float,
    target_lang_code: str,
    stats: dict[str, Any],
) -> str:
    """Recorded surface with low-consensus accepted units deferred one update.

    The per-token confidence features cover the draft continuation beyond the
    prefill (the previous accepted target), so the trim applies only to the
    units newly accepted by this update; the token->surface-unit cut is
    interpolated because Gemma tokens are not zh characters.
    """
    recorded = str(update.get("translation_text") or "")
    metadata = update.get("alignatt_metadata") or {}
    if not isinstance(metadata, dict):
        return recorded
    accepted_count = int(metadata.get("accepted_token_count") or 0)
    if accepted_count <= 0:
        stats["conf_passthrough_update_count"] += 1
        return recorded
    spans = accepted_unit_feature_spans(metadata)
    features = metadata.get("attention_confidence_per_draft_token") or []
    confidences = unit_feature_minima(features, spans, "consensus_ratio")
    keep_tokens = accepted_count
    deferred = False
    for (start, _end), confidence in zip(spans, confidences):
        if confidence is None or confidence < confidence_threshold:
            keep_tokens = start
            deferred = True
            if confidence is None:
                stats["conf_missing_confidence_unit_count"] += 1
            break
    if not deferred:
        stats["conf_passthrough_update_count"] += 1
        return recorded

    partial_units = split_public_emission_units(
        str(update.get("partial_accepted_target") or ""),
        target_lang_code=target_lang_code,
    )
    previous_units = split_public_emission_units(
        previous_partial_accepted,
        target_lang_code=target_lang_code,
    )
    if previous_units and partial_units[: len(previous_units)] == previous_units:
        tail_unit_count = len(partial_units) - len(previous_units)
    else:
        tail_unit_count = len(partial_units)
    if tail_unit_count <= 0:
        stats["conf_passthrough_update_count"] += 1
        return recorded
    keep_unit_count = round(tail_unit_count * keep_tokens / accepted_count)
    trim_unit_count = tail_unit_count - keep_unit_count
    if trim_unit_count <= 0:
        stats["conf_passthrough_update_count"] += 1
        return recorded
    if tail_unit_count != accepted_count:
        stats["interpolated_cut_update_count"] += 1
    recorded_units = split_public_emission_units(
        recorded,
        target_lang_code=target_lang_code,
    )
    kept = recorded_units[: max(0, len(recorded_units) - trim_unit_count)]
    snapped = snap_back_incomplete_ascii_word(kept, recorded_units)
    if len(snapped) != len(kept):
        stats["ascii_word_snap_count"] += 1
    stats["conf_trim_update_count"] += 1
    stats["conf_trim_unit_count"] += len(recorded_units) - len(snapped)
    return join_public_emission_units(snapped, target_lang_code=target_lang_code)


def candidate_surface_for_policy(
    update: dict[str, Any],
    *,
    policy: str,
    draft_unit_history: deque[list[str]],
    final_recorded_units: list[str],
    previous_partial_accepted: str,
    agreement_n: int,
    confidence_threshold: float | None,
    is_final_update: bool,
    target_lang_code: str,
    stats: dict[str, Any],
) -> str:
    recorded = str(update.get("translation_text") or "")
    if policy == "control":
        return recorded

    if is_final_update:
        stats["final_flush_update_count"] += 1
        return recorded

    agreement_units: list[str] | None = None
    if policy in AGREEMENT_POLICIES:
        if len(draft_unit_history) < agreement_n:
            agreement_units = []
            stats["agreement_history_short_update_count"] += 1
        else:
            agreement_units = unit_longest_common_prefix(
                list(draft_unit_history)[-agreement_n:]
            )
            clipped = unit_longest_common_prefix(
                [agreement_units, final_recorded_units]
            )
            stats["agreement_offtrajectory_unit_count"] += len(agreement_units) - len(
                clipped
            )
            agreement_units = clipped
            snapped = snap_back_incomplete_ascii_word(
                agreement_units, final_recorded_units
            )
            if len(snapped) != len(agreement_units):
                stats["ascii_word_snap_count"] += 1
                agreement_units = snapped

    if policy == "la":
        assert agreement_units is not None
        return join_public_emission_units(
            agreement_units, target_lang_code=target_lang_code
        )

    if policy == "conf_sim":
        assert confidence_threshold is not None
        return conf_sim_candidate(
            update,
            previous_partial_accepted=previous_partial_accepted,
            confidence_threshold=confidence_threshold,
            target_lang_code=target_lang_code,
            stats=stats,
        )

    if policy == "hybrid_recorded":
        attention_surface = recorded
    else:  # hybrid_conf
        assert confidence_threshold is not None
        attention_surface = conf_sim_candidate(
            update,
            previous_partial_accepted=previous_partial_accepted,
            confidence_threshold=confidence_threshold,
            target_lang_code=target_lang_code,
            stats=stats,
        )
    assert agreement_units is not None
    attention_units = split_public_emission_units(
        attention_surface, target_lang_code=target_lang_code
    )
    if len(agreement_units) > len(attention_units):
        stats["agreement_branch_win_count"] += 1
        stats["agreement_rescued_unit_count"] += len(agreement_units) - len(
            attention_units
        )
        return join_public_emission_units(
            agreement_units, target_lang_code=target_lang_code
        )
    if len(attention_units) > len(agreement_units):
        stats["attention_branch_win_count"] += 1
        stats["attention_beyond_agreement_unit_count"] += len(attention_units) - len(
            agreement_units
        )
    else:
        stats["branch_tie_count"] += 1
    return attention_surface


def replay_updates_for_policy(
    *,
    stream_updates: list[dict[str, Any]],
    original_hypothesis: list[dict[str, Any]],
    policy: str,
    agreement_n: int,
    confidence_threshold: float | None,
    target_lang_code: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_input: dict[str, list[dict[str, Any]]] = {}
    for update in stream_updates:
        by_input.setdefault(str(update.get("input_name") or ""), []).append(update)
    original_by_input = {
        str(row["source"][0] if isinstance(row.get("source"), list) else row.get("source")): row
        for row in original_hypothesis
    }

    summary: dict[str, Any] = {
        "policy": policy,
        "agreement_n": agreement_n if policy in AGREEMENT_POLICIES else None,
        "confidence_threshold": (
            float(confidence_threshold) if policy in CONFIDENCE_POLICIES else None
        ),
        "input_count": 0,
        "accepted_update_count": 0,
        "rejected_update_count": 0,
        "changed_candidate_update_count": 0,
        "span_reset_update_count": 0,
        "agreement_history_short_update_count": 0,
        "agreement_offtrajectory_unit_count": 0,
        "final_flush_update_count": 0,
        "ascii_word_snap_count": 0,
        "attention_branch_win_count": 0,
        "agreement_branch_win_count": 0,
        "branch_tie_count": 0,
        "agreement_rescued_unit_count": 0,
        "attention_beyond_agreement_unit_count": 0,
        "conf_passthrough_update_count": 0,
        "conf_trim_update_count": 0,
        "conf_trim_unit_count": 0,
        "conf_missing_confidence_unit_count": 0,
        "interpolated_cut_update_count": 0,
        "control_prediction_mismatch_count": 0,
        "control_delays_mismatch_count": 0,
    }

    hypothesis_rows: list[dict[str, Any]] = []
    history_size = max(2, agreement_n)
    for input_name in source_order(original_hypothesis):
        updates = by_input.get(input_name, [])
        emitted_translation = ""
        previous_raw_translation = ""
        previous_partial_accepted = ""
        word_delays_ms: list[float] = []
        word_elapsed_ms: list[float] = []
        accepted_update_count = 0
        draft_unit_history: deque[list[str]] = deque(maxlen=history_size)
        final_recorded_units = (
            split_public_emission_units(
                str(updates[-1].get("translation_text") or ""),
                target_lang_code=target_lang_code,
            )
            if updates
            else []
        )
        for index, update in enumerate(updates):
            is_final_update = index == len(updates) - 1
            draft = str(update.get("partial_draft_target") or "")
            if previous_partial_accepted and not draft.startswith(
                previous_partial_accepted
            ):
                summary["span_reset_update_count"] += 1
            draft_unit_history.append(
                split_public_emission_units(
                    cumulative_draft_surface(update),
                    target_lang_code=target_lang_code,
                )
            )
            candidate_translation = candidate_surface_for_policy(
                update,
                policy=policy,
                draft_unit_history=draft_unit_history,
                final_recorded_units=final_recorded_units,
                previous_partial_accepted=previous_partial_accepted,
                agreement_n=agreement_n,
                confidence_threshold=confidence_threshold,
                is_final_update=is_final_update,
                target_lang_code=target_lang_code,
                stats=summary,
            )
            if candidate_translation != str(update.get("translation_text") or ""):
                summary["changed_candidate_update_count"] += 1
            previous_partial_accepted = str(
                update.get("partial_accepted_target") or ""
            )
            accepts, _added_units = append_only_accepts(
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
            register_translation_words(
                emitted_translation,
                candidate_translation,
                float(update["audio_processed_ms"]),
                word_delays_ms,
                target_lang_code=target_lang_code,
            )
            emitted_translation = candidate_translation
            previous_raw_translation = candidate_translation
            accepted_update_count += 1
            summary["accepted_update_count"] += 1

        original = original_by_input[input_name]
        normalized_elapsed_ms = normalize_computation_aware_timestamps(
            word_delays_ms,
            word_elapsed_ms,
        )
        prediction = prediction_text_from_target_surface(
            emitted_translation,
            target_lang_code=target_lang_code,
        )
        if policy == "control":
            if prediction != str(original.get("prediction") or ""):
                summary["control_prediction_mismatch_count"] += 1
            if [float(v) for v in original.get("delays") or []] != word_delays_ms:
                summary["control_delays_mismatch_count"] += 1
        hypothesis_rows.append(
            {
                "source": [input_name],
                "source_length": original["source_length"],
                "prediction": prediction,
                "delays": word_delays_ms,
                "elapsed": normalized_elapsed_ms,
                "elapsed_wallclock_ms": word_elapsed_ms,
                "elapsed_semantics": HYPOTHESIS_ELAPSED_SEMANTICS_CA_COMPATIBLE,
                "offline_replay_update_count": accepted_update_count,
            }
        )
        summary["input_count"] += 1
    return hypothesis_rows, summary


def policy_tag(
    policy: str, agreement_n: int, confidence_threshold: float | None
) -> str:
    if policy == "control":
        return "control"
    if policy == "la":
        return f"la{agreement_n}"
    if policy == "hybrid_recorded":
        return f"hyb_rec_la{agreement_n}"
    if policy == "conf_sim":
        return f"confsim{threshold_tag(float(confidence_threshold))}"
    return f"hyb_conf{threshold_tag(float(confidence_threshold))}_la{agreement_n}"


def write_replay_artifacts(
    *,
    output_dir: Path,
    artifact_dir: Path,
    policy: str,
    agreement_n: int,
    confidence_threshold: float | None,
    target_lang_code: str,
    hypothesis_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "hypothesis.jsonl", hypothesis_rows)
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    runtime_config = dict(manifest.get("runtime_config") or {})
    runtime_config["offline_replay_from"] = str(artifact_dir)
    manifest.update(
        {
            "generated_at_utc": utc_now_isoformat(),
            "num_inputs": len(hypothesis_rows),
            "target_language_code": target_lang_code,
            "runtime_config": runtime_config,
            "offline_replay": {
                "kind": "agreement_policy",
                "policy": policy,
                "agreement_n": (
                    agreement_n if policy in AGREEMENT_POLICIES else None
                ),
                "confidence_threshold": (
                    float(confidence_threshold)
                    if policy in CONFIDENCE_POLICIES
                    else None
                ),
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


def expand_variants(args: argparse.Namespace) -> list[tuple[str, int, float | None]]:
    policies = args.policy or ["control"]
    agreement_ns = [int(n) for n in (args.agreement_n or [2])]
    thresholds = [float(t) for t in (args.confidence_threshold or [])]
    variants: list[tuple[str, int, float | None]] = []
    for policy in policies:
        if policy == "control":
            variants.append((policy, 2, None))
        elif policy == "la" or policy == "hybrid_recorded":
            variants.extend((policy, n, None) for n in agreement_ns)
        elif policy == "conf_sim":
            if not thresholds:
                raise ValueError("conf_sim requires --confidence-threshold")
            variants.extend((policy, 2, threshold) for threshold in thresholds)
        elif policy == "hybrid_conf":
            if not thresholds:
                raise ValueError("hybrid_conf requires --confidence-threshold")
            variants.extend(
                (policy, n, threshold)
                for threshold in thresholds
                for n in agreement_ns
            )
        else:
            raise ValueError(f"unknown policy {policy!r}")
    return variants


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--policy",
        action="append",
        choices=POLICIES,
        default=[],
        help="Repeatable; defaults to control only.",
    )
    parser.add_argument(
        "--agreement-n",
        type=int,
        action="append",
        default=[],
        help="Repeatable agreement window for la/hybrid policies (default 2).",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        action="append",
        default=[],
        help="Repeatable unit_conf threshold for conf_sim/hybrid_conf.",
    )
    parser.add_argument("--target-lang-code", default="zh")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir.expanduser().resolve()
    stream_updates = load_jsonl(artifact_dir / "stream_updates.jsonl")
    original_hypothesis = load_jsonl(artifact_dir / "hypothesis.jsonl")
    summaries: list[dict[str, Any]] = []
    for policy, agreement_n, confidence_threshold in expand_variants(args):
        hypothesis_rows, summary = replay_updates_for_policy(
            stream_updates=stream_updates,
            original_hypothesis=original_hypothesis,
            policy=policy,
            agreement_n=agreement_n,
            confidence_threshold=confidence_threshold,
            target_lang_code=args.target_lang_code,
        )
        tag = policy_tag(policy, agreement_n, confidence_threshold)
        output_dir = args.output_root / tag
        write_replay_artifacts(
            output_dir=output_dir,
            artifact_dir=artifact_dir,
            policy=policy,
            agreement_n=agreement_n,
            confidence_threshold=confidence_threshold,
            target_lang_code=args.target_lang_code,
            hypothesis_rows=hypothesis_rows,
            summary=summary,
        )
        summaries.append({"output_dir": str(output_dir), **summary})
        print(output_dir)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "offline_replay_points.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output_root / "offline_replay_points.json")


if __name__ == "__main__":
    main()
