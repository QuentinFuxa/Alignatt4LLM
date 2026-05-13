#!/usr/bin/env python3
"""Summarize the MT AlignAtt cutoff microscope into paper artifacts."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cascade.mt.base import AlignAttDecoderPolicy


DEFAULT_OUTPUT_ROOT = Path("outputs/mt_alignatt_cutoff_microscope")
DEFAULT_PAPER_GENERATED = Path("paper/generated")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Microscope run directory. Defaults to the newest directory under outputs/mt_alignatt_cutoff_microscope.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--paper-generated-dir", type=Path, default=DEFAULT_PAPER_GENERATED)
    parser.add_argument("--tex-name", default="mt_alignatt_cutoff_replay.tex")
    parser.add_argument("--json-name", default="mt_alignatt_cutoff_replay_stats.json")
    return parser.parse_args()


def latest_run_dir(output_root: Path) -> Path:
    candidates = sorted(path for path in output_root.iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No microscope run directory found under {output_root}")
    return candidates[-1]


def load_events(run_dir: Path) -> list[dict[str, Any]]:
    events_path = run_dir / "events.jsonl"
    if not events_path.exists():
        raise FileNotFoundError(events_path)
    return [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def pct(value: float, total: float) -> float:
    if total <= 0:
        return 0.0
    return 100.0 * value / total


def fmt_pct(value: float) -> str:
    return f"{value:.1f}\\%"


def median(values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    return float(statistics.median(values))


def token_stability_unit_start_indices(token_rows: list[dict[str, Any]]) -> list[int]:
    starts: list[int] = []
    for idx, row in enumerate(token_rows):
        token = str(row.get("token") or row.get("decoded_piece") or "")
        if AlignAttDecoderPolicy.token_starts_stability_unit(
            token,
            is_first_token=(idx == 0),
        ):
            starts.append(idx)
    return starts


def accepted_count_after_static_cutoff(
    token_rows: list[dict[str, Any]],
    *,
    cutoff_units: int,
) -> int:
    unit_starts = token_stability_unit_start_indices(token_rows)
    if not token_rows or not unit_starts:
        return 0
    if int(cutoff_units) <= 0:
        return len(token_rows)

    keep_units = max(0, len(unit_starts) - int(cutoff_units))
    if keep_units <= 0:
        return 0
    if keep_units < len(unit_starts):
        return int(unit_starts[keep_units])
    return len(token_rows)


def static_cutoff_event_stats(
    event: dict[str, Any],
    *,
    cutoff_units: int,
) -> dict[str, int | None]:
    token_rows = event.get("alignatt_token_diagnostics") or []
    accepted_count = accepted_count_after_static_cutoff(
        token_rows,
        cutoff_units=int(cutoff_units),
    )
    alignatt_count = int(event.get("alignatt_accepted_generated_token_count", 0))
    unsafe_overemit = sum(
        1
        for row in token_rows[:accepted_count]
        if row.get("crosses_policy_frontier") is True
    )
    safe_underemit = max(0, alignatt_count - accepted_count)
    commit_error = unsafe_overemit + safe_underemit
    return {
        "accepted_token_count": accepted_count,
        "unsafe_overemit": unsafe_overemit,
        "safe_underemit": safe_underemit,
        "commit_error": commit_error,
        "exact_match": int(commit_error == 0 and accepted_count == alignatt_count),
    }


def required_exact_cutoff_units(event: dict[str, Any]) -> int | None:
    token_rows = event.get("alignatt_token_diagnostics") or []
    alignatt_count = int(event.get("alignatt_accepted_generated_token_count", 0))
    unit_count = len(token_stability_unit_start_indices(token_rows))
    for cutoff_units in range(0, unit_count + 1):
        if (
            accepted_count_after_static_cutoff(
                token_rows,
                cutoff_units=cutoff_units,
            )
            == alignatt_count
        ):
            return cutoff_units
    return None


def summarize_policies(events: list[dict[str, Any]]) -> dict[str, Any]:
    if not events:
        raise ValueError("No microscope events to summarize")
    cutoff_units = list(range(0, 8))
    draft_total = sum(int(event["draft_generated_token_count"]) for event in events)
    alignatt_total = sum(int(event["alignatt_accepted_generated_token_count"]) for event in events)

    policies: dict[str, dict[str, Any]] = {
        "AlignAtt": {
            "accepted": alignatt_total,
            "accepted_pct_of_draft": pct(alignatt_total, draft_total),
            "unsafe_overemit": 0,
            "safe_underemit": 0,
            "commit_error": 0,
            "unsafe_events": 0,
            "exact_events": None,
        }
    }
    for cutoff in cutoff_units:
        accepted = 0
        unsafe_overemit = 0
        safe_underemit = 0
        commit_error = 0
        unsafe_events = 0
        exact_events = 0
        for event in events:
            event_stats = static_cutoff_event_stats(event, cutoff_units=cutoff)
            event_accepted = int(event_stats["accepted_token_count"] or 0)
            event_unsafe = int(event_stats["unsafe_overemit"] or 0)
            event_underemit = int(event_stats["safe_underemit"] or 0)
            event_error = int(event_stats["commit_error"] or 0)
            accepted += event_accepted
            unsafe_overemit += event_unsafe
            safe_underemit += event_underemit
            commit_error += event_error
            unsafe_events += int(event_unsafe > 0)
            exact_events += int(event_stats["exact_match"] or 0)
        policies[f"cut-last-{cutoff}"] = {
            "accepted": accepted,
            "accepted_pct_of_draft": pct(accepted, draft_total),
            "unsafe_overemit": unsafe_overemit,
            "safe_underemit": safe_underemit,
            "commit_error": commit_error,
            "unsafe_events": unsafe_events,
            "exact_events": exact_events,
        }

    fixed_rows = [
        (name, row)
        for name, row in policies.items()
        if name.startswith("cut-last-")
    ]
    best_commit_error = min(int(row["commit_error"]) for _, row in fixed_rows)
    best_static_cutoffs = [
        int(name.rsplit("-", 1)[-1])
        for name, row in fixed_rows
        if int(row["commit_error"]) == best_commit_error
    ]
    zero_unsafe_cutoffs = [
        int(name.rsplit("-", 1)[-1])
        for name, row in fixed_rows
        if int(row["unsafe_overemit"]) == 0
    ]
    required_exact = [required_exact_cutoff_units(event) for event in events]
    required_exact_present = [
        int(value) for value in required_exact if value is not None
    ]
    source_regressions = summarize_accepted_source_regressions(events)

    return {
        "event_count": len(events),
        "draft_token_count": draft_total,
        "alignatt_accepted_token_count": alignatt_total,
        "cutoff_units": cutoff_units,
        "policies": policies,
        "best_static_cutoff_units": best_static_cutoffs,
        "best_static_commit_error": best_commit_error,
        "first_zero_unsafe_cutoff_units": min(zero_unsafe_cutoffs) if zero_unsafe_cutoffs else None,
        "required_exact_cutoff_units_by_event": required_exact,
        "required_exact_cutoff_min": min(required_exact_present) if required_exact_present else None,
        "required_exact_cutoff_max": max(required_exact_present) if required_exact_present else None,
        "accepted_source_regression_count": source_regressions["regression_count"],
        "accepted_source_regression_events": source_regressions["events_with_regression"],
    }


def summarize_accepted_source_regressions(events: list[dict[str, Any]]) -> dict[str, int]:
    regression_count = 0
    events_with_regression = 0
    for event in events:
        accepted_rows = [
            row
            for row in (event.get("alignatt_token_diagnostics") or [])
            if row.get("accepted_after_stability_trim")
        ]
        unit_indices = [
            row.get("aligned_source_unit_index")
            for row in accepted_rows
            if row.get("aligned_source_unit_index") is not None
        ]
        event_regressions = sum(
            1
            for prev, curr in zip(unit_indices, unit_indices[1:])
            if int(curr) < int(prev)
        )
        regression_count += event_regressions
        events_with_regression += int(event_regressions > 0)
    return {
        "regression_count": regression_count,
        "events_with_regression": events_with_regression,
    }


def summarize_attention(events: list[dict[str, Any]]) -> dict[str, Any]:
    accepted_rows: list[dict[str, Any]] = []
    first_unsafe_rows: list[dict[str, Any]] = []
    cut_last_1_unsafe_rows: list[dict[str, Any]] = []
    first_unsafe_indices: list[int] = []
    draft_counts: list[int] = []
    alignatt_counts: list[int] = []

    for event in events:
        token_rows = event.get("alignatt_token_diagnostics") or []
        accepted_rows.extend(row for row in token_rows if row.get("accepted_after_stability_trim"))
        unsafe_index = event.get("unsafe_target_token_index")
        if unsafe_index is not None and 0 <= int(unsafe_index) < len(token_rows):
            first_unsafe_indices.append(int(unsafe_index))
            first_unsafe_rows.append(token_rows[int(unsafe_index)])
        draft_counts.append(int(event.get("draft_generated_token_count", 0)))
        alignatt_counts.append(int(event.get("alignatt_accepted_generated_token_count", 0)))
        replay = (event.get("cutoff_replays") or {}).get("1") or {}
        for unsafe_idx in replay.get("unsafe_overemitted_token_indices", []):
            if 0 <= int(unsafe_idx) < len(token_rows):
                cut_last_1_unsafe_rows.append(token_rows[int(unsafe_idx)])

    def row_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(rows),
            "median_policy_frontier_distance": median(
                float(row["policy_frontier_distance"])
                for row in rows
                if row.get("policy_frontier_distance") is not None
            ),
            "median_argmax_mass": median(
                float(row["argmax_mass"])
                for row in rows
                if row.get("argmax_mass") is not None
            ),
            "median_accessible_source_mass": median(
                float((row.get("provenance") or {}).get("source_accessible"))
                for row in rows
                if (row.get("provenance") or {}).get("source_accessible") is not None
            ),
            "median_inaccessible_source_mass": median(
                float((row.get("provenance") or {}).get("source_inaccessible"))
                for row in rows
                if (row.get("provenance") or {}).get("source_inaccessible") is not None
            ),
            "policy_frontier_distance_counts": dict(
                sorted(
                    Counter(
                        int(row["policy_frontier_distance"])
                        for row in rows
                        if row.get("policy_frontier_distance") is not None
                    ).items()
                )
            ),
        }

    what_if_totals: dict[str, dict[str, int]] = {}
    for event in events:
        for family, thresholds in (event.get("what_if_counts") or {}).items():
            family_totals = what_if_totals.setdefault(family, {})
            for threshold, count in thresholds.items():
                family_totals[threshold] = family_totals.get(threshold, 0) + int(count)

    return {
        "draft_token_counts_by_event": draft_counts,
        "alignatt_accepted_counts_by_event": alignatt_counts,
        "first_unsafe_indices_by_event": first_unsafe_indices,
        "alignatt_retention_mean_pct": pct(
            sum(alignatt_counts) / max(1, len(alignatt_counts)),
            sum(draft_counts) / max(1, len(draft_counts)),
        ),
        "accepted": row_stats(accepted_rows),
        "first_unsafe": row_stats(first_unsafe_rows),
        "cut_last_1_unsafe": row_stats(cut_last_1_unsafe_rows),
        "what_if_accepted_token_totals": what_if_totals,
    }


def render_table(stats: dict[str, Any]) -> str:
    event_count = int(stats["event_count"])
    draft_total = int(stats["draft_token_count"])
    best_error = int(stats["best_static_commit_error"])
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{l r r r r r r}",
        "\\toprule",
        "Policy & Accepted & Unsafe & Under & Error & Unsafe calls & Exact calls \\\\",
        "\\midrule",
    ]
    for name, row in stats["policies"].items():
        accepted = int(row["accepted"])
        accepted_cell = f"{accepted}/{draft_total} ({fmt_pct(row['accepted_pct_of_draft'])})"
        unsafe = int(row["unsafe_overemit"])
        underemit = int(row["safe_underemit"])
        commit_error = int(row["commit_error"])
        error_cell = str(commit_error)
        if name != "AlignAtt" and commit_error == best_error:
            error_cell = f"\\textbf{{{commit_error}}}"
        unsafe_calls = f"{int(row['unsafe_events'])}/{event_count}"
        if row["exact_events"] is None:
            exact_cell = "--"
        else:
            exact_cell = f"{int(row['exact_events'])}/{event_count}"
        display_name = "\\textsc{AlignAtt}" if name == "AlignAtt" else name
        lines.append(
            f"{display_name} & {accepted_cell} & {unsafe} & {underemit} & "
            f"{error_cell} & {unsafe_calls} & {exact_cell} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "\\caption{\\textbf{Flexibility gap between AlignAtt and fixed target cutoffs.} All policies replay the same six ASR prefixes and Gemma drafts from the first 9.35 seconds of \\texttt{OiqEWDVtWk.wav}. ``Unsafe'' counts accepted tokens whose reconstructed attention crosses the source frontier; ``Under'' counts tokens accepted by \\textsc{AlignAtt} but withheld by the fixed cutoff. Error is their sum.}",
            "\\label{tab:mt-cutoff-replay}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir or latest_run_dir(args.output_root)
    events = load_events(run_dir)
    policy_stats = summarize_policies(events)
    stats = {
        "run_dir": str(run_dir),
        **policy_stats,
        "attention": summarize_attention(events),
    }

    args.paper_generated_dir.mkdir(parents=True, exist_ok=True)
    tex_path = args.paper_generated_dir / args.tex_name
    json_path = args.paper_generated_dir / args.json_name
    tex_path.write_text(render_table(policy_stats), encoding="utf-8")
    json_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"tex": str(tex_path), "json": str(json_path), **stats}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
