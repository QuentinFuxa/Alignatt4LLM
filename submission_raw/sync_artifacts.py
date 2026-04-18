#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ARTIFACT_FILENAMES = (
    "manifest.json",
    "hypothesis.jsonl",
    "stream_updates.jsonl",
)

RESULT_FILENAMES = (
    "evaluation.json",
    "evaluation.report.txt",
    "scores.tsv",
)


@dataclass(frozen=True)
class BundleSpec:
    """Blind-test-set submission artifact bundle (main track, frozen)."""

    track: str
    regime: str
    preset: str
    chunk_ms: int
    source_language_code: str
    target_language_code: str
    source_output_dir: str

    @property
    def submission_subdir(self) -> Path:
        return Path(self.track) / self.regime / (
            f"{self.source_language_code}-{self.target_language_code}"
        )


@dataclass(frozen=True)
class AdditiveBundleSpec:
    """Additive calibration bundle: dev-set inference + scored evaluation.

    Unlike the main-track blind submission bundles, additive bundles are
    produced on the dev-set so that each chunk size can be scored against
    references. Artifacts (inference logs) and results (evaluation scores)
    therefore share the same source directory.
    """

    group: str  # e.g. "chunk850"
    chunk_ms: int
    source_language_code: str
    target_language_code: str
    source_output_dir: str

    @property
    def artifacts_subdir(self) -> Path:
        return Path("additive") / self.group / (
            f"{self.source_language_code}-{self.target_language_code}"
        )

    @property
    def results_subdir(self) -> Path:
        return Path("additive") / self.group / (
            f"{self.source_language_code}-{self.target_language_code}"
        )


BUNDLE_SPECS = (
    BundleSpec(
        track="main",
        regime="low",
        preset="main_low_latency",
        chunk_ms=750,
        source_language_code="en",
        target_language_code="de",
        source_output_dir="outputs/iwslt26_testset_chunk750_borderp1_ende",
    ),
    BundleSpec(
        track="main",
        regime="low",
        preset="main_low_latency",
        chunk_ms=750,
        source_language_code="en",
        target_language_code="it",
        source_output_dir="outputs/iwslt26_testset_chunk750_borderp1_enit",
    ),
    BundleSpec(
        track="main",
        regime="low",
        preset="main_low_latency",
        chunk_ms=750,
        source_language_code="en",
        target_language_code="zh",
        source_output_dir="outputs/iwslt26_testset_chunk750_borderp1_enzh",
    ),
    BundleSpec(
        track="main",
        regime="high",
        preset="main_high_latency",
        chunk_ms=1100,
        source_language_code="en",
        target_language_code="de",
        source_output_dir="outputs/iwslt26_testset_chunk1100_borderp1_ende",
    ),
    BundleSpec(
        track="main",
        regime="high",
        preset="main_high_latency",
        chunk_ms=1100,
        source_language_code="en",
        target_language_code="it",
        source_output_dir="outputs/iwslt26_testset_chunk1100_borderp1_enit",
    ),
    BundleSpec(
        track="main",
        regime="high",
        preset="main_high_latency",
        chunk_ms=1100,
        source_language_code="en",
        target_language_code="zh",
        source_output_dir="outputs/iwslt26_testset_chunk1100_borderp1_enzh",
    ),
)


ADDITIVE_BUNDLE_SPECS = tuple(
    AdditiveBundleSpec(
        group=f"chunk{chunk_ms}",
        chunk_ms=chunk_ms,
        source_language_code="en",
        target_language_code=target,
        source_output_dir=(
            f"outputs/iwslt26_devset_chunk{chunk_ms}_borderp1_en{target}"
        ),
    )
    for chunk_ms in (850, 1500, 1900)
    for target in ("de", "it", "zh")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy the frozen IWSLT submission artifacts and any additive "
            "calibration bundles from outputs/ into a self-contained "
            "submission/ tree. Main-track blind test-set bundles live under "
            "submission/artifacts/main/... Additive dev-set bundles live "
            "under submission/artifacts/additive/<group>/... with their "
            "scored evaluation under submission/results/additive/<group>/..."
        )
    )
    parser.add_argument(
        "--artifact-root",
        default="submission/artifacts",
        help="Destination root for inference artifacts (main + additive).",
    )
    parser.add_argument(
        "--results-root",
        default="submission/results",
        help="Destination root for additive evaluation result files.",
    )
    parser.add_argument(
        "--index-path",
        default="submission/ARTIFACT_INDEX.json",
        help="Path to the generated bundle index JSON.",
    )
    parser.add_argument(
        "--mode",
        choices=("copy", "hardlink"),
        default="copy",
        help="How to materialize files under the submission bundle.",
    )
    parser.add_argument(
        "--skip-additive",
        action="store_true",
        help=(
            "Skip the additive calibration bundles (useful when "
            "re-materializing just the frozen main-track submission)."
        ),
    )
    parser.add_argument(
        "--skip-missing-additive",
        action="store_true",
        help=(
            "Silently skip additive bundles whose source output directory "
            "does not yet exist. Useful while the sweep is still running."
        ),
    )
    return parser.parse_args()


def count_lines(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            count += chunk.count(b"\n")
    return count


def materialize_file(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    if mode == "hardlink":
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def relative_to_repo(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _materialize_main_bundles(
    *,
    specs: Iterable[BundleSpec],
    artifact_root: Path,
    repo_root: Path,
    mode: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in specs:
        source_dir = (repo_root / spec.source_output_dir).resolve()
        destination_dir = artifact_root / spec.submission_subdir

        if not source_dir.is_dir():
            raise FileNotFoundError(f"Missing source artifact directory: {source_dir}")

        if destination_dir.exists():
            shutil.rmtree(destination_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)

        source_files = {name: source_dir / name for name in ARTIFACT_FILENAMES}
        missing_files = [str(path) for path in source_files.values() if not path.is_file()]
        if missing_files:
            raise FileNotFoundError(
                "Missing required artifact files:\n" + "\n".join(missing_files)
            )

        manifest = json.loads(source_files["manifest.json"].read_text())
        runtime_config = manifest.get("runtime_config", {})
        if manifest.get("source_language_code") != spec.source_language_code:
            raise ValueError(
                f"{source_dir} source language mismatch: "
                f"{manifest.get('source_language_code')} != {spec.source_language_code}"
            )
        if manifest.get("target_language_code") != spec.target_language_code:
            raise ValueError(
                f"{source_dir} target language mismatch: "
                f"{manifest.get('target_language_code')} != {spec.target_language_code}"
            )
        if int(runtime_config.get("chunk_ms", -1)) != spec.chunk_ms:
            raise ValueError(
                f"{source_dir} chunk mismatch: "
                f"{runtime_config.get('chunk_ms')} != {spec.chunk_ms}"
            )

        copied_files: dict[str, dict[str, object]] = {}
        for name, source_path in source_files.items():
            destination_path = destination_dir / name
            materialize_file(source_path, destination_path, mode=mode)
            copied_files[name] = {
                "path": relative_to_repo(destination_path, repo_root),
                "bytes": destination_path.stat().st_size,
            }

        hypotheses_count = count_lines(destination_dir / "hypothesis.jsonl")
        stream_updates_count = count_lines(destination_dir / "stream_updates.jsonl")

        rows.append(
            {
                **asdict(spec),
                "source_output_dir": relative_to_repo(source_dir, repo_root),
                "submission_dir": relative_to_repo(destination_dir, repo_root),
                "target_language": manifest.get("target_language"),
                "num_inputs": manifest.get("num_inputs"),
                "num_audios": manifest.get("num_audios"),
                "hypotheses": hypotheses_count,
                "stream_updates": stream_updates_count,
                "generated_at_utc": manifest.get("generated_at_utc"),
                "runtime_config": {
                    "alignment_backend_name": runtime_config.get(
                        "alignment_backend_name"
                    ),
                    "mt_backend_name": runtime_config.get("mt_backend_name"),
                    "chunk_ms": runtime_config.get("chunk_ms"),
                    "translation_alignatt_border_margin": runtime_config.get(
                        "translation_alignatt_border_margin"
                    ),
                    "translation_alignatt_inaccessible_ms": runtime_config.get(
                        "translation_alignatt_inaccessible_ms"
                    ),
                    "translation_alignatt_min_source_mass": runtime_config.get(
                        "translation_alignatt_min_source_mass"
                    ),
                },
                "files": copied_files,
            }
        )
    return rows


def _materialize_additive_bundles(
    *,
    specs: Iterable[AdditiveBundleSpec],
    artifact_root: Path,
    results_root: Path,
    repo_root: Path,
    mode: str,
    skip_missing: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in specs:
        source_dir = (repo_root / spec.source_output_dir).resolve()
        if not source_dir.is_dir():
            if skip_missing:
                print(f"  SKIP missing additive source: {source_dir}")
                continue
            raise FileNotFoundError(f"Missing additive source directory: {source_dir}")

        artifact_files = {name: source_dir / name for name in ARTIFACT_FILENAMES}
        result_files = {name: source_dir / name for name in RESULT_FILENAMES}

        missing_artifacts = [
            str(path) for path in artifact_files.values() if not path.is_file()
        ]
        missing_results = [
            str(path) for path in result_files.values() if not path.is_file()
        ]
        if missing_artifacts:
            raise FileNotFoundError(
                "Missing additive artifact files:\n" + "\n".join(missing_artifacts)
            )
        if missing_results and not skip_missing:
            raise FileNotFoundError(
                "Missing additive result files:\n" + "\n".join(missing_results)
            )

        manifest = json.loads(artifact_files["manifest.json"].read_text())
        runtime_config = manifest.get("runtime_config", {})
        if manifest.get("source_language_code") != spec.source_language_code:
            raise ValueError(
                f"{source_dir} source language mismatch: "
                f"{manifest.get('source_language_code')} != {spec.source_language_code}"
            )
        if manifest.get("target_language_code") != spec.target_language_code:
            raise ValueError(
                f"{source_dir} target language mismatch: "
                f"{manifest.get('target_language_code')} != {spec.target_language_code}"
            )
        if int(runtime_config.get("chunk_ms", -1)) != spec.chunk_ms:
            raise ValueError(
                f"{source_dir} chunk mismatch: "
                f"{runtime_config.get('chunk_ms')} != {spec.chunk_ms}"
            )

        artifacts_dir = artifact_root / spec.artifacts_subdir
        if artifacts_dir.exists():
            shutil.rmtree(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        copied_artifacts: dict[str, dict[str, object]] = {}
        for name, source_path in artifact_files.items():
            destination_path = artifacts_dir / name
            materialize_file(source_path, destination_path, mode=mode)
            copied_artifacts[name] = {
                "path": relative_to_repo(destination_path, repo_root),
                "bytes": destination_path.stat().st_size,
            }

        hypotheses_count = count_lines(artifacts_dir / "hypothesis.jsonl")
        stream_updates_count = count_lines(artifacts_dir / "stream_updates.jsonl")

        copied_results: dict[str, dict[str, object]] = {}
        contract_scores: dict[str, object] = {}
        if all(path.is_file() for path in result_files.values()):
            results_dir = results_root / spec.results_subdir
            if results_dir.exists():
                shutil.rmtree(results_dir)
            results_dir.mkdir(parents=True, exist_ok=True)
            for name, source_path in result_files.items():
                destination_path = results_dir / name
                materialize_file(source_path, destination_path, mode=mode)
                copied_results[name] = {
                    "path": relative_to_repo(destination_path, repo_root),
                    "bytes": destination_path.stat().st_size,
                }
            evaluation = json.loads(result_files["evaluation.json"].read_text())
            contract_scores = dict(evaluation.get("contract_scores", {}))

        rows.append(
            {
                **asdict(spec),
                "source_output_dir": relative_to_repo(source_dir, repo_root),
                "artifacts_dir": relative_to_repo(artifacts_dir, repo_root),
                "results_dir": (
                    relative_to_repo(results_root / spec.results_subdir, repo_root)
                    if copied_results
                    else None
                ),
                "target_language": manifest.get("target_language"),
                "num_inputs": manifest.get("num_inputs"),
                "num_audios": manifest.get("num_audios"),
                "hypotheses": hypotheses_count,
                "stream_updates": stream_updates_count,
                "generated_at_utc": manifest.get("generated_at_utc"),
                "runtime_config": {
                    "alignment_backend_name": runtime_config.get(
                        "alignment_backend_name"
                    ),
                    "mt_backend_name": runtime_config.get("mt_backend_name"),
                    "chunk_ms": runtime_config.get("chunk_ms"),
                    "translation_alignatt_border_margin": runtime_config.get(
                        "translation_alignatt_border_margin"
                    ),
                    "translation_alignatt_inaccessible_ms": runtime_config.get(
                        "translation_alignatt_inaccessible_ms"
                    ),
                    "translation_alignatt_min_source_mass": runtime_config.get(
                        "translation_alignatt_min_source_mass"
                    ),
                },
                "contract_scores": contract_scores,
                "files": {"artifacts": copied_artifacts, "results": copied_results},
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    artifact_root = (repo_root / args.artifact_root).resolve()
    results_root = (repo_root / args.results_root).resolve()
    index_path = (repo_root / args.index_path).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    bundle_rows = _materialize_main_bundles(
        specs=BUNDLE_SPECS,
        artifact_root=artifact_root,
        repo_root=repo_root,
        mode=args.mode,
    )
    print(f"Materialized {len(bundle_rows)} main-track submission bundles.")

    additive_rows: list[dict[str, object]] = []
    if not args.skip_additive:
        additive_rows = _materialize_additive_bundles(
            specs=ADDITIVE_BUNDLE_SPECS,
            artifact_root=artifact_root,
            results_root=results_root,
            repo_root=repo_root,
            mode=args.mode,
            skip_missing=args.skip_missing_additive,
        )
        print(f"Materialized {len(additive_rows)} additive calibration bundles.")

    index_payload = {
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "artifact_root": relative_to_repo(artifact_root, repo_root),
        "results_root": relative_to_repo(results_root, repo_root),
        "materialization_mode": args.mode,
        "bundles": bundle_rows,
        "additive_bundles": additive_rows,
    }
    index_path.write_text(json.dumps(index_payload, indent=2, sort_keys=False) + "\n")
    print(f"Wrote bundle index to {relative_to_repo(index_path, repo_root)}")


if __name__ == "__main__":
    main()
