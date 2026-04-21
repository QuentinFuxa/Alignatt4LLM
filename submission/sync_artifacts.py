#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cascade.submission import SUBMISSION_PRESETS


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

MAIN_TARGET_LANGS = ("de", "it", "zh")
ADDITIVE_CHUNKS = (850, 1500, 1900)


@dataclass(frozen=True)
class MainBundleSpec:
    preset: str
    regime: str
    source_language_code: str
    target_language_code: str
    chunk_ms: int

    @property
    def submission_subdir(self) -> Path:
        return Path("main") / self.regime / f"{self.source_language_code}-{self.target_language_code}"

    def source_output_candidates(self) -> tuple[str, ...]:
        return (
            f"outputs/iwslt26_testset_{self.preset}_en{self.target_language_code}",
            f"outputs/iwslt26_testset_chunk{self.chunk_ms}_borderp1_en{self.target_language_code}",
        )


@dataclass(frozen=True)
class AdditiveBundleSpec:
    group: str
    chunk_ms: int
    source_language_code: str
    target_language_code: str

    @property
    def artifacts_subdir(self) -> Path:
        return Path("additive") / self.group / f"{self.source_language_code}-{self.target_language_code}"

    @property
    def results_subdir(self) -> Path:
        return Path("additive") / self.group / f"{self.source_language_code}-{self.target_language_code}"

    def source_output_candidates(self) -> tuple[str, ...]:
        return (
            f"outputs/iwslt26_devset_additive_{self.group}_en{self.target_language_code}",
            f"outputs/iwslt26_devset_chunk{self.chunk_ms}_borderp1_en{self.target_language_code}",
            f"outputs/iwslt26_devset_chunk{self.chunk_ms}_en{self.target_language_code}",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", default="submission/artifacts")
    parser.add_argument("--results-root", default="submission/results")
    parser.add_argument("--index-path", default="submission/ARTIFACT_INDEX.json")
    parser.add_argument("--mode", choices=("copy", "hardlink"), default="copy")
    parser.add_argument("--skip-additive", action="store_true")
    parser.add_argument("--skip-missing-additive", action="store_true")
    return parser.parse_args()


def count_lines(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            count += chunk.count(b"\n")
    return count


def relative_to_repo(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


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


def resolve_source_dir(repo_root: Path, candidates: tuple[str, ...]) -> Path:
    for candidate in candidates:
        path = repo_root / candidate
        if path.is_dir():
            return path.resolve()
    return (repo_root / candidates[0]).resolve()


def build_main_specs() -> tuple[MainBundleSpec, ...]:
    return tuple(
        MainBundleSpec(
            preset=name,
            regime=preset.latency_regime,
            source_language_code="en",
            target_language_code=target,
            chunk_ms=preset.chunk_ms,
        )
        for name, preset in SUBMISSION_PRESETS.items()
        if preset.track == "main"
        for target in MAIN_TARGET_LANGS
    )


def build_additive_specs() -> tuple[AdditiveBundleSpec, ...]:
    return tuple(
        AdditiveBundleSpec(
            group=f"chunk{chunk_ms}",
            chunk_ms=chunk_ms,
            source_language_code="en",
            target_language_code=target,
        )
        for chunk_ms in ADDITIVE_CHUNKS
        for target in MAIN_TARGET_LANGS
    )


def materialize_main_bundles(
    *,
    specs: tuple[MainBundleSpec, ...],
    artifact_root: Path,
    repo_root: Path,
    mode: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in specs:
        source_dir = resolve_source_dir(repo_root, spec.source_output_candidates())
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Missing source artifact directory: {source_dir}")

        destination_dir = artifact_root / spec.submission_subdir
        if destination_dir.exists():
            shutil.rmtree(destination_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)

        source_files = {name: source_dir / name for name in ARTIFACT_FILENAMES}
        missing_files = [str(path) for path in source_files.values() if not path.is_file()]
        if missing_files:
            raise FileNotFoundError("Missing required artifact files:\n" + "\n".join(missing_files))

        manifest = json.loads(source_files["manifest.json"].read_text(encoding="utf-8"))
        runtime_config = manifest.get("runtime_config", {})
        if int(runtime_config.get("chunk_ms", -1)) != spec.chunk_ms:
            raise ValueError(
                f"{source_dir} chunk mismatch: {runtime_config.get('chunk_ms')} != {spec.chunk_ms}"
            )

        copied_files: dict[str, dict[str, object]] = {}
        for name, source_path in source_files.items():
            destination_path = destination_dir / name
            materialize_file(source_path, destination_path, mode)
            copied_files[name] = {
                "path": relative_to_repo(destination_path, repo_root),
                "bytes": destination_path.stat().st_size,
            }

        rows.append(
            {
                **asdict(spec),
                "source_output_dir": relative_to_repo(source_dir, repo_root),
                "submission_dir": relative_to_repo(destination_dir, repo_root),
                "hypotheses": count_lines(destination_dir / "hypothesis.jsonl"),
                "stream_updates": count_lines(destination_dir / "stream_updates.jsonl"),
                "runtime_config": {
                    "alignment_backend_name": runtime_config.get("alignment_backend_name"),
                    "mt_backend_name": runtime_config.get("mt_backend_name"),
                    "chunk_ms": runtime_config.get("chunk_ms"),
                    "translation_alignatt_border_margin": runtime_config.get(
                        "translation_alignatt_border_margin"
                    ),
                },
                "files": copied_files,
            }
        )
    return rows


def materialize_additive_bundles(
    *,
    specs: tuple[AdditiveBundleSpec, ...],
    artifact_root: Path,
    results_root: Path,
    repo_root: Path,
    mode: str,
    skip_missing: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in specs:
        source_dir = resolve_source_dir(repo_root, spec.source_output_candidates())
        if not source_dir.is_dir():
            if skip_missing:
                continue
            raise FileNotFoundError(f"Missing additive source directory: {source_dir}")

        artifact_destination = artifact_root / spec.artifacts_subdir
        results_destination = results_root / spec.results_subdir
        if artifact_destination.exists():
            shutil.rmtree(artifact_destination)
        if results_destination.exists():
            shutil.rmtree(results_destination)
        artifact_destination.mkdir(parents=True, exist_ok=True)
        results_destination.mkdir(parents=True, exist_ok=True)

        source_artifacts = {name: source_dir / name for name in ARTIFACT_FILENAMES}
        source_results = {name: source_dir / name for name in RESULT_FILENAMES}
        if any(not path.is_file() for path in source_artifacts.values()):
            if skip_missing:
                continue
            raise FileNotFoundError(f"Missing additive artifacts in {source_dir}")
        if any(not path.is_file() for path in source_results.values()):
            if skip_missing:
                continue
            raise FileNotFoundError(f"Missing additive results in {source_dir}")

        copied_artifacts: dict[str, dict[str, object]] = {}
        copied_results: dict[str, dict[str, object]] = {}
        for name, source_path in source_artifacts.items():
            destination_path = artifact_destination / name
            materialize_file(source_path, destination_path, mode)
            copied_artifacts[name] = {
                "path": relative_to_repo(destination_path, repo_root),
                "bytes": destination_path.stat().st_size,
            }
        for name, source_path in source_results.items():
            destination_path = results_destination / name
            materialize_file(source_path, destination_path, mode)
            copied_results[name] = {
                "path": relative_to_repo(destination_path, repo_root),
                "bytes": destination_path.stat().st_size,
            }

        rows.append(
            {
                **asdict(spec),
                "source_output_dir": relative_to_repo(source_dir, repo_root),
                "artifacts_dir": relative_to_repo(artifact_destination, repo_root),
                "results_dir": relative_to_repo(results_destination, repo_root),
                "files": {
                    "artifacts": copied_artifacts,
                    "results": copied_results,
                },
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    artifact_root = repo_root / args.artifact_root
    results_root = repo_root / args.results_root
    index_path = repo_root / args.index_path

    main_rows = materialize_main_bundles(
        specs=build_main_specs(),
        artifact_root=artifact_root,
        repo_root=repo_root,
        mode=args.mode,
    )
    additive_rows: list[dict[str, object]] = []
    if not args.skip_additive:
        additive_rows = materialize_additive_bundles(
            specs=build_additive_specs(),
            artifact_root=artifact_root,
            results_root=results_root,
            repo_root=repo_root,
            mode=args.mode,
            skip_missing=args.skip_missing_additive,
        )

    index_payload = {
        "generated_at_utc": __import__("datetime").datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "main": main_rows,
        "additive": additive_rows,
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index_payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {index_path}")


if __name__ == "__main__":
    main()
