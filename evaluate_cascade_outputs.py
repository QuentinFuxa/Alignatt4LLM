#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from cascade_artifacts import (
    DEFAULT_COMET_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEGMENTATION_PATH,
    DEFAULT_SOURCE_REF_PATH,
    DEFAULT_TARGET_REF_PATH,
    HYPOTHESIS_FILENAME,
    write_evaluation_outputs,
)
from omnisteval import evaluate_instances
from omnisteval.io import format_report, load_resegmentation_inputs
from omnisteval.resegment import resegment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate outputs/cascade_v1 artifacts with OmniSTEval.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory containing hypothesis.jsonl and where evaluation files will be written.",
    )
    parser.add_argument(
        "--speech-segmentation",
        default=DEFAULT_SEGMENTATION_PATH,
        help="Speech segmentation YAML or JSON for long-form resegmentation.",
    )
    parser.add_argument(
        "--target-reference",
        default=DEFAULT_TARGET_REF_PATH,
        help="Target-language reference text file.",
    )
    parser.add_argument(
        "--source-reference",
        default=DEFAULT_SOURCE_REF_PATH,
        help="Source-language reference text file used by XCOMETXL.",
    )
    parser.add_argument(
        "--comet-model",
        default=DEFAULT_COMET_MODEL,
        help="COMET or XCOMET model identifier passed to unbabel-comet.",
    )
    parser.add_argument(
        "--skip-comet",
        action="store_true",
        help="Skip XCOMETXL and only compute local quality plus latency smoke-test metrics.",
    )
    parser.add_argument(
        "--fix-emission-ca",
        action="store_true",
        help="Apply OmniSTEval's fix for cumulative computation-aware timestamps.",
    )
    return parser.parse_args()


def load_lines(path: str) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]


def resolve_hypothesis_path(output_dir: str) -> Path:
    hypothesis_path = Path(output_dir) / HYPOTHESIS_FILENAME
    if not hypothesis_path.exists():
        raise FileNotFoundError(
            f"Missing baseline hypothesis file: {hypothesis_path}. "
            "Run run_cascade_baseline.py from .venv-inference first."
        )
    return hypothesis_path


def read_hypothesis_sources(hypothesis_path: Path) -> set[str]:
    sources: set[str] = set()
    for line in hypothesis_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        source = payload["source"]
        if isinstance(source, list):
            source = source[0]
        source_path = Path(source)
        sources.add(source)
        sources.add(source_path.name)
        sources.add(source_path.stem)
    if not sources:
        raise ValueError(f"No hypothesis sources found in {hypothesis_path}")
    return sources


def filter_longform_reference_inputs(
    segmentation_path: str,
    target_reference_path: str,
    allowed_sources: set[str],
) -> tuple[list[dict], list[str]]:
    segmentation = yaml.safe_load(Path(segmentation_path).read_text(encoding="utf-8"))
    references = load_lines(target_reference_path)
    if len(segmentation) != len(references):
        raise ValueError("Segmentation and target reference files must have matching lengths.")

    filtered_segments: list[dict] = []
    filtered_references: list[str] = []
    for segment, reference in zip(segmentation, references):
        wav_name = str(segment["wav"])
        wav_path = Path(wav_name)
        if wav_name in allowed_sources or wav_path.name in allowed_sources or wav_path.stem in allowed_sources:
            filtered_segments.append(segment)
            filtered_references.append(reference)

    if not filtered_segments:
        raise ValueError(
            "No segments matched the hypothesis sources. "
            f"Sources={sorted(allowed_sources)} segmentation={segmentation_path}"
        )

    return filtered_segments, filtered_references


def main() -> None:
    args = parse_args()
    hypothesis_path = resolve_hypothesis_path(args.output_dir)
    compute_comet = not args.skip_comet
    allowed_sources = read_hypothesis_sources(hypothesis_path)
    filtered_segments, filtered_references = filter_longform_reference_inputs(
        args.speech_segmentation,
        args.target_reference,
        allowed_sources,
    )

    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        filtered_segmentation_path = tmp_path / "filtered.segmentation.json"
        filtered_reference_path = tmp_path / "filtered.target.txt"
        filtered_segmentation_path.write_text(
            json.dumps(filtered_segments, ensure_ascii=False),
            encoding="utf-8",
        )
        filtered_reference_path.write_text(
            "\n".join(filtered_references) + "\n",
            encoding="utf-8",
        )

        ref_words, hyp_words, segmentation, ref_sentences = load_resegmentation_inputs(
            speech_segmentation=str(filtered_segmentation_path),
            text_segmentation=None,
            ref_sentences_file=str(filtered_reference_path),
            hypothesis_file=str(hypothesis_path),
            hypothesis_format="jsonl",
            char_level=False,
            offset_delays=False,
            fix_emission_ca_flag=args.fix_emission_ca,
        )
        instances, instances_dicts = resegment(
            ref_words,
            hyp_words,
            segmentation,
            ref_sentences,
            char_level=False,
            lang="de",
        )

    source_sentences = None
    if compute_comet:
        all_source_sentences = load_lines(args.source_reference)
        if len(all_source_sentences) != len(load_lines(args.target_reference)):
            raise ValueError(
                "The source reference file must contain exactly one line per target reference segment."
            )
        source_sentences = [
            source_sentence
            for segment, source_sentence in zip(
                yaml.safe_load(Path(args.speech_segmentation).read_text(encoding="utf-8")),
                all_source_sentences,
            )
            if str(segment["wav"]) in allowed_sources
            or Path(str(segment["wav"])).name in allowed_sources
            or Path(str(segment["wav"])).stem in allowed_sources
        ]
        if len(source_sentences) != len(ref_sentences):
            raise ValueError(
                "The source reference file must contain exactly one line per target reference segment."
            )

    try:
        scores, instance_report = evaluate_instances(
            instances,
            compute_quality=True,
            compute_latency=True,
            is_longform=True,
            compute_comet=compute_comet,
            comet_model=args.comet_model,
            source_sentences=source_sentences,
        )
    except Exception as exc:
        if compute_comet:
            raise RuntimeError(
                "XCOMETXL evaluation failed. Accept or cache the model for "
                f"{args.comet_model}, or rerun with --skip-comet for an offline smoke test."
            ) from exc
        raise

    contract_scores = {
        "BLEU": scores.get("bleu"),
        "CHRF": scores.get("chrf"),
        "XCOMETXL": scores.get("comet"),
        "LongYAAL CU": scores.get("long_yaal"),
        "LongYAAL CA": scores.get("ca_long_yaal"),
    }
    settings = {
        "hypothesis_file": str(hypothesis_path),
        "speech_segmentation": args.speech_segmentation,
        "target_reference": args.target_reference,
        "matched_sources": sorted(allowed_sources),
        "source_reference": args.source_reference if compute_comet else "skipped",
        "comet_model": args.comet_model if compute_comet else "skipped",
        "fix_emission_ca": args.fix_emission_ca,
        "output_dir": args.output_dir,
    }
    report_text = format_report(
        "Longform speech resegmentation",
        settings,
        scores,
        instance_report,
    )
    written_files = write_evaluation_outputs(
        args.output_dir,
        settings=settings,
        contract_scores=contract_scores,
        raw_scores=scores,
        report_lines=report_text.splitlines(),
        instances_dicts=instances_dicts,
    )

    print(f"Wrote evaluation artifacts to {args.output_dir}")
    for label, path in written_files.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()
