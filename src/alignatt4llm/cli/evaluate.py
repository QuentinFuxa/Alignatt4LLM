#!/usr/bin/env python3
# USE .venv-evaluation for this evaluation.
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import snapshot_download
import yaml

from alignatt4llm.artifacts import (
    DEFAULT_COMET_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEGMENTATION_PATH,
    DEFAULT_SOURCE_REF_PATH,
    DEFAULT_TARGET_LANG_CODE,
    DEFAULT_TARGET_REF_PATH,
    HYPOTHESIS_FILENAME,
    HYPOTHESIS_ELAPSED_SEMANTICS_CA_COMPATIBLE,
    MANIFEST_FILENAME,
    reference_path_for,
    write_evaluation_outputs,
)
from omnisteval import evaluate_instances
from omnisteval.io import format_report, load_resegmentation_inputs
from omnisteval.resegment import resegment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate AlignAtt4LLM inference artifacts with OmniSTEval.",
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
        default=None,
        help=(
            "Target-language reference text file. Defaults to the manifest's "
            "target_language_code when present, else en->de."
        ),
    )
    parser.add_argument(
        "--target-lang-code",
        default=None,
        help=(
            "ISO-ish target language code (e.g. 'de', 'it', 'zh'). Overrides "
            "the manifest value; determines the resegmentation language."
        ),
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
        dest="fix_emission_ca",
        action="store_true",
        help="Force-apply OmniSTEval's fix for legacy cumulative elapsed timestamps.",
    )
    parser.add_argument(
        "--no-fix-emission-ca",
        dest="fix_emission_ca",
        action="store_false",
        help="Disable the automatic compatibility fix for legacy cumulative elapsed timestamps.",
    )
    parser.set_defaults(fix_emission_ca=None)
    return parser.parse_args()


def load_lines(path: str) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]


def resolve_hypothesis_path(output_dir: str) -> Path:
    hypothesis_path = Path(output_dir) / HYPOTHESIS_FILENAME
    if not hypothesis_path.exists():
        raise FileNotFoundError(
            f"Missing baseline hypothesis file: {hypothesis_path}. "
            "Run alignatt-batch or alignatt-compare from .venv-inference first."
        )
    return hypothesis_path


def load_manifest(output_dir: str) -> dict | None:
    manifest_path = Path(output_dir) / MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def resolve_target_lang_code(output_dir: str, cli_override: str | None) -> str:
    if cli_override:
        return cli_override

    manifest = load_manifest(output_dir)
    if manifest is None:
        return DEFAULT_TARGET_LANG_CODE

    code = manifest.get("target_language_code")
    if isinstance(code, str) and code:
        return code
    return DEFAULT_TARGET_LANG_CODE


def resolve_target_reference(cli_override: str | None, target_lang_code: str) -> str:
    if cli_override:
        return cli_override
    if target_lang_code == DEFAULT_TARGET_LANG_CODE:
        return DEFAULT_TARGET_REF_PATH
    return reference_path_for(target_lang_code)


def resolve_fix_emission_ca(output_dir: str, cli_override: bool | None) -> bool:
    if cli_override is not None:
        return cli_override

    manifest = load_manifest(output_dir)
    if manifest is None:
        return False

    runtime_config = manifest.get("runtime_config", {}) or {}
    semantics = runtime_config.get("hypothesis_elapsed_semantics")
    if semantics == HYPOTHESIS_ELAPSED_SEMANTICS_CA_COMPATIBLE:
        return False

    # Legacy cascade_v1 bundles stored raw cumulative wallclock elapsed values
    # in hypothesis.jsonl and need OmniSTEval's compatibility fix at evaluation.
    return manifest.get("schema_version") == "cascade_v1"


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


BLEU_TOKENIZER_BY_LANG = {"zh": "zh", "ja": "ja-mecab", "ko": "ko-mecab"}


def evaluate_base_metrics(
    instances: list, *, target_lang_code: str | None = None
) -> tuple[dict[str, float], list[str], list[str]]:
    tokenizer = BLEU_TOKENIZER_BY_LANG.get((target_lang_code or "").lower(), "13a")
    result = evaluate_instances(
        instances,
        compute_quality=True,
        compute_latency=True,
        is_longform=True,
        compute_comet=False,
        bleu_tokenizer=tokenizer,
    )
    if len(result) == 2:
        scores, instance_report = result
        return scores, instance_report, []
    if len(result) == 3:
        scores, instance_report, normalization_report = result
        return scores, instance_report, normalization_report
    raise ValueError(f"Unexpected OmniSTEval result shape: {len(result)}")


def comet_local_cache_blocker(comet_model: str) -> dict[str, str] | None:
    offline_env = {"1", "true", "yes"}
    if (
        os.environ.get("HF_HUB_OFFLINE", "").lower() not in offline_env
        and os.environ.get("TRANSFORMERS_OFFLINE", "").lower() not in offline_env
    ):
        return None

    try:
        snapshot_download(repo_id=comet_model, local_files_only=True)
    except Exception as exc:
        return {
            "metric": "XCOMETXL",
            "status": "blocked",
            "reason": "comet_model_not_cached_locally",
            "model": comet_model,
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }

    return None


def resolve_comet_checkpoint_path(comet_model: str) -> str:
    from comet import download_model  # type: ignore

    checkpoint_path = Path(comet_model)
    if checkpoint_path.exists():
        return str(checkpoint_path)

    try:
        return str(download_model(comet_model, local_files_only=True))
    except Exception:
        return str(download_model(comet_model))


def compute_comet_system_score(
    *,
    comet_model: str,
    instances: list,
    source_sentences: list[str],
) -> float:
    import torch
    from comet import load_from_checkpoint  # type: ignore

    checkpoint_path = resolve_comet_checkpoint_path(comet_model)
    predict_rows = [
        {
            "src": source_sentence,
            "mt": instance.prediction,
            "ref": instance.reference,
        }
        for source_sentence, instance in zip(source_sentences, instances)
    ]
    accelerators = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    last_error: Exception | None = None

    for accelerator in accelerators:
        model = load_from_checkpoint(checkpoint_path)
        try:
            gpus = 1 if accelerator == "cuda" else 0
            output = model.predict(
                predict_rows,
                batch_size=8,
                gpus=gpus,
                accelerator=accelerator,
                num_workers=1,
                progress_bar=False,
            )
            return float(output.system_score)
        except torch.OutOfMemoryError as exc:
            last_error = exc
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            last_error = exc
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if last_error is None:
        raise RuntimeError("COMET evaluation failed without raising an exception.")
    raise last_error


def try_compute_comet_score(
    instances: list,
    *,
    comet_model: str,
    source_sentences: list[str],
) -> tuple[float | None, dict[str, str] | None]:
    local_cache_blocker = comet_local_cache_blocker(comet_model)
    if local_cache_blocker is not None:
        return None, local_cache_blocker

    try:
        comet_score = compute_comet_system_score(
            comet_model=comet_model,
            instances=instances,
            source_sentences=source_sentences,
        )
    except Exception as exc:
        return None, {
            "metric": "XCOMETXL",
            "status": "blocked",
            "reason": "comet_evaluation_failed",
            "model": comet_model,
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }

    return comet_score, None


def append_metric_blockers(report_lines: list[str], blockers: list[dict[str, str]]) -> list[str]:
    if not blockers:
        return report_lines

    augmented = list(report_lines)
    augmented.extend(["", "Metric blockers", "-" * 64])
    for blocker in blockers:
        augmented.append(
            "  "
            f"{blocker['metric']}: {blocker['status']} | "
            f"{blocker['reason']} | {blocker['model']} | "
            f"{blocker['exception_type']}: {blocker['message']}"
        )
    return augmented


def format_evaluation_report(
    mode_label: str,
    settings: dict,
    scores: dict,
    instance_report: list[str],
    normalization_report: list[str],
) -> str:
    try:
        return format_report(
            mode_label,
            settings,
            scores,
            instance_report,
            normalization_report,
        )
    except TypeError as exc:
        if (
            "normalization_report" not in str(exc)
            and "positional arguments" not in str(exc)
        ):
            raise
        report_text = format_report(mode_label, settings, scores, instance_report)
        if normalization_report:
            report_lines = report_text.splitlines()
            report_lines.extend(["", "Normalization report", "-" * 64, *normalization_report])
            return "\n".join(report_lines)
        return report_text


def main() -> None:
    args = parse_args()
    hypothesis_path = resolve_hypothesis_path(args.output_dir)
    fix_emission_ca = resolve_fix_emission_ca(args.output_dir, args.fix_emission_ca)
    compute_comet = not args.skip_comet
    target_lang_code = resolve_target_lang_code(args.output_dir, args.target_lang_code)
    target_reference_path = resolve_target_reference(args.target_reference, target_lang_code)
    allowed_sources = read_hypothesis_sources(hypothesis_path)
    filtered_segments, filtered_references = filter_longform_reference_inputs(
        args.speech_segmentation,
        target_reference_path,
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

        char_level = target_lang_code == "zh"
        ref_words, hyp_words, segmentation, ref_sentences = load_resegmentation_inputs(
            speech_segmentation=str(filtered_segmentation_path),
            text_segmentation=None,
            ref_sentences_file=str(filtered_reference_path),
            hypothesis_file=str(hypothesis_path),
            hypothesis_format="jsonl",
            char_level=char_level,
            offset_delays=False,
            fix_emission_ca_flag=fix_emission_ca,
        )
        instances, instances_dicts = resegment(
            ref_words,
            hyp_words,
            segmentation,
            ref_sentences,
            char_level=char_level,
            lang=target_lang_code,
        )

    source_sentences = None
    if compute_comet:
        all_source_sentences = load_lines(args.source_reference)
        if len(all_source_sentences) != len(load_lines(target_reference_path)):
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

    scores, instance_report, normalization_report = evaluate_base_metrics(
        instances,
        target_lang_code=target_lang_code,
    )
    metric_blockers: list[dict[str, str]] = []
    if compute_comet:
        comet_score, comet_blocker = try_compute_comet_score(
            instances,
            comet_model=args.comet_model,
            source_sentences=source_sentences,
        )
        if comet_blocker is not None:
            metric_blockers.append(comet_blocker)
        if comet_score is not None:
            scores["comet"] = comet_score

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
        "target_reference": target_reference_path,
        "target_lang_code": target_lang_code,
        "matched_sources": sorted(allowed_sources),
        "source_reference": args.source_reference if compute_comet else "skipped",
        "comet_model": args.comet_model if compute_comet else "skipped",
        "fix_emission_ca": fix_emission_ca,
        "output_dir": args.output_dir,
    }
    report_text = format_evaluation_report(
        "Longform speech resegmentation",
        settings,
        scores,
        instance_report,
        normalization_report,
    )
    report_lines = report_text.splitlines()
    report_lines = append_metric_blockers(report_lines, metric_blockers)
    written_files = write_evaluation_outputs(
        args.output_dir,
        settings=settings,
        contract_scores=contract_scores,
        raw_scores=scores,
        report_lines=report_lines,
        instances_dicts=instances_dicts,
        metric_blockers=metric_blockers,
    )

    print(f"Wrote evaluation artifacts to {args.output_dir}")
    for label, path in written_files.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()
