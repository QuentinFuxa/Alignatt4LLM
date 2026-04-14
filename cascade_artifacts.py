from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


ARTIFACT_SCHEMA_VERSION = "cascade_v1"
DEFAULT_WAV_PATH = "test-set/audio/ccpXHNfaoy.wav"
DEFAULT_OUTPUT_DIR = "outputs/cascade_v1"
DEFAULT_SEGMENTATION_PATH = "test-set/audio-segments.yaml"
DEFAULT_SOURCE_REF_PATH = "test-set/ref/en.txt"
DEFAULT_TARGET_REF_PATH = "test-set/ref/de.txt"
DEFAULT_COMET_MODEL = "Unbabel/XCOMET-XL"

MANIFEST_FILENAME = "manifest.json"
HYPOTHESIS_FILENAME = "hypothesis.jsonl"
STREAM_UPDATES_FILENAME = "stream_updates.jsonl"
FINAL_ASR_FILENAME = "transcript.en.txt"
FINAL_TRANSLATION_FILENAME = "translation.de.txt"
RESEGMENTED_INSTANCES_FILENAME = "instances.resegmented.jsonl"
EVALUATION_JSON_FILENAME = "evaluation.json"
EVALUATION_REPORT_FILENAME = "evaluation.report.txt"
SCORES_TSV_FILENAME = "scores.tsv"


@dataclass
class StreamUpdate:
    update_idx: int
    audio_processed_ms: float
    wallclock_elapsed_ms: float
    asr_text: str
    translation_text: str
    new_words: list[str] = field(default_factory=list)


@dataclass
class InferenceArtifacts:
    wav_path: str
    chunk_ms: int
    source_language: str
    target_language: str
    latency_unit: str
    audio_duration_ms: float
    final_asr_text: str
    final_translation_text: str
    translation_word_delays_ms: list[float]
    translation_word_elapsed_ms: list[float]
    updates: list[StreamUpdate]
    runtime_config: dict[str, Any]

    def hypothesis_record(self) -> dict[str, Any]:
        return {
            "source": [Path(self.wav_path).name],
            "source_length": self.audio_duration_ms,
            "prediction": self.final_translation_text,
            "delays": self.translation_word_delays_ms,
            "elapsed": self.translation_word_elapsed_ms,
        }

    def manifest_record(self) -> dict[str, Any]:
        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "generated_at_utc": utc_now_isoformat(),
            "kind": "inference",
            "wav_path": self.wav_path,
            "chunk_ms": self.chunk_ms,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "latency_unit": self.latency_unit,
            "audio_duration_ms": self.audio_duration_ms,
            "files": {
                "hypothesis_jsonl": HYPOTHESIS_FILENAME,
                "stream_updates_jsonl": STREAM_UPDATES_FILENAME,
                "transcript_en_txt": FINAL_ASR_FILENAME,
                "translation_de_txt": FINAL_TRANSLATION_FILENAME,
            },
            "runtime_config": self.runtime_config,
        }


def utc_now_isoformat() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_text(path: Path, value: str) -> None:
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def ensure_output_dir(output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def write_inference_artifacts(artifacts: InferenceArtifacts, output_dir: str | Path) -> dict[str, str]:
    output_path = ensure_output_dir(output_dir)

    write_json(output_path / MANIFEST_FILENAME, artifacts.manifest_record())
    write_jsonl(output_path / HYPOTHESIS_FILENAME, [artifacts.hypothesis_record()])
    write_jsonl(
        output_path / STREAM_UPDATES_FILENAME,
        [asdict(update) for update in artifacts.updates],
    )
    write_text(output_path / FINAL_ASR_FILENAME, artifacts.final_asr_text)
    write_text(output_path / FINAL_TRANSLATION_FILENAME, artifacts.final_translation_text)

    return {
        "manifest": str(output_path / MANIFEST_FILENAME),
        "hypothesis": str(output_path / HYPOTHESIS_FILENAME),
        "stream_updates": str(output_path / STREAM_UPDATES_FILENAME),
        "transcript": str(output_path / FINAL_ASR_FILENAME),
        "translation": str(output_path / FINAL_TRANSLATION_FILENAME),
    }


def write_evaluation_outputs(
    output_dir: str | Path,
    *,
    settings: dict[str, Any],
    contract_scores: dict[str, float | None],
    raw_scores: dict[str, float],
    report_lines: list[str],
    instances_dicts: list[dict[str, Any]],
) -> dict[str, str]:
    output_path = ensure_output_dir(output_dir)

    write_jsonl(output_path / RESEGMENTED_INSTANCES_FILENAME, instances_dicts)
    write_scores_tsv(output_path / SCORES_TSV_FILENAME, contract_scores)
    write_text(output_path / EVALUATION_REPORT_FILENAME, "\n".join(report_lines).strip())
    write_json(
        output_path / EVALUATION_JSON_FILENAME,
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "generated_at_utc": utc_now_isoformat(),
            "kind": "evaluation",
            "settings": settings,
            "contract_scores": contract_scores,
            "raw_scores": raw_scores,
            "report_lines": report_lines,
            "files": {
                "instances_resegmented_jsonl": RESEGMENTED_INSTANCES_FILENAME,
                "scores_tsv": SCORES_TSV_FILENAME,
                "evaluation_report_txt": EVALUATION_REPORT_FILENAME,
            },
        },
    )

    return {
        "instances": str(output_path / RESEGMENTED_INSTANCES_FILENAME),
        "scores": str(output_path / SCORES_TSV_FILENAME),
        "report": str(output_path / EVALUATION_REPORT_FILENAME),
        "evaluation": str(output_path / EVALUATION_JSON_FILENAME),
    }


def write_scores_tsv(path: Path, contract_scores: dict[str, float | None]) -> None:
    ordered_metrics = [
        "BLEU",
        "CHRF",
        "XCOMETXL",
        "LongYAAL CU",
        "LongYAAL CA",
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("metric\tvalue\n")
        for metric in ordered_metrics:
            value = contract_scores.get(metric)
            rendered = "NA" if value is None else f"{value:.4f}"
            handle.write(f"{metric}\t{rendered}\n")

