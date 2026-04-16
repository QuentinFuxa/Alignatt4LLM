from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from cascade_text_surface import (
    is_char_level_target_lang,
    split_target_emission_units,
)


ARTIFACT_SCHEMA_VERSION = "cascade_v1"
DEFAULT_WAV_PATH = "test-set/audio/ccpXHNfaoy.wav"
DEFAULT_OUTPUT_DIR = "outputs/cascade_v1"
DEFAULT_SEGMENTATION_PATH = "test-set/audio-segments.yaml"
DEFAULT_SOURCE_LANG_CODE = "en"
DEFAULT_TARGET_LANG_CODE = "de"
DEFAULT_SOURCE_REF_PATH = f"test-set/ref/{DEFAULT_SOURCE_LANG_CODE}.txt"
DEFAULT_TARGET_REF_PATH = f"test-set/ref/{DEFAULT_TARGET_LANG_CODE}.txt"
DEFAULT_COMET_MODEL = "Unbabel/XCOMET-XL"

MANIFEST_FILENAME = "manifest.json"
HYPOTHESIS_FILENAME = "hypothesis.jsonl"
STREAM_UPDATES_FILENAME = "stream_updates.jsonl"


def final_asr_filename(source_lang_code: str = DEFAULT_SOURCE_LANG_CODE) -> str:
    return f"transcript.{source_lang_code}.txt"


def final_translation_filename(target_lang_code: str = DEFAULT_TARGET_LANG_CODE) -> str:
    return f"translation.{target_lang_code}.txt"


def reference_path_for(lang_code: str) -> str:
    return f"test-set/ref/{lang_code}.txt"


# Legacy default-language aliases preserved for callers that have not yet been
# parameterised by target language. Prefer ``final_*_filename`` helpers above.
FINAL_ASR_FILENAME = final_asr_filename()
FINAL_TRANSLATION_FILENAME = final_translation_filename()
RESEGMENTED_INSTANCES_FILENAME = "instances.resegmented.jsonl"
EVALUATION_JSON_FILENAME = "evaluation.json"
EVALUATION_REPORT_FILENAME = "evaluation.report.txt"
SCORES_TSV_FILENAME = "scores.tsv"
HYPOTHESIS_ELAPSED_SEMANTICS_CA_COMPATIBLE = "ca_compatible_incremental"
STREAM_UPDATE_ELAPSED_SEMANTICS_WALLCLOCK = "wallclock_elapsed_since_run_start"


def normalize_computation_aware_timestamps(
    delays_ms: list[float],
    elapsed_wallclock_ms: list[float],
) -> list[float]:
    if len(delays_ms) != len(elapsed_wallclock_ms):
        raise ValueError(
            "Computation-aware timestamps require matching delay and elapsed lengths: "
            f"{len(delays_ms)} != {len(elapsed_wallclock_ms)}"
        )
    if not delays_ms:
        return []

    # OmniSTEval expects computation-aware timestamps on the source-time axis.
    # Our runtime stores cumulative wallclock elapsed values, so we convert them
    # into incremental CA-compatible timestamps while preserving monotonicity.
    normalized = [float(elapsed_wallclock_ms[0])]
    for index in range(1, len(delays_ms)):
        candidate = (
            float(elapsed_wallclock_ms[index])
            - float(elapsed_wallclock_ms[index - 1])
            + float(delays_ms[index - 1])
        )
        normalized.append(max(candidate, normalized[-1]))
    return normalized


@dataclass
class StreamUpdate:
    update_idx: int
    audio_processed_ms: float
    wallclock_elapsed_ms: float
    asr_text: str
    translation_text: str
    new_words: list[str] = field(default_factory=list)
    is_eos: bool = False
    raw_translation_text: str | None = None
    emission_policy_action: str | None = None
    translation_prompt_num_cached_tokens: int | None = None
    translation_prompt_num_tokens: int | None = None
    partial_accepted_target: str | None = None
    partial_accepted_token_count: int | None = None
    partial_draft_target: str | None = None
    alignatt_metadata: dict[str, Any] | None = None
    translation_timings_ms: dict[str, float] | None = None


@dataclass
class InferenceArtifacts:
    wav_path: str
    chunk_ms: int
    translation_variant: str | None
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
    run_provenance: dict[str, Any] = field(default_factory=dict)
    source_language_code: str = DEFAULT_SOURCE_LANG_CODE
    target_language_code: str = DEFAULT_TARGET_LANG_CODE

    def hypothesis_record(self) -> dict[str, Any]:
        normalized_elapsed_ms = normalize_computation_aware_timestamps(
            self.translation_word_delays_ms,
            self.translation_word_elapsed_ms,
        )
        units = split_target_emission_units(
            self.final_translation_text,
            target_lang_code=self.target_language_code,
        )
        prediction = (
            "".join(units)
            if is_char_level_target_lang(self.target_language_code)
            else " ".join(units)
        )
        return {
            "source": [Path(self.wav_path).name],
            "source_length": self.audio_duration_ms,
            "prediction": prediction,
            "delays": self.translation_word_delays_ms,
            "elapsed": normalized_elapsed_ms,
            "elapsed_wallclock_ms": self.translation_word_elapsed_ms,
            "elapsed_semantics": HYPOTHESIS_ELAPSED_SEMANTICS_CA_COMPATIBLE,
        }

    def final_asr_filename(self) -> str:
        return final_asr_filename(self.source_language_code)

    def final_translation_filename(self) -> str:
        return final_translation_filename(self.target_language_code)

    def manifest_record(self) -> dict[str, Any]:
        runtime_config = dict(self.runtime_config)
        runtime_config.setdefault(
            "hypothesis_elapsed_semantics",
            HYPOTHESIS_ELAPSED_SEMANTICS_CA_COMPATIBLE,
        )
        runtime_config.setdefault(
            "stream_update_elapsed_semantics",
            STREAM_UPDATE_ELAPSED_SEMANTICS_WALLCLOCK,
        )
        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "generated_at_utc": utc_now_isoformat(),
            "kind": "inference",
            "wav_path": self.wav_path,
            "chunk_ms": self.chunk_ms,
            "translation_variant": self.translation_variant,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "source_language_code": self.source_language_code,
            "target_language_code": self.target_language_code,
            "latency_unit": self.latency_unit,
            "audio_duration_ms": self.audio_duration_ms,
            "files": {
                "hypothesis_jsonl": HYPOTHESIS_FILENAME,
                "stream_updates_jsonl": STREAM_UPDATES_FILENAME,
                "transcript_txt": self.final_asr_filename(),
                "translation_txt": self.final_translation_filename(),
            },
            "runtime_config": runtime_config,
            "run_provenance": dict(self.run_provenance),
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

    asr_path = output_path / artifacts.final_asr_filename()
    translation_path = output_path / artifacts.final_translation_filename()

    write_json(output_path / MANIFEST_FILENAME, artifacts.manifest_record())
    write_jsonl(output_path / HYPOTHESIS_FILENAME, [artifacts.hypothesis_record()])
    write_jsonl(
        output_path / STREAM_UPDATES_FILENAME,
        [asdict(update) for update in artifacts.updates],
    )
    write_text(asr_path, artifacts.final_asr_text)
    write_text(translation_path, artifacts.final_translation_text)

    return {
        "manifest": str(output_path / MANIFEST_FILENAME),
        "hypothesis": str(output_path / HYPOTHESIS_FILENAME),
        "stream_updates": str(output_path / STREAM_UPDATES_FILENAME),
        "transcript": str(asr_path),
        "translation": str(translation_path),
    }


def write_evaluation_outputs(
    output_dir: str | Path,
    *,
    settings: dict[str, Any],
    contract_scores: dict[str, float | None],
    raw_scores: dict[str, float],
    report_lines: list[str],
    instances_dicts: list[dict[str, Any]],
    metric_blockers: list[dict[str, Any]] | None = None,
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
            "metric_blockers": metric_blockers or [],
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
