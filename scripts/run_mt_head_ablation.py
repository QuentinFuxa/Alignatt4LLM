#!/usr/bin/env python3
"""Run the Gemma MT head-filtering ablation with hot model reuse.

This script compares the shipped per-direction top-8 MT AlignAtt heads against
an "average all heads" counterfactual reconstructed from the stored
``ts_matrix`` ranking payload. It keeps the maintained runtime fixed:

- ASR: ``qwen_forced``
- MT: ``gemma_vllm_alignatt``
- Presets: `gemma_low_latency` / `gemma_high_latency` from `cascade.presets`

Only the MT head set changes. Baseline top-8 metrics are read from the compact
archived submission score JSON unless ``--baseline-root`` points to a legacy
evaluation-bundle directory.

The inference path runs in one long-lived Python process and reuses the hot
Qwen + Gemma engines across directions to avoid repeated model reloads.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cascade.incremental_output import (
    append_only_incremental_output,
    empty_incremental_output,
)
from cascade.runtime import (
    CascadeRuntimeConfig,
    LANGUAGE_CODE_TO_NAME,
    LANGUAGE_NAME_TO_CODE,
    LoadedModelBundle,
    alignatt_heads_path_for,
    target_lang_code_for,
)
from cascade.presets import get_runtime_preset
from cascade.text_surface import (
    join_public_emission_units,
    split_public_emission_units,
)
from run_simulstream_batch import (
    git_sha,
    resolve_input_paths,
    run_batch_inference,
    run_single_audio,
)
from cascade.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    HYPOTHESIS_ELAPSED_SEMANTICS_CA_COMPATIBLE,
    HYPOTHESIS_FILENAME,
    MANIFEST_FILENAME,
    STREAM_UPDATE_ELAPSED_SEMANTICS_WALLCLOCK,
    STREAM_UPDATES_FILENAME,
    ensure_output_dir,
    utc_now_isoformat,
    write_json,
    write_jsonl,
)


DEFAULT_DIRECTIONS = ("en-de", "en-zh", "en-it")
DEFAULT_REGIMES = ("low",)
DEFAULT_BASELINE_ROOT = REPO_ROOT / "docs" / "archive" / "2026-05-submission-scores.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "mt_head_ablation"
DEFAULT_SMOKE_WAV = REPO_ROOT / "data" / "smoke" / "alignatt_smoke18.wav"
DEFAULT_DEV_AUDIO_DIR = REPO_ROOT / "data" / "devset" / "audio"


class ReusableCascadeProcessor:
    """Minimal processor adapter that keeps a LoadedModelBundle hot."""

    def __init__(self, runtime_config: CascadeRuntimeConfig):
        self._runtime_config = runtime_config
        self._default_paper_context_path = runtime_config.paper_context_path
        self._bundle = LoadedModelBundle(runtime_config)
        self._bundle.load()
        self._session = self._bundle.new_session()
        self._emitted_units: list[str] = []
        self._target_lang_code = target_lang_code_for(runtime_config.target_lang)

    @property
    def session(self):
        return self._session

    def reconfigure(self, runtime_config: CascadeRuntimeConfig) -> None:
        self._runtime_config = runtime_config
        self._bundle.config = runtime_config
        self._bundle.ensure_alignment_backend()
        mt_backend = self._bundle.ensure_mt_backend()
        mt_backend.runtime_config = runtime_config
        mt_backend.refresh_alignatt_artifacts()
        self._target_lang_code = target_lang_code_for(runtime_config.target_lang)
        self.clear()

    def clear(self) -> None:
        self._runtime_config.paper_context_path = self._default_paper_context_path
        self._session = self._bundle.new_session()
        self._session.clear()
        self._emitted_units = []

    def tokens_to_string(self, tokens: list[str]) -> str:
        return join_public_emission_units(tokens, target_lang_code=self._target_lang_code)

    def process_chunk(self, waveform):
        session_result = self._session.process_audio_chunk(waveform)
        if session_result is None:
            return empty_incremental_output()
        translation, _ = self._session.apply_translation_emit_policy(
            self._current_emitted_text(),
            session_result.raw_translation_text,
            is_final=False,
        )
        return self._compute_incremental_output(translation)

    def end_of_stream(self):
        final_result = self._session.finalize_stream()
        translation, _ = self._session.apply_translation_emit_policy(
            self._current_emitted_text(),
            final_result.raw_translation_text,
            is_final=True,
        )
        return self._compute_incremental_output(translation)

    def _current_emitted_text(self) -> str:
        return self.tokens_to_string(self._emitted_units)

    def _compute_incremental_output(self, new_translation: str):
        new_units = split_public_emission_units(
            new_translation,
            target_lang_code=self._target_lang_code,
        )
        previous_units = self._emitted_units
        if not new_units:
            return empty_incremental_output()
        if len(new_units) < len(previous_units):
            return empty_incremental_output()
        if new_units[: len(previous_units)] != previous_units:
            return empty_incremental_output()
        added_units = new_units[len(previous_units) :]
        if not added_units:
            return empty_incremental_output()
        self._emitted_units = list(new_units)
        added_string = self.tokens_to_string(added_units)
        return append_only_incremental_output(
            new_tokens=[added_string],
            new_string=added_string,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--directions",
        nargs="+",
        default=list(DEFAULT_DIRECTIONS),
        help="Language directions like en-de en-zh en-it.",
    )
    parser.add_argument(
        "--regimes",
        nargs="+",
        default=list(DEFAULT_REGIMES),
        choices=("low", "high"),
        help="Latency regimes to evaluate.",
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_DEV_AUDIO_DIR),
        help="Input audio directory for the full ablation.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory for ablation outputs.",
    )
    parser.add_argument(
        "--baseline-root",
        default=str(DEFAULT_BASELINE_ROOT),
        help=(
            "Compact archived scores JSON, or a legacy directory containing "
            "<regime>/<direction>/evaluation.json bundles."
        ),
    )
    parser.add_argument(
        "--smoke-wav",
        default=str(DEFAULT_SMOKE_WAV),
        help="Optional single clip to validate the all-head path before the full sweep.",
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Skip the single-audio validation pass.",
    )
    parser.add_argument(
        "--evaluation-python",
        default=str(REPO_ROOT / ".venv-evaluation" / "bin" / "python"),
        help="Python executable for evaluate_cascade_outputs.py.",
    )
    parser.add_argument(
        "--inference-python",
        default=str(REPO_ROOT / ".venv-inference" / "bin" / "python"),
        help="Kept for provenance; the script itself should already run there.",
    )
    parser.add_argument("--single-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--summarize-existing", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--direction", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--regime", default=None, choices=("low", "high"), help=argparse.SUPPRESS)
    parser.add_argument("--heads-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--top-k", default=None, type=int, help=argparse.SUPPRESS)
    parser.add_argument("--smoke-only", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def load_head_payload_by_direction(direction: str) -> tuple[Path, dict[str, Any]]:
    source_code, target_code = direction.split("-", 1)
    head_path = REPO_ROOT / (
        "data/alignatt_heads/"
        f"translation_heads_google_gemma-4-E4B-it_{source_code}-{target_code}.json"
    )
    payload = json.loads(head_path.read_text(encoding="utf-8"))
    return head_path, payload


def materialize_all_heads_payload(
    *,
    direction: str,
    output_root: Path,
) -> tuple[Path, int]:
    source_path, payload = load_head_payload_by_direction(direction)
    ts_matrix = payload["ts_matrix"]
    num_layers = int(payload["num_layers"])
    num_heads = int(payload["num_heads"])
    all_heads: list[dict[str, Any]] = []
    for layer in range(num_layers):
        for head in range(num_heads):
            all_heads.append(
                {
                    "layer": layer,
                    "head": head,
                    "ts": round(float(ts_matrix[layer][head]), 6),
                    "count": int(payload.get("used_pairs", 0)),
                }
            )
    all_heads.sort(key=lambda row: (float(row["ts"]), -int(row["layer"]), -int(row["head"])), reverse=True)

    full_payload = dict(payload)
    full_payload["token_alignment_heads"] = all_heads
    full_payload["regime"] = "all_heads_from_ts_matrix"
    full_payload["source_payload"] = str(source_path.relative_to(REPO_ROOT))
    full_payload["all_head_count"] = len(all_heads)

    output_path = output_root / "head_files" / f"{direction}_all_heads.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(full_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path, len(all_heads)


def build_runtime_config(
    *,
    direction: str,
    regime: str,
    heads_path: Path,
    top_k: int,
) -> tuple[CascadeRuntimeConfig, int]:
    source_code, target_code = direction.split("-", 1)
    preset_name = "gemma_low_latency" if regime == "low" else "gemma_high_latency"
    preset = get_runtime_preset(preset_name)
    source_lang = LANGUAGE_CODE_TO_NAME[source_code]
    target_lang = LANGUAGE_CODE_TO_NAME[target_code]
    config = CascadeRuntimeConfig(
        source_lang=source_lang,
        target_lang=target_lang,
        alignment_backend_name=preset.alignment_backend_name,
        mt_backend_name=preset.mt_backend_name,
    )
    config.apply_overrides(
        min_start_seconds=preset.min_start_seconds,
        max_history_utterances=preset.max_history_utterances,
        partial_max_new_tokens=preset.partial_max_new_tokens,
        translation_alignatt_min_source_mass=preset.translation_alignatt_min_source_mass,
        translation_alignatt_border_margin=preset.translation_alignatt_border_margin,
        translation_alignatt_inaccessible_ms=preset.translation_alignatt_inaccessible_ms,
        translation_alignatt_argmax_mass_threshold=preset.translation_alignatt_argmax_mass_threshold,
        translation_alignatt_heads_path=str(heads_path),
        translation_alignatt_top_k_heads=int(top_k),
        mt_vllm_enforce_eager=preset.mt_vllm_enforce_eager,
        mt_vllm_cudagraph_mode=preset.mt_vllm_cudagraph_mode,
        mt_vllm_enable_prefix_caching=preset.mt_vllm_enable_prefix_caching,
        mt_vllm_gpu_memory_utilization=preset.mt_vllm_gpu_memory_utilization,
        paper_context_mode=preset.paper_context_mode,
        paper_context_top_k=preset.paper_context_top_k,
        paper_context_max_chars=preset.paper_context_max_chars,
        paper_context_history_window_words=preset.paper_context_history_window_words,
    )
    return config, int(preset.chunk_ms)


def build_processor_config(
    *,
    direction: str,
    regime: str,
    heads_path: Path,
    top_k: int,
) -> tuple[SimpleNamespace, int]:
    runtime_config, chunk_ms = build_runtime_config(
        direction=direction,
        regime=regime,
        heads_path=heads_path,
        top_k=top_k,
    )
    source_code, target_code = direction.split("-", 1)
    processor_config = SimpleNamespace(
        source_lang_code=source_code,
        target_lang_code=target_code,
        chunk_ms=chunk_ms,
        speech_chunk_size=chunk_ms / 1000.0,
        alignment_backend_name=runtime_config.alignment_backend_name,
        mt_backend_name=runtime_config.mt_backend_name,
        min_start_seconds=runtime_config.min_start_seconds,
        max_history_utterances=runtime_config.max_history_utterances,
        partial_max_new_tokens=runtime_config.partial_max_new_tokens,
        translation_alignatt_min_source_mass=runtime_config.translation_alignatt_min_source_mass,
        translation_alignatt_border_margin=runtime_config.translation_alignatt_border_margin,
        translation_alignatt_inaccessible_ms=runtime_config.translation_alignatt_inaccessible_ms,
        translation_alignatt_argmax_mass_threshold=runtime_config.translation_alignatt_argmax_mass_threshold,
        translation_alignatt_heads_path=str(heads_path),
        translation_alignatt_top_k_heads=int(top_k),
        translation_alignatt_filter_width=runtime_config.translation_alignatt_filter_width,
        translation_alignatt_probe_mode=runtime_config.translation_alignatt_probe_mode,
        mt_vllm_enforce_eager=runtime_config.mt_vllm_enforce_eager,
        mt_vllm_cudagraph_mode=runtime_config.mt_vllm_cudagraph_mode,
        mt_vllm_enable_prefix_caching=runtime_config.mt_vllm_enable_prefix_caching,
        mt_vllm_gpu_memory_utilization=runtime_config.mt_vllm_gpu_memory_utilization,
    )
    return processor_config, chunk_ms


def run_batch_with_hot_bundle(
    *,
    processor: ReusableCascadeProcessor,
    input_paths: list[str],
    output_dir: Path,
    chunk_ms: int,
    source_lang_code: str,
    target_lang_code: str,
) -> dict[str, Any]:
    print(
        f"Will process {len(input_paths)} media files for {source_lang_code}->{target_lang_code} "
        f"with all-head MT AlignAtt",
        flush=True,
    )
    all_hypothesis_records: list[dict[str, Any]] = []
    all_stream_updates: list[dict[str, Any]] = []
    per_input_results: list[dict[str, Any]] = []
    batch_start = perf_counter()

    for idx, input_path in enumerate(input_paths):
        print(f"\n[{idx+1}/{len(input_paths)}] {Path(input_path).name} ...", flush=True)
        result = run_single_audio(
            processor,
            input_path,
            chunk_ms,
            target_lang_code,
        )
        all_hypothesis_records.append(result["hypothesis_record"])
        all_stream_updates.extend(result["stream_updates"])
        per_input_results.append(
            {
                "input": result["input_name"],
                "audio_s": round(result["audio_duration_ms"] / 1000.0, 1),
                "rtf": round(result["rtf"], 4),
                "updates": result["num_updates"],
                "paper_context_path": None,
            }
        )
        print(
            f"  RTF={result['rtf']:.3f}  updates={result['num_updates']}  "
            f"wallclock={result['total_wallclock_s']:.1f}s",
            flush=True,
        )

    batch_wallclock_s = perf_counter() - batch_start
    total_audio_s = sum(entry["audio_s"] for entry in per_input_results)
    batch_rtf = batch_wallclock_s / total_audio_s if total_audio_s > 0 else 0.0

    runtime_config = {
        "chunk_ms": int(chunk_ms),
        "alignment_backend_name": processor.session.config.alignment_backend_name,
        "mt_backend_name": processor.session.config.mt_backend_name,
        "min_start_seconds": processor.session.config.min_start_seconds,
        "max_history_utterances": processor.session.config.max_history_utterances,
        "partial_max_new_tokens": processor.session.config.partial_max_new_tokens,
        "translation_alignatt_min_source_mass": processor.session.config.translation_alignatt_min_source_mass,
        "translation_alignatt_border_margin": processor.session.config.translation_alignatt_border_margin,
        "translation_alignatt_inaccessible_ms": processor.session.config.translation_alignatt_inaccessible_ms,
        "translation_alignatt_argmax_mass_threshold": processor.session.config.translation_alignatt_argmax_mass_threshold,
        "hypothesis_elapsed_semantics": HYPOTHESIS_ELAPSED_SEMANTICS_CA_COMPATIBLE,
        "stream_update_elapsed_semantics": STREAM_UPDATE_ELAPSED_SEMANTICS_WALLCLOCK,
        "translation_alignatt_heads_path": processor.session.config.translation_alignatt_heads_path,
        "translation_alignatt_top_k_heads": processor.session.config.translation_alignatt_top_k_heads,
        "translation_alignatt_filter_width": processor.session.config.translation_alignatt_filter_width,
        "translation_alignatt_probe_mode": processor.session.config.translation_alignatt_probe_mode,
        "gemma_audio_alignment_heads_path": processor.session.config.gemma_audio_alignment_heads_path,
        "translation_emit_policy": processor.session.config.translation_emit_policy,
        "translation_max_tail_rewrite_words": processor.session.config.translation_max_tail_rewrite_words,
        "temperature": processor.session.config.temperature,
        "repetition_penalty": processor.session.config.repetition_penalty,
        "mt_vllm_enforce_eager": processor.session.config.mt_vllm_enforce_eager,
        "mt_vllm_cudagraph_mode": processor.session.config.mt_vllm_cudagraph_mode,
        "mt_vllm_enable_prefix_caching": processor.session.config.mt_vllm_enable_prefix_caching,
        "mt_vllm_gpu_memory_utilization": processor.session.config.mt_vllm_gpu_memory_utilization,
    }

    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "generated_at_utc": utc_now_isoformat(),
        "kind": "inference_batch",
        "num_inputs": len(input_paths),
        "input_paths": input_paths,
        "num_audios": len(input_paths),
        "wav_paths": input_paths,
        "source_language": LANGUAGE_CODE_TO_NAME.get(source_lang_code, source_lang_code),
        "target_language": LANGUAGE_CODE_TO_NAME.get(target_lang_code, target_lang_code),
        "source_language_code": source_lang_code,
        "target_language_code": target_lang_code,
        "runtime_config": runtime_config,
        "run_provenance": {
            "git_sha": git_sha(),
            "framework_mode": "reusable_loaded_model_bundle",
            "script": "scripts/run_mt_head_ablation.py",
        },
        "speed": {
            "batch_wallclock_s": round(batch_wallclock_s, 2),
            "batch_rtf": round(batch_rtf, 4),
            "total_audio_s": round(total_audio_s, 1),
            "per_input": per_input_results,
            "per_audio": per_input_results,
        },
    }

    output_path = ensure_output_dir(str(output_dir))
    write_json(Path(output_path) / MANIFEST_FILENAME, manifest)
    write_jsonl(Path(output_path) / HYPOTHESIS_FILENAME, all_hypothesis_records)
    write_jsonl(Path(output_path) / STREAM_UPDATES_FILENAME, all_stream_updates)

    print(
        f"\nBatch complete: {len(input_paths)} inputs, {total_audio_s:.0f}s total audio",
        flush=True,
    )
    print(
        f"Batch wallclock: {batch_wallclock_s:.1f}s  RTF: {batch_rtf:.4f}",
        flush=True,
    )
    print(f"Artifacts: {output_dir}", flush=True)
    return manifest


def evaluate_output_dir(
    *,
    output_dir: Path,
    evaluation_python: str,
    target_lang_code: str,
) -> dict[str, Any]:
    cmd = [
        evaluation_python,
        "evaluate_cascade_outputs.py",
        "--output-dir",
        str(output_dir),
        "--target-lang-code",
        target_lang_code,
    ]
    subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        check=True,
    )
    evaluation_path = output_dir / "evaluation.json"
    return json.loads(evaluation_path.read_text(encoding="utf-8"))


def load_baseline_evaluation(*, baseline_root: Path, regime: str, direction: str) -> dict[str, Any]:
    if baseline_root.is_file():
        archive = json.loads(baseline_root.read_text(encoding="utf-8"))
        try:
            return archive["regimes"][regime][direction]
        except KeyError as exc:
            raise KeyError(
                f"Missing archived baseline scores for {regime} {direction} "
                f"in {baseline_root}"
            ) from exc

    path = baseline_root / regime / direction / "evaluation.json"
    return json.loads(path.read_text(encoding="utf-8"))


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def optional_delta(left: Any, right: Any) -> float | None:
    left_value = optional_float(left)
    right_value = optional_float(right)
    if left_value is None or right_value is None:
        return None
    return left_value - right_value


def compute_delta_table_row(
    *,
    regime: str,
    direction: str,
    baseline_eval: dict[str, Any],
    all_heads_eval: dict[str, Any],
    all_head_count: int,
) -> dict[str, Any]:
    base_scores = baseline_eval["contract_scores"]
    all_scores = all_heads_eval["contract_scores"]
    return {
        "regime": regime,
        "direction": direction,
        "all_head_count": int(all_head_count),
        "baseline_bleu": float(base_scores["BLEU"]),
        "all_heads_bleu": float(all_scores["BLEU"]),
        "delta_bleu": float(base_scores["BLEU"]) - float(all_scores["BLEU"]),
        "baseline_chrf": float(base_scores["CHRF"]),
        "all_heads_chrf": float(all_scores["CHRF"]),
        "delta_chrf": float(base_scores["CHRF"]) - float(all_scores["CHRF"]),
        "baseline_xcometxl": optional_float(base_scores["XCOMETXL"]),
        "all_heads_xcometxl": optional_float(all_scores["XCOMETXL"]),
        "delta_xcometxl": optional_delta(base_scores["XCOMETXL"], all_scores["XCOMETXL"]),
        "baseline_longyaal_cu_ms": float(base_scores["LongYAAL CU"]),
        "all_heads_longyaal_cu_ms": float(all_scores["LongYAAL CU"]),
        "delta_longyaal_cu_ms": float(base_scores["LongYAAL CU"]) - float(all_scores["LongYAAL CU"]),
        "baseline_longyaal_ca_ms": optional_float(base_scores.get("LongYAAL CA")),
        "all_heads_longyaal_ca_ms": optional_float(all_scores.get("LongYAAL CA")),
        "delta_longyaal_ca_ms": optional_delta(
            base_scores.get("LongYAAL CA"),
            all_scores.get("LongYAAL CA"),
        ),
    }


def format_optional_tsv(value: Any) -> str:
    maybe_float = optional_float(value)
    if maybe_float is None:
        return ""
    return f"{maybe_float:.6f}"


def format_optional_console(value: Any, *, precision: int) -> str:
    maybe_float = optional_float(value)
    if maybe_float is None:
        return "n/a"
    return f"{maybe_float:+.{precision}f}"


def write_summary_files(summary_rows: list[dict[str, Any]], output_root: Path) -> None:
    summary_root = output_root / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)
    json_path = summary_root / "summary.json"
    json_path.write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    tsv_lines = [
        "\t".join(
            [
                "regime",
                "direction",
                "all_head_count",
                "baseline_bleu",
                "all_heads_bleu",
                "delta_bleu",
                "baseline_chrf",
                "all_heads_chrf",
                "delta_chrf",
                "baseline_xcometxl",
                "all_heads_xcometxl",
                "delta_xcometxl",
                "baseline_longyaal_cu_ms",
                "all_heads_longyaal_cu_ms",
                "delta_longyaal_cu_ms",
                "baseline_longyaal_ca_ms",
                "all_heads_longyaal_ca_ms",
                "delta_longyaal_ca_ms",
            ]
        )
    ]
    for row in summary_rows:
        tsv_lines.append(
            "\t".join(
                [
                    str(row["regime"]),
                    str(row["direction"]),
                    str(row["all_head_count"]),
                    f"{row['baseline_bleu']:.6f}",
                    f"{row['all_heads_bleu']:.6f}",
                    f"{row['delta_bleu']:.6f}",
                    f"{row['baseline_chrf']:.6f}",
                    f"{row['all_heads_chrf']:.6f}",
                    f"{row['delta_chrf']:.6f}",
                    format_optional_tsv(row["baseline_xcometxl"]),
                    format_optional_tsv(row["all_heads_xcometxl"]),
                    format_optional_tsv(row["delta_xcometxl"]),
                    f"{row['baseline_longyaal_cu_ms']:.6f}",
                    f"{row['all_heads_longyaal_cu_ms']:.6f}",
                    f"{row['delta_longyaal_cu_ms']:.6f}",
                    format_optional_tsv(row["baseline_longyaal_ca_ms"]),
                    format_optional_tsv(row["all_heads_longyaal_ca_ms"]),
                    format_optional_tsv(row["delta_longyaal_ca_ms"]),
                ]
            )
        )
    (summary_root / "summary.tsv").write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")


def print_summary(summary_rows: list[dict[str, Any]]) -> None:
    print("\nMT head-filtering summary (top-8 baseline minus all-head averaging):", flush=True)
    for row in summary_rows:
        print(
            f"  {row['regime']:>4} {row['direction']}: "
            f"delta_BLEU={row['delta_bleu']:+.2f}  "
            f"delta_chrF={row['delta_chrf']:+.2f}  "
            f"delta_XCOMETXL={format_optional_console(row['delta_xcometxl'], precision=4)}  "
            f"delta_LongYAAL_CU_ms={format_optional_console(row['delta_longyaal_cu_ms'], precision=1)}  "
            f"delta_LongYAAL_CA_ms={format_optional_console(row['delta_longyaal_ca_ms'], precision=1)}",
            flush=True,
        )


def summarize_existing_runs(
    *,
    args: argparse.Namespace,
    baseline_root: Path,
    output_root: Path,
    head_files: dict[str, tuple[Path, int]],
) -> None:
    summary_rows: list[dict[str, Any]] = []
    for regime in args.regimes:
        for direction in args.directions:
            _heads_path, head_count = head_files[direction]
            run_dir = output_root / regime / direction / "all_heads"
            evaluation_path = run_dir / "evaluation.json"
            if not evaluation_path.exists():
                raise FileNotFoundError(
                    f"Missing evaluation bundle for {regime} {direction}: {evaluation_path}"
                )
            all_heads_eval = json.loads(evaluation_path.read_text(encoding="utf-8"))
            baseline_eval = load_baseline_evaluation(
                baseline_root=baseline_root,
                regime=regime,
                direction=direction,
            )
            summary_rows.append(
                compute_delta_table_row(
                    regime=regime,
                    direction=direction,
                    baseline_eval=baseline_eval,
                    all_heads_eval=all_heads_eval,
                    all_head_count=head_count,
                )
            )

    write_summary_files(summary_rows, output_root=output_root)
    print_summary(summary_rows)


def launch_isolated_run(
    *,
    args: argparse.Namespace,
    direction: str,
    regime: str,
    heads_path: Path,
    top_k: int,
    smoke_only: bool,
) -> Path:
    run_dir = Path(args.output_root) / regime / direction / "all_heads"
    cmd = [
        args.inference_python,
        str(Path(__file__).resolve()),
        "--single-run",
        "--direction",
        direction,
        "--regime",
        regime,
        "--heads-path",
        str(heads_path),
        "--top-k",
        str(int(top_k)),
        "--output-root",
        str(args.output_root),
        "--input-dir",
        str(args.input_dir),
        "--baseline-root",
        str(args.baseline_root),
        "--evaluation-python",
        str(args.evaluation_python),
        "--inference-python",
        str(args.inference_python),
    ]
    if smoke_only:
        cmd.extend(["--smoke-only", "--smoke-wav", str(args.smoke_wav)])
    subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        check=True,
    )
    return run_dir


def run_single_mode(args: argparse.Namespace) -> None:
    if not args.direction or not args.regime or not args.heads_path or args.top_k is None:
        raise SystemExit("--single-run requires --direction, --regime, --heads-path, and --top-k.")
    processor_config, _chunk_ms = build_processor_config(
        direction=args.direction,
        regime=args.regime,
        heads_path=Path(args.heads_path),
        top_k=int(args.top_k),
    )
    source_code, target_code = args.direction.split("-", 1)
    input_paths = (
        [str(Path(args.smoke_wav))]
        if args.smoke_only
        else resolve_input_paths(inputs=None, input_dir=args.input_dir)
    )
    run_dir = Path(args.output_root) / args.regime / args.direction / "all_heads"
    if args.smoke_only:
        run_dir = run_dir / "smoke"
    run_batch_inference(
        processor_config=processor_config,
        input_paths=input_paths,
        output_dir=str(run_dir),
        source_lang_code=source_code,
        target_lang_code=target_code,
    )
    if not args.smoke_only:
        evaluate_output_dir(
            output_dir=run_dir,
            evaluation_python=args.evaluation_python,
            target_lang_code=target_code,
        )


def run_smoke_validation(
    *,
    processor: ReusableCascadeProcessor,
    smoke_wav: Path,
    direction: str,
    chunk_ms: int,
) -> None:
    print(
        f"Smoke validation: {smoke_wav.name} with all-head MT for {direction} at chunk_ms={chunk_ms}",
        flush=True,
    )
    result = run_single_audio(
        processor,
        str(smoke_wav),
        int(chunk_ms),
        direction.split("-", 1)[1],
    )
    print(
        f"Smoke pass complete: updates={result['num_updates']} "
        f"rtf={result['rtf']:.3f} wallclock={result['total_wallclock_s']:.1f}s",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.single_run:
        run_single_mode(args)
        return
    output_root = Path(args.output_root)
    baseline_root = Path(args.baseline_root)

    head_files: dict[str, tuple[Path, int]] = {
        direction: materialize_all_heads_payload(direction=direction, output_root=output_root)
        for direction in args.directions
    }

    if args.summarize_existing:
        summarize_existing_runs(
            args=args,
            baseline_root=baseline_root,
            output_root=output_root,
            head_files=head_files,
        )
        return

    first_direction = args.directions[0]

    if not args.skip_smoke:
        first_heads_path, first_head_count = head_files[first_direction]
        launch_isolated_run(
            args=args,
            direction=first_direction,
            regime=args.regimes[0],
            heads_path=first_heads_path,
            top_k=first_head_count,
            smoke_only=True,
        )

    summary_rows: list[dict[str, Any]] = []
    for regime in args.regimes:
        for direction in args.directions:
            heads_path, head_count = head_files[direction]
            print(
                f"\n=== Running {regime} {direction} with {head_count} MT heads ===",
                flush=True,
            )
            run_dir = launch_isolated_run(
                args=args,
                direction=direction,
                regime=regime,
                heads_path=heads_path,
                top_k=head_count,
                smoke_only=False,
            )
            _source_code, target_code = direction.split("-", 1)
            all_heads_eval = json.loads(
                (run_dir / "evaluation.json").read_text(encoding="utf-8")
            )
            baseline_eval = load_baseline_evaluation(
                baseline_root=baseline_root,
                regime=regime,
                direction=direction,
            )
            summary_rows.append(
                compute_delta_table_row(
                    regime=regime,
                    direction=direction,
                    baseline_eval=baseline_eval,
                    all_heads_eval=all_heads_eval,
                    all_head_count=head_count,
                )
            )

    write_summary_files(summary_rows, output_root=output_root)
    print_summary(summary_rows)


if __name__ == "__main__":
    main()
