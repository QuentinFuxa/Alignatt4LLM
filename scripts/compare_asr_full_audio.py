#!/usr/bin/env python3
"""Unified ASR evaluation harness for Gemma/Qwen short-form and streaming runs.

This script deliberately stays on the ASR side only. It provides one
reproducible harness for two evaluation regimes:

- ``segmented_ref``: evaluate independent reference segments without any
  streaming logic, to estimate the short-form ceiling.
- ``streaming_full``: evaluate the real SimulStream ASR runtime on a full or
  truncated talk, with commit events, cut metrics, latency, and per-update
  streaming traces.

Typical usage:

    PYTHONPATH=. .venv-inference/bin/python scripts/compare_asr_full_audio.py run \
        --wav data/devset/audio/ccpXHNfaoy.wav \
        --eval-mode both \
        --output-dir outputs/asr_compare_ccpXHNfaoy

    PYTHONPATH=. python3 scripts/compare_asr_full_audio.py plot \
        --summary outputs/asr_compare_ccpXHNfaoy/summary.json
"""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path
import re
from statistics import mean, median
import string
import sys
from time import perf_counter
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SAMPLE_RATE = 16000
DEFAULT_WAV = "data/devset/audio/ccpXHNfaoy.wav"
DEFAULT_SEGMENTS = "data/devset/audio-segments.yaml"
DEFAULT_SOURCE_REF = "data/devset/ref/en.txt"
DEFAULT_OUTPUT_DIR = "outputs/asr_compare_ccpXHNfaoy"
DEFAULT_BACKENDS = ("qwen_forced", "gemma_vllm_qk_fast")
ALL_BACKENDS = ("qwen_forced", "gemma_vllm_qk_fast")
EVAL_MODE_CHOICES = ("segmented_ref", "streaming_full", "both")
VLLM_PATH_MODE_CHOICES = ("shipping", "chat", "generate")
GEMMA_SAMPLING_MODE_CHOICES = ("shipping", "hf_model_card")

PUNCT_TABLE = str.maketrans("", "", string.punctuation + "“”‘’")
PUNCT_TRAILING = string.punctuation + "”’)]}"
PUNCT_LEADING = "\"'`“”‘’([{"
META_RESPONSE_PATTERNS = (
    "please provide the audio",
    "please provide the speech",
    "please provide the speech segment",
    "based on the prompt",
    "cannot perform the transcription",
    "can't perform the transcription",
    "i cannot transcribe",
    "i can't transcribe",
    "unable to transcribe",
    "would you like me to transcribe",
    "need the audio",
    "need a speech segment",
    "provide a verbatim, word-for-word transcription",
    "word-for-word transcription of the audio",
    "only output the transcription",
    "follow these specific instructions for formatting the answer",
)


def _strip_word_surface(raw: str) -> str:
    start = 0
    end = len(raw)
    while start < end and raw[start] in PUNCT_LEADING:
        start += 1
    while end > start and raw[end - 1] in PUNCT_TRAILING:
        end -= 1
    if start >= end:
        return ""
    return raw[start:end]


def lexical_word_spans(text: str) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for match in re.finditer(r"\S+", text):
        raw = match.group(0)
        leading = 0
        trailing = len(raw)
        while leading < trailing and raw[leading] in PUNCT_LEADING:
            leading += 1
        while trailing > leading and raw[trailing - 1] in PUNCT_TRAILING:
            trailing -= 1
        if leading >= trailing:
            continue
        spans.append(
            {
                "surface": raw[leading:trailing],
                "surface_lower": raw[leading:trailing].lower(),
                "char_start": match.start() + leading,
                "char_end": match.start() + trailing,
            }
        )
    return spans


def tokenize_words(text: str) -> list[str]:
    return [span["surface_lower"] for span in lexical_word_spans(text)]


def normalize_for_error_rate(text: str) -> list[str]:
    return text.lower().translate(PUNCT_TABLE).split()


def levenshtein_distance(ref: list[str], hyp: list[str]) -> int:
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    previous = list(range(len(hyp) + 1))
    for i, ref_item in enumerate(ref, start=1):
        current = [i] + [0] * len(hyp)
        for j, hyp_item in enumerate(hyp, start=1):
            cost = 0 if ref_item == hyp_item else 1
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            )
        previous = current
    return previous[-1]


def compute_wer(reference_text: str, hypothesis_text: str) -> float:
    ref = normalize_for_error_rate(reference_text)
    hyp = normalize_for_error_rate(hypothesis_text)
    return levenshtein_distance(ref, hyp) / max(1, len(ref))


def compute_cer(reference_text: str, hypothesis_text: str) -> float:
    ref = list("".join(normalize_for_error_rate(reference_text)))
    hyp = list("".join(normalize_for_error_rate(hypothesis_text)))
    return levenshtein_distance(ref, hyp) / max(1, len(ref))


def build_word_projection(ref_words: list[str], hyp_words: list[str]) -> dict[int, int]:
    matcher = difflib.SequenceMatcher(a=ref_words, b=hyp_words, autojunk=False)
    hyp_to_ref: dict[int, int] = {}
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for delta in range(i2 - i1):
                hyp_to_ref[j1 + delta] = i1 + delta
            continue
        if tag == "replace":
            paired = min(i2 - i1, j2 - j1)
            for delta in range(paired):
                hyp_to_ref[j1 + delta] = i1 + delta
    return hyp_to_ref


def cumulative_word_boundaries(texts: list[str]) -> list[int]:
    total = 0
    boundaries: list[int] = []
    for text in texts:
        total += len(tokenize_words(text))
        boundaries.append(total)
    return boundaries


def greedy_match_boundaries(
    predicted: list[int],
    reference: list[int],
    *,
    tolerance_words: int,
) -> dict[str, Any]:
    used_reference: set[int] = set()
    matches: list[dict[str, int]] = []
    for predicted_boundary in sorted(predicted):
        best_idx: int | None = None
        best_distance: int | None = None
        for ref_idx, reference_boundary in enumerate(reference):
            if ref_idx in used_reference:
                continue
            distance = abs(predicted_boundary - reference_boundary)
            if distance > tolerance_words:
                continue
            if best_idx is None or distance < best_distance:
                best_idx = ref_idx
                best_distance = distance
        if best_idx is None:
            continue
        used_reference.add(best_idx)
        matches.append(
            {
                "predicted_boundary_words": int(predicted_boundary),
                "reference_boundary_words": int(reference[best_idx]),
                "reference_boundary_index": int(best_idx),
                "distance_words": int(best_distance),
            }
        )
    precision = len(matches) / max(1, len(predicted))
    recall = len(matches) / max(1, len(reference))
    f1 = 0.0 if (precision + recall) == 0.0 else 2.0 * precision * recall / (precision + recall)
    return {
        "predicted_count": len(predicted),
        "reference_count": len(reference),
        "matched_count": len(matches),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matches": matches,
    }


def summarize_lags_s(lags_s: list[float]) -> dict[str, float | None]:
    if not lags_s:
        return {
            "mean_s": None,
            "median_s": None,
            "p90_s": None,
            "mean_abs_s": None,
            "median_abs_s": None,
            "p90_abs_s": None,
            "max_abs_s": None,
        }
    sorted_signed = sorted(lags_s)
    sorted_abs = sorted(abs(x) for x in lags_s)
    p90_index = min(len(sorted_signed) - 1, int(0.9 * len(sorted_signed)))
    p90_abs_index = min(len(sorted_abs) - 1, int(0.9 * len(sorted_abs)))
    return {
        "mean_s": mean(lags_s),
        "median_s": median(lags_s),
        "p90_s": sorted_signed[p90_index],
        "mean_abs_s": mean(sorted_abs),
        "median_abs_s": median(sorted_abs),
        "p90_abs_s": sorted_abs[p90_abs_index],
        "max_abs_s": sorted_abs[-1],
    }


def is_meta_response(text: str) -> bool:
    normalized = " ".join(str(text).strip().lower().split())
    if not normalized:
        return False
    return any(pattern in normalized for pattern in META_RESPONSE_PATTERNS)


def _sanitize_run_component(value: Any) -> str:
    text = str(value).strip().replace(".", "p")
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", text).strip("-") or "x"


def label_for_backend(backend_name: str) -> str:
    if backend_name == "qwen_forced":
        return "Qwen3 ASR + Forced Aligner"
    if backend_name == "gemma_vllm_qk_fast":
        return "Gemma 4 ASR vLLM AlignAtt"
    return backend_name


def build_run_id(
    *,
    backend_name: str,
    eval_mode: str,
    gemma_vllm_path_mode: str,
    gemma_sampling_mode: str,
    asr_alignatt_frame_threshold: int,
    asr_alignatt_rewind_threshold: int,
) -> str:
    parts = [backend_name]
    if backend_name == "gemma_vllm_qk_fast":
        if gemma_vllm_path_mode != "shipping":
            parts.append(f"path-{gemma_vllm_path_mode}")
        if gemma_sampling_mode != "shipping":
            parts.append(f"sm-{gemma_sampling_mode}")
        if eval_mode == "streaming_full":
            parts.append(f"ft-{int(asr_alignatt_frame_threshold)}")
            parts.append(f"rt-{int(asr_alignatt_rewind_threshold)}")
    return "__".join(_sanitize_run_component(part) for part in parts)


def build_run_label(
    *,
    backend_name: str,
    run_id: str,
) -> str:
    base = label_for_backend(backend_name)
    if run_id == backend_name:
        return base
    suffix = run_id[len(backend_name) :].lstrip("_")
    suffix = suffix.replace("__", " | ").replace("_", " ")
    return f"{base} | {suffix}"


def segment_selection_rank(run: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(run.get("meta_response_count", 0)),
        float(run["metrics"]["wer"]),
        float(run.get("mean_segment_wer", run["metrics"]["wer"])),
        float(run["metrics"]["cer"]),
    )


def streaming_selection_rank(
    run: dict[str, Any],
    *,
    tolerance_words: int,
) -> tuple[float, float, float, float]:
    tol_key = f"tolerance_{int(tolerance_words)}w"
    return (
        float(run.get("meta_response_count", 0)),
        abs(
            float(run["metrics"]["predicted_boundary_count"])
            - float(run["metrics"]["reference_boundary_count"])
        ),
        float(run["metrics"]["wer"]),
        -float(run["metrics"]["boundary_metrics"][tol_key]["f1"]),
    )


def rank_to_jsonable(rank: tuple[float, ...]) -> list[float]:
    return [float(item) for item in rank]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _reference_wav_candidates(wav_name: str) -> list[str]:
    candidates = [wav_name]
    if wav_name.endswith("_short60s.wav"):
        candidates.append(wav_name.replace("_short60s.wav", ".wav"))
    unique: list[str] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def load_reference_segments(
    *,
    wav_path: str,
    segments_path: str,
    source_ref_path: str,
) -> dict[str, Any]:
    import soundfile as sf
    import yaml

    requested_wav_name = Path(wav_path).name
    clip_duration_s = float(sf.info(wav_path).duration)

    segments = yaml.safe_load(Path(segments_path).read_text(encoding="utf-8"))
    references = Path(source_ref_path).read_text(encoding="utf-8").splitlines()
    if len(segments) != len(references):
        raise ValueError("audio-segments.yaml and source reference must have matching lengths.")

    selected_reference_wav_name: str | None = None
    matching_pairs: list[tuple[dict[str, Any], str]] = []
    for candidate_name in _reference_wav_candidates(requested_wav_name):
        candidate_pairs = [
            (dict(segment), text.strip())
            for segment, text in zip(segments, references)
            if Path(str(segment["wav"])).name == candidate_name
        ]
        if candidate_pairs:
            selected_reference_wav_name = candidate_name
            matching_pairs = candidate_pairs
            break
    if not matching_pairs or selected_reference_wav_name is None:
        raise ValueError(
            f"No reference segments found for {requested_wav_name} in {segments_path}."
        )

    clipped_pairs: list[tuple[dict[str, Any], str]] = []
    dropped_tail_segments = 0
    for segment, text in matching_pairs:
        end_time_s = float(segment["offset"]) + float(segment["duration"])
        if end_time_s <= clip_duration_s + 1e-6:
            clipped_pairs.append((segment, text))
        else:
            dropped_tail_segments += 1
    if not clipped_pairs:
        raise ValueError(
            f"Reference for {requested_wav_name} exists, but no segment fits inside "
            f"the {clip_duration_s:.3f}s clip."
        )

    filtered_segments = [pair[0] for pair in clipped_pairs]
    filtered_texts = [pair[1] for pair in clipped_pairs]
    end_times_s = [
        float(segment["offset"]) + float(segment["duration"])
        for segment in filtered_segments
    ]
    full_text = " ".join(text for text in filtered_texts if text).strip()
    full_text_spans = lexical_word_spans(full_text)

    timed_words: list[dict[str, Any]] = []
    for segment, text in zip(filtered_segments, filtered_texts):
        segment_spans = lexical_word_spans(text)
        if not segment_spans:
            continue
        segment_start_s = float(segment["offset"])
        segment_end_s = segment_start_s + float(segment["duration"])
        step_s = (segment_end_s - segment_start_s) / max(1, len(segment_spans))
        for word_idx, span in enumerate(segment_spans):
            timed_words.append(
                {
                    "surface": span["surface"],
                    "surface_lower": span["surface_lower"],
                    "absolute_start_time_s": segment_start_s + word_idx * step_s,
                    "absolute_end_time_s": segment_start_s + (word_idx + 1) * step_s,
                }
            )
    if len(full_text_spans) != len(timed_words):
        raise ValueError(
            "Reference word timeline mismatch: lexical token count differs between "
            "the concatenated transcript and the per-segment reconstruction."
        )

    word_timeline: list[dict[str, Any]] = []
    for word_index, (span, timed_word) in enumerate(
        zip(full_text_spans, timed_words),
        start=1,
    ):
        word_timeline.append(
            {
                "word_index": int(word_index),
                "text": timed_word["surface"],
                "text_lower": timed_word["surface_lower"],
                "char_start": int(span["char_start"]),
                "char_end": int(span["char_end"]),
                "absolute_start_time_s": float(timed_word["absolute_start_time_s"]),
                "absolute_end_time_s": float(timed_word["absolute_end_time_s"]),
            }
        )

    return {
        "requested_wav_name": requested_wav_name,
        "reference_wav_name": selected_reference_wav_name,
        "clip_duration_s": clip_duration_s,
        "segment_count": len(filtered_segments),
        "tail_dropped_segment_count": dropped_tail_segments,
        "segments": filtered_segments,
        "texts": filtered_texts,
        "full_text": full_text,
        "word_count": len(tokenize_words(full_text)),
        "boundary_word_positions": cumulative_word_boundaries(filtered_texts),
        "boundary_end_times_s": end_times_s,
        "word_timeline": word_timeline,
    }


def public_reference_summary(reference_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "requested_wav_name": reference_payload["requested_wav_name"],
        "reference_wav_name": reference_payload["reference_wav_name"],
        "clip_duration_s": float(reference_payload["clip_duration_s"]),
        "segment_count": int(reference_payload["segment_count"]),
        "tail_dropped_segment_count": int(reference_payload["tail_dropped_segment_count"]),
        "word_count": int(reference_payload["word_count"]),
        "boundary_word_positions": list(reference_payload["boundary_word_positions"]),
        "boundary_end_times_s": list(reference_payload["boundary_end_times_s"]),
    }


def compare_backend_to_reference(
    *,
    backend_payload: dict[str, Any],
    reference_payload: dict[str, Any],
    tolerance_words: int,
) -> dict[str, Any]:
    reference_text = reference_payload["full_text"]
    backend_text = backend_payload["final_asr_text"]
    reference_words = tokenize_words(reference_text)
    predicted_words = tokenize_words(backend_text)
    hyp_to_ref = build_word_projection(reference_words, predicted_words)

    predicted_segment_texts = backend_payload["committed_texts"]
    predicted_boundaries_hyp = cumulative_word_boundaries(predicted_segment_texts)
    projected_boundaries_ref: list[int] = []
    projected_boundary_details: list[dict[str, Any]] = []

    for boundary_idx, predicted_boundary in enumerate(predicted_boundaries_hyp):
        if predicted_boundary <= 0:
            continue
        hyp_word_idx = predicted_boundary - 1
        ref_word_idx = hyp_to_ref.get(hyp_word_idx)
        if ref_word_idx is None:
            projected_boundary_details.append(
                {
                    "predicted_segment_index": int(boundary_idx),
                    "predicted_boundary_words": int(predicted_boundary),
                    "projected_reference_boundary_words": None,
                }
            )
            continue
        projected_boundary = ref_word_idx + 1
        projected_boundaries_ref.append(projected_boundary)
        projected_boundary_details.append(
            {
                "predicted_segment_index": int(boundary_idx),
                "predicted_boundary_words": int(predicted_boundary),
                "projected_reference_boundary_words": int(projected_boundary),
            }
        )

    reference_boundaries = list(reference_payload["boundary_word_positions"])
    exact_match = greedy_match_boundaries(
        projected_boundaries_ref,
        reference_boundaries,
        tolerance_words=0,
    )
    tolerance_match = greedy_match_boundaries(
        projected_boundaries_ref,
        reference_boundaries,
        tolerance_words=int(tolerance_words),
    )

    commit_events = backend_payload["commit_events"]
    commit_times_s = [float(event["end_time_s"]) for event in commit_events]
    projected_boundary_to_time_s: dict[int, float] = {}
    projected_boundary_to_segment_index: dict[int, int] = {}
    for projected_detail, end_time_s in zip(projected_boundary_details, commit_times_s):
        projected = projected_detail["projected_reference_boundary_words"]
        if projected is None:
            continue
        projected_boundary_to_time_s[int(projected)] = float(end_time_s)
        projected_boundary_to_segment_index[int(projected)] = int(
            projected_detail["predicted_segment_index"]
        )

    matched_lags_s: list[float] = []
    matched_lag_points: list[dict[str, Any]] = []
    for match in tolerance_match["matches"]:
        reference_boundary_words = int(match["reference_boundary_words"])
        reference_boundary_index = int(match["reference_boundary_index"])
        predicted_time_s = projected_boundary_to_time_s.get(reference_boundary_words)
        if predicted_time_s is None:
            continue
        reference_time_s = float(reference_payload["boundary_end_times_s"][reference_boundary_index])
        lag_s = predicted_time_s - reference_time_s
        matched_lags_s.append(lag_s)
        matched_lag_points.append(
            {
                "reference_segment_index": reference_boundary_index + 1,
                "reference_boundary_words": reference_boundary_words,
                "reference_end_time_s": reference_time_s,
                "predicted_end_time_s": predicted_time_s,
                "lag_s": lag_s,
                "distance_words": int(match["distance_words"]),
                "predicted_segment_index": projected_boundary_to_segment_index[reference_boundary_words] + 1,
            }
        )

    return {
        "wer": compute_wer(reference_text, backend_text),
        "cer": compute_cer(reference_text, backend_text),
        "reference_word_count": len(reference_words),
        "predicted_word_count": len(predicted_words),
        "predicted_boundary_count": len(predicted_boundaries_hyp),
        "reference_boundary_count": len(reference_boundaries),
        "projected_boundary_details": projected_boundary_details,
        "boundary_metrics": {
            "exact": exact_match,
            f"tolerance_{int(tolerance_words)}w": tolerance_match,
        },
        "lag_summary": summarize_lags_s(matched_lags_s),
        "lag_points": matched_lag_points,
    }


def maybe_configure_gemma_vllm_experiment(
    config,
    *,
    gemma_vllm_path_mode: str,
    gemma_sampling_mode: str,
) -> None:
    if config.alignment_backend_name != "gemma_vllm_qk_fast":
        return
    if gemma_vllm_path_mode not in VLLM_PATH_MODE_CHOICES:
        raise ValueError(
            f"Unknown gemma_vllm_path_mode {gemma_vllm_path_mode!r}; "
            f"expected one of {VLLM_PATH_MODE_CHOICES}."
        )
    if gemma_sampling_mode != "shipping":
        setattr(config, "gemma_vllm_sampling_mode", gemma_sampling_mode)


def maybe_warmup_backend(
    *,
    backend_name: str,
    alignment_backend,
    warmup_seconds: float,
) -> float:
    if (
        backend_name == "gemma_vllm_qk_fast"
        and warmup_seconds > 0.0
        and hasattr(alignment_backend, "warmup")
    ):
        warmup_start = perf_counter()
        alignment_backend.warmup(duration_seconds=float(warmup_seconds))
        return perf_counter() - warmup_start
    return 0.0


def run_segmented_reference_eval(
    *,
    wav_path: str,
    backend_name: str,
    reference_payload: dict[str, Any],
    run_id: str,
    run_label: str,
    min_start_seconds: float,
    warmup_seconds: float,
    match_tolerance_words: int,
    gemma_vllm_path_mode: str,
    gemma_sampling_mode: str,
    gemma_max_model_len: int | None,
    segment_max_count: int | None,
    segment_indices: list[int] | None,
) -> dict[str, Any]:
    import numpy as np

    from cascade.audio import load_audio_mono_16khz
    from cascade.runtime import CascadeRuntimeConfig, LoadedModelBundle

    config = CascadeRuntimeConfig(
        source_lang="English",
        target_lang="German",
        alignment_backend_name=backend_name,
    )
    config.min_start_seconds = float(min_start_seconds)
    if backend_name == "gemma_vllm_qk_fast":
        if gemma_max_model_len is not None:
            config.gemma_max_model_len = int(gemma_max_model_len)
        maybe_configure_gemma_vllm_experiment(
            config,
            gemma_vllm_path_mode=gemma_vllm_path_mode,
            gemma_sampling_mode=gemma_sampling_mode,
        )

    bundle = LoadedModelBundle(config)
    load_start = perf_counter()
    alignment_backend = bundle.ensure_alignment_backend()
    load_s = perf_counter() - load_start
    warmup_s = maybe_warmup_backend(
        backend_name=backend_name,
        alignment_backend=alignment_backend,
        warmup_seconds=warmup_seconds,
    )
    if hasattr(alignment_backend, "reset_caches"):
        alignment_backend.reset_caches()

    audio = load_audio_mono_16khz(wav_path)
    processing_start = perf_counter()
    segment_details: list[dict[str, Any]] = []
    committed_texts: list[str] = []
    commit_events: list[dict[str, Any]] = []

    pairs = list(zip(reference_payload["segments"], reference_payload["texts"]))
    if segment_indices:
        selected_indices = {int(index) for index in segment_indices}
        pairs = [
            pair
            for segment_index, pair in enumerate(pairs, start=1)
            if segment_index in selected_indices
        ]
    if segment_max_count is not None:
        pairs = pairs[: int(segment_max_count)]

    for segment_index, (segment, reference_text) in enumerate(pairs, start=1):
        start_s = float(segment["offset"])
        duration_s = float(segment["duration"])
        end_s = start_s + duration_s
        start_sample = int(round(start_s * SAMPLE_RATE))
        end_sample = int(round(end_s * SAMPLE_RATE))
        audio_slice = np.asarray(audio[start_sample:end_sample], dtype=np.float32)

        error_payload: dict[str, Any] | None = None
        hypothesis_text = ""
        result_diagnostics: dict[str, Any] | None = None
        if hasattr(alignment_backend, "reset_streaming_state"):
            alignment_backend.reset_streaming_state()
        try:
            result = alignment_backend.transcribe_and_align(
                audio_slice,
                sample_rate=SAMPLE_RATE,
                language="English",
            )
            if result is not None:
                hypothesis_text = str(result.text).strip()
                result_diagnostics = dict(result.diagnostics)
        except Exception as exc:  # noqa: BLE001 - keep the artifact durable
            error_payload = {"type": type(exc).__name__, "message": str(exc)}

        segment_detail = {
            "segment_index": int(segment_index),
            "offset_s": start_s,
            "duration_s": duration_s,
            "reference_text": reference_text,
            "hypothesis_text": hypothesis_text,
            "wer": compute_wer(reference_text, hypothesis_text),
            "cer": compute_cer(reference_text, hypothesis_text),
            "reference_word_count": len(tokenize_words(reference_text)),
            "hypothesis_word_count": len(tokenize_words(hypothesis_text)),
            "is_meta_response": bool(is_meta_response(hypothesis_text)),
            "diagnostics": result_diagnostics,
            "error": error_payload,
        }
        segment_details.append(segment_detail)
        committed_texts.append(hypothesis_text)
        commit_events.append(
            {
                "segment_index": int(segment_index),
                "text": hypothesis_text,
                "end_time_s": end_s,
                "audio_processed_s": end_s,
                "wallclock_s": None,
                "is_eos_flush": False,
            }
        )

    processing_s = perf_counter() - processing_start
    evaluated_duration_s = sum(float(segment["duration"]) for segment, _text in pairs)
    final_asr_text = " ".join(text for text in committed_texts if text).strip()
    payload = {
        "run_id": run_id,
        "run_label": run_label,
        "backend_name": backend_name,
        "eval_mode": "segmented_ref",
        "wav_path": wav_path,
        "reference_wav_name": reference_payload["reference_wav_name"],
        "audio_duration_s": float(evaluated_duration_s),
        "load_s": load_s,
        "warmup_s": warmup_s,
        "processing_s": processing_s,
        "rtf_wallclock": processing_s / max(evaluated_duration_s, 1e-9),
        "segment_count": len(segment_details),
        "committed_segment_count": len(committed_texts),
        "update_count": len(committed_texts),
        "final_asr_text": final_asr_text,
        "committed_texts": committed_texts,
        "commit_events": commit_events,
        "segment_details": segment_details,
        "experiment": {
            "gemma_vllm_path_mode": gemma_vllm_path_mode,
            "gemma_sampling_mode": gemma_sampling_mode,
            "gemma_max_model_len": gemma_max_model_len,
            "segment_indices": (
                None if not segment_indices else [int(index) for index in segment_indices]
            ),
        },
    }
    payload["metrics"] = compare_backend_to_reference(
        backend_payload=payload,
        reference_payload={
            **reference_payload,
            "segments": [segment for segment, _text in pairs],
            "texts": [text for _segment, text in pairs],
            "full_text": " ".join(text for _segment, text in pairs if text).strip(),
            "boundary_word_positions": cumulative_word_boundaries(
                [text for _segment, text in pairs]
            ),
            "boundary_end_times_s": [
                float(segment["offset"]) + float(segment["duration"])
                for segment, _text in pairs
            ],
        },
        tolerance_words=match_tolerance_words,
    )
    payload["mean_segment_wer"] = mean(
        detail["wer"] for detail in segment_details
    ) if segment_details else 0.0
    payload["mean_segment_cer"] = mean(
        detail["cer"] for detail in segment_details
    ) if segment_details else 0.0
    payload["meta_response_count"] = sum(
        1 for detail in segment_details if detail["is_meta_response"]
    )
    payload["top_worst_segments"] = sorted(
        [
            {
                "segment_index": detail["segment_index"],
                "offset_s": detail["offset_s"],
                "duration_s": detail["duration_s"],
                "wer": detail["wer"],
                "cer": detail["cer"],
                "is_meta_response": detail["is_meta_response"],
                "reference_text": detail["reference_text"],
                "hypothesis_text": detail["hypothesis_text"],
                "error": detail["error"],
            }
            for detail in segment_details
        ],
        key=lambda detail: (
            not detail["is_meta_response"],
            -float(detail["wer"]),
            -float(detail["cer"]),
            -float(detail["duration_s"]),
        ),
    )[:5]
    payload["selection_rank"] = rank_to_jsonable(segment_selection_rank(payload))
    return payload


def run_backend_streaming_eval(
    *,
    wav_path: str,
    backend_name: str,
    reference_payload: dict[str, Any],
    run_id: str,
    run_label: str,
    chunk_ms: int,
    min_start_seconds: float,
    warmup_seconds: float,
    asr_alignatt_frame_threshold: int,
    asr_alignatt_rewind_threshold: int,
    match_tolerance_words: int,
    gemma_vllm_path_mode: str,
    gemma_sampling_mode: str,
    gemma_max_model_len: int | None,
    gemma_audio_alignment_top_k_heads: int | None = None,
) -> dict[str, Any]:
    import numpy as np

    from cascade.audio import load_audio_mono_16khz
    from cascade.runtime import CascadeRuntimeConfig, LoadedModelBundle

    config = CascadeRuntimeConfig(
        source_lang="English",
        target_lang="German",
        alignment_backend_name=backend_name,
    )
    config.min_start_seconds = float(min_start_seconds)
    config.asr_alignatt_frame_threshold = int(asr_alignatt_frame_threshold)
    config.asr_alignatt_rewind_threshold = int(asr_alignatt_rewind_threshold)
    if gemma_audio_alignment_top_k_heads is not None:
        config.gemma_audio_alignment_top_k_heads = int(gemma_audio_alignment_top_k_heads)

    if backend_name == "gemma_vllm_qk_fast":
        if gemma_max_model_len is not None:
            config.gemma_max_model_len = int(gemma_max_model_len)
        maybe_configure_gemma_vllm_experiment(
            config,
            gemma_vllm_path_mode=gemma_vllm_path_mode,
            gemma_sampling_mode=gemma_sampling_mode,
        )

    bundle = LoadedModelBundle(config)
    load_start = perf_counter()
    alignment_backend = bundle.ensure_alignment_backend()
    load_s = perf_counter() - load_start
    warmup_s = maybe_warmup_backend(
        backend_name=backend_name,
        alignment_backend=alignment_backend,
        warmup_seconds=warmup_seconds,
    )

    session = bundle.new_session()

    audio = load_audio_mono_16khz(wav_path)
    chunk_size = int(SAMPLE_RATE * chunk_ms / 1000)
    audio_duration_s = len(audio) / SAMPLE_RATE

    commit_events: list[dict[str, Any]] = []
    stream_trace: list[dict[str, Any]] = []
    processing_start = perf_counter()
    last_commit_count = len(session.state.utt_sources) - 1
    last_trace_len = 0
    update_count = 0
    last_chunk_idx = 0

    for chunk_idx, stop_sample in enumerate(
        range(chunk_size, len(audio) + chunk_size, chunk_size),
        start=1,
    ):
        stop_sample = min(stop_sample, len(audio))
        last_chunk_idx = chunk_idx
        session.state.source = np.asarray(audio[:stop_sample], dtype=np.float32)
        if session.current_audio_seconds() < config.min_start_seconds:
            continue

        current_asr = session.transcribe_audio()
        trace_snapshot = session.asr_stream_trace()
        new_rows = trace_snapshot[last_trace_len:]
        for row in new_rows:
            enriched = dict(row)
            enriched["chunk_idx"] = int(chunk_idx)
            enriched["wallclock_s"] = perf_counter() - processing_start
            stream_trace.append(enriched)
        last_trace_len = len(trace_snapshot)
        if current_asr:
            update_count += 1

        commit_count = len(session.state.utt_sources) - 1
        if commit_count > last_commit_count:
            wallclock_s = perf_counter() - processing_start
            for segment_idx in range(last_commit_count + 1, commit_count + 1):
                commit_events.append(
                    {
                        "segment_index": int(segment_idx),
                        "text": session.state.utt_sources[segment_idx].strip(),
                        "end_time_s": session.state.utt_timestamps[segment_idx] / SAMPLE_RATE,
                        "audio_processed_s": stop_sample / SAMPLE_RATE,
                        "wallclock_s": wallclock_s,
                        "is_eos_flush": False,
                    }
                )
            last_commit_count = commit_count

    session.state.source = np.asarray(audio, dtype=np.float32)
    final_asr = session.transcribe_audio(is_final_chunk=True) or session.render_public_asr_text()
    final_wallclock_s = perf_counter() - processing_start
    trace_snapshot = session.asr_stream_trace()
    new_rows = trace_snapshot[last_trace_len:]
    for row in new_rows:
        enriched = dict(row)
        enriched["chunk_idx"] = int(max(1, last_chunk_idx))
        enriched["wallclock_s"] = final_wallclock_s
        stream_trace.append(enriched)
    final_commit_count = len(session.state.utt_sources) - 1
    if final_commit_count > last_commit_count:
        for segment_idx in range(last_commit_count + 1, final_commit_count + 1):
            commit_events.append(
                {
                    "segment_index": int(segment_idx),
                    "text": session.state.utt_sources[segment_idx].strip(),
                    "end_time_s": session.state.utt_timestamps[segment_idx] / SAMPLE_RATE,
                    "audio_processed_s": audio_duration_s,
                    "wallclock_s": final_wallclock_s,
                    "is_eos_flush": True,
                }
            )

    committed_texts = [segment.strip() for segment in session.state.utt_sources[1:] if segment.strip()]
    payload = {
        "run_id": run_id,
        "run_label": run_label,
        "backend_name": backend_name,
        "eval_mode": "streaming_full",
        "wav_path": wav_path,
        "reference_wav_name": reference_payload["reference_wav_name"],
        "chunk_ms": int(chunk_ms),
        "min_start_seconds": float(min_start_seconds),
        "asr_alignatt_frame_threshold": int(asr_alignatt_frame_threshold),
        "asr_alignatt_rewind_threshold": int(asr_alignatt_rewind_threshold),
        "load_s": load_s,
        "warmup_s": warmup_s,
        "processing_s": final_wallclock_s,
        "rtf_wallclock": final_wallclock_s / max(audio_duration_s, 1e-9),
        "audio_duration_s": audio_duration_s,
        "update_count": int(update_count),
        "committed_segment_count": len(committed_texts),
        "final_asr_text": final_asr,
        "committed_texts": committed_texts,
        "commit_events": commit_events,
        "stream_trace": stream_trace,
        "experiment": {
            "asr_alignatt_frame_threshold": int(asr_alignatt_frame_threshold),
            "asr_alignatt_rewind_threshold": int(asr_alignatt_rewind_threshold),
            "gemma_vllm_path_mode": gemma_vllm_path_mode,
            "gemma_sampling_mode": gemma_sampling_mode,
            "gemma_max_model_len": gemma_max_model_len,
        },
    }
    payload["metrics"] = compare_backend_to_reference(
        backend_payload=payload,
        reference_payload=reference_payload,
        tolerance_words=match_tolerance_words,
    )
    payload["meta_response_count"] = sum(
        1
        for row in stream_trace
        if is_meta_response(row.get("hypothesis_text", ""))
    )
    payload["selection_rank"] = rank_to_jsonable(
        streaming_selection_rank(payload, tolerance_words=match_tolerance_words)
    )
    return payload


def compact_run_for_summary(
    run: dict[str, Any],
    *,
    tolerance_words: int,
) -> dict[str, Any]:
    base = {
        "run_id": run["run_id"],
        "run_label": run["run_label"],
        "backend_name": run["backend_name"],
        "eval_mode": run["eval_mode"],
    }
    if run.get("error") is not None:
        base["error"] = dict(run["error"])
        return base

    compact = {
        **base,
        "load_s": run.get("load_s"),
        "warmup_s": run.get("warmup_s"),
        "processing_s": run.get("processing_s"),
        "rtf_wallclock": run.get("rtf_wallclock"),
        "meta_response_count": run.get("meta_response_count"),
        "selection_rank": list(run.get("selection_rank", [])),
        "metrics": {
            "wer": run["metrics"]["wer"],
            "cer": run["metrics"]["cer"],
            "predicted_boundary_count": run["metrics"]["predicted_boundary_count"],
            "reference_boundary_count": run["metrics"]["reference_boundary_count"],
            "boundary_metrics": {
                "exact": {
                    "f1": run["metrics"]["boundary_metrics"]["exact"]["f1"],
                    "precision": run["metrics"]["boundary_metrics"]["exact"]["precision"],
                    "recall": run["metrics"]["boundary_metrics"]["exact"]["recall"],
                },
                f"tolerance_{int(tolerance_words)}w": {
                    "f1": run["metrics"]["boundary_metrics"][f"tolerance_{int(tolerance_words)}w"]["f1"],
                    "precision": run["metrics"]["boundary_metrics"][f"tolerance_{int(tolerance_words)}w"]["precision"],
                    "recall": run["metrics"]["boundary_metrics"][f"tolerance_{int(tolerance_words)}w"]["recall"],
                },
            },
            "lag_summary": run["metrics"]["lag_summary"],
        },
        "experiment": dict(run.get("experiment", {})),
    }
    if run["eval_mode"] == "segmented_ref":
        compact["segment_count"] = run.get("segment_count")
        compact["mean_segment_wer"] = run.get("mean_segment_wer")
        compact["mean_segment_cer"] = run.get("mean_segment_cer")
    else:
        compact["chunk_ms"] = run.get("chunk_ms")
        compact["update_count"] = run.get("update_count")
        compact["committed_segment_count"] = run.get("committed_segment_count")
    return compact


def write_summary_text(summary: dict[str, Any], output_path: Path) -> None:
    tolerance_words = int(summary["match_tolerance_words"])
    lines = [
        f"WAV: {summary['wav_path']}",
        f"Reference WAV: {summary['reference']['reference_wav_name']}",
        f"Clip duration: {summary['reference']['clip_duration_s']:.3f}s",
        f"Reference segments: {summary['reference']['segment_count']}  words={summary['reference']['word_count']}",
        f"Boundary matching tolerance: ±{tolerance_words} words",
        "",
    ]

    for section_name in ("segmented_ref", "streaming_full"):
        section = summary.get(section_name)
        if section is None:
            continue
        lines.append(f"[{section_name}]")
        lines.append(f"Selection rule: {' > '.join(section['selection_rule'])}")
        if section.get("best_run_id") is not None:
            lines.append(f"Best run: {section['best_run_id']}")
        for run in section["runs"]:
            if run.get("error") is not None:
                lines.append(f"  - {run['run_id']}: ERROR {run['error']['type']}: {run['error']['message']}")
                continue
            tol_key = f"tolerance_{tolerance_words}w"
            tol_metrics = run["metrics"]["boundary_metrics"][tol_key]
            lag_summary = run["metrics"]["lag_summary"]
            line = (
                f"  - {run['run_id']}: WER={run['metrics']['wer']:.4f} "
                f"CER={run['metrics']['cer']:.4f} "
                f"F1@±{tolerance_words}w={tol_metrics['f1']:.4f}"
            )
            if run.get("rtf_wallclock") is not None:
                line += f" RTF={run['rtf_wallclock']:.4f}"
            if lag_summary["mean_abs_s"] is not None:
                line += f" mean|lag|={lag_summary['mean_abs_s']:.3f}s"
            if run.get("meta_response_count") is not None:
                line += f" meta={run['meta_response_count']}"
            if run.get("mean_segment_wer") is not None:
                line += f" mean-seg-WER={run['mean_segment_wer']:.4f}"
            lines.append(line)
        lines.append("")

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def render_plot(*, summary: dict[str, Any], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    tolerance_words = int(summary["match_tolerance_words"])
    section = summary.get("streaming_full") or summary.get("segmented_ref")
    if section is None:
        raise ValueError("summary.json does not contain any plottable evaluation section.")
    runs = [run for run in section["runs"] if run.get("error") is None]
    if not runs:
        raise ValueError("No successful runs available for plotting.")

    tol_key = f"tolerance_{tolerance_words}w"
    labels = [run["run_label"] for run in runs]
    wers = [float(run["metrics"]["wer"]) for run in runs]
    f1s = [float(run["metrics"]["boundary_metrics"][tol_key]["f1"]) for run in runs]
    rtfs = [float(run.get("rtf_wallclock") or 0.0) for run in runs]
    colors = {
        "qwen_forced": "#C44E52",
        "gemma_vllm_qk_fast": "#55A868",
    }
    bar_colors = [colors.get(run["backend_name"], "#8172B3") for run in runs]
    positions = list(range(len(runs)))

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.8), sharey=True)
    metric_specs = (
        ("WER", wers, False),
        (f"Boundary F1 @ ±{tolerance_words} words", f1s, True),
        ("Real-Time Factor", rtfs, False),
    )

    for ax, (title, values, higher_is_better) in zip(axes, metric_specs):
        ax.barh(positions, values, color=bar_colors, alpha=0.92)
        ax.set_title(title, fontsize=11)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=9.5)
        ax.grid(axis="x", alpha=0.2, linestyle=":")
        ax.set_axisbelow(True)
        if higher_is_better:
            ax.set_xlim(0.0, max(1.0, max(values) * 1.15))
        else:
            ax.set_xlim(0.0, max(values) * 1.15 if max(values) > 0 else 1.0)
        for y_pos, value in zip(positions, values):
            ax.text(
                value + max(ax.get_xlim()[1] * 0.015, 0.005),
                y_pos,
                f"{value:.3f}",
                va="center",
                ha="left",
                fontsize=9,
            )
        ax.invert_yaxis()

    wav_name = Path(summary["wav_path"]).name
    fig.suptitle(
        f"ASR Comparison on {wav_name} ({section['section_name']})",
        fontsize=13,
        y=0.98,
    )
    fig.text(
        0.5,
        0.03,
        "Runs are ordered by the section selection rule shown in summary.txt.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0.03, 0.06, 0.995, 0.93))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_run_parser(subparsers) -> None:
    run = subparsers.add_parser("run", help="Run segmented and/or streaming ASR evaluation")
    run.add_argument("--wav", default=DEFAULT_WAV)
    run.add_argument("--segments", default=DEFAULT_SEGMENTS)
    run.add_argument("--source-ref", default=DEFAULT_SOURCE_REF)
    run.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    run.add_argument(
        "--eval-mode",
        choices=EVAL_MODE_CHOICES,
        default="both",
        help="Which evaluation regime to run.",
    )
    run.add_argument(
        "--backends",
        nargs="+",
        choices=ALL_BACKENDS,
        default=list(DEFAULT_BACKENDS),
        help="Subset of backends to evaluate.",
    )
    run.add_argument("--chunk-ms", type=int, default=800)
    run.add_argument("--min-start-seconds", type=float, default=2.0)
    run.add_argument(
        "--asr-alignatt-frame-threshold",
        type=int,
        default=4,
        help=(
            "AlignAtt token-level frontier gate in audio frames (simul_whisper "
            "§4). Lower = more aggressive commit, higher = safer."
        ),
    )
    run.add_argument(
        "--asr-alignatt-rewind-threshold",
        type=int,
        default=200,
        help=(
            "Attention-collapse guard: abort the chunk if a generated token's "
            "argmax rewinds more than this many frames before the running "
            "reference."
        ),
    )
    run.add_argument(
        "--gemma-audio-alignment-top-k-heads",
        type=int,
        default=None,
        help=(
            "Override the number of top-ranked audio-alignment heads averaged "
            "for the AlignAtt walk. Lower values drop garbage heads from the "
            "average and can sharpen the per-token argmaxes."
        ),
    )
    run.add_argument("--match-tolerance-words", type=int, default=3)
    run.add_argument(
        "--gemma-warmup-seconds",
        type=float,
        default=18.0,
        help="Warmup pass for gemma_vllm_qk_fast before measured runs.",
    )
    run.add_argument(
        "--gemma-vllm-path-mode",
        choices=VLLM_PATH_MODE_CHOICES,
        default="shipping",
        help="Gemma vLLM invocation path ablation for no-prefix runs.",
    )
    run.add_argument(
        "--gemma-sampling-mode",
        choices=GEMMA_SAMPLING_MODE_CHOICES,
        default="shipping",
        help="Private harness override for Gemma vLLM decoding (shipping vs HF model-card sampling).",
    )
    run.add_argument(
        "--gemma-max-model-len",
        type=int,
        default=None,
        help="Optional ASR-only override for Gemma vLLM max_model_len.",
    )
    run.add_argument(
        "--segment-max-count",
        type=int,
        default=None,
        help="Optional cap on the number of reference segments for quick probes.",
    )
    run.add_argument(
        "--segment-indices",
        nargs="+",
        type=int,
        default=None,
        help="Optional 1-based reference segment indices for targeted segmented_ref reruns.",
    )
    run.add_argument(
        "--skip-plot",
        action="store_true",
        help="Write JSON/TXT only; skip the optional PNG rendering step.",
    )
    run.set_defaults(func=cmd_run)


def build_plot_parser(subparsers) -> None:
    plot = subparsers.add_parser("plot", help="Render the PNG figure from summary.json")
    plot.add_argument("--summary", required=True)
    plot.add_argument("--output", default=None)
    plot.set_defaults(func=cmd_plot)


def cmd_run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    reference = load_reference_segments(
        wav_path=args.wav,
        segments_path=args.segments,
        source_ref_path=args.source_ref,
    )

    segmented_runs: list[dict[str, Any]] = []
    streaming_runs: list[dict[str, Any]] = []

    for backend_name in args.backends:
        run_id = build_run_id(
            backend_name=backend_name,
            eval_mode="segmented_ref",
            gemma_vllm_path_mode=args.gemma_vllm_path_mode,
            gemma_sampling_mode=args.gemma_sampling_mode,
            asr_alignatt_frame_threshold=args.asr_alignatt_frame_threshold,
            asr_alignatt_rewind_threshold=args.asr_alignatt_rewind_threshold,
        )
        run_label = build_run_label(backend_name=backend_name, run_id=run_id)

        if args.eval_mode in ("segmented_ref", "both"):
            print(f"\n[{run_id}] segmented_ref", flush=True)
            try:
                run_payload = run_segmented_reference_eval(
                    wav_path=args.wav,
                    backend_name=backend_name,
                    reference_payload=reference,
                    run_id=run_id,
                    run_label=run_label,
                    min_start_seconds=args.min_start_seconds,
                    warmup_seconds=(
                        args.gemma_warmup_seconds if backend_name == "gemma_vllm_qk_fast" else 0.0
                    ),
                    match_tolerance_words=args.match_tolerance_words,
                    gemma_vllm_path_mode=args.gemma_vllm_path_mode,
                    gemma_sampling_mode=args.gemma_sampling_mode,
                    gemma_max_model_len=args.gemma_max_model_len,
                    segment_max_count=args.segment_max_count,
                    segment_indices=args.segment_indices,
                )
                printable = {
                    "run_id": run_id,
                    "wer": round(run_payload["metrics"]["wer"], 6),
                    "mean_segment_wer": round(run_payload["mean_segment_wer"], 6),
                    "meta_response_count": int(run_payload["meta_response_count"]),
                }
            except Exception as exc:  # noqa: BLE001 - keep the sweep durable
                run_payload = {
                    "run_id": run_id,
                    "run_label": run_label,
                    "backend_name": backend_name,
                    "eval_mode": "segmented_ref",
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
                printable = {"run_id": run_id, "error": run_payload["error"]}
            segmented_runs.append(run_payload)
            write_json(runs_dir / f"{run_id}__segmented_ref.json", run_payload)
            print(json.dumps(printable, indent=2, ensure_ascii=False))

        if args.eval_mode in ("streaming_full", "both"):
            stream_run_id = build_run_id(
                backend_name=backend_name,
                eval_mode="streaming_full",
                gemma_vllm_path_mode=args.gemma_vllm_path_mode,
                gemma_sampling_mode=args.gemma_sampling_mode,
                asr_alignatt_frame_threshold=args.asr_alignatt_frame_threshold,
                asr_alignatt_rewind_threshold=args.asr_alignatt_rewind_threshold,
            )
            stream_run_label = build_run_label(
                backend_name=backend_name,
                run_id=stream_run_id,
            )
            print(f"\n[{stream_run_id}] streaming_full", flush=True)
            try:
                run_payload = run_backend_streaming_eval(
                    wav_path=args.wav,
                    backend_name=backend_name,
                    reference_payload=reference,
                    run_id=stream_run_id,
                    run_label=stream_run_label,
                    chunk_ms=args.chunk_ms,
                    min_start_seconds=args.min_start_seconds,
                    warmup_seconds=(
                        args.gemma_warmup_seconds if backend_name == "gemma_vllm_qk_fast" else 0.0
                    ),
                    asr_alignatt_frame_threshold=args.asr_alignatt_frame_threshold,
                    asr_alignatt_rewind_threshold=args.asr_alignatt_rewind_threshold,
                    match_tolerance_words=args.match_tolerance_words,
                    gemma_vllm_path_mode=args.gemma_vllm_path_mode,
                    gemma_sampling_mode=args.gemma_sampling_mode,
                    gemma_max_model_len=args.gemma_max_model_len,
                    gemma_audio_alignment_top_k_heads=args.gemma_audio_alignment_top_k_heads,
                )
                printable = {
                    "run_id": stream_run_id,
                    "wer": round(run_payload["metrics"]["wer"], 6),
                    "rtf_wallclock": round(run_payload["rtf_wallclock"], 6),
                    "predicted_boundary_count": int(run_payload["metrics"]["predicted_boundary_count"]),
                    "meta_response_count": int(run_payload["meta_response_count"]),
                }
            except Exception as exc:  # noqa: BLE001 - keep the sweep durable
                run_payload = {
                    "run_id": stream_run_id,
                    "run_label": stream_run_label,
                    "backend_name": backend_name,
                    "eval_mode": "streaming_full",
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
                printable = {"run_id": stream_run_id, "error": run_payload["error"]}
            streaming_runs.append(run_payload)
            write_json(runs_dir / f"{stream_run_id}__streaming_full.json", run_payload)
            print(json.dumps(printable, indent=2, ensure_ascii=False))

    segment_eval_payload: dict[str, Any] | None = None
    if segmented_runs:
        successful_segmented = [run for run in segmented_runs if run.get("error") is None]
        successful_segmented.sort(key=segment_selection_rank)
        if successful_segmented:
            best_segmented_run_id = successful_segmented[0]["run_id"]
        else:
            best_segmented_run_id = None
        segment_eval_payload = {
            "kind": "segment_eval",
            "wav_path": args.wav,
            "reference": public_reference_summary(reference),
            "selection_rule": [
                "meta_response_count",
                "full_wer",
                "mean_segment_wer",
                "full_cer",
            ],
            "best_run_id": best_segmented_run_id,
            "runs": segmented_runs,
        }
        write_json(output_dir / "segment_eval.json", segment_eval_payload)

    stream_trace_payload: dict[str, Any] | None = None
    streaming_summary_runs: list[dict[str, Any]] = []
    if streaming_runs:
        successful_streaming = [run for run in streaming_runs if run.get("error") is None]
        successful_streaming.sort(
            key=lambda run: streaming_selection_rank(
                run,
                tolerance_words=args.match_tolerance_words,
            )
        )
        if successful_streaming:
            best_streaming_run_id = successful_streaming[0]["run_id"]
        else:
            best_streaming_run_id = None
        stream_trace_payload = {
            "kind": "stream_trace",
            "wav_path": args.wav,
            "reference": public_reference_summary(reference),
            "runs": [
                {
                    "run_id": run["run_id"],
                    "backend_name": run["backend_name"],
                    "eval_mode": run["eval_mode"],
                    "stream_trace": run.get("stream_trace", []),
                }
                for run in streaming_runs
                if run.get("error") is None
            ],
        }
        write_json(output_dir / "stream_trace.json", stream_trace_payload)
        streaming_summary_runs = [
            compact_run_for_summary(run, tolerance_words=args.match_tolerance_words)
            for run in successful_streaming
        ] + [
            compact_run_for_summary(run, tolerance_words=args.match_tolerance_words)
            for run in streaming_runs
            if run.get("error") is not None
        ]
    segmented_summary_runs: list[dict[str, Any]] = []
    if segmented_runs:
        successful_segmented = [run for run in segmented_runs if run.get("error") is None]
        successful_segmented.sort(key=segment_selection_rank)
        segmented_summary_runs = [
            compact_run_for_summary(run, tolerance_words=args.match_tolerance_words)
            for run in successful_segmented
        ] + [
            compact_run_for_summary(run, tolerance_words=args.match_tolerance_words)
            for run in segmented_runs
            if run.get("error") is not None
        ]

    summary = {
        "kind": "asr_compare_harness",
        "wav_path": args.wav,
        "segments_path": args.segments,
        "source_ref_path": args.source_ref,
        "match_tolerance_words": int(args.match_tolerance_words),
        "reference": public_reference_summary(reference),
        "segmented_ref": (
            None
            if segment_eval_payload is None
            else {
                "section_name": "segmented_ref",
                "selection_rule": segment_eval_payload["selection_rule"],
                "best_run_id": segment_eval_payload["best_run_id"],
                "runs": segmented_summary_runs,
            }
        ),
        "streaming_full": (
            None
            if not streaming_runs
            else {
                "section_name": "streaming_full",
                "selection_rule": [
                    "meta_response_count",
                    "abs(predicted_boundary_count - reference_boundary_count)",
                    "full_wer",
                    f"-boundary_f1_±{int(args.match_tolerance_words)}w",
                ],
                "best_run_id": (
                    None
                    if not streaming_summary_runs
                    else next(
                        (
                            run["run_id"]
                            for run in streaming_summary_runs
                            if run.get("error") is None
                        ),
                        None,
                    )
                ),
                "runs": streaming_summary_runs,
            }
        ),
    }
    summary_path = output_dir / "summary.json"
    write_json(summary_path, summary)
    write_summary_text(summary, output_dir / "summary.txt")
    print(f"\nWrote summary: {summary_path}")

    if not args.skip_plot:
        try:
            output_path = output_dir / "comparison_plot.png"
            render_plot(summary=summary, output_path=output_path)
            print(f"Wrote plot: {output_path}")
        except ModuleNotFoundError as exc:
            print(
                "Skipping plot rendering because matplotlib is unavailable in this interpreter: "
                f"{exc}"
            )


def cmd_plot(args: argparse.Namespace) -> None:
    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    output_path = Path(args.output) if args.output else summary_path.with_name("comparison_plot.png")
    render_plot(summary=summary, output_path=output_path)
    print(f"Wrote plot: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    build_run_parser(subparsers)
    build_plot_parser(subparsers)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
