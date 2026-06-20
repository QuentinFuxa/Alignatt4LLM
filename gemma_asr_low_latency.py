#!/usr/bin/env python3
"""Standalone Gemma AlignAtt low-latency ASR runner.

Defaults lock the paper's `tab:asr-frontier` Config B operating point:

    f  (frame threshold)   = 4       # 160 ms guard from audio frontier
    rho (rewind threshold) = 50      # 2000 ms anchored-rewind window
    k  (top-k heads)       = 6       # layer-23 English heads, monotonicity-ranked
    chunk_ms               = 800
    min_start_seconds      = 2.0
    probe                  = qk_fast
    dtype                  = bfloat16
    temperature            = 0.0     # greedy
    max_model_len          = 1024

Override chunk_ms and f via ``--chunk-ms`` / ``--frame-threshold`` to
slide along the latency/accuracy Pareto without editing the file.

Streams every input wav through a single loaded backend and emits one
`hypothesis.jsonl` (OmniSTEval-compatible) plus a `manifest.json`, so the
output directory can be fed directly to `evaluate_cascade_outputs.py
--skip-comet` for BLEU / chrF / LongYAAL CU / LongYAAL CA. SimulEval
latency semantics: ``delays[i]`` is the chunk-boundary audio time at
which word i was emitted, not alignatt's acoustic end estimate.

No metrics are computed here --- just streaming inference.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from cascade.artifacts import (  # noqa: E402
    HYPOTHESIS_ELAPSED_SEMANTICS_CA_COMPATIBLE,
    build_asr_hypothesis_record,
)
from cascade.audio import load_audio_mono_16khz  # noqa: E402
from cascade.runtime import (  # noqa: E402
    CascadeRuntimeConfig,
    LoadedModelBundle,
)


SAMPLE_RATE = 16000

# ------------------------------------------------------------------
# Config B --- sub-1 s CU-LongYAAL Gemma AlignAtt operating point.
# ------------------------------------------------------------------
# These are the paper-validated defaults; ``--chunk-ms`` and
# ``--frame-threshold`` let a caller slide along the latency/accuracy
# Pareto without editing the file.
CHUNK_MS = 800
MIN_START_SECONDS = 2.0
COMMIT_POLICY = "frontier_flush"
FRAME_THRESHOLD = 4        # f
REWIND_THRESHOLD = 50      # rho
TOP_K_HEADS = 6            # k
# Experimental rescue for pathological empty-stop chunks. Disabled by default:
# on the smoke clip it can unblock a few stalls, but the accepted retries still
# corrupt the forced prefix with plausible-yet-wrong continuations.
EOS_ONLY_RESCUE = False
EOS_ONLY_RESCUE_MAX_NEW_TOKENS = 3
ALIGNMENT_BACKEND = "gemma_vllm_qk_fast"


def build_config() -> CascadeRuntimeConfig:
    config = CascadeRuntimeConfig(
        source_lang="English",
        target_lang="English",
        alignment_backend_name=ALIGNMENT_BACKEND,
    )
    config.min_start_seconds = float(MIN_START_SECONDS)
    config.asr_alignatt_commit_policy = str(COMMIT_POLICY)
    config.asr_alignatt_frame_threshold = int(FRAME_THRESHOLD)
    config.asr_alignatt_rewind_threshold = int(REWIND_THRESHOLD)
    config.gemma_audio_alignment_top_k_heads = int(TOP_K_HEADS)
    config.gemma_audio_eos_only_rescue_enabled = bool(EOS_ONLY_RESCUE)
    config.gemma_audio_eos_only_rescue_max_new_tokens = int(
        EOS_ONLY_RESCUE_MAX_NEW_TOKENS
    )
    return config


def stream_one_wav(bundle: LoadedModelBundle, wav_path: Path) -> dict:
    """Stream ``wav_path`` through a fresh session and collect the ASR trace.

    Returns a dict with the raw trace, the per-token commit log (alignatt's
    acoustic ``end_time_s`` per committed token), and the committed
    prediction, enough to build an OmniSTEval-compatible hypothesis record.
    """
    session = bundle.new_session()
    audio = load_audio_mono_16khz(str(wav_path))
    audio_duration_s = len(audio) / SAMPLE_RATE
    chunk_size = int(SAMPLE_RATE * CHUNK_MS / 1000)

    trace: list[dict] = []
    last_trace_len = 0
    t0 = perf_counter()

    for stop_sample in range(chunk_size, len(audio) + chunk_size, chunk_size):
        stop_sample = min(stop_sample, len(audio))
        session.state.source = np.asarray(audio[:stop_sample], dtype=np.float32)
        if session.current_audio_seconds() < MIN_START_SECONDS:
            continue
        session.transcribe_audio()
        snapshot = session.asr_stream_trace()
        wallclock_s = perf_counter() - t0
        for row in snapshot[last_trace_len:]:
            enriched = dict(row)
            enriched["wallclock_s"] = wallclock_s
            trace.append(enriched)
        last_trace_len = len(snapshot)

    # Final flush with is_final_chunk=True so any pending tail is committed.
    session.state.source = np.asarray(audio, dtype=np.float32)
    session.transcribe_audio(is_final_chunk=True)
    final_wallclock_s = perf_counter() - t0
    snapshot = session.asr_stream_trace()
    for row in snapshot[last_trace_len:]:
        enriched = dict(row)
        enriched["wallclock_s"] = final_wallclock_s
        trace.append(enriched)

    return {
        "wav_path": str(wav_path),
        "audio_duration_s": audio_duration_s,
        "processing_s": final_wallclock_s,
        "rtf_wallclock": final_wallclock_s / max(audio_duration_s, 1e-9),
        "final_asr_text": session.render_public_asr_text(),
        "stream_trace": trace,
        "per_token_commits": session.per_token_commits(),
    }


def build_hypothesis_record(run: dict) -> dict:
    """Delegate to the shared ASR emission-time hypothesis builder."""
    return build_asr_hypothesis_record(
        per_token_commits=run.get("per_token_commits") or [],
        stream_trace=run.get("stream_trace") or [],
        wav_name=Path(run["wav_path"]).name,
        audio_duration_s=float(run["audio_duration_s"]),
        processing_s=float(run["processing_s"]),
    )


def write_artifacts(records: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hypothesis.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "asr_enen_lowlat_v1",
        "kind": "inference",
        "chunk_ms": CHUNK_MS,
        "min_start_seconds": MIN_START_SECONDS,
        "source_language_code": "en",
        "target_language_code": "en",
        "alignment_backend": ALIGNMENT_BACKEND,
        "alignatt": {
            "commit_policy": COMMIT_POLICY,
            "frame_threshold": FRAME_THRESHOLD,
            "rewind_threshold": REWIND_THRESHOLD,
            "top_k_heads": TOP_K_HEADS,
            "aggregation": "median_argmax",
            "eos_only_rescue": EOS_ONLY_RESCUE,
            "eos_only_rescue_max_new_tokens": EOS_ONLY_RESCUE_MAX_NEW_TOKENS,
            "probe_mode": "qk_fast",
        },
        "files": {"hypothesis_jsonl": "hypothesis.jsonl"},
        "runtime_config": {
            "hypothesis_elapsed_semantics": HYPOTHESIS_ELAPSED_SEMANTICS_CA_COMPATIBLE,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    global CHUNK_MS, COMMIT_POLICY, FRAME_THRESHOLD, REWIND_THRESHOLD
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wavs", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=CHUNK_MS,
        help=f"Audio chunk length in ms (default {CHUNK_MS}).",
    )
    parser.add_argument(
        "--commit-policy",
        choices=("frontier_flush", "rewind_abort"),
        default=COMMIT_POLICY,
        help=(
            "ASR AlignAtt commit rule. `frontier_flush` commits the maximal "
            "monotone prefix each chunk and only keeps the final frontier "
            "band; `rewind_abort` preserves the legacy whole-chunk abort path."
        ),
    )
    parser.add_argument(
        "--frame-threshold",
        type=int,
        default=FRAME_THRESHOLD,
        help=(
            f"AlignAtt frame threshold f (40 ms/frame; default "
            f"{FRAME_THRESHOLD}). Larger values trade latency for accuracy."
        ),
    )
    parser.add_argument(
        "--rewind-threshold",
        type=int,
        default=REWIND_THRESHOLD,
        help=(
            f"Legacy rewind-abort threshold rho in frames (40 ms/frame; "
            f"default {REWIND_THRESHOLD}). Ignored by `frontier_flush`."
        ),
    )
    args = parser.parse_args()

    CHUNK_MS = int(args.chunk_ms)
    COMMIT_POLICY = str(args.commit_policy)
    FRAME_THRESHOLD = int(args.frame_threshold)
    REWIND_THRESHOLD = int(args.rewind_threshold)

    print(
        f"[config] backend={ALIGNMENT_BACKEND} chunk={CHUNK_MS}ms "
        f"min_start={MIN_START_SECONDS}s policy={COMMIT_POLICY} f={FRAME_THRESHOLD} "
        f"rho={REWIND_THRESHOLD} k={TOP_K_HEADS} agg=median_argmax "
        f"eos_rescue={EOS_ONLY_RESCUE}/{EOS_ONLY_RESCUE_MAX_NEW_TOKENS}",
        flush=True,
    )

    bundle = LoadedModelBundle(build_config())
    load_start = perf_counter()
    bundle.ensure_alignment_backend()
    print(f"[load] backend ready in {perf_counter() - load_start:.1f}s", flush=True)

    records: list[dict] = []
    per_wav_dir = args.output_dir / "per_wav"
    per_wav_dir.mkdir(parents=True, exist_ok=True)

    for idx, wav in enumerate(args.wavs, start=1):
        print(f"[stream] ({idx}/{len(args.wavs)}) {wav.name}", flush=True)
        t0 = perf_counter()
        run = stream_one_wav(bundle, wav)
        rec = build_hypothesis_record(run)
        records.append(rec)

        (per_wav_dir / f"{wav.stem}.json").write_text(
            json.dumps(run, indent=2), encoding="utf-8"
        )
        n_words = len(rec["prediction"].split())
        print(
            f"    duration={run['audio_duration_s']:.1f}s "
            f"rtf={run['rtf_wallclock']:.3f} words={n_words} "
            f"wall={perf_counter() - t0:.1f}s",
            flush=True,
        )

    write_artifacts(records, args.output_dir)
    print(
        f"\nwrote {args.output_dir}/hypothesis.jsonl ({len(records)} entries)\n"
        f"  -> feed this dir to evaluate_cascade_outputs.py --skip-comet"
    )


if __name__ == "__main__":
    main()
