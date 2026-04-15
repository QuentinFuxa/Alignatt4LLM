#!/usr/bin/env python3
"""Run a single-audio latency experiment in the warm inference kernel.

This script hot-reloads the cascade modules inside the persistent
.venv-inference kernel, then runs one audio with latency-optimized
configuration overrides and evaluates the LongYAAL CU.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import queue
import subprocess

from jupyter_client import BlockingKernelClient

from cascade_artifacts import DEFAULT_WAV_PATH


DEFAULT_CONNECTION_FILE = "/home/.local/share/jupyter/runtime/kernel-cascade-simultaneous.json"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--connection-file", default=DEFAULT_CONNECTION_FILE)
    parser.add_argument("--wav-path", default=DEFAULT_WAV_PATH)
    parser.add_argument("--chunk-ms", default=400, type=int)
    parser.add_argument("--min-start-seconds", default=2.0, type=float)
    parser.add_argument("--partial-max-new-tokens", default=16, type=int)
    parser.add_argument("--partial-followup-max-new-tokens", default=8, type=int)
    parser.add_argument("--rewind-threshold", default=8, type=int)
    parser.add_argument("--inaccessible-ms", default=0.0, type=float)
    parser.add_argument("--scheduler-stall-seconds", default=1.2, type=float)
    parser.add_argument("--max-history-utterances", default=0, type=int)
    parser.add_argument("--tag", default=None)
    return parser.parse_args()


def execute_kernel_code(client: BlockingKernelClient, code: str, *, timeout_s: int) -> list[str]:
    msg_id = client.execute(code, store_history=False)
    stream_chunks: list[str] = []

    while True:
        try:
            message = client.get_iopub_msg(timeout=timeout_s)
        except queue.Empty as exc:
            raise RuntimeError("Timed out while waiting for kernel output.") from exc

        if message.get("parent_header", {}).get("msg_id") != msg_id:
            continue

        msg_type = message["msg_type"]
        content = message["content"]
        if msg_type == "stream":
            text = content.get("text", "")
            if text:
                print(text, end="")
                stream_chunks.append(text)
        elif msg_type in ("execute_result", "display_data"):
            data = content.get("data", {})
            text = data.get("text/plain")
            if text:
                print(text)
                stream_chunks.append(f"{text}\n")
        elif msg_type == "error":
            traceback = "\n".join(content.get("traceback", []))
            raise RuntimeError(f"Kernel execution failed:\n{traceback}")
        elif msg_type == "status" and content.get("execution_state") == "idle":
            return stream_chunks


def hot_reload_core_in_kernel(client: BlockingKernelClient) -> None:
    code = """
import importlib
import qwen3asr_gemma_cascade_core as core

saved_asr = core.asr
saved_mt_backend = core.mt_backend
saved_gemma_llm = core.gemma_llm
saved_gemma_tokenizer = core.gemma_tokenizer
saved_speech_id = core.state.speech_id

import cascade_translation_variants
import cascade_mt_backend
import cascade_emission
import cascade_source_frontier
import cascade_source_text
import cascade_text_surface
cascade_translation_variants = importlib.reload(cascade_translation_variants)
cascade_mt_backend = importlib.reload(cascade_mt_backend)
cascade_emission = importlib.reload(cascade_emission)
cascade_source_frontier = importlib.reload(cascade_source_frontier)
cascade_source_text = importlib.reload(cascade_source_text)
cascade_text_surface = importlib.reload(cascade_text_surface)
core = importlib.reload(core)

core.asr = saved_asr
core.mt_backend = saved_mt_backend
core.gemma_llm = saved_gemma_llm
core.gemma_tokenizer = saved_gemma_tokenizer
core.state = core.CascadeState(speech_id=saved_speech_id)

core.load_models()
print("__HOT_RELOAD_OK__")
"""
    outputs = execute_kernel_code(client, code, timeout_s=3600)
    if "__HOT_RELOAD_OK__" not in "".join(outputs):
        raise RuntimeError("Hot reload did not complete successfully.")


def run_baseline_in_kernel(
    client: BlockingKernelClient,
    *,
    wav_path: str,
    chunk_ms: int,
    min_start_seconds: float,
    partial_max_new_tokens: int,
    partial_followup_max_new_tokens: int,
    rewind_threshold: int,
    inaccessible_ms: float,
    scheduler_stall_seconds: float,
    max_history_utterances: int,
    output_dir: str,
) -> None:
    overrides = {
        "min_start_seconds": float(min_start_seconds),
        "partial_max_new_tokens": int(partial_max_new_tokens),
        "partial_followup_max_new_tokens": int(partial_followup_max_new_tokens),
        "translation_alignatt_rewind_threshold": int(rewind_threshold),
        "translation_alignatt_inaccessible_ms": float(inaccessible_ms),
        "translation_scheduler_stall_seconds": float(scheduler_stall_seconds),
        "max_history_utterances": int(max_history_utterances),
    }
    code = f"""
import json
import qwen3asr_gemma_cascade_core as core

written = core.run_baseline(
    wav_path=r"{wav_path}",
    output_dir=r"{output_dir}",
    chunk_ms={int(chunk_ms)},
    runtime_overrides={json.dumps(overrides)},
)
print("__WRITTEN__")
print(json.dumps(written))
"""
    execute_kernel_code(client, code, timeout_s=3600)


def evaluate_output_dir(output_dir: Path) -> dict:
    command = [
        "/home/cascade_simultaneous/.venv-evaluation/bin/python",
        "/home/cascade_simultaneous/evaluate_cascade_outputs.py",
        "--output-dir",
        str(output_dir),
        "--skip-comet",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")
    evaluation_path = output_dir / "evaluation.json"
    return json.loads(evaluation_path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    tag = args.tag or f"latency_exp_{utc_stamp()}"
    output_dir = Path(f"/home/cascade_simultaneous/outputs/{tag}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"=== Latency experiment: chunk_ms={args.chunk_ms} "
        f"min_start={args.min_start_seconds} "
        f"partial_max={args.partial_max_new_tokens} "
        f"followup_max={args.partial_followup_max_new_tokens} "
        f"rewind={args.rewind_threshold} "
        f"inaccessible_ms={args.inaccessible_ms} ==="
    )

    client = BlockingKernelClient(connection_file=args.connection_file)
    client.load_connection_file()
    client.start_channels()
    try:
        client.wait_for_ready(timeout=120)
        hot_reload_core_in_kernel(client)
        run_baseline_in_kernel(
            client,
            wav_path=args.wav_path,
            chunk_ms=args.chunk_ms,
            min_start_seconds=args.min_start_seconds,
            partial_max_new_tokens=args.partial_max_new_tokens,
            partial_followup_max_new_tokens=args.partial_followup_max_new_tokens,
            rewind_threshold=args.rewind_threshold,
            inaccessible_ms=args.inaccessible_ms,
            scheduler_stall_seconds=args.scheduler_stall_seconds,
            max_history_utterances=args.max_history_utterances,
            output_dir=str(output_dir),
        )
        evaluation = evaluate_output_dir(output_dir)
        scores = evaluation["contract_scores"]
        print("\n=== Results ===")
        for metric in ("BLEU", "CHRF", "LongYAAL CU", "LongYAAL CA"):
            print(f"  {metric}: {scores.get(metric)}")
    finally:
        client.stop_channels()


if __name__ == "__main__":
    main()
