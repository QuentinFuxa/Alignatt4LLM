#!/usr/bin/env python3
"""Run a single-audio latency experiment in the warm inference kernel.

The harness hot-reloads the cascade Python modules inside the persistent
`.venv-inference` kernel. Only the expensive hot weights (ASR + Gemma) and
the tokenizer are carried across runs; the backend object, its policy, and
the prompt KV cache are rebuilt from the freshly reloaded modules every
time so that code edits actually take effect and cache state cannot leak
between runs.

Defaults track the current best provisional operating point from PLAN.md so
that invoking the harness without flags reproduces the recommended
configuration instead of a degraded one.
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


def collect_local_provenance(args: argparse.Namespace) -> dict:
    def _git(*cmd: str) -> str:
        try:
            out = subprocess.run(
                ["git", *cmd],
                cwd="/home/cascade_simultaneous",
                check=True,
                capture_output=True,
                text=True,
            )
            return out.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""

    commit_sha = _git("rev-parse", "HEAD")
    dirty_files = _git("status", "--porcelain")
    return {
        "git_commit_sha": commit_sha,
        "git_dirty": bool(dirty_files),
        "git_dirty_files": [line for line in dirty_files.splitlines() if line.strip()],
        "cli_args": {key: value for key, value in vars(args).items()},
        "harness_started_at_utc": utc_stamp(),
        "cache_reset_policy": "rebuild_backend_each_run",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--connection-file", default=DEFAULT_CONNECTION_FILE)
    parser.add_argument("--wav-path", default=DEFAULT_WAV_PATH)
    parser.add_argument("--chunk-ms", default=450, type=int)
    parser.add_argument("--min-start-seconds", default=2.0, type=float)
    parser.add_argument("--partial-max-new-tokens", default=16, type=int)
    parser.add_argument("--partial-followup-max-new-tokens", default=8, type=int)
    parser.add_argument("--rewind-threshold", default=8, type=int)
    parser.add_argument("--inaccessible-ms", default=0.0, type=float)
    parser.add_argument("--scheduler-stall-seconds", default=1.2, type=float)
    parser.add_argument("--max-history-utterances", default=1, type=int)
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


def hot_reload_core_in_kernel(client: BlockingKernelClient) -> dict:
    """Reload Python modules, rebuild the backend, reset caches.

    Returns a provenance dict describing what happened in the kernel, so the
    manifest can faithfully record what code/config served this run.
    """
    code = """
import importlib
import json

import qwen3asr_gemma_cascade_core as core

saved_asr = core.asr
saved_mt_backend = core.mt_backend
saved_speech_id = core.state.speech_id

import cascade_artifacts
import cascade_translation_variants
import cascade_mt_backend
import cascade_emission
import cascade_source_frontier
import cascade_source_text
import cascade_text_surface
cascade_artifacts = importlib.reload(cascade_artifacts)
cascade_translation_variants = importlib.reload(cascade_translation_variants)
cascade_mt_backend = importlib.reload(cascade_mt_backend)
cascade_emission = importlib.reload(cascade_emission)
cascade_source_frontier = importlib.reload(cascade_source_frontier)
cascade_source_text = importlib.reload(cascade_source_text)
cascade_text_surface = importlib.reload(cascade_text_surface)
core = importlib.reload(core)

core.asr = saved_asr
core.state = core.CascadeState(speech_id=saved_speech_id)

# Rebuild backend under the freshly reloaded class, preserving hot weights.
# Fall back to a full load_models() if this is the first run in this kernel.
if saved_mt_backend is None:
    core.load_models()
else:
    core.rebuild_mt_backend_preserving_weights(existing_backend=saved_mt_backend)
core.mt_backend.reset_caches()

provenance = {
    "kernel_backend_class": type(core.mt_backend).__name__,
    "kernel_backend_module": type(core.mt_backend).__module__,
    "kernel_model_loaded": core.mt_backend.model is not None,
    "kernel_prompt_cache_empty": not core.mt_backend.prompt_cache.full_prompt_ids,
}
print("__HOT_RELOAD_PROVENANCE__" + json.dumps(provenance) + "__END__")
print("__HOT_RELOAD_OK__")
"""
    outputs = execute_kernel_code(client, code, timeout_s=3600)
    joined = "".join(outputs)
    if "__HOT_RELOAD_OK__" not in joined:
        raise RuntimeError("Hot reload did not complete successfully.")
    start = joined.find("__HOT_RELOAD_PROVENANCE__")
    end = joined.find("__END__", start)
    provenance: dict = {}
    if start != -1 and end != -1:
        try:
            provenance = json.loads(joined[start + len("__HOT_RELOAD_PROVENANCE__") : end])
        except json.JSONDecodeError:
            provenance = {}
    return provenance


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
    run_provenance: dict,
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
    overrides_json = json.dumps(overrides)
    provenance_json = json.dumps(run_provenance)
    code = f"""
import json
import qwen3asr_gemma_cascade_core as core

core.mt_backend.reset_caches()
written = core.run_baseline(
    wav_path=r"{wav_path}",
    output_dir=r"{output_dir}",
    chunk_ms={int(chunk_ms)},
    runtime_overrides=json.loads({overrides_json!r}),
    run_provenance=json.loads({provenance_json!r}),
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

    local_provenance = collect_local_provenance(args)

    print(
        f"=== Latency experiment: chunk_ms={args.chunk_ms} "
        f"min_start={args.min_start_seconds} "
        f"partial_max={args.partial_max_new_tokens} "
        f"followup_max={args.partial_followup_max_new_tokens} "
        f"rewind={args.rewind_threshold} "
        f"inaccessible_ms={args.inaccessible_ms} "
        f"history={args.max_history_utterances} ==="
    )
    print(f"git_commit_sha={local_provenance['git_commit_sha']} dirty={local_provenance['git_dirty']}")

    client = BlockingKernelClient(connection_file=args.connection_file)
    client.load_connection_file()
    client.start_channels()
    try:
        client.wait_for_ready(timeout=120)
        kernel_provenance = hot_reload_core_in_kernel(client)
        run_provenance = {**local_provenance, **kernel_provenance}
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
            run_provenance=run_provenance,
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
