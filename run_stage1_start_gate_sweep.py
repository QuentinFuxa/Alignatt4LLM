#!/usr/bin/env python3
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


def sweep_tag(value: float) -> str:
    return str(value).replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Stage 1 min_start_seconds sweep inside the warmed inference kernel "
            "and evaluate each run locally."
        ),
    )
    parser.add_argument(
        "--connection-file",
        default=DEFAULT_CONNECTION_FILE,
        help="Jupyter kernel connection file for the warmed .venv-inference kernel.",
    )
    parser.add_argument("--wav-path", default=DEFAULT_WAV_PATH, help="Input WAV file.")
    parser.add_argument("--chunk-ms", default=800, type=int, help="Streaming chunk size.")
    parser.add_argument(
        "--min-start-seconds",
        nargs="+",
        type=float,
        default=[5.0, 3.0, 2.0, 1.5],
        help="Sweep values for config.min_start_seconds.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Directory where per-run artifacts and the sweep summary will be written.",
    )
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


def run_single_baseline_in_kernel(
    client: BlockingKernelClient,
    *,
    wav_path: str,
    chunk_ms: int,
    min_start_seconds: float,
    output_dir: str,
) -> None:
    code = f"""
import json
import qwen3asr_gemma_cascade_core as core

written = core.run_baseline(
    wav_path=r\"{wav_path}\",
    output_dir=r\"{output_dir}\",
    chunk_ms={int(chunk_ms)},
    runtime_overrides={{\"min_start_seconds\": {float(min_start_seconds)}}},
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


def write_summary(output_root: Path, rows: list[dict]) -> None:
    summary_json_path = output_root / "summary.json"
    summary_tsv_path = output_root / "summary.tsv"

    summary_json_path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "\t".join(
            [
                "min_start_seconds",
                "output_dir",
                "BLEU",
                "CHRF",
                "LongYAAL_CU",
                "LongYAAL_CA",
            ]
        )
    ]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    str(row["min_start_seconds"]),
                    str(row["output_dir"]),
                    f'{row["BLEU"]:.4f}',
                    f'{row["CHRF"]:.4f}',
                    f'{row["LongYAAL_CU"]:.4f}',
                    f'{row["LongYAAL_CA"]:.4f}',
                ]
            )
        )

    summary_tsv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote sweep summary to {summary_tsv_path}")


def main() -> None:
    args = parse_args()
    output_root = Path(
        args.output_root
        or f"/home/cascade_simultaneous/outputs/stage1_start_gate_sweep_{utc_stamp()}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    client = BlockingKernelClient(connection_file=args.connection_file)
    client.load_connection_file()
    client.start_channels()

    try:
        client.wait_for_ready(timeout=120)
        hot_reload_core_in_kernel(client)

        summary_rows: list[dict] = []
        for min_start_seconds in args.min_start_seconds:
            run_output_dir = output_root / f"min_start_{sweep_tag(min_start_seconds)}"
            print(
                f"\n=== Stage 1 run: min_start_seconds={min_start_seconds} "
                f"chunk_ms={args.chunk_ms} ==="
            )
            run_single_baseline_in_kernel(
                client,
                wav_path=args.wav_path,
                chunk_ms=args.chunk_ms,
                min_start_seconds=float(min_start_seconds),
                output_dir=str(run_output_dir),
            )
            evaluation = evaluate_output_dir(run_output_dir)
            scores = evaluation["contract_scores"]
            summary_rows.append(
                {
                    "min_start_seconds": float(min_start_seconds),
                    "output_dir": str(run_output_dir),
                    "BLEU": float(scores["BLEU"]),
                    "CHRF": float(scores["CHRF"]),
                    "LongYAAL_CU": float(scores["LongYAAL CU"]),
                    "LongYAAL_CA": float(scores["LongYAAL CA"]),
                }
            )

        write_summary(output_root, summary_rows)
    finally:
        client.stop_channels()


if __name__ == "__main__":
    main()
