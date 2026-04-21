#!/usr/bin/env python3
"""Export a frozen standalone SimulStream bundle under dist/."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cascade.submission import get_submission_preset


TARGETS = ("de", "it", "zh")
MAIN_PRESETS = ("main_low_latency", "main_high_latency")


README_TEXT = """# Standalone SimulStream Bundle

This directory is generated from the active repo with:

```bash
.venv-inference/bin/python submission/export_standalone_bundle.py
```

It contains the maintained runtime only:

- `cascade/`
- `data/alignatt_heads/`
- `render_preset_yaml.py`
- `docker-entrypoint.sh`
- `bin/run_simulstream_inference.sh`
- `bin/render_submission_preset.sh`
- rendered main-track configs for `en->{de,it,zh}`

Models are not bundled. The runtime expects local Hugging Face snapshots or
the usual `CASCADE_QWEN_ASR_SNAPSHOT`, `CASCADE_QWEN_ALIGNER_SNAPSHOT`, and
`CASCADE_GEMMA_SNAPSHOT` overrides.
"""


DOCKERFILE_TEXT = """FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \\
    PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    UV_LINK_MODE=copy \\
    HF_HOME=/root/.cache/huggingface \\
    HF_HUB_OFFLINE=1 \\
    TRANSFORMERS_OFFLINE=1 \\
    VLLM_USE_DEEP_GEMM=0 \\
    VLLM_MOE_USE_DEEP_GEMM=0

RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    ca-certificates \\
    curl \\
    ffmpeg \\
    git \\
 && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app
COPY . /app

RUN chmod +x /app/setup_inference_qwen_asr_vllm.sh \\
    /app/bin/run_simulstream_inference.sh \\
    /app/bin/render_submission_preset.sh \\
    /app/docker-entrypoint.sh \\
 && /app/setup_inference_qwen_asr_vllm.sh /opt/cascade-venv

ENV PATH="/opt/cascade-venv/bin:${PATH}" \\
    PYTHONPATH="/app:${PYTHONPATH}"

ENTRYPOINT ["/app/docker-entrypoint.sh"]
"""


RUN_SIMULSTREAM_TEXT = """#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${CASCADE_ENV_DIR:-$ROOT_DIR/.venv-inference}"
SIMULSTREAM_BIN="$ENV_DIR/bin/simulstream_inference"

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <speech_processor.yaml> <wavlist.txt> <metrics.jsonl>" >&2
  exit 1
fi

CONFIG_PATH="$1"
WAVLIST_PATH="$2"
METRICS_LOG_PATH="$3"

if [ ! -f "$CONFIG_PATH" ]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi

if [ ! -f "$WAVLIST_PATH" ]; then
  echo "Wav list not found: $WAVLIST_PATH" >&2
  exit 1
fi

SOURCE_LANG_CODE="$(awk -F': *' '$1 == "source_lang_code" {print $2; exit}' "$CONFIG_PATH")"
TARGET_LANG_CODE="$(awk -F': *' '$1 == "target_lang_code" {print $2; exit}' "$CONFIG_PATH")"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"

exec "$SIMULSTREAM_BIN" \\
  --speech-processor-config "$CONFIG_PATH" \\
  --wav-list-file "$WAVLIST_PATH" \\
  --src-lang "$SOURCE_LANG_CODE" \\
  --tgt-lang "$TARGET_LANG_CODE" \\
  --metrics-log-file "$METRICS_LOG_PATH"
"""


RENDER_PRESET_TEXT = """#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${CASCADE_ENV_DIR:-$ROOT_DIR/.venv-inference}"
PYTHON_BIN="$ENV_DIR/bin/python"

if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
  echo "Usage: $0 <preset> <source_lang_code> <target_lang_code> <output.yaml> [paper_context.json]" >&2
  exit 1
fi

PRESET="$1"
SOURCE_LANG_CODE="$2"
TARGET_LANG_CODE="$3"
OUTPUT_PATH="$4"
PAPER_CONTEXT_PATH="${5:-}"

CMD=(
  "$PYTHON_BIN"
  "$ROOT_DIR/render_preset_yaml.py"
  --preset "$PRESET"
  --source-lang-code "$SOURCE_LANG_CODE"
  --target-lang-code "$TARGET_LANG_CODE"
  --output "$OUTPUT_PATH"
)

if [ -n "$PAPER_CONTEXT_PATH" ]; then
  CMD+=(--paper-context-path "$PAPER_CONTEXT_PATH")
fi

"${CMD[@]}"
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="dist/standalone_bundle")
    return parser.parse_args()


def write_text(path: Path, content: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    if executable:
        path.chmod(0o755)


def copy_any(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def render_main_configs(output_dir: Path) -> None:
    for preset_name in MAIN_PRESETS:
        preset = get_submission_preset(preset_name)
        for target in TARGETS:
            cfg = preset.build_speech_processor_config(
                source_lang_code="en",
                target_lang_code=target,
                paper_context_path=None,
            )
            config_path = output_dir / "configs" / preset_name / f"en-{target}.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                yaml.safe_dump(vars(cfg), sort_keys=False),
                encoding="utf-8",
            )


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = (repo_root / args.output_dir).resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copy_any(repo_root / "cascade", output_dir / "cascade")
    copy_any(repo_root / "data" / "alignatt_heads", output_dir / "data" / "alignatt_heads")
    copy_any(repo_root / "patch_qwen_asr_for_transformers5.py", output_dir / "patch_qwen_asr_for_transformers5.py")
    copy_any(repo_root / "setup_inference_qwen_asr_vllm.sh", output_dir / "setup_inference_qwen_asr_vllm.sh")
    copy_any(repo_root / "pyproject.toml", output_dir / "pyproject.toml")
    copy_any(repo_root / "uv.lock", output_dir / "uv.lock")
    copy_any(repo_root / "submission" / "render_preset_yaml.py", output_dir / "render_preset_yaml.py")
    copy_any(repo_root / "submission" / "docker-entrypoint.sh", output_dir / "docker-entrypoint.sh")

    render_main_configs(output_dir)
    write_text(output_dir / "README.md", README_TEXT)
    write_text(output_dir / "Dockerfile", DOCKERFILE_TEXT)
    write_text(output_dir / "bin" / "run_simulstream_inference.sh", RUN_SIMULSTREAM_TEXT, executable=True)
    write_text(output_dir / "bin" / "render_submission_preset.sh", RENDER_PRESET_TEXT, executable=True)
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
