#!/usr/bin/env bash
set -euo pipefail

# IWSLT 2026 Simultaneous submission entry point.
#
# Invokes the official `simulstream_inference` CLI with one of our frozen
# presets (`cascade_submission.SUBMISSION_PRESETS`) rendered to a YAML
# `speech_processor_config` on the fly, so organisers can run:
#
#   docker run --gpus all --rm \
#     -e PRESET=main_low_latency -e SRC_LANG=English -e TGT_LANG=German \
#     -v /host/wavs:/io/wavs:ro -v /host/out:/io/out \
#     cascade-simul-iwslt26 \
#     /io/wavs/wavlist.txt /io/out/metrics.jsonl
#
# wavlist.txt paths must be relative to the wavlist file location (SimulStream
# contract: see simulstream/client/wav_reader_client.py::load_wav_file_list).

PRESET="${PRESET:-${CASCADE_SUBMISSION_PRESET:-main_low_latency}}"
SRC_LANG="${SRC_LANG:-${CASCADE_SOURCE_LANG:-English}}"
TGT_LANG="${TGT_LANG:-${CASCADE_TARGET_LANG:-German}}"
SRC_LANG_CODE="${SRC_LANG_CODE:-en}"
TGT_LANG_CODE="${TGT_LANG_CODE:-de}"
PAPER_CONTEXT_PATH="${PAPER_CONTEXT_PATH:-}"

WAV_LIST_FILE="${1:-${WAV_LIST_FILE:-}}"
METRICS_LOG_FILE="${2:-${METRICS_LOG_FILE:-/io/out/metrics.jsonl}}"

if [ -z "$WAV_LIST_FILE" ]; then
  echo "Usage: docker run ... cascade-simul-iwslt26 <wavlist.txt> [<metrics.jsonl>]" >&2
  echo "  or set env var WAV_LIST_FILE" >&2
  exit 2
fi

export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0

CONFIG_DIR="$(mktemp -d)"
CONFIG_FILE="${CONFIG_DIR}/speech_processor.yaml"

python /app/submission_raw/render_preset_yaml.py \
  --preset "$PRESET" \
  --source-lang-code "$SRC_LANG_CODE" \
  --target-lang-code "$TGT_LANG_CODE" \
  ${PAPER_CONTEXT_PATH:+--paper-context-path "$PAPER_CONTEXT_PATH"} \
  --output "$CONFIG_FILE"

mkdir -p "$(dirname "$METRICS_LOG_FILE")"

exec python -m simulstream.inference \
  --speech-processor-config "$CONFIG_FILE" \
  --wav-list-file "$WAV_LIST_FILE" \
  --src-lang "$SRC_LANG" \
  --tgt-lang "$TGT_LANG" \
  --metrics-log-file "$METRICS_LOG_FILE"
