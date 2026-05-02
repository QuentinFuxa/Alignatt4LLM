# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/opt/huggingface \
    VLLM_USE_DEEP_GEMM=0 \
    VLLM_MOE_USE_DEEP_GEMM=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
 && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app
COPY pyproject.toml uv.lock setup_inference_qwen_asr_vllm.sh patch_qwen_asr_for_transformers5.py ./
COPY cascade ./cascade
COPY submission/docker-entrypoint.sh submission/render_preset_yaml.py submission/download_model_snapshots.py ./submission/
COPY submission/http_proxy_processor.yaml ./submission/
COPY data/alignatt_heads/translation_heads_google_gemma-4-E4B-it_en-de.json ./data/alignatt_heads/
COPY data/alignatt_heads/translation_heads_google_gemma-4-E4B-it_en-it.json ./data/alignatt_heads/
COPY data/alignatt_heads/translation_heads_google_gemma-4-E4B-it_en-zh.json ./data/alignatt_heads/

RUN chmod +x /app/setup_inference_qwen_asr_vllm.sh /app/submission/docker-entrypoint.sh \
 && /app/setup_inference_qwen_asr_vllm.sh /opt/cascade-venv

RUN --mount=type=secret,id=hf_token,required=false \
    /opt/cascade-venv/bin/python /app/submission/download_model_snapshots.py \
      --output-dir /opt/cascade-models \
      --token-file /run/secrets/hf_token

ENV PATH="/opt/cascade-venv/bin:${PATH}" \
    PYTHONPATH="/app:${PYTHONPATH}" \
    CASCADE_QWEN_ASR_SNAPSHOT="/opt/cascade-models/qwen3-asr-1.7b" \
    CASCADE_QWEN_ALIGNER_SNAPSHOT="/opt/cascade-models/qwen3-forced-aligner-0.6b" \
    CASCADE_GEMMA_SNAPSHOT="/opt/cascade-models/gemma-4-e4b-it" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 8080
ENTRYPOINT ["/app/submission/docker-entrypoint.sh"]
