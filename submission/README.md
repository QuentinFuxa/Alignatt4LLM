# Submission Workspace

`submission/` is the maintained Docker submission surface.

It contains:

- `render_preset_yaml.py` to render a validated preset as a SimulStream `speech_processor.yaml`
- `docker-entrypoint.sh` for direct inference and HTTP server modes
- `download_model_snapshots.py` for H100 Docker builds
- `http_proxy_processor.yaml` for host-side SimulStream HTTP proxy evaluation
- frozen dev-set summary logs under `submission/dev_logs/`

## Source Of Truth

Submission presets come only from `cascade/submission.py`:

- `main_low_latency`: `chunk_ms=850`, `translation_alignatt_border_margin=1`
- `main_high_latency`: `chunk_ms=1500`, `translation_alignatt_border_margin=1`

The maintained DockerHub image supports only `en-de`, `en-it`, and `en-zh`.
Extra-context presets and CS->EN artifacts are intentionally excluded from the
submission surface.

## Build On H100

```bash
DOCKER_BUILDKIT=1 docker build \
  --secret id=hf_token,src="$HF_TOKEN_FILE" \
  -t "$DOCKERHUB_REPO:latest" .
```

The build downloads and bundles the exact Qwen ASR, Qwen ForcedAligner, and
Gemma snapshots used by the runtime. The resulting image runs with
`HF_HUB_OFFLINE=1` and does not require model downloads at evaluation time.

## Direct Inference

```bash
docker run --gpus all --rm \
  -e PRESET=main_low_latency \
  -e TGT_LANG_CODE=de \
  -v /host/wavs:/io/wavs:ro \
  -v /host/out:/io/out \
  "$DOCKERHUB_REPO:latest" \
  infer /io/wavs/wavlist.txt /io/out/metrics.jsonl
```

`infer` may be omitted; it is the default mode.

## HTTP Server Mode

```bash
docker run --gpus all --rm -p 8080:8080 \
  -e PRESET=main_low_latency \
  -e TGT_LANG_CODE=de \
  "$DOCKERHUB_REPO:latest" serve
```

Then run the host-side SimulStream client with `submission/http_proxy_processor.yaml`.

## Frozen Evidence

`submission/dev_logs/` keeps the six validated main-track dev-set summary
bundles. The full `.jsonl` hypotheses and stream updates are not tracked; use
the manifests, reports, and `scores.tsv` files as retained provenance.
