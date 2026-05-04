# Submission

## Maintained Surface

- Presets are defined in `cascade/submission.py`
- Supported Docker directions are exactly `en-de`, `en-it`, and `en-zh`
- Low latency is `main_low_latency` with `chunk_ms=850`
- High latency is `main_high_latency` with `chunk_ms=1500`
- Extra-context presets and CS->EN are not part of the DockerHub delivery

## DockerHub Image

The root `Dockerfile` builds the submission image from the active runtime only:

- `cascade/`
- `submission/docker-entrypoint.sh`
- `submission/render_preset_yaml.py`
- `submission/download_model_snapshots.py`
- EN->DE/IT/ZH MT AlignAtt head JSONs

Model snapshots are downloaded during the H100 build with a BuildKit secret:

```bash
export DOCKERHUB_REPO="dockerhub-user/cascade-simul-iwslt26"
export HF_TOKEN_FILE="$HOME/.cache/huggingface/token"
submission/build_push_dockerhub_h100.sh
```

The helper tags both `$DOCKERHUB_REPO:$IMAGE_TAG` and
`$DOCKERHUB_REPO:latest`, validates one clip when `VALIDATION_WAV` exists, and
pushes by default. Use `PUSH=0` to keep the image local.

For JarvisLabs CLI 0.2.0b13, the bare VM is exposed as the `vm` template:

```bash
jl instance create --gpu H100 --template vm --storage 300 --region IN2 --name alignatt-iwslt26-h100 --yes
jl instance upload <machine_id> .
jl instance ssh <machine_id>
```

The image bundles these revisions and runs offline afterwards:

- `Qwen/Qwen3-ASR-1.7B@7278e1e70fe206f11671096ffdd38061171dd6e5`
- `Qwen/Qwen3-ForcedAligner-0.6B@c7cbfc2048c462b0d63a45797104fc9db3ad62b7`
- `google/gemma-4-E4B-it@83df0a889143b1dbfc61b591bbc639540fd9ce4c`

## Runtime Modes

Direct inference is the default mode:

```bash
docker run --gpus all --rm --ipc=host \
  -e PRESET=main_low_latency \
  -e TGT_LANG_CODE=de \
  -v /host/wavs:/io/wavs:ro \
  -v /host/out:/io/out \
  "$DOCKERHUB_REPO:latest" \
  infer /io/wavs/wavlist.txt /io/out/metrics.jsonl
```

HTTP server mode exposes the same speech processor on port `8080`:

```bash
docker run --gpus all --rm --ipc=host -p 8080:8080 \
  -e PRESET=main_low_latency \
  -e TGT_LANG_CODE=de \
  "$DOCKERHUB_REPO:latest" serve
```

From the host, use `submission/http_proxy_processor.yaml` with
`simulstream_inference` to evaluate through the HTTP proxy.

## Evidence Kept In Tree

- `submission/dev_logs/` contains the six validated main-track dev-set summary bundles
- `submission/DEV_LOG_INDEX.json` indexes the retained manifest and evaluation files
- The full `.jsonl` hypotheses and stream updates are intentionally not tracked
