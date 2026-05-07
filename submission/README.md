# Submission Workspace

`submission/` is the maintained Docker submission surface.

It contains:

- `render_preset_yaml.py` to render a validated preset as a SimulStream `speech_processor.yaml`
- `docker-entrypoint.sh` for direct inference and HTTP server modes
- `download_model_snapshots.py` for H100 Docker builds
- `build_push_dockerhub_h100.sh` for H100 build, one-clip validation, and DockerHub push
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
export DOCKERHUB_REPO="dockerhub-user/cascade-simul-iwslt26"
export HF_TOKEN_FILE="$HOME/.cache/huggingface/token"
submission/build_push_dockerhub_h100.sh
```

The build downloads and bundles the exact Qwen ASR, Qwen ForcedAligner, and
Gemma snapshots used by the submitted runtime. The resulting image runs with
`HF_HUB_OFFLINE=1` and does not require model downloads at evaluation time.

Set `PUSH=0` for a build-only dry run. Set `VALIDATION_WAV=/path/to/clip.wav`
to force the helper to run one direct-inference smoke clip before pushing.

### JarvisLabs H100 VM

JarvisLabs CLI 0.2.0b13 exposes VM mode as the `vm` template:

```bash
jl gpus
jl instance create \
  --gpu H100 \
  --template vm \
  --storage 300 \
  --region IN2 \
  --name alignatt-iwslt26-h100 \
  --yes
jl instance upload <machine_id> .
jl instance ssh <machine_id>
```

On the VM:

```bash
cd ~/Alignatt4LLM
export DOCKERHUB_REPO="dockerhub-user/cascade-simul-iwslt26"
export HF_TOKEN="hf_..."
export DOCKERHUB_USERNAME="dockerhub-user"
export DOCKERHUB_TOKEN="dockerhub-access-token"
submission/build_push_dockerhub_h100.sh
```

Pause or destroy the instance after the push:

```bash
jl instance pause <machine_id>
# or, once artifacts are no longer needed:
jl instance destroy <machine_id>
```

## Direct Inference

```bash
docker run --gpus all --rm --ipc=host \
  -e PRESET=main_low_latency \
  -e TGT_LANG_CODE=de \
  -v /host/wavs:/io/wavs:ro \
  -v /host/out:/io/out \
  "$DOCKERHUB_REPO:latest" \
  infer /io/wavs/wavlist.txt /io/out/metrics.jsonl
```

`infer` may be omitted; it is the default mode.

For the maintained main-track Docker surface, run one image per target language
by changing only `TGT_LANG_CODE`:

- `de` for English->German
- `it` for English->Italian
- `zh` for English->Chinese

## HTTP Server Mode

```bash
docker run --gpus all --rm --ipc=host -p 8080:8080 \
  -e PRESET=main_low_latency \
  -e TGT_LANG_CODE=de \
  "$DOCKERHUB_REPO:latest" serve
```

Then run the host-side SimulStream client with `submission/http_proxy_processor.yaml`.

## Frozen Evidence

`submission/dev_logs/` keeps the six validated main-track dev-set summary
bundles. The full `.jsonl` hypotheses and stream updates are not tracked; use
the manifests, reports, and `scores.tsv` files as retained provenance.
