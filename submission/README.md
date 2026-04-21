# Submission Workspace

`submission/` is the single maintained submission surface for the repo.

It now contains:

- frozen dev-set logs under `submission/dev_logs/`
- `render_preset_yaml.py` to materialize a validated preset as a SimulStream `speech_processor.yaml`
- `docker-entrypoint.sh` for the root Docker image
- `sync_artifacts.py` to re-materialize a self-contained submission bundle from `outputs/`
- `export_standalone_bundle.py` to generate a frozen standalone bundle under `dist/`

## Source of truth

Submission presets come only from [`cascade/submission.py`](/home/cascade_simultaneous/cascade/submission.py):

- `main_low_latency`: `chunk_ms=850`, `translation_alignatt_border_margin=1`
- `main_high_latency`: `chunk_ms=1500`, `translation_alignatt_border_margin=1`
- `context_low_latency`: `main_low_latency` + `paper_context_mode=title_abstract`
- `context_high_latency`: `main_high_latency` + `paper_context_mode=title_abstract`

The maintained MT preset surface intentionally excludes any MT-side anti-rewind
threshold. Target-language reorderings, especially EN->ZH, make that heuristic
counterproductive for partial acceptance.

## Root Docker path

The repo root `Dockerfile` renders one frozen preset on startup, then invokes
the official `simulstream.inference` CLI.

Example:

```bash
docker run --gpus all --rm \
  -e PRESET=main_low_latency \
  -e SRC_LANG=English \
  -e TGT_LANG=German \
  -e SRC_LANG_CODE=en \
  -e TGT_LANG_CODE=de \
  -v /host/wavs:/io/wavs:ro \
  -v /host/out:/io/out \
  cascade-simul-iwslt26 \
  /io/wavs/wavlist.txt /io/out/metrics.jsonl
```

## Standalone bundle export

Generate a frozen standalone bundle in `dist/standalone_bundle`:

```bash
.venv-inference/bin/python submission/export_standalone_bundle.py
```

That export contains:

- the active `cascade/` package
- `data/alignatt_heads/`
- the submission renderer and entrypoint
- generated main-track YAML configs for `en->{de,it,zh}`
- a minimal Docker recipe and shell launchers

## Artifact sync

To rebuild a self-contained `submission/artifacts/` + `submission/results/`
tree from already-produced runs:

```bash
.venv-inference/bin/python submission/sync_artifacts.py
```

Main-track bundles are derived from the preset metadata and look first for the
new preset-named output directories, then fall back to the older chunk-named
directories if they already exist.

## Included frozen evidence

`submission/dev_logs/` keeps the six validated dev-set runs for the maintained
main-track regimes:

- `submission/dev_logs/low/en-de`
- `submission/dev_logs/low/en-it`
- `submission/dev_logs/low/en-zh`
- `submission/dev_logs/high/en-de`
- `submission/dev_logs/high/en-it`
- `submission/dev_logs/high/en-zh`
