# Submission

## Maintained path

- Presets are defined in `cascade/submission.py`
- Root Docker uses `submission/docker-entrypoint.sh`
- YAML rendering uses `submission/render_preset_yaml.py`
- Artifact sync uses `submission/sync_artifacts.py`
- Standalone export uses `submission/export_standalone_bundle.py`

## Naming

- Main test-set outputs should be named by preset, for example `outputs/iwslt26_testset_main_low_latency_ende`
- `submission/sync_artifacts.py` still accepts older chunk-based output directories as a compatibility fallback

## Evidence kept in-tree

- `submission/dev_logs/` contains the six validated main-track dev-set bundles
- `submission/DEV_LOG_INDEX.json` indexes those frozen dev logs

## Export

Generate the standalone bundle with:

```bash
.venv-inference/bin/python submission/export_standalone_bundle.py
```
