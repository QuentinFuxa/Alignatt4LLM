## What was validated

The exported image archive was validated in three ways on this host:

1. Exact-image structural validation with `skopeo inspect --config` on:
   - `submission/cascade-simul-iwslt26_submission-20260418-850low-1500high.tar`
2. Rootfs smoke test from the exact repacked image contents:
   - `python /app/submission_raw/render_preset_yaml.py --preset main_low_latency ...`
   - produced `chunk_ms: 850`
3. Runtime CLI smoke test from the exact image rootfs:
   - `python -m simulstream.inference --help`
   - started successfully from the embedded `/opt/cascade-venv`

## Important host limitation

This development machine blocks Linux namespace operations needed by the local Docker daemon when registering layers:

```text
unshare: operation not permitted
failed to register layer: unshare: operation not permitted
```

Because of that kernel restriction, `docker build`, `docker load`, and `docker run` could not be completed natively on this host even for `hello-world`.

The fallback validation above was therefore performed against the exported image contents directly:

- OCI image rebuilt from the Dockerfile steps
- Docker archive exported with `skopeo`
- config checked with `skopeo inspect --config`
- entrypoint dependencies smoke-tested inside the repacked rootfs with `chroot`

## Smoke command outputs

Rendered preset excerpt:

```yaml
chunk_ms: 850
alignment_backend_name: qwen_forced
mt_backend_name: gemma_vllm_alignatt
translation_alignatt_border_margin: 1
```

`simulstream.inference` help was available and reported:

```text
Simulstream version: 0.2.0
```
