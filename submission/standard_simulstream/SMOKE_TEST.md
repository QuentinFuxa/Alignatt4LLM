# Smoke Test

Validated on `2026-04-18` on the development host with the bundled launcher and
the canonical `simulstream_inference` CLI.

Command used:

```bash
tmpdir="$(mktemp -d)"
ln -s /home/cascade_simultaneous/tmp/alignatt_smoke18.wav "$tmpdir/alignatt_smoke18.wav"
printf 'alignatt_smoke18.wav\n' > "$tmpdir/wavlist.txt"

CASCADE_ENV_DIR=/home/cascade_simultaneous/.venv-inference \
  ./bin/run_simulstream_inference.sh \
  configs/main_low_latency/en-de.yaml \
  "$tmpdir/wavlist.txt" \
  "$tmpdir/metrics.jsonl"
```

Observed result:

- exit code `0`
- `simulstream.inference` loaded the speech processor in `97.258 s`
- the run streamed one file end to end and wrote `24` JSONL records
- first non-empty emission at `2.55 s`: `Hallo, ich bin`

The launcher exports `VLLM_USE_DEEP_GEMM=0` and `VLLM_MOE_USE_DEEP_GEMM=0`,
which is required on this host for the vLLM startup path used by the bundle.
