# IWSLT 2026 Simultaneous — submission bundle

Contact: Quentin Fuxa (quentin.fuxa@gmail.com)
Track: Main, LOW-latency regime (LongYAAL CU ≤ 2 s)
System: `qwen_forced` ASR (Qwen3-ASR-1.7B + Qwen3-ForcedAligner-0.6B) →
`gemma_vllm_alignatt` MT (Gemma-4-E4B-it with vLLM + MT-side AlignAtt).

## Frozen preset

`main_low_latency`:

- `chunk_ms = 750`
- `translation_alignatt_border_margin = 1`
- `translation_alignatt_inaccessible_ms = 0`
- everything else: `cascade_submission.SubmissionPreset` defaults.

## Validated dev-set numbers (MCIF, 21 clips / 919 refs)

| Direction | BLEU  | chrF  | COMET  | LongYAAL CU | LongYAAL CA | Empty |
|-----------|-------|-------|--------|-------------|-------------|-------|
| en -> de  | 27.35 | 61.46 | 0.8669 | 1707 ms     | 1343 ms     | 0/919 |
| en -> it  | 38.37 | 66.82 | 0.7875 | 1675 ms     | 1321 ms     | 0/919 |
| en -> zh  | 35.02 | 33.88 | 0.7308 | 1672 ms     | 1498 ms     | 0/919 |

All three directions live on the LOW side of the 2 s LongYAAL CU boundary with
no empty predictions.

## Artefact locations

Each test-set direction produces three files under
`outputs/iwslt26_testset_chunk750_borderp1_en<xx>/`:

- `manifest.json` — runtime config + provenance
- `hypothesis.jsonl` — one record per test-set talk (21 talks)
- `stream_updates.jsonl` — per-chunk streaming updates for LongYAAL

Counts per direction (IWSLT blind test-set, 21 ACL talks each):

| Direction | hypotheses | stream updates |
|-----------|------------|----------------|
| en -> de  | 21         | 6 775          |
| en -> it  | 21         | 7 330          |
| en -> zh  | 21         | 7 254          |

## Reproducing from the repo

```bash
# 1) install the inference env (see setup_inference_qwen_asr_vllm.sh)
./setup_inference_qwen_asr_vllm.sh .venv-inference

# 2) re-run the frozen preset on the test-set per direction
for TGT in de it zh; do
  VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 \
    .venv-inference/bin/python run_iwslt_submission.py batch \
      --preset main_low_latency \
      --source en \
      --target $TGT \
      --input-dir test-set/audio \
      --output-dir outputs/iwslt26_testset_chunk750_borderp1_en$TGT
done
```

`scripts/run_testset_submission.sh` wraps this loop.

## Docker path

See `submission/README.md` for `docker build` / `docker run` instructions. The
container renders the frozen preset into a simulstream YAML and calls
`simulstream.inference` the way IWSLT organizers expect. The runtime relies on
an HF cache mounted at `/root/.cache/huggingface` (Qwen3-ASR-1.7B,
Qwen3-ForcedAligner-0.6B, gemma-4-E4B-it; ~21 GB). The Dockerfile itself does
not bake model snapshots — they are mounted at run time.
