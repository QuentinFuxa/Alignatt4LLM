# IWSLT 2026 Simultaneous — submission bundle

Contact: Quentin Fuxa (quentin.fuxa@gmail.com)
Track: Main, both latency regimes (LOW ≤ 2 s, HIGH 2-4 s)
System: `qwen_forced` ASR (Qwen3-ASR-1.7B + Qwen3-ForcedAligner-0.6B) →
`gemma_vllm_alignatt` MT (Gemma-4-E4B-it with vLLM + MT-side AlignAtt).

## Frozen presets

Both presets share the same alignment mechanism; only `chunk_ms` differs.

`main_low_latency`:

- `chunk_ms = 750`
- `translation_alignatt_border_margin = 1`
- `translation_alignatt_inaccessible_ms = 0`

`main_high_latency`:

- `chunk_ms = 1100`
- `translation_alignatt_border_margin = 1`
- `translation_alignatt_inaccessible_ms = 0`

Everything else: `cascade_submission.SubmissionPreset` defaults.

## Validated dev-set numbers (MCIF, 21 clips / 919 refs)

LOW regime (`main_low_latency`, chunk_ms=750):

| Direction | BLEU  | chrF  | COMET  | LongYAAL CU | LongYAAL CA | Empty |
|-----------|-------|-------|--------|-------------|-------------|-------|
| en -> de  | 27.35 | 61.46 | 0.8669 | 1707 ms     | 1343 ms     | 0/919 |
| en -> it  | 38.37 | 66.82 | 0.7875 | 1675 ms     | 1321 ms     | 0/919 |
| en -> zh  | 35.02 | 33.88 | 0.7308 | 1672 ms     | 1498 ms     | 0/919 |

HIGH regime (`main_high_latency`, chunk_ms=1100):

| Direction | BLEU  | chrF  | COMET  | LongYAAL CU | LongYAAL CA | Empty |
|-----------|-------|-------|--------|-------------|-------------|-------|
| en -> de  | 29.99 | 62.78 | 0.8922 | 2414 ms     | 2037 ms     | 0/919 |
| en -> it  | 42.75 | 69.24 | 0.8268 | 2315 ms     | 1947 ms     | 0/919 |
| en -> zh  | 37.91 | 36.12 | 0.7636 | 2258 ms     | 2075 ms     | 0/919 |

Both presets achieve zero empty predictions. HIGH regime gains +0.025 / +0.039
/ +0.033 COMET over LOW across the three directions.

## Artefact locations

Each test-set direction produces three files under
`outputs/iwslt26_testset_chunk<chunk_ms>_borderp1_en<xx>/`:

- `manifest.json` — runtime config + provenance
- `hypothesis.jsonl` — one record per test-set talk (21 talks)
- `stream_updates.jsonl` — per-chunk streaming updates for LongYAAL

Counts per direction (IWSLT blind test-set, 21 ACL talks each):

LOW regime (`outputs/iwslt26_testset_chunk750_borderp1_*`):

| Direction | hypotheses | stream updates |
|-----------|------------|----------------|
| en -> de  | 21         | 6 775          |
| en -> it  | 21         | 7 330          |
| en -> zh  | 21         | 7 254          |

HIGH regime (`outputs/iwslt26_testset_chunk1100_borderp1_*`):

| Direction | hypotheses | stream updates |
|-----------|------------|----------------|
| en -> de  | 21         | 5 151          |
| en -> it  | 21         | 5 458          |
| en -> zh  | 21         | 5 550          |

## Reproducing from the repo

```bash
# 1) install the inference env (see setup_inference_qwen_asr_vllm.sh)
./setup_inference_qwen_asr_vllm.sh .venv-inference

# 2a) LOW regime submission (main_low_latency)
bash scripts/run_testset_submission.sh

# 2b) HIGH regime submission (main_high_latency)
PRESET=main_high_latency \
OUTPUT_PREFIX=outputs/iwslt26_testset_chunk1100_borderp1 \
  bash scripts/run_testset_submission.sh
```

Both wrap `run_iwslt_submission.py batch` and keep the three directions
sequential on a single GPU.

## Docker path

See `submission/README.md` for `docker build` / `docker run` instructions. The
container renders the frozen preset into a simulstream YAML and calls
`simulstream.inference` the way IWSLT organizers expect. The runtime relies on
an HF cache mounted at `/root/.cache/huggingface` (Qwen3-ASR-1.7B,
Qwen3-ForcedAligner-0.6B, gemma-4-E4B-it; ~21 GB). The Dockerfile itself does
not bake model snapshots — they are mounted at run time.
