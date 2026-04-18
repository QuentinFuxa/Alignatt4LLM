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

The submission bundle vendors the six frozen main-track blind test-set
artifact sets under:

- `submission/artifacts/main/low/en-de/`
- `submission/artifacts/main/low/en-it/`
- `submission/artifacts/main/low/en-zh/`
- `submission/artifacts/main/high/en-de/`
- `submission/artifacts/main/high/en-it/`
- `submission/artifacts/main/high/en-zh/`

Each direction/regime directory contains:

- `manifest.json` — runtime config + provenance
- `hypothesis.jsonl` — one record per test-set talk (21 talks)
- `stream_updates.jsonl` — per-chunk streaming updates for LongYAAL

The source generation runs remain in
`outputs/iwslt26_testset_chunk<chunk_ms>_borderp1_en<xx>/`.

### Additive calibration bundles

Additional `chunk_ms` calibration points are vendored as dev-set bundles
(they cannot be blind-scored on the test set). Both the inference artifacts
and the scored evaluation are preserved so the latency/quality trade-off can
be re-audited without re-running inference:

- `submission/artifacts/additive/chunk850/en-{de,it,zh}/`
- `submission/artifacts/additive/chunk1900/en-{de,it,zh}/`
- `submission/results/additive/chunk850/en-{de,it,zh}/`
- `submission/results/additive/chunk1900/en-{de,it,zh}/`

Artifact directories keep the same trio as the main-track bundles
(`manifest.json`, `hypothesis.jsonl`, `stream_updates.jsonl`), while the
corresponding results directory mirrors the layout produced by
`evaluate_cascade_outputs.py`:

- `evaluation.json` — contract + raw metrics with metric-blocker ledger
- `evaluation.report.txt` — human-readable OmniSTEval summary
- `scores.tsv` — tabular contract scores for downstream plotting

Source dev-set runs remain in
`outputs/iwslt26_devset_chunk<chunk_ms>_borderp1_en<xx>/`. Refresh any
vendored copy with:

```bash
.venv-inference/bin/python submission/sync_artifacts.py
```

`sync_artifacts.py` is append-only for the main-track blind bundles: it
re-materialises the six frozen `750/1100` directories verbatim from
their source, and adds the additive bundles under `additive/chunk<ms>/`
without overwriting or renaming any existing main-track files.

Counts per direction (IWSLT blind test-set, 21 ACL talks each):

LOW regime (`submission/artifacts/main/low/*`):

| Direction | hypotheses | stream updates |
|-----------|------------|----------------|
| en -> de  | 21         | 6 775          |
| en -> it  | 21         | 7 330          |
| en -> zh  | 21         | 7 254          |

HIGH regime (`submission/artifacts/main/high/*`):

| Direction | hypotheses | stream updates |
|-----------|------------|----------------|
| en -> de  | 21         | 5 151          |
| en -> it  | 21         | 5 458          |
| en -> zh  | 21         | 5 550          |

## Dev-set calibration points (scored on MCIF dev-set, 919 refs)

`border_margin=1` is fixed across every row; only `chunk_ms` varies.
`750` / `1100` are the two frozen submission regimes; `850`, `1500`,
and `1900` are additive calibration points that extend the curve
without changing any other runtime knob.

| chunk_ms | Direction | BLEU  | chrF  | COMET  | LongYAAL CU | LongYAAL CA |
|----------|-----------|-------|-------|--------|-------------|-------------|
| 750      | en -> de  | 27.35 | 61.46 | 0.8669 | 1707 ms     | 1343 ms     |
| 850      | en -> de  | 28.76 | 62.14 | 0.8752 | 1998 ms     | 1629 ms     |
| 1100     | en -> de  | 29.99 | 62.78 | 0.8922 | 2414 ms     | 2037 ms     |
| 1500     | en -> de  | 32.63 | 64.21 | 0.9018 | 3528 ms     | 3136 ms     |
| 1900     | en -> de  | 33.99 | 64.88 | 0.9128 | 4766 ms     | 4362 ms     |
| 750      | en -> it  | 38.37 | 66.82 | 0.7875 | 1675 ms     | 1321 ms     |
| 850      | en -> it  | 40.10 | 68.02 | 0.8052 | 1984 ms     | 1621 ms     |
| 1100     | en -> it  | 42.75 | 69.24 | 0.8268 | 2315 ms     | 1947 ms     |
| 1500     | en -> it  | 44.46 | 70.06 | 0.8407 | 3484 ms     | 3096 ms     |
| 1900     | en -> it  | 45.38 | 70.70 | 0.8515 | 4623 ms     | 4238 ms     |
| 750      | en -> zh  | 35.02 | 33.88 | 0.7308 | 1672 ms     | 1498 ms     |
| 850      | en -> zh  | 36.01 | 34.97 | 0.7432 | 1947 ms     | 1767 ms     |
| 1100     | en -> zh  | 37.91 | 36.12 | 0.7636 | 2258 ms     | 2075 ms     |
| 1500     | en -> zh  | 39.86 | 37.81 | 0.7781 | 3272 ms     | 3085 ms     |
| 1900     | en -> zh  | 41.55 | 38.97 | 0.7891 | 4220 ms     | 4023 ms     |

Observations:

- Quality and latency scale monotonically with `chunk_ms` on every
  direction / every metric, with zero empty predictions at any row.
- `chunk_ms=850` sits right at the LOW/HIGH boundary (CU ≈ 2000 ms) and
  buys ~+1-2 BLEU / +0.008-0.018 COMET over the frozen `chunk_ms=750`
  LOW preset, at the cost of ~+290-310 ms CU.
- `chunk_ms=1500` is a mid-HIGH calibration anchor: CU ≈ 3.3-3.5 s (well
  inside the 2-4 s IWSLT HIGH band), +1.9-2.6 BLEU / +0.01-0.02 COMET
  over the frozen `chunk_ms=1100` HIGH preset while still respecting the
  latency envelope.
- `chunk_ms=1900` overshoots the HIGH band (CU ≈ 4.2-4.8 s, above the
  IWSLT 2-4 s envelope) but serves as the top-of-curve reference point:
  +3-4 BLEU / +0.02-0.03 COMET over the frozen `chunk_ms=1100` HIGH
  preset. Useful as a paper calibration anchor, not as a submission.

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
