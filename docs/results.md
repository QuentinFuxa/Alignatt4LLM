# Results

## Current Surfaces

- Gemma baseline preset: `gemma_low_latency` (`chunk_ms=850`)
- Gemma high-latency preset: `gemma_high_latency` (`chunk_ms=1500`)
- Stable MT baseline: `gemma_vllm_alignatt`
- Active EN->ZH route: `milmmt_vllm_alignatt`
- Fixed-cutoff counterfactual policy: `cut_last_target_units`

## Historical Submitted Anchors

The submitted paper used Qwen forced ASR plus Gemma vLLM AlignAtt MT. These
numbers remain reference anchors, not active optimization targets.

Low-regime dev logs (`chunk_ms=850`):

- en->de: BLEU `28.76`, chrF `62.14`, XCOMET-XL `0.8752`, LongYAAL CU `1997.8 ms`
- en->it: BLEU `40.10`, chrF `68.02`, XCOMET-XL `0.8052`, LongYAAL CU `1983.7 ms`
- en->zh: BLEU `36.01`, chrF `34.97`, XCOMET-XL `0.7432`, LongYAAL CU `1947.0 ms`

High-regime dev logs (`chunk_ms=1500`):

- en->de: BLEU `32.63`, chrF `64.21`, XCOMET-XL `0.9018`, LongYAAL CU `3528.2 ms`
- en->it: BLEU `44.46`, chrF `70.06`, XCOMET-XL `0.8407`, LongYAAL CU `3484.3 ms`
- en->zh: BLEU `39.86`, chrF `37.81`, XCOMET-XL `0.7781`, LongYAAL CU `3271.9 ms`

See the arXiv paper at https://arxiv.org/abs/2606.03967,
`docs/archive/2026-05-submission.md`, and
`docs/archive/2026-05-submission-scores.json` for the compact submission-era
record.

## EN->ZH MiLMMT Calibration Anchor

Single-audio microscope on local dev-set clip `ccpXHNfaoy.wav`
with `qwen_forced` ASR + `milmmt_vllm_alignatt` MT, `chunk_ms=750`,
`border_margin=0`, `top_k_heads=8`, no COMET:

- `min_source_mass=0.000`: BLEU `30.71`, chrF `28.05`, LongYAAL CU `1401 ms`
- `min_source_mass=0.020`: BLEU `35.79`, chrF `31.30`, LongYAAL CU `1832 ms`
- `min_source_mass=0.035`: BLEU `35.92`, chrF `31.74`, LongYAAL CU `1941 ms`
- `min_source_mass=0.050`: BLEU `37.24`, chrF `32.59`, LongYAAL CU `2127 ms`

These are calibration notes for the active MiLMMT route. They do not supersede
full dev-set scoring.

## Cutoff Comparison Workflow

Use `scripts/run_mt_cutoff_policy_sweep.py` to produce real streaming outputs
for AlignAtt and fixed `cut_last_target_units` policies, then score each output
with `evaluate_cascade_outputs.py` and summarize with
`scripts/report_mt_cutoff_policy_tradeoff.py`.

Claims should be made from scored output directories and their manifests, not
from one-off console logs.
