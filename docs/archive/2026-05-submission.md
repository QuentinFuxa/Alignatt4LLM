# 2026-05 Submission Archive

The IWSLT 2026 paper and Docker-oriented submission surface are historical.
The public paper artifact is the arXiv record:
https://arxiv.org/abs/2606.03967.

Active development now uses `cascade/presets.py` and the research-facing preset
names `gemma_low_latency` and `gemma_high_latency`.

## Preset Mapping

- Historical `main_low_latency` -> active `gemma_low_latency`
- Historical `main_high_latency` -> active `gemma_high_latency`

The legacy names may still resolve through `get_runtime_preset(...)` for old
artifacts and wrappers, but they should not be used in new docs or commands.

## Submitted Runtime

- ASR: `qwen_forced`
- MT: `gemma_vllm_alignatt`
- Directions: `en-de`, `en-it`, `en-zh`
- Low regime: `chunk_ms=850`
- High regime: `chunk_ms=1500`

## Retained Score Anchors

Low regime:

- en->de: BLEU `28.76`, chrF `62.14`, XCOMET-XL `0.8752`, LongYAAL CU `1997.8 ms`
- en->it: BLEU `40.10`, chrF `68.02`, XCOMET-XL `0.8052`, LongYAAL CU `1983.7 ms`
- en->zh: BLEU `36.01`, chrF `34.97`, XCOMET-XL `0.7432`, LongYAAL CU `1947.0 ms`

High regime:

- en->de: BLEU `32.63`, chrF `64.21`, XCOMET-XL `0.9018`, LongYAAL CU `3528.2 ms`
- en->it: BLEU `44.46`, chrF `70.06`, XCOMET-XL `0.8407`, LongYAAL CU `3484.3 ms`
- en->zh: BLEU `39.86`, chrF `37.81`, XCOMET-XL `0.7781`, LongYAAL CU `3271.9 ms`

The submitted paper source, generated PDF, historical Docker helpers, and full
dev-log bundles have been removed from the active tree. The machine-readable
score anchors are preserved in
`docs/archive/2026-05-submission-scores.json`.
