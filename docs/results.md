# Results

## Current maintained surfaces

- Batch runner default: `chunk_ms=800`
- Submission low regime: `chunk_ms=850`
- Submission high regime: `chunk_ms=1500`

## Reference point

Historical en→de reference on `data/devset/audio/ccpXHNfaoy.wav` at `chunk_ms=450`:

- BLEU `27.5`
- chrF `63.5`
- COMET `0.861`
- LongYAAL CU `1766 ms`
- LongYAAL CA `1473 ms`
- RTF `0.401`

## Submission dev-log anchors

LOW (`main_low_latency`, `chunk_ms=850`):

- en→de: BLEU `28.76`, chrF `62.14`, XCOMET-XL `0.8752`, LongYAAL CU `1997.8 ms`
- en→it: BLEU `40.10`, chrF `68.02`, XCOMET-XL `0.8052`, LongYAAL CU `1983.7 ms`
- en→zh: BLEU `36.01`, chrF `34.97`, XCOMET-XL `0.7432`, LongYAAL CU `1947.0 ms`

HIGH (`main_high_latency`, `chunk_ms=1500`):

- en→de: BLEU `32.63`, chrF `64.21`, XCOMET-XL `0.9018`, LongYAAL CU `3528.2 ms`
- en→it: BLEU `44.46`, chrF `70.06`, XCOMET-XL `0.8407`, LongYAAL CU `3484.3 ms`
- en→zh: BLEU `39.86`, chrF `37.81`, XCOMET-XL `0.7781`, LongYAAL CU `3271.9 ms`

Use the manifest inside each output directory as the exact provenance record.
