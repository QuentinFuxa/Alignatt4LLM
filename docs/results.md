# Results

## Current maintained surfaces

- Batch runner default: `chunk_ms=800`
- Submission low regime: `chunk_ms=850`
- Submission high regime: `chunk_ms=1500`
- Submitted MT route: `gemma_vllm_alignatt`
- Experimental MT route: `milmmt_vllm_alignatt`
- Official organizer baseline outputs are parsed by
  `scripts/parse_official_baseline_outputs.py` from
  `https://github.com/user-attachments/files/26411361/outputs.zip`.

## Reference point

Historical en→de reference on `data/devset/audio/ccpXHNfaoy.wav` at `chunk_ms=450`:

- BLEU `27.5`
- chrF `63.5`
- COMET `0.861`
- LongYAAL CU `1766 ms`
- LongYAAL CA `1473 ms`
- RTF `0.401`

## Submission dev-log anchors

Frozen low-regime dev logs (`chunk_ms=850`):

- en→de: BLEU `28.76`, chrF `62.14`, XCOMET-XL `0.8752`, LongYAAL CU `1997.8 ms`
- en→it: BLEU `40.10`, chrF `68.02`, XCOMET-XL `0.8052`, LongYAAL CU `1983.7 ms`
- en→zh: BLEU `36.01`, chrF `34.97`, XCOMET-XL `0.7432`, LongYAAL CU `1947.0 ms`

Frozen high-regime dev logs (`chunk_ms=1500`):

- en→de: BLEU `32.63`, chrF `64.21`, XCOMET-XL `0.9018`, LongYAAL CU `3528.2 ms`
- en→it: BLEU `44.46`, chrF `70.06`, XCOMET-XL `0.8407`, LongYAAL CU `3484.3 ms`
- en→zh: BLEU `39.86`, chrF `37.81`, XCOMET-XL `0.7781`, LongYAAL CU `3271.9 ms`

Use the manifest inside each output directory as the exact provenance record.

## EN->ZH MiLMMT provenance-mass calibration

Single-audio microscope on `/home/dev-set/mcif-long-trans/audio/ccpXHNfaoy.wav`
with `qwen_forced` ASR + `milmmt_vllm_alignatt` MT, `chunk_ms=750`,
`border_margin=0`, `top_k_heads=8`, no COMET:

- `min_source_mass=0.000`: BLEU `30.71`, chrF `28.05`, LongYAAL CU `1401 ms`
- `min_source_mass=0.020`: BLEU `35.79`, chrF `31.30`, LongYAAL CU `1832 ms`
- `min_source_mass=0.035`: BLEU `35.92`, chrF `31.74`, LongYAAL CU `1941 ms`
- `min_source_mass=0.050`: BLEU `37.24`, chrF `32.59`, LongYAAL CU `2127 ms`

These are experimental MiLMMT calibration notes only. They do not replace the
submitted Gemma presets or the frozen submission dev-log anchors above.
