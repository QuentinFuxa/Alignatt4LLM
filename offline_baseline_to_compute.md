# Offline Baseline To Compute

This note defines what to compute if we decide to add an offline baseline to
Table 1 or to the appendix.

## Short answer

Compute an **offline MT baseline**, not an ASR-only baseline.

The most useful and defensible variant is:

> **Qwen final ASR + Gemma offline MT**

This keeps the cascade setting, keeps ASR errors, and isolates the effect of
the simultaneous MT policy. In the paper, this can be described as running the
same ASR front-end to completion and then translating the final source text with
Gemma in final/offline mode, without AlignAtt truncation.

Do **not** add an ASR-only offline row to Table 1. ASR-only gives WER/CER, while
Table 1 reports translation quality and latency. It can be useful as a diagnostic
appendix result, but it does not answer the baseline question for speech
translation.

## Priority order

### 1. Preferred: final Qwen ASR text -> Gemma offline MT

Use this if final Qwen transcripts are already available from the dev/test runs.

Inputs:

- final source text from Qwen ASR, one segment or utterance per line;
- target references in `data/devset/ref/{de,it,zh}.txt`;
- Gemma MT backend with `is_partial=False`;
- evaluation environment for BLEU, chrF, and XCOMET.

What this measures:

- offline MT quality given the same ASR front-end;
- how much quality is lost by the streaming MT policy and AlignAtt gating;
- not streaming latency.

How to report it:

- row label: `Qwen final ASR + Gemma offline MT`;
- BLEU/chrF/XCOMET: report normally;
- LongYAAL/CU or CA: mark as `--` or `offline`, because it is not a simultaneous
  system.

This is the row to put in Table 1 if we can compute it cleanly.

### 2. Fallback: gold English source -> Gemma offline MT

Use this if final Qwen ASR transcripts are not available quickly.

Inputs:

- `data/devset/ref/en.txt` as source text;
- `data/devset/ref/{de,it,zh}.txt` as target references;
- Gemma MT backend with `is_partial=False`.

What this measures:

- oracle offline MT upper bound;
- translation-only quality without ASR errors;
- whether Gemma itself can reach higher quality when the streaming constraint is
  removed.

How to report it:

- row label: `Gemma offline MT (gold source)` or `Gemma offline MT oracle`;
- do **not** call it a cascade baseline;
- preferably put it in the appendix unless Table 1 has enough space and the label
  is very explicit;
- LongYAAL/CU or CA should be `--` or `offline`.

This is useful, but less directly comparable than priority 1.

### 3. Avoid for Table 1: ASR offline only

ASR offline alone would be:

> full-audio Qwen ASR -> WER/CER against English reference

This can explain whether ASR is the bottleneck, but it does not produce German,
Italian, or Chinese translations. Therefore it should not be added to Table 1
unless followed by offline MT.

## Context-window constraint

Do not translate whole talks in one Gemma call unless we have checked that every
talk fits safely in the model context window.

Safer choices:

1. translate official MCIF/source segments independently;
2. translate final ASR segments independently if the ASR output is segmented;
3. only use talk-level translation if the longest source text fits comfortably.

Segment-level offline MT is acceptable if we label it clearly. It avoids context
overflow and is fast to compute.

## Practical command sketch

The existing MT backend supports final/offline decoding:

- `run_mt_backend_parity.py --is-final` calls the Gemma MT backend with
  `is_partial=False`;
- in `cascade/mt/gemma_vllm_backend.py`, `is_partial=False` decodes a full
  translation and skips the AlignAtt observer.

For a smoke test on one sentence:

```bash
.venv-inference/bin/python run_mt_backend_parity.py \
  --source-text "This is a short source sentence." \
  --target-lang-code de \
  --is-final \
  --output tmp/offline_mt_smoke_de.json
```

For the actual baseline, run the same final-mode MT over all source lines for
each target language (`de`, `it`, `zh`) and write one hypothesis line per source
line. Then score against:

```text
data/devset/ref/de.txt
data/devset/ref/it.txt
data/devset/ref/zh.txt
```

Recommended output layout:

```text
outputs/offline_mt/
  qwen_final_asr/
    en-de/hyp.txt
    en-it/hyp.txt
    en-zh/hyp.txt
  gold_source/
    en-de/hyp.txt
    en-it/hyp.txt
    en-zh/hyp.txt
```

## Acceptance checks before putting numbers in the paper

- The source/hypothesis/reference files have the same number of lines.
- The source condition is named exactly: `final Qwen ASR` or `gold source`.
- Gemma was run in final/offline mode (`is_partial=False`), not streaming mode.
- No LongYAAL/CU value is reported for the offline row unless we deliberately
  define a separate offline latency convention.
- If using gold source, the row is described as an oracle MT upper bound, not as
  a submitted cascade system.

## What to add to the paper

If priority 1 is computed:

> We additionally report an offline cascade variant in which Qwen ASR is run to
> completion and the final transcript is translated by Gemma without AlignAtt
> gating. This isolates the quality cost of the simultaneous MT policy.

If only priority 2 is computed:

> We additionally report an oracle offline MT variant using the reference English
> source segments. This measures the translation model upper bound without ASR
> errors or simultaneous decoding constraints.

