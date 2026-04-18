# Current Plan

Date: 2026-04-18

This file tracks only the remaining work. Completed work is already frozen in
`DECISIONS.md`, `submission/SUBMISSION_BUNDLE.md`, `submission/README.md`, and
the vendored artifacts under `submission/`.

## Current frozen state

Main-track validated presets:

- LOW regime: `main_low_latency`
  - `chunk_ms=750`
  - `translation_alignatt_border_margin=1`
  - full dev-set validated on `en->de`, `en->it`, `en->zh`
- HIGH regime: `main_high_latency`
  - `chunk_ms=1100`
  - `translation_alignatt_border_margin=1`
  - full dev-set validated on `en->de`, `en->it`, `en->zh`

Main-track submission bundle already materialized under `submission/`:

- `submission/artifacts/main/low/en-de/`
- `submission/artifacts/main/low/en-it/`
- `submission/artifacts/main/low/en-zh/`
- `submission/artifacts/main/high/en-de/`
- `submission/artifacts/main/high/en-it/`
- `submission/artifacts/main/high/en-zh/`
- `submission/ARTIFACT_INDEX.json`

These existing `750 ms` and `1100 ms` bundles are now treated as immutable.

## Active objective

Extend `submission/` with additional chunk-size bundles and their scored
results, **without overwriting or mutating the existing frozen 750/1100
content**.

## Operating rules

- Append-only inside `submission/`: never overwrite, rename, or delete the
  existing `main/low` and `main/high` bundles.
- New chunk sizes must be added as clearly separate additive bundles.
- For every new chunk size, vendor both:
  - artifacts: `manifest.json`, `hypothesis.jsonl`, `stream_updates.jsonl`
  - results: `evaluation.json`, `evaluation.report.txt`, `scores.tsv`
- Keep provenance explicit: each additive bundle should still point back to its
  originating `outputs/...` directory.
- Keep runs sequential on one GPU and avoid broad sweeps before one candidate
  is working cleanly.

## Additive work items

### 1. Add `chunk_ms=850`

AJOUTER dans `submission/` les artifacts ET les résultats pour
`chunk_ms=850ms`.

Recommended destination shape:

- `submission/artifacts/additive/chunk850/...`
- `submission/results/additive/chunk850/...`

Do not overwrite the existing `750 ms` LOW bundle.

### 2. Add `chunk_ms=1900`

AJOUTER dans `submission/` les artifacts ET les résultats pour
`chunk_ms=1900ms`.

Recommended destination shape:

- `submission/artifacts/additive/chunk1900/...`
- `submission/results/additive/chunk1900/...`

Do not overwrite the existing `1100 ms` HIGH bundle.

## Short checklist

- Keep the frozen `750 ms` and `1100 ms` submission bundles untouched.
- Add `850 ms`, `1900 ms`
  `submission/`.
- Always vendor both raw artifacts and scored result files.
- Update the submission-side index or bundle notes only by extension, never by
  replacement.
