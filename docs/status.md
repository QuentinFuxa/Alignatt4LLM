# Status

## Public Branch State

- This branch is the public-facing research-code surface for AlignAtt4LLM.
- The paper source and generated PDF are not vendored here. Use the arXiv
  record: https://arxiv.org/abs/2606.03967.
- The Docker submission surface is historical and has been removed from the
  active tree.
- Active runtime code lives in `cascade/`; active presets live in
  `cascade/presets.py`.
- The canonical MiLMMT backend spelling is `milmmt_vllm_alignatt`.
- Gemma and MiLMMT MT routes share the Gemma-family vLLM implementation in
  `cascade/mt/gemma_vllm_backend.py`.

## Active Workflows

- End-to-end streaming: `run_simulstream_batch.py`
- Single-audio backend comparison: `run_simulstream_compare.py`
- EN->ZH MiLMMT calibration: `scripts/run_alignatt_lean_sweep.py`
- EN->ZH candidate promotion ranking:
  `scripts/report_enzh_candidate_promotions.py`
- EN->ZH guarded permissiveness ranking:
  `scripts/report_enzh_combined_guard_diagnostics.py`
- AlignAtt vs fixed cutoff:
  `scripts/run_mt_cutoff_policy_sweep.py` and
  `scripts/report_mt_cutoff_policy_tradeoff.py`

## Decisions

- New commands should use `gemma_low_latency` / `gemma_high_latency`.
- Old preset names are retained only inside compatibility code and archive
  material.
- The submitted Gemma cascade is now a historical baseline, not the organizing
  frame for new work.
- MiLMMT is the active MT improvement route, especially for EN->ZH.
- Official organizer baseline outputs are parsed on demand instead of vendored.
- Full streaming runs remain expensive experiments; validate on one local clip
  before launching a broader sweep.

## EN->ZH Evidence Discipline

- Compare against the no-context public baseline unless a report explicitly
  says it studies the with-context baseline.
- AlignAtt must be more permissive than local agreement at comparable ASR chunk
  size; quality gains that mainly come from waiting longer are not acceptable
  evidence.
- Clean AlignAtt evidence must be separated from guarded policy variants such
  as source regression gates, token-argmax frontier gates, minimum source-unit
  waits, source-LCP stability, and provenance caps.
- Claims should be tied to manifests, score files, and exact commands.
- Streaming output is append-only: do not emit `deleted_tokens` or
  `deleted_string`.

## Public Release Checks

- Publish from a sanitized branch whose reachable history does not include
  removed private packaging or paper-source material.
- Run the maintained lightweight test suite before pushing.
- Run an A100 inference smoke before using the public branch for new result
  claims.
