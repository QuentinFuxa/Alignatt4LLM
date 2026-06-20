# `tools/`

Utilities that support experiments around the public `alignatt4llm` package.
The stable user-facing entrypoints are the `alignatt-*` console scripts defined
in `pyproject.toml`; files here are research and reporting helpers.

## `tools/research/`

Experiment launchers and calibration scripts. This includes ASR comparison
probes, additive chunk sweeps, MT cutoff-policy sweeps, head-set construction,
and local test-set wrappers.

## `tools/reports/`

Post-processing scripts for scored outputs, diagnostic plots, replay analyses,
and quality-latency summaries. Reports should always cite the manifest and
score files that produced them.

## `tools/bootstrap/`

Environment setup helpers for the A100 inference stack, including the Qwen ASR
patch used by the maintained inference environment.
