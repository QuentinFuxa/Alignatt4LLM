Welcome on the A100 machine.

This repo is now in the post-submission research phase. The public paper lives
on arXiv at https://arxiv.org/abs/2606.03967; paper sources and generated PDFs
are not part of the active tree. Active work should target reusable evidence
for the next iteration, not submission packaging.

## Research focus

- EN->ZH quality: make the `milmmt_vllm_alignatt` route clearly stronger than
  the Gemma MT route under comparable latency.
- EN->ZH latency: AlignAtt is expected to be more permissive than local
  agreement. For a comparable ASR input chunk size, an AlignAtt point should
  have lower LongYAAL CU than the local-agreement baseline; a 960 ms AlignAtt
  point landing near the 1.20 s / 1.28 s baseline latency regime is a failure
  signal, not a success.
- EN->ZH comparisons should use the public no-context baseline unless a report
  explicitly says it is studying the with-context baseline.
- Policy evidence: make AlignAtt vs fixed `cut_last_target_units` comparisons
  measurable, reproducible, and not benchmark-specific.

## Core rules

- Do not add hardcoded lexical repairs, phrase tables, or dataset-specific hacks.
- Prefer principled runtime, architecture, or policy changes over local patches.
- Rechallenge any AlignAtt configuration that becomes more conservative than
  local agreement. Do not accept quality gains that come mainly from delaying
  source commits or target emission unless the latency tradeoff still preserves
  AlignAtt's intended permissiveness.
- The first sanity check for an EN->ZH AlignAtt point is same-chunk latency
  against the public no-context local-agreement baseline. If it is not lower
  LongYAAL CU at the same ASR chunk size, pause quality tuning and audit the
  runtime knobs, manifests, ASR commit path, and MT acceptance path before
  launching broader sweeps.
- A guarded low-latency probe can guide experiments, but it is not the main
  clean AlignAtt claim. A promoted EN->ZH claim must state whether it is clean
  frontier AlignAtt or a guarded policy variant, and it must pass the
  same-chunk no-context latency check before being treated as evidence that
  AlignAtt beats local agreement.
- Distinguish pure AlignAtt evidence from auxiliary guard variants. Source
  LCP stability and source-LCP append slack, source regression gates,
  token-argmax frontier gates, minimum accessible source units, ASR punctuation
  delays, and source-bound prefill/split experiments are principled diagnostics
  or policy variants, but they must be reported as such and not described as
  the clean AlignAtt result.
- If source regression is tested, prefer auditing whether it is acting like a
  local-agreement hard wait. The `trim_target_unit` source-regression action is
  a guarded diagnostic for this: it lets the draft continue and trims only the
  accepted target-unit suffix whose AlignAtt source progression regresses,
  instead of stopping generation at the first regressive token. It must still
  beat the same-chunk no-context local-agreement latency before it is useful.
  The more permissive `trim_unrecovered` diagnostic keeps bounded local
  regressions that recover later in the same draft and trims only an
  unrecovered regressive suffix; use it to test whether source regression is
  accidentally behaving like local agreement.
- If `translation_alignatt_token_argmax_frontier_gate` is enabled, treat it as an
  auxiliary guard and challenge it separately when same-chunk CU is too high. A
  hard stop on the first future-source argmax can become local-agreement-like if
  later tokens in the same draft recover; first test the existing frontier
  patience axis and recoverable-frontier diagnostics before introducing any new
  runtime knob.
- If `translation_alignatt_min_accessible_source_units` is enabled, the
  historical `block` mode is a conservative diagnostic, not the preferred
  low-latency AlignAtt shape. When same-chunk CU is too high, test
  `translation_alignatt_min_accessible_source_units_mode=target_unit_cap` before
  scaling broader guarded sweeps: it keeps a bounded target-unit prefix instead
  of waiting for the full source-unit minimum.
- Surface normalization or exact prefix-repeat deduplication may be useful for
  append-only runtime hygiene, but it is not an AlignAtt mechanism. Artifacts
  explicitly named as surface/dedup probes are diagnostics and must not be used
  as clean AlignAtt evidence.
- Runs with provenance inaccessible/margin caps, including
  `translation_alignatt_max_inaccessible_source_mass < 1.0`, are guarded
  policy variants. The maintained runtime, preset, and runner defaults must
  keep this cap disabled at `1.0`; lower caps are explicit diagnostics and must
  not be used for the main clean claim.
- A source-accessible mass floor (`translation_alignatt_min_source_mass`) can
  be a clean AlignAtt source-accessibility variant when source regression,
  token-argmax gates, minimum source units, source LCP, and provenance
  inaccessible/margin caps are disabled. Report it as a source-mass-floor clean
  variant, not as the pure frontier result. The maintained runtime, preset, and
  runner defaults must keep this floor at `0.0`; positive floors are explicit
  experiment axes.
- An accepted-prefix source-accessible mass floor
  (`translation_alignatt_min_accepted_accessible_source_mass`) can also be a
  clean token-frontier source-mass axis when the same auxiliary guards are
  disabled. It trims prompt/non-source-dominated accepted target suffixes at
  target stability-unit boundaries. When enabled, the accepted prefix must pass
  the global source-accessible mean, the recent-unit mean, and the weakest
  recent target-unit mean; this prevents one grounded unit from hiding a
  prompt-only suffix. It must not be used to justify local-agreement-like
  waiting. The maintained default is `0.0`, and positive values must still pass
  the same-chunk no-context latency check.
- The `unit_mass` acceptance variant is also a clean source-mass AlignAtt
  candidate only when `translation_alignatt_min_source_mass > 0` and the same
  auxiliary guards are disabled. It is intentionally source-mass permissive,
  not an argmax-frontier unit policy; adding an argmax frontier check to it
  would make it more local-agreement-like. Do not treat `unit_mass` with a zero
  source mass floor as meaningful clean evidence.
- The `unit_mass_source_bearing` acceptance variant is a clean source-bearing
  unit AlignAtt candidate only when
  `translation_alignatt_source_bearing_min_source_mass > 0`,
  `translation_alignatt_min_source_mass == 0`, and the same auxiliary guards are
  disabled. Use it to test whether blocking ungrounded target continuations can
  recover quality without local-agreement-style waiting. A source-bearing unit
  must contain at least one source-bearing token, and each source-bearing token
  must still satisfy the AlignAtt source frontier or the configured soft
  frontier. Accepting a target unit whose source mass is entirely below
  threshold, or whose source-bearing token points beyond the allowed frontier,
  is not valid clean evidence.
  The maintained defaults keep
  `translation_alignatt_source_bearing_min_source_mass=0.005` and
  `translation_alignatt_source_bearing_hard_inaccessible_cap=1.0`; lower caps
  are guarded diagnostics and must not be interpreted as clean source-bearing
  evidence.
- The `unit_argmax`, `unit_consensus`, and `unit_conf` acceptance variants are
  clean unit-frontier AlignAtt candidates only when
  `translation_alignatt_min_source_mass == 0` and the same auxiliary guards are
  disabled. If a source-mass floor is set with any of them, report that as a
  guarded/unused-floor diagnostic, not as clean AlignAtt evidence. `unit_conf`
  adds a pure attention-confidence gate on top of the argmax frontier: a unit
  is deferred when fewer than `translation_alignatt_min_alignment_confidence`
  of the selected heads agree (within one source token) with the consensus
  argmax. The maintained default is `0.0` (gate disabled, identical to
  `unit_argmax`); positive floors are clean attention-only experiment axes and
  must still pass the same-chunk no-context latency check.
- `translation_alignatt_source_frontier_action=trim_unrecovered` is a clean
  recoverable-frontier AlignAtt variant only when the same auxiliary guards are
  disabled. It lets the draft recover from bounded within-draft future-frontier
  blips and trims only an unrecovered target-unit suffix. It is not local
  agreement and must not be combined with hard source waiting for the main
  claim; it still must pass the same-chunk no-context latency check.
- Do not overproduce tests; protect reusable mechanisms and real invariants.
- Streaming output is append-only: never emit `deleted_tokens` or `deleted_string`.
- Keep claims tied to manifests, score files, and exact commands.

## Active runtime

- Main runtime: `cascade/runtime.py`
- Runtime presets: `cascade/presets.py`
- SimulStream processor: `cascade/simulstream_processor.py`
- Canonical batch runner: `run_simulstream_batch.py`
- Canonical single-audio validation loop: `run_simulstream_compare.py` on
  `data/smoke/alignatt_smoke18.wav`

## Supported backends

- ASR `qwen_forced` — stable default.
- ASR `gemma_vllm_qk_fast` — Gemma AlignAtt ASR research path; standalone
  entrypoint: `gemma_asr_low_latency.py`.
- MT `gemma_vllm_alignatt` — stable Gemma baseline route.
- MT `milmmt_vllm_alignatt` — active MiLMMT improvement route.

Canonical active presets are `gemma_low_latency` and `gemma_high_latency`.
Use these names in new commands, docs, manifests, and result tables.

## Operational constraints

- Model reloads are expensive; reuse hot bundles whenever possible.
- Treat full streaming runs as costly experiments, not cheap probes.
- Validate on one clip before launching any broader sweep.
- Keep backend comparisons sequential and isolated on a single GPU.
- Use `.venv-inference` for runtime work and `.venv-evaluation` for scoring.

## Data and docs

- Tracked dev-set: `data/devset/`
- Compatibility alias: `dev-set/` -> `data/devset/`
- Local untracked test-set: `data/testset/`
- AlignAtt heads: `data/alignatt_heads/`
- Smoke fixtures: `data/smoke/`
- Current docs: `docs/system.md`, `docs/results.md`, `docs/status.md`
- Historical submission note: `docs/archive/2026-05-submission.md`
- Historical cleanup notes: `docs/archive/2026-04-history.md`

The Docker submission surface is historical and has been removed from the
active tree. Do not recreate `submission/` or a root `Dockerfile` unless the
project explicitly returns to packaging work.
