# Bilan mid-run — IWSLT 2026 submission prep, 2026-04-17 afternoon

Scope: everything I measured or produced during this Claude session on the
H100 machine. Purpose: hand off a clean picture to the next agent working on
low-latency knob tuning, so they don't redo the same mistakes.

**This file deliberately lists my errors too.** I was corrected twice today
on flow / mechanism, and those corrections matter for whoever tunes next.

---

## 0. Hard deadline context

- IWSLT 2026 Simultaneous Translation submission deadline: **Friday
  2026-04-17 23:59 AoE.**
- IWSLT requires: (a) dev-set SimulStream logs per direction (to assign a
  latency regime), (b) test-set outputs, (c) Docker image or log submission
  with README.
- IWSLT quality metrics: **COMET-XL primary**, chrF + BLEURT secondary.
  BLEU is NOT part of the primary ranking.
- Latency regimes (by non-computation-aware LongYAAL): `0-2 s` = low,
  `2-4 s` = high.
- Target hardware: H100 80 GB. We are on that hardware.

---

## 1. What is already produced and submittable

All paths under `outputs/`. All runs use
`alignment_backend_name=qwen_forced` + `mt_backend_name=gemma_vllm_alignatt`,
`chunk_ms=450`, `min_start_seconds=2.0`, `asr_commit_mode=punctuation_lcp`,
`translation_alignatt_rewind_threshold=8`, `translation_alignatt_border_margin=0`,
`translation_alignatt_min_source_mass=0.0` unless stated otherwise.

### 1.1 Dev-set (MCIF, 21 clips, 919 ref segments)

Directory `dev-set/` (renamed from `test-set/` today — see §7 infra notes).

| Direction | Preset                  | BLEU  | chrF  | COMET  | LongYAAL CU | LongYAAL CA | Empty |
|-----------|-------------------------|-------|-------|--------|-------------|-------------|-------|
| en→de     | `main_low_latency`      | 27.91 | 62.01 | 0.8618 | 2263.8 ms   | 1920.7 ms   | 1/919 |
| en→it     | `main_low_latency`      | 38.83 | 68.11 | 0.7781 | 2306.3 ms   | 1981.0 ms   | 0/919 |
| en→zh     | `main_low_latency` + **V3 prompt** | 40.85 | 37.30 | 0.7276 | 2219.8 ms   | 2059.4 ms   | 0/919 |

Outputs:
- `outputs/iwslt26_devset_main_low_ende/`
- `outputs/iwslt26_devset_main_low_enit/`
- `outputs/iwslt26_devset_main_low_enzh_promptV3/` (the V3-prompt run is the one to keep; the earlier `_enzh` and `_enzh_simplified` are superseded).

**Regime**: all three are in the HIGH regime (CU > 2000 ms). Baseline IWSLT
en→zh is at CU 1909 ms (just below 2 s, LOW). We are currently on the wrong
side of the low/high boundary. Breaking into LOW is the remaining lever.

### 1.2 Test-set (official IWSLT blind, 21 ACL talks, 5.2–11.2 min each)

Directory `test-set/` (contains the real IWSLT blind audio + PDFs, downloaded
and unpacked today from the SharePoint zip).

| Direction | Preset                  | Wallclock | RTF   | Updates | Needs rerun ? |
|-----------|-------------------------|-----------|-------|---------|---------------|
| en→de     | `main_low_latency`      | 33.2 min  | 0.254 |  8 522  | No            |
| en→it     | `main_low_latency`      | 33.3 min  | 0.254 |  8 714  | No            |
| en→zh     | `main_low_latency` (old "Chinese" prompt) | 34.8 min | 0.266 | 9 676 | **Yes — current run predates the V3 zh prompt fix** |

Outputs:
- `outputs/iwslt26_testset_main_low_ende/`
- `outputs/iwslt26_testset_main_low_enit/`
- `outputs/iwslt26_testset_main_low_enzh/` — **stale, re-run required once the dev-set policy tuning converges**.

Nothing test-set is scorable locally (blind set, no refs). IWSLT does the
scoring.

### 1.3 Paper artifacts (extra-context sub-track)

24 JSON artifacts in `data/paper_artifacts/` (21 test-set + 3 dev-set). Not
used in any submission run, because the context sub-track was not chosen.
Kept available in case the extra-context track is attempted.

Two artifacts have weak parses (`lwDNGJCFIK`: empty title/abstract;
`lJJuRCeung`: truncated title). Fine for now, would need re-extraction if
context track is activated.

---

## 2. Gemma MT prompt evolution

Three prompt variants tested. All other parameters held constant.

### 2.1 V1 — original
System prompt ended with:
> *"Translate from the beginning of the current sentence, preserve names and
> technical terms when they are already clear, and let the runtime decide
> which drafted tokens are committed."*

Language field plumbing: `"zh" → "Chinese"` in `LANGUAGE_CODE_TO_NAME`.
Consequence: Gemma drifted between simplified and traditional characters on
zh (measured: **2.35 %** of CJK chars were Traditional-only).

### 2.2 V2 — "Simplified Chinese" language label only
Changed `LANGUAGE_NAME_TO_CODE["Simplified Chinese"] = "zh"` so the Gemma
prompt is rendered with `"Simplified Chinese"` everywhere `{target_lang}`
appears.

Effect on full dev-set en→zh: BLEU 38.70 → **40.42** (+1.72), COMET basically
unchanged (0.7273 → 0.7259, within noise). Trad drift dropped from 2.35 % →
**0.00 %** (round-trip zhconv check). Kept.

### 2.3 V3 — preserve-names + target-idioms rule
Swapped the old last line for three explicit lines:

> *"Translate from the beginning of the current sentence.*
> *Keep {source_lang} personal names and technical acronyms verbatim in the
> output; render other proper nouns in their established {target_lang} form.*
> *Use established {target_lang} terminology for domain concepts.*
> *Let the runtime decide which drafted tokens are committed."*

Effect on full dev-set en→zh vs V2: BLEU +0.43 (40.42 → 40.85), chrF +0.63,
COMET +0.0017. Small but consistent. **Kept.** This is the prompt used for
the numbers in §1.1.

**Design note**: the rule is deliberately language-agnostic (applies to de /
it / zh uniformly). I did NOT check V3 on en→de / en→it full dev-sets — the
en→de / en→it numbers in §1.1 were produced with V1. Running V3 on de / it
is a fast, cheap verification (≈35 min per direction) that should probably
happen before submission freezes.

---

## 4. Latency attempts (what I got wrong, what's actually true)

### 4.1 `min_start_seconds: 2.0 → 1.0` — no effect

Full dev-set en→de, otherwise identical config:

| Setting                | BLEU  | chrF  | COMET  | CU       | CA       |
|------------------------|-------|-------|--------|----------|----------|
| `min_start_seconds=2.0`| 27.91 | 62.01 | 0.8618 | 2263.8   | 1920.7   |
| `min_start_seconds=1.0`| 27.67 | 61.92 | 0.8599 | **2262.4** | 1942.8 |

**Δ CU: −1.4 ms.** Not a latency knob for this cascade — the ASR simply does
not commit anything in the first 2 s of audio anyway, so the floor never
fires.

### 4.2 Where I was wrong about the bottleneck

I initially claimed the CU bottleneck was "ASR punctuation_lcp waits for
stable sentence-terminal punctuation before committing, ≈400-600 ms of
sentence-end wait." User corrected me, correctly, that

> "LCP is only for ASR commits, not for MT. MT consumes everything every
> round."

After reading `cascade/runtime.py:render_translation` (line 995+) I confirm:

- On every chunk, `should_run_partial_mt` can fire a `translate_with_mt(
  is_partial=True, source_text=<full partial ASR hypothesis>, ...)`.
- The MT is given the **entire current ASR partial prefix** — not just what
  `punctuation_lcp` has committed.
- `accessible_source_token_count` on the prompt source map is driven by
  audio timestamps (`cascade_source_frontier.py:81-89`): a source unit is
  accessible iff `unit.end_ms ≤ current_audio_ms − inaccessible_ms`. With
  our default `inaccessible_ms=0`, effectively **every source unit present
  in the partial ASR is accessible**.
- AlignAtt's `source_frontier` stop reason therefore triggers only when the
  MT draft's attention argmax points to a position `≥` the number of
  source tokens — i.e., beyond the end of what has been transcribed so far
  (MT hallucinating forward). It is rarely the dominant gate when
  `inaccessible_ms=0`.

**Punctuation_lcp therefore primarily controls what segments are appended
to the committed utterance history fed back into the MT as
`translation_history`.** It does not gate the partial MT call itself.

### 4.3 The actual latency knobs (revised, honest)

Given §4.2, the real parameters governing how many target tokens come out
per chunk are:

- `partial_max_new_tokens` (default 16, our preset 16): the max draft
  length on the **first** partial MT call after an ASR event.
- `partial_followup_max_new_tokens` (default 8, our preset 8): max draft
  length on **subsequent** partial MT calls before the commit boundary.
- `translation_alignatt_rewind_threshold` (default 8, our preset 8): max
  backward jump of the attention argmax between consecutive drafted tokens
  before we call `alignatt:rewind` and throw away the rest. Only relevant
  when argmax is actually unstable.
- `translation_alignatt_border_margin` (**new knob I added today**, default
  0): speculative look-ahead beyond `accessible_source_token_count`. Only
  helps when the argmax overshoots the tail; given our `inaccessible_ms=0`
  that probably means the model is hallucinating future content, so the
  realistic useful range is small (1–3). **Not validated against BLEU /
  COMET yet.**
- `chunk_ms` (default 450): controls how frequently the cascade loop ticks
  and how much new audio the ASR sees per tick. Not directly a latency
  knob; smaller chunks = more tick overhead; larger chunks = coarser commit
  timing.
- `asr_commit_mode` (default `punctuation_lcp`): switching to
  `stable_and_accessible K=3` on `RESULTS.md` numbers lowered CU to
  1919 ms (LOW regime) but dropped BLEU from 27.51 → 18.71 because it
  discards the punctuation signal our Qwen3 ASR emits reliably. Keep it.

### 4.4 Attempted tunings that were interrupted / inconclusive

- `chunk_ms=300` dev-set en→de: started, user stopped early ("j'y crois
  pas"). No aggregated numbers.
- `chunk_ms=350`: started, stopped before any clip finished.
- `border_margin=4` 2-clip smoke: launched and interrupted by the user
  before a meaningful measurement was produced. The knob itself is coded
  and plumbed end-to-end (CLI flag, override_keys, runtime_config field,
  policy check). **Ready to test.**

### 4.5 What I recommend the next agent actually tries

Policy-principled, one knob moved at a time, dev-set en→de as the
reference, all other knobs held at the current preset:

1. `partial_max_new_tokens ∈ {24, 32, 48}` — emits more target tokens per
   partial call. Direct leverage on CU: more tokens accepted per tick.
2. `partial_followup_max_new_tokens ∈ {12, 16}` — same logic for
   subsequent-partial calls.
3. Only *after* the above two have a clean Pareto: sweep
   `translation_alignatt_border_margin ∈ {1, 2, 3}` on top of the best
   point, to see if speculative emission past the frontier is a real lever
   or cosmetic.
4. `chunk_ms ∈ {400, 500}` only if the above top out above 2 s CU — small
   chunk changes to test whether the ASR tick cadence is the last gate.

**Hard constraint**: do not drop `asr_commit_mode` from `punctuation_lcp`
— the 11 BLEU penalty swamps any latency win at our COMET range.

---

## 5. Known correctness issues in the metrics

Two things that need to be mentioned honestly:

### 5.1 Misdiagnosed `trad-drift ratio` metric

Earlier in the session I reported en→zh "19 % traditional-only chars" using
a hand-made char set. That set had both simplified AND traditional
characters in the wrong buckets, so it **double-counted**. The correct
measurement is via `zhconv.convert(text, "zh-cn")` round trip:

- V1 Chinese prompt: 678 / 28 830 CJK chars changed under t2s → **2.35 %**
  trad-only. Not 19 %.
- V2/V3 Simplified Chinese prompt: 0 / 28 828 → **0.00 %** trad-only.

So the real trad-drift on V1 was small, and the V2 "Simplified Chinese"
prompt removes it completely. The BLEU gain from V1 → V2 (+1.72) is not
attributable to trad-drift removal alone — the prompt label change
likely also nudges character-level vocabulary choice within simplified. To
confirm, compare sentence-level output diffs V1 → V2; I did not do that.

### 5.2 Obsolete offline rescore latency metrics are bogus

The historical offline rescore run was produced by
re-translating the Gemma V3 run's final accepted English prefix offline, then synthesising per-character emission delays by linear
interpolation between Gemma's first and last emission times. This
preserves BLEU / chrF / COMET validity but the reported LongYAAL CU /
CA numbers (10 453 ms / 9 145 ms) are artefacts of the interpolation —
ignore them.

### 5.3 No per-clip COMET breakdown

`evaluation.json` only carries aggregate scalars. I did not produce
per-clip COMET numbers (the eval doesn't expose them by default). If the
next agent needs to identify specific weak clips, they'll need to run
XCOMET-XL per clip (small wrapper needed).

---

## 6. Infrastructure state

### 6.1 Folder rename

Today I renamed the original MCIF folder `test-set/` → `dev-set/` and
downloaded the real IWSLT blind test set into a fresh `test-set/`. All
active code paths (`cascade_artifacts.py`, `run_simulstream_batch.py`,
`scripts/*.py`, README) were updated to point at `dev-set/`. Historical
docs (`DECISIONS.md`, `PLAN.md`, `docs/archive/`) still contain the old
`test-set/` paths — this is deliberate (append-only audit log). The
bundles live under:

- `dev-set/audio/` + `dev-set/pdf/` + `dev-set/ref/` (MCIF, 21 clips, has
  references, scorable)
- `test-set/audio/` + `test-set/pdf/` (real IWSLT blind, 21 en ACL talks,
  5–11 min each, no references)
- `test-set/cs_en_slides/` (IWSLT cs-en Linguistic Mondays, 13 clips, 1
  partially truncated: `R-4.wav` is half-length; rest intact)
- `zip_downloaded/` for the raw downloaded archives (kept for traceability)

`.dockerignore` now excludes both `dev-set/`, `test-set/`, `zip_downloaded/`
and the old exclusions.

### 6.2 Docker

`Dockerfile` is in place but NOT built or validated today. Known things
it needs:

- `VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0` already in the ENV (fixed
  today, otherwise crashes on H100 warmup).
- `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1` → models need to be baked
  into the cache OR mounted at runtime. Currently the image does not bake
  them (`COPY . /app` doesn't include the HF cache). For submission, either
  bake the three snapshots (Qwen3-ASR-1.7B + Qwen3-ForcedAligner-0.6B +
  Gemma-4-E4B-it, ≈10 GB) or document the mount clearly.
- Entry point `/app/submission/docker-entrypoint.sh` reads
  `CASCADE_SUBMISSION_PRESET` / `CASCADE_SOURCE_LANG` / `CASCADE_TARGET_LANG`.
  README in `submission/README.md` documents the 4 directions.
- **Not yet smoke-tested end-to-end** — a `docker build` + a single
  websocket request should be the next infra item after the knob tuning
  lands.

### 6.3 cs→en status

- Main cs→en (Parliament) zip (`iwslt26-encs-testset-blind.zip`) is
  truncated; only 3 short clips extractable.
- cs→en extra-context (Linguistic seminars, 13 clips) is extracted, one
  clip (`R-4.wav`) is half-truncated, rest intact.
- **No cs→en run has been executed today.** Not launched, not scored.
- The cascade is runtime-validated for cs→en on a single clip
  (`csJIsDTYMW.wav`) per `RESULTS.md`, but no BLEU on the dev set has ever
  been produced from this repo.
- Deprioritized given the user classified cs→en as "prio 2, small
  direction, can skip if needed."

---

## 7. What the next agent should NOT re-do

- Don't change the Chinese prompt again. V3 is the anchor; any further
  change is a new experiment, not an undo.
- Don't revert `LANGUAGE_NAME_TO_CODE["Simplified Chinese"] = "zh"`. It's
  what unlocked the V2 → V3 BLEU gain.
- Don't touch the rename `test-set/ ↔ dev-set/`. All active code has been
  migrated.
- Don't re-enable HY-MT for submission. It is wired in, working, and
  should stay dormant unless someone proves the COMET-vs-BLEU trade is
  acceptable for IWSLT's specific ranking.
- Don't start full dev-set runs before a one- or two-clip smoke confirms
  a knob change is going the right direction. Every full dev-set run is
  ≈33 min of GPU you don't get back.

---

## 8. Open questions the next agent has to settle

1. Can `partial_max_new_tokens` alone push CU under 2 s on dev-set en→de
   without dropping BLEU below ≈26 ?
2. Does `border_margin > 0` have any measurable effect when
   `inaccessible_ms=0` (hint from §4.2: probably small).
3. Is the en→it / en→zh CU behaviour the same shape as en→de under the
   same knobs, or direction-dependent ?
4. Does V3 prompt hurt en→de / en→it ? (**Not verified**; it was only
   validated on en→zh.)
5. Does the test-set en→zh need a rerun with V3 prompt ? (**Yes**, once
   the knob tuning lands.)
6. Should we submit (a) LOW regime with small BLEU hit, or (b) HIGH
   regime with current BLEU if LOW is unreachable ? The user has already
   signalled they want LOW; this bilan documents the state, not that
   decision.

---

*Bilan written at mid-session handoff. Author: Claude. No shortcuts, no
rhetorical polish — if something is unknown or untested it is flagged as
such.*
