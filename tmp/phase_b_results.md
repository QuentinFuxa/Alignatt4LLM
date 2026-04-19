# Phase A / B results log (2026-04-17)

Preset knobs (post-edit, common to all runs below unless noted):
- partial_max_new_tokens=24 (was 16)
- partial_followup_max_new_tokens=16 (was 8)
- translation_scheduler_stall_seconds=0.8 (was 1.2)
- translation_alignatt_rewind_threshold=8
- translation_alignatt_border_margin=0
- translation_alignatt_min_source_mass=0.0
- asr_commit_mode=punctuation_lcp
- chunk_ms=450
- min_start_seconds=2.0

Baseline (pre-edit preset, for reference):
- en→de full dev-set: BLEU 27.91, chrF 62.01, COMET 0.8618, CU 2263.8, CA 1920.7 (outputs/iwslt26_devset_main_low_ende)
- en→it full dev-set: BLEU 38.83, chrF 68.11, COMET 0.7781, CU 2306.3, CA 1981.0 (outputs/iwslt26_devset_main_low_enit)
- en→zh V3 full dev-set: BLEU 40.85, chrF 37.30, COMET 0.7276, CU 2219.8, CA 2059.4 (outputs/iwslt26_devset_main_low_enzh_promptV3)

## Phase A — shared knob smoke (2 clips ccpXHNfaoy + OiqEWDVtWk, en→de, τ=0.0)

| Run | BLEU | chrF | COMET | CU   | CA   | dir |
|-----|------|------|-------|------|------|-----|
| phase_a_smoke_ende_tau0 | ? | ? | ? | ? | ? | outputs/phase_a_smoke_ende_tau0 |

## Phase B — en→de τ sweep (2 clips)

| τ   | BLEU | chrF | COMET | CU   | CA   | dir |
|-----|------|------|-------|------|------|-----|
| 0.00 | (same as Phase A) |  |  |  |  | phase_a_smoke_ende_tau0 |
| 0.05 | ? | ? | ? | ? | ? | phase_b_ende_tau005 |
| 0.10 | ? | ? | ? | ? | ? | phase_b_ende_tau010 |
| 0.15 | ? | ? | ? | ? | ? | phase_b_ende_tau015 |

Shortlist for full dev-set en→de: TBD after sweep.

## Phase B — en→it τ sweep (2 clips)

| τ   | BLEU | chrF | COMET | CU   | CA   | dir |
|-----|------|------|-------|------|------|-----|
| 0.00 | ? | ? | ? | ? | ? | phase_b_enit_tau000 |
| 0.05 | ? | ? | ? | ? | ? | phase_b_enit_tau005 |
| 0.10 | ? | ? | ? | ? | ? | phase_b_enit_tau010 |

## Phase B — en→zh τ sweep (2 clips, V3 prompt)

| τ   | BLEU | chrF | COMET | CU   | CA   | dir |
|-----|------|------|-------|------|------|-----|
| 0.00 | ? | ? | ? | ? | ? | phase_b_enzh_tau000 |
| 0.05 | ? | ? | ? | ? | ? | phase_b_enzh_tau005 |
| 0.10 | ? | ? | ? | ? | ? | phase_b_enzh_tau010 |

## Phase B — full dev-set confirmations

(populate after shortlist per direction)

## Final test-set

(populate after best_d chosen per direction)
