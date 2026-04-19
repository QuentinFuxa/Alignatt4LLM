"""Build train/val/test split manifest for the Gemma feature aligner.

Groups acl-speech gold segments by talk, selects clips with good duration
(5-15s), and assigns to train/val/test ensuring different talks per split.

Talk boundaries (from English text analysis):
  Talk 410: sent_0   - sent_99   (Math Word Problem Solving, Allan, ByteDance)
  Talk 468: sent_100  - sent_183  (Antoine, Maastricht University)
  Talk 567: sent_184  - sent_239  (VALSE benchmark)
  Talk 597: sent_240  - sent_330  (Kamezawa, University of Tokyo)
  Talk 111: sent_331  - sent_415  (Asaf Harari)

Split strategy:
  Train: talks 410 + 468 (2 speakers)
  Val:   talk 567         (1 speaker, different from train/test)
  Test:  talk 111         (1 speaker, different from train/val)
"""

import json
import random
from pathlib import Path

import soundfile as sf

from dedicated_audio_gemma_aligner.paths import ACL_GOLD_DIR, ACL_TEXT_FILE, SMALL_SPLIT_MANIFEST

GOLD_DIR = ACL_GOLD_DIR
TEXT_FILE = ACL_TEXT_FILE
OUTPUT = SMALL_SPLIT_MANIFEST

TALKS = {
    "410": (0, 99),
    "468": (100, 183),
    "567": (184, 239),
    "597": (240, 330),
    "111": (331, 415),
}

MIN_DURATION = 4.0
MAX_DURATION = 18.0

SPLIT_CONFIG = {
    "train": {"talks": ["410", "468"], "count": 15},
    "val":   {"talks": ["567"],        "count": 4},
    "test":  {"talks": ["111"],        "count": 5},
}

random.seed(42)


def main():
    with open(TEXT_FILE) as f:
        text_lines = [l.strip() for l in f]

    candidates = {}
    for talk_id, (start, end) in TALKS.items():
        clips = []
        for idx in range(start, end + 1):
            wav_path = GOLD_DIR / f"sent_{idx}.wav"
            if not wav_path.exists():
                continue
            info = sf.info(str(wav_path))
            dur = info.duration
            if dur < MIN_DURATION or dur > MAX_DURATION:
                continue
            clips.append({
                "sent_idx": idx,
                "audio": str(wav_path),
                "tag": f"acl_{talk_id}_sent{idx}",
                "talk_id": talk_id,
                "duration_s": round(dur, 2),
                "reference_text": text_lines[idx] if idx < len(text_lines) else None,
            })
        candidates[talk_id] = clips

    for talk_id, clips in candidates.items():
        print(f"Talk {talk_id}: {len(clips)} clips in duration range [{MIN_DURATION}, {MAX_DURATION}]s")

    manifest = {"splits": {}, "metadata": {
        "source": "acl-speech gold segments",
        "min_duration_s": MIN_DURATION,
        "max_duration_s": MAX_DURATION,
        "seed": 42,
        "talk_boundaries": TALKS,
    }}

    for split_name, cfg in SPLIT_CONFIG.items():
        pool = []
        for talk_id in cfg["talks"]:
            pool.extend(candidates[talk_id])
        random.shuffle(pool)

        selected = sorted(pool[:cfg["count"]], key=lambda c: c["sent_idx"])
        manifest["splits"][split_name] = selected
        print(f"\n{split_name}: {len(selected)} clips")
        for c in selected:
            print(f"  {c['tag']}: {c['duration_s']}s - {c['reference_text'][:60]}...")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved manifest -> {OUTPUT}")


if __name__ == "__main__":
    main()
