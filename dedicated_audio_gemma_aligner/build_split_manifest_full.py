"""Build full-scale train/test split using all ACL gold segments.

Train: talks 410, 468, 567, 597 (~303 clips)
Test:  talk 111 (~70 clips, different speaker, fully held out)
"""

import json
from pathlib import Path

import soundfile as sf

from dedicated_audio_gemma_aligner.paths import ACL_GOLD_DIR, ACL_TEXT_FILE, FULL_SPLIT_MANIFEST

GOLD_DIR = ACL_GOLD_DIR
TEXT_FILE = ACL_TEXT_FILE
OUTPUT = FULL_SPLIT_MANIFEST

TALKS = {
    "410": (0, 99),
    "468": (100, 183),
    "567": (184, 239),
    "597": (240, 330),
    "111": (331, 415),
}

MIN_DURATION = 3.0
MAX_DURATION = 18.0

TRAIN_TALKS = ["410", "468", "567", "597"]
TEST_TALKS = ["111"]


def main():
    with open(TEXT_FILE) as f:
        text_lines = [l.strip() for l in f]

    splits = {"train": [], "test": []}

    for talk_id, (start, end) in TALKS.items():
        split = "train" if talk_id in TRAIN_TALKS else "test"
        for idx in range(start, end + 1):
            wav_path = GOLD_DIR / f"sent_{idx}.wav"
            if not wav_path.exists():
                continue
            info = sf.info(str(wav_path))
            if info.duration < MIN_DURATION or info.duration > MAX_DURATION:
                continue
            splits[split].append({
                "sent_idx": idx,
                "audio": str(wav_path),
                "tag": f"acl_{talk_id}_sent{idx}",
                "talk_id": talk_id,
                "duration_s": round(info.duration, 2),
                "reference_text": text_lines[idx] if idx < len(text_lines) else None,
            })

    manifest = {"splits": splits, "metadata": {
        "source": "acl-speech gold segments (full scale)",
        "min_duration_s": MIN_DURATION,
        "max_duration_s": MAX_DURATION,
        "train_talks": TRAIN_TALKS,
        "test_talks": TEST_TALKS,
        "talk_boundaries": TALKS,
    }}

    for split_name, clips in splits.items():
        talks = set(c["talk_id"] for c in clips)
        total_dur = sum(c["duration_s"] for c in clips)
        print(f"{split_name}: {len(clips)} clips, talks={talks}, {total_dur/60:.1f} min")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved -> {OUTPUT}")


if __name__ == "__main__":
    main()
