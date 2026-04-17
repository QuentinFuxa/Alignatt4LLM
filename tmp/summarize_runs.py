#!/usr/bin/env python3
"""Summarize Phase B run metrics from outputs/ directories."""
import json, sys
from pathlib import Path

cols = ["dir", "target", "clips", "BLEU", "chrF", "COMET", "CU", "CA"]
rows = []
for d in sorted(Path("outputs").iterdir()):
    ev = d / "evaluation.json"
    if not ev.exists():
        continue
    try:
        data = json.loads(ev.read_text())
    except Exception:
        continue
    scores = data.get("contract_scores", {})
    settings = data.get("settings", {})
    tgt = settings.get("target_lang_code", "?")
    total = 0
    try:
        instances_path = d / "instances.resegmented.jsonl"
        if instances_path.exists():
            with instances_path.open() as f:
                total = sum(1 for _ in f)
    except Exception:
        pass
    def fmt(v, fmtspec):
        if v is None:
            return "-"
        return format(v, fmtspec)
    rows.append([
        d.name, tgt, total,
        fmt(scores.get('BLEU'), ".2f"),
        fmt(scores.get('CHRF'), ".2f"),
        fmt(scores.get('XCOMETXL'), ".4f"),
        fmt(scores.get('LongYAAL CU'), ".0f"),
        fmt(scores.get('LongYAAL CA'), ".0f"),
    ])

filter_substr = sys.argv[1] if len(sys.argv) > 1 else ""
filtered = [r for r in rows if filter_substr in r[0]]
print("\t".join(cols))
for r in filtered:
    print("\t".join(str(x) for x in r))
