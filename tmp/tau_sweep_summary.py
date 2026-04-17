#!/usr/bin/env python3
"""Compare τ sweep candidates for a given direction, highlight LOW candidates."""
import json, sys
from pathlib import Path

direction = sys.argv[1] if len(sys.argv) > 1 else "ende"
low_gate = 2000.0

print(f"=== τ sweep summary: {direction} ===")
print(f"{'run':50s}  {'τ':>6s}  {'BLEU':>6s}  {'chrF':>6s}  {'COMET':>7s}  {'CU':>5s}  {'CA':>5s}  LOW?")

best_comet_low = None
for d in sorted(Path("outputs").iterdir()):
    name = d.name
    if not (f"phase_b_{direction}" in name or (f"phase_a_smoke_{direction}" in name)):
        continue
    ev = d / "evaluation.json"
    if not ev.exists():
        print(f"{name:50s}  (no evaluation.json)")
        continue
    try:
        data = json.loads(ev.read_text())
    except Exception:
        continue
    scores = data.get("contract_scores", {})
    tau_str = "-"
    if "tau" in name:
        try:
            tau_str = f"{int(name.split('tau')[-1].split('_')[0])/100:.2f}"
        except Exception:
            pass
    elif "smoke" in name and "tau0" in name:
        tau_str = "0.00"
    cu = scores.get('LongYAAL CU')
    ca = scores.get('LongYAAL CA')
    bleu = scores.get('BLEU')
    chrf = scores.get('CHRF')
    comet = scores.get('XCOMETXL')
    low_flag = "LOW" if (cu is not None and cu < low_gate) else "HIGH"
    print(
        f"{name:50s}  {tau_str:>6s}  "
        f"{(bleu or 0):6.2f}  {(chrf or 0):6.2f}  {(comet or 0):7.4f}  "
        f"{(cu or 0):5.0f}  {(ca or 0):5.0f}  {low_flag}"
    )
    if low_flag == "LOW" and comet is not None:
        if best_comet_low is None or comet > best_comet_low[1]:
            best_comet_low = (name, comet, tau_str, cu)

print("")
if best_comet_low:
    print(f"BEST LOW-regime candidate: {best_comet_low[0]} (τ={best_comet_low[2]}, COMET={best_comet_low[1]:.4f}, CU={best_comet_low[3]:.0f})")
else:
    print("No LOW-regime candidate found.")
