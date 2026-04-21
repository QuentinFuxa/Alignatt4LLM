#!/usr/bin/env python3
"""Per-audio ASR comparison scatter for the 21-clip dev-set.

Reads the per-audio JSON payloads produced by
``scripts/compare_asr_per_audio_batch.py`` for the two backends and
emits a standalone TikZ figure (ready to ``\\input`` from the paper)
that places every audio as a pair of points in (mean boundary lag,
WER) space -- one colour per backend -- with a faint connector
between the paired points so the reader can eyeball per-audio deltas.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


BACKEND_CONFIG = {
    "qwen_forced": {
        "label": "Qwen3-ASR + forced aligner",
        "short": "Qwen3",
        "colour": "orange!75!black",
        "mark": "*",
    },
    "gemma_vllm_qk_fast": {
        "label": "Gemma-4 E4B + AlignAtt (ours)",
        "short": "Gemma-AlignAtt",
        "colour": "blue!60!black",
        "mark": "triangle*",
    },
}


def _load_per_audio(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for json_path in sorted(path.glob("*.json")):
        payload = json.loads(json_path.read_text())
        wav_name = Path(payload.get("wav_path", json_path.stem)).stem
        records[wav_name] = payload
    return records


def _collect_points(payload: dict) -> dict:
    metrics = payload.get("metrics", {}) or {}
    lag_summary = metrics.get("lag_summary") or {}
    lag_points = metrics.get("lag_points") or []
    return {
        "wer": float(metrics.get("wer") or 0.0),
        "cer": float(metrics.get("cer") or 0.0),
        "mean_lag_s": (
            float(lag_summary.get("mean_s")) if lag_summary.get("mean_s") is not None else None
        ),
        "median_lag_s": (
            float(lag_summary.get("median_s"))
            if lag_summary.get("median_s") is not None
            else None
        ),
        "mean_abs_lag_s": (
            float(lag_summary.get("mean_abs_s"))
            if lag_summary.get("mean_abs_s") is not None
            else None
        ),
        "matched_count": len(lag_points),
        "audio_duration_s": float(payload.get("audio_duration_s") or 0.0),
        "rtf_wallclock": float(payload.get("rtf_wallclock") or 0.0),
    }


def _summary_row(backend: str, points: list[dict]) -> dict:
    if not points:
        return {"backend": backend, "audio_count": 0}
    wers = [p["wer"] for p in points]
    cers = [p["cer"] for p in points]
    mean_lags = [p["mean_lag_s"] for p in points if p["mean_lag_s"] is not None]
    abs_lags = [p["mean_abs_lag_s"] for p in points if p["mean_abs_lag_s"] is not None]
    return {
        "backend": backend,
        "audio_count": len(points),
        "wer_mean": sum(wers) / len(wers),
        "wer_median": sorted(wers)[len(wers) // 2],
        "cer_mean": sum(cers) / len(cers),
        "lag_mean_s": (sum(mean_lags) / len(mean_lags)) if mean_lags else None,
        "abs_lag_mean_s": (sum(abs_lags) / len(abs_lags)) if abs_lags else None,
    }


def _axis_bounds(values: list[float], pad_frac: float = 0.08) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    lo, hi = min(values), max(values)
    if math.isclose(lo, hi):
        pad = max(abs(lo) * 0.2, 0.5)
        return lo - pad, hi + pad
    span = hi - lo
    return lo - span * pad_frac, hi + span * pad_frac


def _nice_step(span: float, target_ticks: int = 5) -> float:
    if span <= 0:
        return 1.0
    raw = span / target_ticks
    pow10 = 10 ** math.floor(math.log10(raw))
    for mult in (1, 2, 2.5, 5, 10):
        step = mult * pow10
        if step >= raw:
            return step
    return 10 * pow10


def _ticks(lo: float, hi: float, step: float) -> list[float]:
    start = math.floor(lo / step) * step
    ticks: list[float] = []
    v = start
    while v <= hi + step * 0.5:
        if lo - step * 0.5 <= v <= hi + step * 0.5:
            ticks.append(v)
        v += step
    return ticks


def render_tikz_figure(
    *,
    per_backend: dict[str, dict[str, dict]],
    summary: dict[str, dict],
    output_path: Path,
    axis_width_cm: float = 7.6,
    axis_height_cm: float = 5.4,
) -> None:
    ordered_backends = [b for b in BACKEND_CONFIG if b in per_backend]
    all_wavs = sorted({wav for b in ordered_backends for wav in per_backend[b]})

    # Build per-backend lag/WER arrays for axis fitting.
    lag_values: list[float] = []
    wer_values: list[float] = []
    per_backend_points: dict[str, list[dict]] = {}
    for backend in ordered_backends:
        points: list[dict] = []
        for wav in all_wavs:
            payload = per_backend[backend].get(wav)
            if payload is None:
                continue
            stats = _collect_points(payload)
            if stats["mean_lag_s"] is None:
                continue
            stats["wav"] = wav
            points.append(stats)
            lag_values.append(stats["mean_lag_s"])
            wer_values.append(stats["wer"] * 100.0)
        per_backend_points[backend] = points

    # Axis bounds: lag clamped so we see both backends fairly.
    lag_lo, lag_hi = _axis_bounds(lag_values, pad_frac=0.1)
    wer_lo, wer_hi = _axis_bounds(wer_values, pad_frac=0.15)
    wer_lo = max(0.0, wer_lo)
    lag_step = _nice_step(lag_hi - lag_lo, target_ticks=5)
    wer_step = _nice_step(wer_hi - wer_lo, target_ticks=5)
    lag_ticks = _ticks(lag_lo, lag_hi, lag_step)
    wer_ticks = _ticks(wer_lo, wer_hi, wer_step)

    def sx(v: float) -> float:
        return (v - lag_lo) / (lag_hi - lag_lo) * axis_width_cm

    def sy(v: float) -> float:
        return (v - wer_lo) / (wer_hi - wer_lo) * axis_height_cm

    lines: list[str] = []
    lines.append("% Auto-generated by scripts/plot_asr_comparison.py -- do not edit by hand.")
    lines.append("\\begin{figure}[t]")
    lines.append("\\centering")
    lines.append(
        "\\resizebox{\\linewidth}{!}{"
        "\\begin{tikzpicture}[x=1cm,y=1cm, every node/.style={font=\\scriptsize}]"
    )

    frame_x0 = -1.4
    frame_x1 = axis_width_cm + 3.9
    frame_y0 = -1.5
    frame_y1 = axis_height_cm + 1.8
    lines.append(
        f"\\draw[rounded corners=2pt, fill=white, draw=black!20] "
        f"({frame_x0:.2f},{frame_y0:.2f}) rectangle ({frame_x1:.2f},{frame_y1:.2f});"
    )

    # Plot area frame.
    lines.append(
        f"\\draw[black!40, line width=0.3pt] (0,0) rectangle "
        f"({axis_width_cm:.2f},{axis_height_cm:.2f});"
    )

    # Grid + x ticks.
    for tick in lag_ticks:
        x = sx(tick)
        if 0 <= x <= axis_width_cm + 1e-6:
            lines.append(
                f"\\draw[black!12, line width=0.25pt] ({x:.2f},0) -- ({x:.2f},{axis_height_cm:.2f});"
            )
            lines.append(
                f"\\draw[black!55] ({x:.2f},0) -- ({x:.2f},-0.10);"
            )
            lines.append(
                f"\\node[anchor=north, text=black!70] at ({x:.2f},-0.12) {{{tick:.1f}}};"
            )
    # Grid + y ticks.
    for tick in wer_ticks:
        y = sy(tick)
        if 0 <= y <= axis_height_cm + 1e-6:
            lines.append(
                f"\\draw[black!12, line width=0.25pt] (0,{y:.2f}) -- ({axis_width_cm:.2f},{y:.2f});"
            )
            lines.append(
                f"\\draw[black!55] (0,{y:.2f}) -- (-0.10,{y:.2f});"
            )
            lines.append(
                f"\\node[anchor=east, text=black!70] at (-0.15,{y:.2f}) {{{tick:.0f}}};"
            )

    # Axis labels.
    lines.append(
        f"\\node[anchor=north, text=black!80] at "
        f"({axis_width_cm / 2:.2f},-0.75) "
        f"{{mean boundary lag (s), predicted $-$ reference end time}};"
    )
    lines.append(
        f"\\node[anchor=south, rotate=90, text=black!80] at "
        f"(-0.95,{axis_height_cm / 2:.2f}) {{WER (\\%)}};"
    )
    # Figure title.
    lines.append(
        f"\\node[anchor=west, font=\\bfseries\\small] at ({frame_x0 + 0.25:.2f},{axis_height_cm + 1.3:.2f}) "
        f"{{Per-audio ASR latency vs.\\ accuracy on the 21-clip dev-set}};"
    )

    # Zero-lag reference line.
    zero_x = sx(0.0)
    if 0 <= zero_x <= axis_width_cm:
        lines.append(
            f"\\draw[black!40, dashed, line width=0.3pt] ({zero_x:.2f},0) -- ({zero_x:.2f},{axis_height_cm:.2f});"
        )
        lines.append(
            f"\\node[anchor=south, text=black!50, font=\\tiny] at ({zero_x:.2f},{axis_height_cm + 0.05:.2f}) {{lag $=0$}};"
        )

    # Connector lines between paired per-audio points.
    wav_to_backend_point = {
        backend: {p["wav"]: p for p in pts} for backend, pts in per_backend_points.items()
    }
    if len(ordered_backends) == 2:
        b0, b1 = ordered_backends
        for wav in all_wavs:
            p0 = wav_to_backend_point[b0].get(wav)
            p1 = wav_to_backend_point[b1].get(wav)
            if p0 is None or p1 is None:
                continue
            x0, y0 = sx(p0["mean_lag_s"]), sy(p0["wer"] * 100.0)
            x1, y1 = sx(p1["mean_lag_s"]), sy(p1["wer"] * 100.0)
            lines.append(
                f"\\draw[black!25, line width=0.25pt] ({x0:.2f},{y0:.2f}) -- ({x1:.2f},{y1:.2f});"
            )

    # Points.
    for backend in ordered_backends:
        cfg = BACKEND_CONFIG[backend]
        for p in per_backend_points[backend]:
            x, y = sx(p["mean_lag_s"]), sy(p["wer"] * 100.0)
            lines.append(
                f"\\filldraw[fill={cfg['colour']}, draw={cfg['colour']}, line width=0.25pt] "
                f"({x:.2f},{y:.2f}) circle (0.08);"
            )

    # Backend mean markers (diamond + label).
    for backend in ordered_backends:
        s = summary.get(backend)
        if not s or s.get("lag_mean_s") is None:
            continue
        cfg = BACKEND_CONFIG[backend]
        x, y = sx(s["lag_mean_s"]), sy(s["wer_mean"] * 100.0)
        lines.append(
            f"\\filldraw[fill={cfg['colour']}, draw=white, line width=0.6pt] "
            f"({x:.2f},{y:.2f}) -- ({x + 0.18:.2f},{y:.2f}) -- "
            f"({x:.2f},{y + 0.18:.2f}) -- ({x - 0.18:.2f},{y:.2f}) -- cycle;"
        )

    # Legend.
    legend_x = axis_width_cm + 0.4
    legend_y = axis_height_cm - 0.15
    lines.append(
        f"\\node[anchor=north west, font=\\bfseries\\scriptsize] at "
        f"({legend_x:.2f},{legend_y + 0.85:.2f}) {{Backend}};"
    )
    line_y = legend_y
    for backend in ordered_backends:
        cfg = BACKEND_CONFIG[backend]
        s = summary.get(backend, {}) or {}
        wer_m = s.get("wer_mean")
        lag_m = s.get("lag_mean_s")
        lines.append(
            f"\\filldraw[fill={cfg['colour']}, draw={cfg['colour']}, line width=0.25pt] "
            f"({legend_x + 0.05:.2f},{line_y:.2f}) circle (0.09);"
        )
        label = cfg["label"]
        lines.append(
            f"\\node[anchor=west, text=black!85] at ({legend_x + 0.25:.2f},{line_y:.2f}) {{{label}}};"
        )
        if wer_m is not None and lag_m is not None:
            lines.append(
                f"\\node[anchor=west, text=black!55, font=\\tiny] at "
                f"({legend_x + 0.25:.2f},{line_y - 0.28:.2f}) "
                f"{{mean WER {wer_m * 100:.1f}\\%, mean lag {lag_m:+.2f}\\,s}}"
                f";"
            )
        line_y -= 0.80
    # Diamond legend entry.
    lines.append(
        f"\\filldraw[fill=black!30, draw=white, line width=0.5pt] "
        f"({legend_x + 0.05:.2f},{line_y:.2f}) -- ({legend_x + 0.20:.2f},{line_y:.2f}) -- "
        f"({legend_x + 0.05:.2f},{line_y + 0.15:.2f}) -- ({legend_x - 0.10:.2f},{line_y:.2f}) -- cycle;"
    )
    lines.append(
        f"\\node[anchor=west, text=black!65, font=\\tiny] at "
        f"({legend_x + 0.25:.2f},{line_y:.2f}) {{dataset mean}};"
    )
    line_y -= 0.45
    lines.append(
        f"\\draw[black!25, line width=0.35pt] ({legend_x - 0.10:.2f},{line_y:.2f}) -- ({legend_x + 0.25:.2f},{line_y:.2f});"
    )
    lines.append(
        f"\\node[anchor=west, text=black!65, font=\\tiny] at "
        f"({legend_x + 0.30:.2f},{line_y:.2f}) {{same audio}};"
    )

    lines.append("\\end{tikzpicture}}")

    caption_parts = [
        "\\textbf{Per-audio ASR latency vs.\\ accuracy on the 21-clip IWSLT 2026 dev-set.}",
        "Each dot is one audio; circles of matching colour denote the same clip transcribed by the two ASR frontends.",
        "The x-axis reports the mean boundary lag (predicted segment end time $-$ reference end time, in seconds; negative means the predicted commit preceded the reference boundary within the 3-word matching tolerance).",
        "The y-axis is segment-level WER against the MCIF English reference.",
        "Diamonds mark the dataset mean for each backend.",
    ]
    lines.append("\\caption{" + " ".join(caption_parts) + "}")
    lines.append("\\label{fig:asr-per-audio-compare}")
    lines.append("\\end{figure}")

    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing per-backend subfolders of per-audio JSON payloads.",
    )
    parser.add_argument(
        "--output-tex",
        required=True,
        help="Path to write the generated TikZ figure.",
    )
    parser.add_argument(
        "--output-summary-json",
        default=None,
        help="Optional path to write an aggregated JSON summary of both backends.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=list(BACKEND_CONFIG.keys()),
        help="Backend folder names to include, in plotting order.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    per_backend: dict[str, dict[str, dict]] = {}
    for backend in args.backends:
        backend_dir = input_dir / backend
        if not backend_dir.is_dir():
            raise SystemExit(f"missing backend directory: {backend_dir}")
        records = _load_per_audio(backend_dir)
        if not records:
            raise SystemExit(f"no per-audio JSONs under {backend_dir}")
        per_backend[backend] = records

    summary = {
        backend: _summary_row(
            backend,
            [_collect_points(payload) for payload in records.values()],
        )
        for backend, records in per_backend.items()
    }

    output_tex = Path(args.output_tex)
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    render_tikz_figure(
        per_backend=per_backend,
        summary=summary,
        output_path=output_tex,
    )
    print(f"wrote {output_tex}")

    if args.output_summary_json:
        Path(args.output_summary_json).write_text(json.dumps(summary, indent=2))
        print(f"wrote {args.output_summary_json}")


if __name__ == "__main__":
    main()
