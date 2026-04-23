#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any


PAPER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PAPER_DIR.parent
GENERATED_DIR = PAPER_DIR / "generated"
CANDIDATE_BANK_PATH = PAPER_DIR / "candidate_bank_en_fr.json"
SECTION3_FIG_PATH = GENERATED_DIR / "decoder_only_alignatt_heads.tex"
QUALITATIVE_RESULTS_PATH = GENERATED_DIR / "qualitative_search_results.json"
QUALITATIVE_BEST_PATH = GENERATED_DIR / "qualitative_best_snapshot.json"
QUALITATIVE_FIG_PATH = GENERATED_DIR / "mt_selective_reconstruction_example.tex"
QUALITATIVE_PROVENANCE_PATH = GENERATED_DIR / "qualitative_provenance_summary.json"
QUALITATIVE_PROVENANCE_TEX_PATH = GENERATED_DIR / "qualitative_provenance_summary.tex"
BENCHMARK_RESULTS_PATH = GENERATED_DIR / "mt_capture_speed_benchmark.json"
BENCHMARK_FIG_PATH = GENERATED_DIR / "mt_capture_speed_benchmark.tex"
MT_HEAD_FILTERING_RESULTS_PATH = GENERATED_DIR / "mt_head_filtering_ablation.json"
MT_HEAD_FILTERING_TABLE_PATH = GENERATED_DIR / "mt_head_filtering_ablation.tex"
MT_E2E_HEAD_ABLATION_BUNDLES = [
    {
        "bundle_dir": REPO_ROOT / "outputs" / "iwslt26_devset_chunk1100_borderp1_rerun20260423_ende",
        "setting": "Top-8 heads",
        "tag": "top8",
    },
    {
        "bundle_dir": (
            REPO_ROOT
            / "outputs"
            / "mt_head_ablation_low_all_heads_chunk1100_current"
            / "low"
            / "en-de"
            / "all_heads"
        ),
        "setting": "All 336 heads",
        "tag": "all_heads",
    },
]
MT_E2E_HEAD_ABLATION_RESULTS_PATH = GENERATED_DIR / "mt_e2e_head_ablation.json"
MT_E2E_HEAD_ABLATION_TABLE_PATH = GENERATED_DIR / "mt_e2e_head_ablation.tex"
LOW_CONFIG_SNAPSHOT_BUNDLES = [
    {
        "bundle_dir": REPO_ROOT / "outputs" / "iwslt26_devset_chunk1100_borderp1_rerun20260423_ende",
        "label": r"EN$\to$DE",
        "tag": "maintained-low",
    }
]
LOW_CONFIG_SNAPSHOT_RESULTS_PATH = GENERATED_DIR / "low_regime_config_snapshot.json"
LOW_CONFIG_SNAPSHOT_TABLE_PATH = GENERATED_DIR / "low_regime_config_snapshot.tex"
BENCHMARK_JSON_BEGIN = "__BENCHMARK_JSON_BEGIN__"
BENCHMARK_JSON_END = "__BENCHMARK_JSON_END__"
PROVISIONAL_EN_FR_HEADS = (
    REPO_ROOT
    / "data/alignatt_heads/translation_heads_google_gemma-4-E4B-it_en-fr_provisional_shared_kernel.json"
)
GEMMA_ASR_HEADS_PATH = (
    REPO_ROOT
    / "data/alignatt_heads/audio_alignment_heads_google_gemma-4-E4B-it_en_forced.json"
)
GEMMA_ASR_FULL_RANKING_PATH = (
    REPO_ROOT
    / "data/alignatt_heads/audio_alignment_heads_google_gemma-4-E4B-it_en_forced.full_ranking.json"
)

GEMMA_LAYER_COUNT = 42
GEMMA_HEAD_COUNT = 8
GEMMA_FULL_LAYERS = {5, 11, 17, 23, 29, 35, 41}
GEMMA_SHARED_KV_START = 24
TARGET_LANGUAGE_TO_HEADS = {
    "French": PROVISIONAL_EN_FR_HEADS,
    "German": REPO_ROOT / "data/alignatt_heads/translation_heads_google_gemma-4-E4B-it_en-de.json",
    "Italian": REPO_ROOT / "data/alignatt_heads/translation_heads_google_gemma-4-E4B-it_en-it.json",
}
TARGET_LANGUAGE_TO_CODE = {
    "French": "fr",
    "German": "de",
    "Italian": "it",
}
BENCHMARK_REPEATS = 3

BENCHMARK_SUITE = [
    {
        "id": "short_01",
        "length_bin": "short",
        "source_text": "the model gives itself more context before it answers",
        "assistant_prefill": "",
        "accessible_units": 6,
    },
    {
        "id": "short_01_prefill",
        "length_bin": "short",
        "source_text": "the model gives itself more context before it answers",
        "assistant_prefill": "Das Modell gibt sich",
        "accessible_units": 6,
    },
    {
        "id": "short_02",
        "length_bin": "short",
        "source_text": "we carefully keep only the useful rows",
        "assistant_prefill": "",
        "accessible_units": 5,
    },
    {
        "id": "short_02_prefill",
        "length_bin": "short",
        "source_text": "we carefully keep only the useful rows",
        "assistant_prefill": "Wir behalten nur",
        "accessible_units": 5,
    },
    {
        "id": "short_03",
        "length_bin": "short",
        "source_text": "the policy now rejects the risky word",
        "assistant_prefill": "",
        "accessible_units": 5,
    },
    {
        "id": "short_03_prefill",
        "length_bin": "short",
        "source_text": "the policy now rejects the risky word",
        "assistant_prefill": "Die Policy weist",
        "accessible_units": 5,
    },
    {
        "id": "medium_01",
        "length_bin": "medium",
        "source_text": "we selectively reconstruct only the source slice that drives the acceptance rule",
        "assistant_prefill": "",
        "accessible_units": 9,
    },
    {
        "id": "medium_01_prefill",
        "length_bin": "medium",
        "source_text": "we selectively reconstruct only the source slice that drives the acceptance rule",
        "assistant_prefill": "Wir rekonstruieren selektiv nur",
        "accessible_units": 9,
    },
    {
        "id": "medium_02",
        "length_bin": "medium",
        "source_text": "the observer returns a compact and engine native capture of query and key tensors",
        "assistant_prefill": "",
        "accessible_units": 10,
    },
    {
        "id": "medium_02_prefill",
        "length_bin": "medium",
        "source_text": "the observer returns a compact and engine native capture of query and key tensors",
        "assistant_prefill": "Der Beobachter liefert eine",
        "accessible_units": 10,
    },
    {
        "id": "medium_03",
        "length_bin": "medium",
        "source_text": "we explicitly compare only the seams that expose a real capture path",
        "assistant_prefill": "",
        "accessible_units": 8,
    },
    {
        "id": "medium_03_prefill",
        "length_bin": "medium",
        "source_text": "we explicitly compare only the seams that expose a real capture path",
        "assistant_prefill": "Wir vergleichen ausdrücklich nur",
        "accessible_units": 8,
    },
    {
        "id": "long_01",
        "length_bin": "long",
        "source_text": "the paper reports a clear and reproducible comparison of the three capture seams under compiled inference",
        "assistant_prefill": "",
        "accessible_units": 11,
    },
    {
        "id": "long_01_prefill",
        "length_bin": "long",
        "source_text": "the paper reports a clear and reproducible comparison of the three capture seams under compiled inference",
        "assistant_prefill": "Der Beitrag berichtet über einen",
        "accessible_units": 11,
    },
    {
        "id": "long_02",
        "length_bin": "long",
        "source_text": "we intentionally place the accepted prefix before we continue the translation with a new draft",
        "assistant_prefill": "",
        "accessible_units": 11,
    },
    {
        "id": "long_02_prefill",
        "length_bin": "long",
        "source_text": "we intentionally place the accepted prefix before we continue the translation with a new draft",
        "assistant_prefill": "Wir stellen das akzeptierte Präfix",
        "accessible_units": 11,
    },
]


def ensure_repo_imports() -> None:
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def tex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def rgb_mix(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = clamp01(t)
    return tuple(int(round((1.0 - t) * x + t * y)) for x, y in zip(a, b))


def tikz_color(rgb: tuple[int, int, int]) -> str:
    return f"{{rgb,255:red,{rgb[0]}; green,{rgb[1]}; blue,{rgb[2]}}}"


def word_count(text: str) -> int:
    return len([piece for piece in text.strip().split() if piece])


def stable_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def load_head_payload(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if "token_alignment_heads" not in payload:
        raise ValueError(f"{path} does not contain token_alignment_heads")
    return payload


def head_grid_from_payload(payload: dict[str, Any]) -> list[list[float]]:
    grid = [[0.0 for _ in range(GEMMA_HEAD_COUNT)] for _ in range(GEMMA_LAYER_COUNT)]
    for head in payload.get("all_heads_ranked", []):
        layer = int(head["layer"])
        query_head = int(head["head"])
        grid[layer][query_head] = float(head["ts"])
    return grid


def average_head_grids(payloads: list[dict[str, Any]]) -> list[list[float]]:
    grid = [[0.0 for _ in range(GEMMA_HEAD_COUNT)] for _ in range(GEMMA_LAYER_COUNT)]
    for payload in payloads:
        matrix = payload.get("ts_matrix")
        if matrix is None:
            raise ValueError("payload missing dense ts_matrix")
        for layer in range(GEMMA_LAYER_COUNT):
            for head in range(GEMMA_HEAD_COUNT):
                grid[layer][head] += float(matrix[layer][head])
    if payloads:
        denom = float(len(payloads))
        for layer in range(GEMMA_LAYER_COUNT):
            for head in range(GEMMA_HEAD_COUNT):
                grid[layer][head] /= denom
    return grid


def load_multilingual_mean_grid() -> tuple[list[list[float]], list[tuple[int, int]]]:
    payloads = [
        load_head_payload(REPO_ROOT / "data/alignatt_heads/translation_heads_google_gemma-4-E4B-it_en-de.json"),
        load_head_payload(REPO_ROOT / "data/alignatt_heads/translation_heads_google_gemma-4-E4B-it_en-it.json"),
        load_head_payload(REPO_ROOT / "data/alignatt_heads/translation_heads_google_gemma-4-E4B-it_en-zh.json"),
    ]
    shared_payload = load_head_payload(REPO_ROOT / "data/alignatt_heads/translation_heads_shared_kernel_top8.json")
    selected = [
        (int(head["layer"]), int(head["head"]))
        for head in shared_payload["token_alignment_heads"]
    ]
    return average_head_grids(payloads), selected


def load_asr_grid_and_selected() -> tuple[list[list[float]], list[tuple[int, int]]]:
    selected_payload = load_head_payload(GEMMA_ASR_HEADS_PATH)
    selected = [
        (int(head["layer"]), int(head["head"]))
        for head in selected_payload["token_alignment_heads"]
    ]
    full_ranking = load_json(GEMMA_ASR_FULL_RANKING_PATH)
    grid = [[0.0 for _ in range(GEMMA_HEAD_COUNT)] for _ in range(GEMMA_LAYER_COUNT)]
    for entry in full_ranking:
        layer = int(entry["layer"])
        head = int(entry["head"])
        grid[layer][head] = float(entry["ts"])
    return grid, selected


def build_section3_figure() -> None:
    mt_grid, mt_selected = load_multilingual_mean_grid()
    asr_grid, asr_selected = load_asr_grid_and_selected()

    observed_values = [
        value
        for grid in (mt_grid, asr_grid)
        for row in grid
        for value in row
        if value > 0.0
    ]
    scale_min = 0.0
    scale_max = max(observed_values) if observed_values else 1.0

    background = (243, 246, 250)
    low = (251, 236, 203)
    mid = (238, 159, 82)
    high = (178, 38, 35)
    label_layers = {0, *GEMMA_FULL_LAYERS}

    def score_fill_rgb(value: float) -> tuple[int, int, int]:
        if value <= 0.0 or scale_max <= scale_min:
            return background
        normalized = clamp01((value - scale_min) / (scale_max - scale_min))
        normalized = math.pow(normalized, 0.9)
        if normalized < 0.55:
            return rgb_mix(low, mid, normalized / 0.55)
        return rgb_mix(mid, high, (normalized - 0.55) / 0.45)

    mt_y0 = 0.0
    asr_y0 = 11.0
    panel_height = 8.0
    legend_y = -2.6
    box_y_min = -4.6
    box_y_max = asr_y0 + panel_height + 3.0  # 22.0

    lines: list[str] = []
    lines.append(r"\begin{figure*}[t]")
    lines.append(r"\centering")
    lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.append(r"\begin{tikzpicture}[x=0.35cm,y=0.35cm, every node/.style={font=\scriptsize}]")
    lines.append(
        rf"\draw[rounded corners=2pt, fill=black!2, draw=black!20] (-3.2,{box_y_min:.2f}) rectangle (43.2,{box_y_max:.2f});"
    )

    def emit_panel(
        grid: list[list[float]],
        selected: list[tuple[int, int]],
        y0: float,
        panel_title: str,
        *,
        show_layer_labels: bool,
        show_shared_kv_bracket: bool,
    ) -> None:
        selected_set = set(selected)
        y1 = y0 + panel_height

        title_y = y1 + 1.5 if show_shared_kv_bracket else y1 + 0.7
        lines.append(
            rf"\node[anchor=west, font=\bfseries\scriptsize] at (-2.8,{title_y:.2f}) {{{panel_title}}};"
        )

        for layer in GEMMA_FULL_LAYERS:
            lines.append(
                rf"\fill[black!7] ({layer - 0.02:.2f},{y0 - 0.02:.2f}) rectangle "
                rf"({layer + 1.02:.2f},{y1 + 0.02:.2f});"
            )

        if show_shared_kv_bracket:
            kv_x0 = float(GEMMA_SHARED_KV_START)
            kv_x1 = float(GEMMA_LAYER_COUNT)
            y_lo = y1 + 0.15
            y_hi = y1 + 0.55
            lines.append(rf"\fill[blue!6] ({kv_x0:.2f},{y_lo:.2f}) rectangle ({kv_x1:.2f},{y_hi:.2f});")
            lines.append(
                rf"\draw[blue!55!black, line width=0.7pt] ({kv_x0:.2f},{y_lo:.2f}) -- ({kv_x0:.2f},{y_hi:.2f});"
            )
            lines.append(
                rf"\draw[blue!55!black, line width=0.7pt] ({kv_x1:.2f},{y_lo:.2f}) -- ({kv_x1:.2f},{y_hi:.2f});"
            )
            lines.append(
                rf"\draw[blue!55!black, line width=0.7pt] ({kv_x0:.2f},{y_hi:.2f}) -- ({kv_x1:.2f},{y_hi:.2f});"
            )
            lines.append(
                rf"\node[anchor=south, font=\tiny, text=blue!55!black] at "
                rf"({(kv_x0 + kv_x1) / 2:.2f},{y_hi + 0.05:.2f}) "
                rf"{{late shared-KV block (L{GEMMA_SHARED_KV_START}--L{GEMMA_LAYER_COUNT - 1})}};"
            )

        for layer in range(GEMMA_LAYER_COUNT):
            for head in range(GEMMA_HEAD_COUNT):
                fill_rgb = score_fill_rgb(grid[layer][head])
                draw = "black!55"
                width = "0.25pt"
                if (layer, head) in selected_set:
                    draw = "blue!55!black"
                    width = "0.9pt"
                lines.append(
                    rf"\filldraw[fill={tikz_color(fill_rgb)}, draw={draw}, line width={width}] "
                    rf"({layer:.2f},{y0 + head:.2f}) rectangle ({layer + 1:.2f},{y0 + head + 1:.2f});"
                )
        lines.append(
            rf"\draw[draw=black!45, line width=0.5pt] (0,{y0:.2f}) rectangle (42,{y1:.2f});"
        )

        for head in range(GEMMA_HEAD_COUNT):
            lines.append(
                rf"\node[anchor=east, text=black!70, font=\tiny] at (-0.2,{y0 + head + 0.5:.2f}) {{H{head}}};"
            )
        if show_layer_labels:
            for layer in range(GEMMA_LAYER_COUNT):
                if layer in label_layers:
                    lines.append(
                        rf"\node[anchor=north, text=black!70, font=\tiny] at "
                        rf"({layer + 0.5:.2f},{y0 - 0.2:.2f}) {{L{layer}}};"
                    )

    emit_panel(
        asr_grid,
        asr_selected,
        asr_y0,
        "ASR alignment heads (English, forced-alignment TS)",
        show_layer_labels=False,
        show_shared_kv_bracket=True,
    )
    emit_panel(
        mt_grid,
        mt_selected,
        mt_y0,
        r"MT alignment heads (en$\to\{$de,\,it,\,zh$\}$ mean TS)",
        show_layer_labels=True,
        show_shared_kv_bracket=False,
    )

    lines.append(rf"\node[anchor=east, font=\tiny, text=black!70] at (5.0,{legend_y:.2f}) {{TS scale}};")
    gradient_x0 = 5.4
    gradient_x1 = 12.6
    n_steps = 32
    for idx in range(n_steps):
        value = scale_min + (scale_max - scale_min) * idx / max(1, n_steps - 1)
        fill_rgb = score_fill_rgb(value)
        x_left = gradient_x0 + (gradient_x1 - gradient_x0) * idx / n_steps
        x_right = gradient_x0 + (gradient_x1 - gradient_x0) * (idx + 1) / n_steps
        lines.append(
            rf"\filldraw[fill={tikz_color(fill_rgb)}, draw={tikz_color(fill_rgb)}] "
            rf"({x_left:.2f},{legend_y - 0.35:.2f}) rectangle ({x_right:.2f},{legend_y + 0.35:.2f});"
        )
    lines.append(
        rf"\draw[draw=black!35, line width=0.25pt] "
        rf"({gradient_x0:.2f},{legend_y - 0.35:.2f}) rectangle ({gradient_x1:.2f},{legend_y + 0.35:.2f});"
    )
    lines.append(
        rf"\node[anchor=north, font=\tiny, text=black!60] at "
        rf"({gradient_x0:.2f},{legend_y - 0.45:.2f}) {{{scale_min:.2f}}};"
    )
    lines.append(
        rf"\node[anchor=north, font=\tiny, text=black!60] at "
        rf"({gradient_x1:.2f},{legend_y - 0.45:.2f}) {{{scale_max:.2f}}};"
    )

    lines.append(
        rf"\filldraw[fill=white, draw=blue!55!black, line width=0.9pt] "
        rf"(14.2,{legend_y - 0.35:.2f}) rectangle (15.0,{legend_y + 0.35:.2f});"
    )
    lines.append(
        rf"\node[anchor=west, font=\tiny, text=black!70] at "
        rf"(15.2,{legend_y:.2f}) {{retained online}};"
    )

    lines.append(
        rf"\filldraw[fill=black!7, draw=black!20] "
        rf"(21.2,{legend_y - 0.35:.2f}) rectangle (22.0,{legend_y + 0.35:.2f});"
    )
    lines.append(
        rf"\node[anchor=west, font=\tiny, text=black!70] at "
        rf"(22.2,{legend_y:.2f}) {{full-attention layer}};"
    )

    lines.append(r"\end{tikzpicture}%")
    lines.append(r"}")
    lines.append(
        r"\caption{\textbf{Architecture-aware view of Gemma AlignAtt heads.} "
        r"Each heatmap covers Gemma's 42 layers (horizontal axis) and 8 query heads "
        r"(vertical axis); color encodes mean token-alignment score TS on a shared "
        r"$[0, \mathrm{max}]$ scale, so empty cells read as low rather than "
        r"``below threshold''. \emph{Top:} audio-to-text alignment heads, scored by "
        r"forced-alignment MAE on English dev audio. \emph{Bottom:} source-to-target "
        r"translation heads, averaged over en$\to$\{de,\,it,\,zh\}. Blue outlines mark "
        r"the per-role retained heads ($\mathcal{H}_{\mathrm{ASR}}$ and "
        r"$\mathcal{H}_{\mathrm{MT}}$); grey vertical stripes mark full-attention layers; "
        r"the blue bracket delimits the late shared-KV block, where observer cost "
        r"follows KV-group ownership rather than query-head count. The two retained "
        r"sets concentrate in disjoint depth regions, consistent with per-role "
        r"functional specialization.}"
    )
    lines.append(r"\label{fig:decoder-only-heads}")
    lines.append(r"\end{figure*}")
    SECTION3_FIG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_mt_head_filtering_table() -> None:
    payload = load_json(MT_HEAD_FILTERING_RESULTS_PATH)
    rows = list(payload.get("summary_rows", []))
    if not rows:
        raise ValueError(f"{MT_HEAD_FILTERING_RESULTS_PATH} does not contain summary_rows")

    direction_labels = {
        "en-de": r"EN$\to$DE",
        "en-zh": r"EN$\to$ZH",
        "en-it": r"EN$\to$IT",
    }

    def format_points(value: float) -> str:
        return f"{100.0 * float(value):.2f}"

    def format_count(value: int) -> str:
        return f"{int(value):,}"

    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\resizebox{\columnwidth}{!}{%")
    lines.append(r"\begin{tabular}{l r r r r}")
    lines.append(r"\toprule")
    lines.append(r"Pair & Top-8 TS & All-336 TS & Gain & Aligned tokens \\")
    lines.append(r"\midrule")
    for row in rows:
        direction = str(row["direction"])
        lines.append(
            " & ".join(
                [
                    direction_labels.get(direction, tex_escape(direction)),
                    format_points(float(row["source_top_k_score"])),
                    format_points(float(row["source_all_heads_score"])),
                    rf"\textbf{{+{float(row['source_delta_points']):.2f}}}",
                    format_count(int(row["used_target_tokens"])),
                ]
            )
            + r" \\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(
        r"\caption{\textbf{MT head-set filtering ablation on held-out word-aligned "
        r"dev examples.} To make the all-head baseline maximally charitable, we "
        r"average the candidate head set first and then restrict the argmax to the "
        r"source-prompt token zone before scoring against gold aligned source "
        r"tokens. Scores are reported in points ($100\times\mathrm{TS}$). Even "
        r"under this source-only argmax, the per-direction retained top-8 heads "
        r"still preserve a markedly stronger alignment signal than uniform "
        r"all-head averaging.}"
    )
    lines.append(r"\label{tab:mt-head-filtering}")
    lines.append(r"\end{table}")
    MT_HEAD_FILTERING_TABLE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_mt_e2e_head_ablation_table() -> None:
    def format_score(value: float, *, decimals: int = 2) -> str:
        return f"{float(value):.{decimals}f}"

    def format_xcomet(value: float) -> str:
        return f"{float(value):.3f}"

    def format_latency_seconds(value_ms: float) -> str:
        return f"{float(value_ms) / 1000.0:.2f}"

    rows: list[dict[str, Any]] = []
    for spec in MT_E2E_HEAD_ABLATION_BUNDLES:
        bundle_dir = Path(spec["bundle_dir"])
        manifest = load_json(bundle_dir / "manifest.json")
        evaluation = load_json(bundle_dir / "evaluation.json")
        runtime = manifest["runtime_config"]
        scores = evaluation["contract_scores"]
        rows.append(
            {
                "setting": spec["setting"],
                "tag": spec["tag"],
                "bundle_dir": str(bundle_dir.relative_to(REPO_ROOT)),
                "chunk_ms": runtime["chunk_ms"],
                "translation_alignatt_top_k_heads": runtime["translation_alignatt_top_k_heads"],
                "translation_alignatt_border_margin": runtime["translation_alignatt_border_margin"],
                "min_start_seconds": runtime["min_start_seconds"],
                "translation_alignatt_rewind_threshold": runtime.get(
                    "translation_alignatt_rewind_threshold"
                ),
                "xcometxl": scores["XCOMETXL"],
                "longyaal_cu_ms": scores["LongYAAL CU"],
                "longyaal_ca_ms": scores["LongYAAL CA"],
            }
        )

    if len(rows) != 2:
        raise ValueError("Expected exactly two EN->DE MCIF comparison rows")
    rows_by_tag = {row["tag"]: row for row in rows}
    top8_row = rows_by_tag["top8"]
    all_heads_row = rows_by_tag["all_heads"]
    write_json(
        MT_E2E_HEAD_ABLATION_RESULTS_PATH,
        {
            "rows": rows,
            "deltas": {
                "xcometxl": all_heads_row["xcometxl"] - top8_row["xcometxl"],
                "longyaal_cu_ms": all_heads_row["longyaal_cu_ms"] - top8_row["longyaal_cu_ms"],
                "longyaal_ca_ms": all_heads_row["longyaal_ca_ms"] - top8_row["longyaal_ca_ms"],
            },
        },
    )

    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{l c c c}")
    lines.append(r"\toprule")
    lines.append(
        r"Setting & XCOMET-XL $\uparrow$ & CU (s) $\downarrow$ & CA (s) $\downarrow$ \\"
    )
    lines.append(r"\midrule")
    lines.append(
        " & ".join(
            [
                top8_row["setting"],
                format_xcomet(top8_row["xcometxl"]),
                rf"\textbf{{{format_latency_seconds(top8_row['longyaal_cu_ms'])}}}",
                rf"\textbf{{{format_latency_seconds(top8_row['longyaal_ca_ms'])}}}",
            ]
        )
        + r" \\"
    )
    lines.append(
        " & ".join(
            [
                all_heads_row["setting"],
                rf"\textbf{{{format_xcomet(all_heads_row['xcometxl'])}}}",
                format_latency_seconds(all_heads_row["longyaal_cu_ms"]),
                format_latency_seconds(all_heads_row["longyaal_ca_ms"]),
            ]
        )
        + r" \\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{\textbf{EN$\to$DE full-MCIF MT head-set comparison at "
        r"$\Delta_{\mathrm{chunk}}=1100$ ms.} Both rows use the same maintained "
        r"runtime; only the MT head set changes. All-head replay is slightly "
        r"better on XCOMET-XL, but increases CU and especially CA because the "
        r"observer becomes less selective while the runtime must reconstruct "
        r"more attention at each MT step.}"
    )
    lines.append(r"\label{tab:mt-e2e-head-filtering}")
    lines.append(r"\end{table}")
    MT_E2E_HEAD_ABLATION_TABLE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_low_regime_config_snapshot_table() -> None:
    def format_score(value: float, *, decimals: int = 2) -> str:
        return f"{float(value):.{decimals}f}"

    def format_xcomet(value: float) -> str:
        return f"{float(value):.3f}"

    def format_latency_seconds(value_ms: float) -> str:
        return f"{float(value_ms) / 1000.0:.2f}"

    def make_shortstack(lines: list[str]) -> str:
        return r"\shortstack[l]{" + r"\\".join(lines) + "}"

    rows: list[dict[str, Any]] = []
    for spec in LOW_CONFIG_SNAPSHOT_BUNDLES:
        bundle_dir = Path(spec["bundle_dir"])
        manifest = load_json(bundle_dir / "manifest.json")
        evaluation = load_json(bundle_dir / "evaluation.json")
        runtime = manifest["runtime_config"]
        scores = evaluation["contract_scores"]
        rows.append(
            {
                "label": spec["label"],
                "tag": spec["tag"],
                "bundle_dir": str(bundle_dir.relative_to(REPO_ROOT)),
                "alignment_backend_name": runtime["alignment_backend_name"],
                "mt_backend_name": runtime["mt_backend_name"],
                "chunk_ms": runtime["chunk_ms"],
                "translation_alignatt_top_k_heads": runtime["translation_alignatt_top_k_heads"],
                "translation_alignatt_border_margin": runtime["translation_alignatt_border_margin"],
                "min_start_seconds": runtime["min_start_seconds"],
                "bleu": scores["BLEU"],
                "chrf": scores["CHRF"],
                "xcometxl": scores["XCOMETXL"],
                "longyaal_cu_ms": scores["LongYAAL CU"],
                "longyaal_ca_ms": scores["LongYAAL CA"],
            }
        )

    write_json(LOW_CONFIG_SNAPSHOT_RESULTS_PATH, {"rows": rows})

    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\resizebox{\columnwidth}{!}{%")
    lines.append(r"\begin{tabular}{l l l}")
    lines.append(r"\toprule")
    lines.append(r"Language & Config & Scores \\")
    lines.append(r"\midrule")
    for row in rows:
        config_cell = make_shortstack(
            [
                rf"\texttt{{{tex_escape(row['alignment_backend_name'])}}} + \texttt{{{tex_escape(row['mt_backend_name'])}}}",
                rf"$\Delta_{{\mathrm{{chunk}}}}={int(row['chunk_ms'])}$ ms; top-$k={int(row['translation_alignatt_top_k_heads'])}$",
                rf"border margin $={int(row['translation_alignatt_border_margin'])}$; min-start $={float(row['min_start_seconds']):.1f}$ s",
            ]
        )
        score_cell = make_shortstack(
            [
                rf"BLEU {format_score(row['bleu'])}; chrF {format_score(row['chrf'])}",
                rf"XCOMET-XL {format_xcomet(row['xcometxl'])}",
                rf"LongYAAL CU/CA {format_latency_seconds(row['longyaal_cu_ms'])} / {format_latency_seconds(row['longyaal_ca_ms'])} s",
            ]
        )
        lines.append(" & ".join([row["label"], config_cell, score_cell]) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(
        r"\caption{\textbf{Current low-regime configuration snapshot for the refreshed EN$\to$DE rerun.} "
        r"The table records the exact maintained low-latency operating point we use to "
        r"recalibrate the system near the 2\,s boundary: the deployed Qwen3-ASR forced "
        r"alignment backend, the Gemma AlignAtt MT backend, the larger 1.1\,s cascade "
        r"chunk, and the retained top-8 MT head set with border margin 1.}"
    )
    lines.append(r"\label{tab:low-config-snapshot}")
    lines.append(r"\end{table}")
    LOW_CONFIG_SNAPSHOT_TABLE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_runtime_config(
    *,
    target_lang: str,
    heads_path: Path,
    max_new_tokens: int = 16,
) -> Any:
    ensure_repo_imports()
    from cascade.runtime import CascadeRuntimeConfig, temporary_runtime_config

    config = CascadeRuntimeConfig(
        source_lang="English",
        target_lang=target_lang,
        translation_alignatt_heads_path=str(heads_path),
        translation_alignatt_top_k_heads=8,
        translation_alignatt_filter_width=7,
        translation_alignatt_probe_mode="qk_fast",
        translation_alignatt_inaccessible_ms=0.0,
        translation_alignatt_border_margin=0,
        translation_alignatt_min_source_mass=0.0,
        partial_max_new_tokens=max_new_tokens,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.0,
        mt_vllm_enforce_eager=False,
        mt_vllm_enable_prefix_caching=False,
        mt_vllm_cudagraph_mode="full",
        mt_vllm_gpu_memory_utilization=0.5,
        gemma_max_model_len=1024,
    )
    return config, temporary_runtime_config


def build_frontier_with_accessible_units(source_text: str, accessible_units: int) -> Any:
    ensure_repo_imports()
    from cascade.source_frontier import build_source_accessibility_frontier

    total_units = word_count(source_text)
    unit_ms = 320.0
    timestamps = [
        (idx * unit_ms, (idx + 1) * unit_ms)
        for idx in range(total_units)
    ]
    current_audio_ms = max(0.0, float(accessible_units) * unit_ms)
    return build_source_accessibility_frontier(
        source_text=source_text,
        word_timestamps_ms=timestamps,
        current_audio_ms=current_audio_ms,
        inaccessible_ms=0.0,
        is_final=False,
    )


def group_generated_units(
    *,
    tokenizer,
    policy,
    generated_ids: list[int],
    source_map,
    source_rows,
    provenance_mass,
    aligned_source_positions,
) -> list[dict[str, Any]]:
    if not generated_ids:
        return []
    token_strings = [str(token) for token in tokenizer.convert_ids_to_tokens(generated_ids)]
    groups: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for token_idx, (token_id, token_str) in enumerate(zip(generated_ids, token_strings)):
        starts_new = policy.token_starts_stability_unit(token_str, is_first_token=(token_idx == 0))
        if starts_new or current is None:
            current = {
                "token_ids": [],
                "token_indices": [],
            }
            groups.append(current)
        current["token_ids"].append(int(token_id))
        current["token_indices"].append(token_idx)

    result: list[dict[str, Any]] = []
    for unit_idx, group in enumerate(groups):
        token_indices = group["token_indices"]
        token_ids = group["token_ids"]
        text = tokenizer.decode(token_ids, skip_special_tokens=False)
        visible_text = text.replace("▁", " ").strip()
        if not visible_text:
            visible_text = text.strip()
        source_unit_weights = [0.0 for _ in source_map.source_unit_spans]
        source_token_mass_total = 0.0
        non_source = 0.0
        suffix = 0.0
        accessible = 0.0
        inaccessible = 0.0
        aligned_positions: list[float] = []
        for token_index in token_indices:
            row_tensor = source_rows[token_index]
            row_mean = row_tensor.mean(dim=0).detach().cpu().tolist()
            source_token_mass_total += float(sum(row_mean))
            for unit in source_map.source_unit_spans:
                if not unit.prompt_token_positions:
                    continue
                local_positions = [
                    pos
                    for pos, prompt_pos in enumerate(source_map.source_token_positions)
                    if prompt_pos in unit.prompt_token_positions
                ]
                if local_positions:
                    source_unit_weights[unit.unit_index] += sum(row_mean[pos] for pos in local_positions)
            if token_index < len(provenance_mass):
                prov = provenance_mass[token_index]
                accessible += float(prov[0])
                inaccessible += float(prov[1])
                non_source += float(prov[2])
                suffix += float(prov[3])
            if token_index < len(aligned_source_positions):
                aligned = aligned_source_positions[token_index]
                if aligned is not None:
                    aligned_positions.append(float(aligned))
        if token_indices:
            count = float(len(token_indices))
            source_unit_weights = [value / count for value in source_unit_weights]
            accessible /= count
            inaccessible /= count
            non_source /= count
            suffix /= count
        total_mass = sum(source_unit_weights)
        result.append(
            {
                "unit_index": unit_idx,
                "text": visible_text,
                "token_count": len(token_indices),
                "source_unit_weights": source_unit_weights,
                "source_mass_total": total_mass,
                "provenance": {
                    "source_accessible": accessible,
                    "source_inaccessible": inaccessible,
                    "non_source_prompt": non_source,
                    "suffix": suffix,
                },
                "mean_aligned_source_position": stable_mean(aligned_positions) if aligned_positions else None,
                "starts_new_unit": True,
            }
        )
    return result


def serialize_source_map(source_map) -> dict[str, Any]:
    return {
        "source_text": source_map.source_text,
        "source_token_positions": [int(pos) for pos in source_map.source_token_positions],
        "accessible_source_token_count": int(source_map.accessible_source_token_count),
        "accessible_unit_count": int(source_map.accessible_unit_count),
        "total_unit_count": int(source_map.total_unit_count),
        "current_audio_ms": float(source_map.current_audio_ms),
        "inaccessible_ms": float(source_map.inaccessible_ms),
        "is_final": bool(source_map.is_final),
        "source_unit_spans": [
            {
                "unit_index": int(unit.unit_index),
                "text": unit.text,
                "prompt_token_positions": [int(pos) for pos in unit.prompt_token_positions],
                "is_accessible": bool(unit.is_accessible),
                "start_ms": None if unit.start_ms is None else float(unit.start_ms),
                "end_ms": None if unit.end_ms is None else float(unit.end_ms),
            }
            for unit in source_map.source_unit_spans
        ],
    }


def inversion_score(mean_positions: list[float]) -> float:
    if len(mean_positions) < 2:
        return 0.0
    decreases = 0
    total_pairs = 0
    magnitude = 0.0
    for idx in range(1, len(mean_positions)):
        prev = mean_positions[idx - 1]
        cur = mean_positions[idx]
        if cur < prev:
            decreases += 1
            magnitude += prev - cur
        total_pairs += 1
    if total_pairs == 0:
        return 0.0
    return float(decreases / total_pairs + 0.1 * magnitude / max(1.0, len(mean_positions)))


def english_copy_ratio(text: str) -> float:
    words = [piece.strip(" ,.;:!?").lower() for piece in text.split()]
    if not words:
        return 1.0
    ascii_words = [word for word in words if word.isascii()]
    if not ascii_words:
        return 0.0
    suspicious = 0
    french_articles = {"le", "la", "les", "de", "des", "du", "un", "une", "et", "à", "au", "aux"}
    for word in ascii_words:
        if word in french_articles:
            continue
        if word in {"model", "policy", "draft", "answer", "question", "source", "prefix", "translation"}:
            suspicious += 1
    return suspicious / max(1, len(ascii_words))


def qualitative_score(snapshot: dict[str, Any]) -> float:
    target_units = snapshot["target_units"]
    if not target_units:
        return -1.0
    accessible_count = int(snapshot["source_map"]["accessible_source_token_count"])
    aligned_positions = [
        unit["mean_aligned_source_position"]
        for unit in target_units
        if unit["mean_aligned_source_position"] is not None
    ]
    source_focus = stable_mean(
        [
            unit["provenance"]["source_accessible"] + unit["provenance"]["source_inaccessible"]
            for unit in target_units
        ]
    )
    prompt_leak = stable_mean(
        [unit["provenance"]["non_source_prompt"] for unit in target_units]
    )
    suffix_mass = stable_mean([unit["provenance"]["suffix"] for unit in target_units])
    prefix_words = word_count(snapshot["assistant_prefill"])
    accepted_total_words = word_count(snapshot["acceptance_text"])
    accepted_draft_units = max(0, accepted_total_words - prefix_words)
    accepted_units = target_units[:accepted_draft_units]
    accepted_accessible_ratio = stable_mean(
        [
            1.0
            if (
                unit["mean_aligned_source_position"] is not None
                and unit["mean_aligned_source_position"] < accessible_count
            )
            else 0.0
            for unit in accepted_units
        ]
    )
    accepted_source_focus = stable_mean(
        [
            unit["provenance"]["source_accessible"] + unit["provenance"]["source_inaccessible"]
            for unit in accepted_units
        ]
    )
    blocked_idx = snapshot.get("blocked_source_unit_index")
    frontier_idx = int(snapshot["source_map"]["accessible_unit_count"])
    boundary_block = 1.0 if isinstance(blocked_idx, int) and blocked_idx >= frontier_idx else 0.0
    family = str(snapshot.get("family", ""))
    score = 0.0
    score += 2.8 * min(accepted_draft_units, 3)
    score += 4.0 * accepted_accessible_ratio
    score += 1.6 * accepted_source_focus
    score += 0.8 * source_focus
    score += 0.8 * boundary_block
    score += 0.6 if snapshot["stop_reason"] == "alignatt:source_frontier" else -0.2
    score -= 1.5 * inversion_score(aligned_positions)
    score -= 0.8 * prompt_leak
    score -= 0.5 * suffix_mass
    score -= 0.8 * english_copy_ratio(snapshot["draft_text"])
    score += 0.4 if snapshot["assistant_prefill"].strip() else -2.0
    if snapshot["stable_rerun_match"]:
        score += 0.5
    if accepted_draft_units == 0:
        score -= 4.0
    if family.startswith("perspective_"):
        score -= 1.5
    if family in {"postnominal_adjective", "adverbial_reordering"}:
        score += 0.4
    return score


def accepted_draft_units(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    accepted_total_words = word_count(snapshot["acceptance_text"])
    prefix_words = word_count(snapshot["assistant_prefill"])
    accepted_unit_count = max(0, accepted_total_words - prefix_words)
    return list(snapshot["target_units"][:accepted_unit_count])


def summarize_target_units(units: list[dict[str, Any]]) -> dict[str, float | int]:
    if not units:
        return {
            "unit_count": 0,
            "source_accessible": 0.0,
            "source_inaccessible": 0.0,
            "source_total": 0.0,
            "non_source_prompt": 0.0,
            "suffix": 0.0,
        }
    source_accessible = stable_mean(
        [float(unit["provenance"]["source_accessible"]) for unit in units]
    )
    source_inaccessible = stable_mean(
        [float(unit["provenance"]["source_inaccessible"]) for unit in units]
    )
    non_source_prompt = stable_mean(
        [float(unit["provenance"]["non_source_prompt"]) for unit in units]
    )
    suffix = stable_mean([float(unit["provenance"]["suffix"]) for unit in units])
    return {
        "unit_count": len(units),
        "source_accessible": source_accessible,
        "source_inaccessible": source_inaccessible,
        "source_total": source_accessible + source_inaccessible,
        "non_source_prompt": non_source_prompt,
        "suffix": suffix,
    }


def write_qualitative_provenance_summary(summary: dict[str, Any]) -> None:
    def fmt_pct(value: float) -> str:
        return f"{100.0 * float(value):.0f}\\%"

    all_units = summary["all_draft_units"]
    accepted_units = summary["accepted_draft_units"]
    lines = [
        r"\paragraph{Residual provenance in decoder-only AlignAtt.}",
        (
            r"Unlike encoder--decoder AlignAtt, the source-restricted rows of "
            r"Eq.~\eqref{eq:virtual-xattn} are not normalized over the source alone: "
            r"attention can also land on the accepted target prefix, the rest of the "
            r"prompt template, and the speculative suffix. On the French qualitative "
            r"probe bank used to mine Fig.~\ref{fig:mt-selective-reconstruction} "
            rf"({int(summary['snapshot_count'])} scored prefix-continuation probes on an NVIDIA A40), "
            r"drafted target units allocate on average "
            rf"{fmt_pct(all_units['source_accessible'])} to accessible source tokens, "
            rf"{fmt_pct(all_units['source_inaccessible'])} to still-inaccessible source tokens, "
            rf"{fmt_pct(all_units['non_source_prompt'])} to non-source prompt positions "
            r"(system/template plus accepted prefix), and "
            rf"{fmt_pct(all_units['suffix'])} to the speculative suffix. Restricting to "
            r"the target units that AlignAtt actually accepts shifts these shares to "
            rf"{fmt_pct(accepted_units['source_accessible'])}/"
            rf"{fmt_pct(accepted_units['source_inaccessible'])}/"
            rf"{fmt_pct(accepted_units['non_source_prompt'])}/"
            rf"{fmt_pct(accepted_units['suffix'])}. These residual masses are therefore "
            r"not implementation noise but a structural difference from standard AlignAtt."
        ),
    ]
    QUALITATIVE_PROVENANCE_TEX_PATH.write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


@dataclass
class ProbeExecution:
    target_lang: str
    heads_path: Path
    backend: Any
    variant: Any


def load_vllm_execution(target_lang: str, max_new_tokens: int = 16) -> ProbeExecution:
    ensure_repo_imports()
    from cascade.mt.base import build_mt_backend
    from cascade.runtime import gemma_model_name, temporary_runtime_config
    from cascade.translation_variants import TRANSLATION_VARIANTS, DEFAULT_TRANSLATION_VARIANT_ID

    heads_path = TARGET_LANGUAGE_TO_HEADS[target_lang]
    config, temp_cfg = build_runtime_config(
        target_lang=target_lang,
        heads_path=heads_path,
        max_new_tokens=max_new_tokens,
    )
    with temp_cfg(
        config,
        source_lang="English",
        target_lang=target_lang,
        translation_alignatt_heads_path=str(heads_path),
    ):
        backend = build_mt_backend(model_name=gemma_model_name, runtime_config=config)
        backend.load()
    variant = TRANSLATION_VARIANTS[DEFAULT_TRANSLATION_VARIANT_ID]
    return ProbeExecution(target_lang=target_lang, heads_path=heads_path, backend=backend, variant=variant)


def run_vllm_probe_snapshot(
    execution: ProbeExecution,
    *,
    source_text: str,
    accessible_units: int,
    assistant_prefill: str,
) -> dict[str, Any]:
    ensure_repo_imports()
    from vllm import SamplingParams
    from cascade.mt.base import (
        compute_prefix_online_alignatt_source_argmaxes,
        source_local_position_to_unit_index,
    )
    from cascade.mt.gemma_vllm_observer import reconstruct_mt_attention_rows

    backend = execution.backend
    variant = execution.variant
    backend.reset_caches()
    frontier = build_frontier_with_accessible_units(source_text, accessible_units)
    rendered = variant.render_messages(
        source_lang="English",
        target_lang=execution.target_lang,
        text=source_text,
        source_frontier=frontier,
        source_history=[],
        translation_history=[],
        is_partial=True,
        assistant_prefill=assistant_prefill,
    )
    prompt_package = backend.render_prompt_package(rendered)
    source_map = prompt_package.source_map
    if source_map is None or not source_map.source_token_positions:
        raise RuntimeError("Could not build prompt source map for qualitative probe.")

    prompt_token_ids = list(prompt_package.prompt_token_ids)
    max_new_tokens = backend.compute_max_tokens(
        prompt_tokens=len(prompt_token_ids),
        source_text=rendered.source_text,
        is_partial=True,
        assistant_prefill=assistant_prefill,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=int(max_new_tokens),
        repetition_penalty=1.0,
        stop_token_ids=list(backend.resolve_generation_stop_token_ids()) or None,
        skip_special_tokens=False,
    )

    prepare_diag = backend._prepare_mt_observer(prompt_length=len(prompt_token_ids))
    outputs = backend.llm.generate(
        [{"prompt_token_ids": prompt_token_ids}],
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    completion = outputs[0].outputs[0]
    generated_ids = [int(token_id) for token_id in completion.token_ids]
    special_ids = {
        int(token_id)
        for token_id in getattr(backend.tokenizer, "all_special_ids", []) or []
    }
    while generated_ids and generated_ids[-1] in special_ids:
        generated_ids.pop()
    capture_payload = backend._fetch_mt_observer_payload()
    reconstruction = reconstruct_mt_attention_rows(
        capture_payload,
        alignatt_heads=backend.alignatt_heads,
        source_positions=source_map.source_token_positions,
        accessible_source_token_count=source_map.accessible_source_token_count,
    )
    source_rows = reconstruction.source_attention_rows_per_token
    provenance_mass = reconstruction.provenance_mass_per_token
    operating_count = min(len(source_rows), len(generated_ids))
    source_rows = source_rows[:operating_count]
    provenance_mass = provenance_mass[:operating_count]
    generated_ids = generated_ids[:operating_count]
    aligned_positions = compute_prefix_online_alignatt_source_argmaxes(
        source_rows,
        filter_width=backend.policy.alignatt_filter_width(),
    ) if source_rows else []
    mean_source_attention_rows = [
        row.mean(dim=0).detach().cpu().tolist()
        for row in source_rows
    ]
    per_head_source_attention_rows = [
        row.detach().cpu().tolist()
        for row in source_rows
    ]

    accepted_candidate_ids: list[int] = []
    unsafe_reason: str | None = None
    blocked_source_local_position: int | None = None
    blocked_source_unit_index: int | None = None
    stop_reason = getattr(completion, "finish_reason", None)
    unsafe_target_token_index: int | None = None
    unsafe_token_id: int | None = None
    for token_index, (token_id, source_position) in enumerate(zip(generated_ids, aligned_positions)):
        unsafe_reason, _ = backend.policy.should_stop_in_loop(
            current_source_local_position=source_position,
            accessible_source_token_count=source_map.accessible_source_token_count,
        )
        if unsafe_reason == "source_frontier":
            unsafe_target_token_index = token_index
            unsafe_token_id = int(token_id)
            blocked_source_local_position = source_position
            blocked_source_unit_index = source_local_position_to_unit_index(source_map, source_position)
            stop_reason = "alignatt:source_frontier"
            break
        accepted_candidate_ids.append(int(token_id))

    acceptance = backend.policy.finalize_partial(
        accepted_candidate_ids=accepted_candidate_ids,
        aligned_source_local_positions=aligned_positions,
        source_map=source_map,
        unsafe_reason=unsafe_reason,
        unsafe_target_token_index=unsafe_target_token_index,
        unsafe_token_id=unsafe_token_id,
        blocked_source_local_position=blocked_source_local_position,
        blocked_source_unit_index=blocked_source_unit_index,
        stop_reason=stop_reason,
        probe_backend="paper_vllm_probe",
    )
    accepted_generated_token_ids = [int(token_id) for token_id in acceptance.accepted_generated_ids]
    draft_text = backend.decode_candidate_text(
        generated_ids=generated_ids,
        assistant_prefill=assistant_prefill,
        variant=variant,
        is_partial=True,
    )
    acceptance_text = backend.decode_candidate_text(
        generated_ids=accepted_generated_token_ids,
        assistant_prefill=assistant_prefill,
        variant=variant,
        is_partial=True,
    )
    target_units = group_generated_units(
        tokenizer=backend.tokenizer,
        policy=backend.policy,
        generated_ids=generated_ids,
        source_map=source_map,
        source_rows=source_rows,
        provenance_mass=provenance_mass,
        aligned_source_positions=aligned_positions,
    )
    return {
        "target_lang": execution.target_lang,
        "heads_path": str(execution.heads_path),
        "source_text": source_text,
        "accessible_units": accessible_units,
        "assistant_prefill": assistant_prefill,
        "prompt_messages": rendered.messages,
        "prompt_text": prompt_package.prompt_text,
        "prompt_token_ids": [int(token_id) for token_id in prompt_token_ids],
        "prompt_token_count": len(prompt_token_ids),
        "source_map": serialize_source_map(source_map),
        "prepare_diagnostics": prepare_diag,
        "observer_diagnostics": reconstruction.diagnostics,
        "generated_token_ids": generated_ids,
        "accepted_generated_token_ids": accepted_generated_token_ids,
        "generated_token_strings": list(backend.tokenizer.convert_ids_to_tokens(generated_ids)),
        "draft_text": draft_text,
        "acceptance_text": acceptance_text,
        "stop_reason": stop_reason,
        "unsafe_reason": unsafe_reason,
        "blocked_source_local_position": blocked_source_local_position,
        "blocked_source_unit_index": blocked_source_unit_index,
        "aligned_source_local_positions": aligned_positions,
        "provenance_per_token": [
            {
                "source_accessible": float(row[0]),
                "source_inaccessible": float(row[1]),
                "non_source_prompt": float(row[2]),
                "suffix": float(row[3]),
            }
            for row in provenance_mass
        ],
        "source_attention_rows_per_token": mean_source_attention_rows,
        "source_attention_rows_by_head_per_token": per_head_source_attention_rows,
        "source_units": [
            {
                "unit_index": int(unit.unit_index),
                "text": unit.text,
                "is_accessible": bool(unit.is_accessible),
            }
            for unit in source_map.source_unit_spans
        ],
        "accessible_source_token_count": int(source_map.accessible_source_token_count),
        "accessible_source_unit_count": int(source_map.accessible_unit_count),
        "target_units": target_units,
        "accepted_prefix_word_count": word_count(acceptance_text),
        "draft_word_count": word_count(draft_text),
        "source_word_count": len(source_map.source_unit_spans),
    }


def search_qualitative_example() -> None:
    candidates = load_json(CANDIDATE_BANK_PATH)
    attempted_languages = ["French"]
    overall_results: dict[str, Any] = {"attempted_languages": attempted_languages, "languages": {}}
    best_snapshot: dict[str, Any] | None = None
    best_score = float("-inf")

    for target_lang in attempted_languages:
        execution = load_vllm_execution(target_lang)
        language_rows: list[dict[str, Any]] = []
        all_draft_units: list[dict[str, Any]] = []
        accepted_units: list[dict[str, Any]] = []
        stop_reason_counts: Counter[str] = Counter()
        for candidate in candidates:
            source_text = str(candidate["source_text"]).strip()
            source_words = word_count(source_text)
            frontier_points = sorted(
                {
                    max(2, min(source_words - 1, math.floor(source_words * ratio)))
                    for ratio in (0.55, 0.65, 0.75)
                }
            )
            for accessible_units in frontier_points:
                base_snapshot = run_vllm_probe_snapshot(
                    execution,
                    source_text=source_text,
                    accessible_units=accessible_units,
                    assistant_prefill="",
                )
                accepted_prefix = base_snapshot["acceptance_text"].strip()
                if not accepted_prefix or accepted_prefix == base_snapshot["draft_text"].strip():
                    language_rows.append(
                        {
                            "candidate_id": candidate["id"],
                            "family": candidate["family"],
                            "source_text": source_text,
                            "accessible_units": accessible_units,
                            "status": "no_prefill_progress",
                            "stop_reason": base_snapshot["stop_reason"],
                            "acceptance_text": base_snapshot["acceptance_text"],
                            "draft_text": base_snapshot["draft_text"],
                        }
                    )
                    continue
                with_prefill = run_vllm_probe_snapshot(
                    execution,
                    source_text=source_text,
                    accessible_units=accessible_units,
                    assistant_prefill=accepted_prefix,
                )
                rerun = run_vllm_probe_snapshot(
                    execution,
                    source_text=source_text,
                    accessible_units=accessible_units,
                    assistant_prefill=accepted_prefix,
                )
                with_prefill["stable_rerun_match"] = (
                    with_prefill["draft_text"] == rerun["draft_text"]
                    and with_prefill["stop_reason"] == rerun["stop_reason"]
                    and with_prefill["aligned_source_local_positions"] == rerun["aligned_source_local_positions"]
                )
                with_prefill["candidate_id"] = candidate["id"]
                with_prefill["family"] = candidate["family"]
                with_prefill["score"] = qualitative_score(with_prefill)
                all_draft_units.extend(with_prefill["target_units"])
                accepted_units.extend(accepted_draft_units(with_prefill))
                stop_reason_counts[str(with_prefill["stop_reason"])] += 1
                language_rows.append(
                    {
                        "candidate_id": candidate["id"],
                        "family": candidate["family"],
                        "source_text": source_text,
                        "accessible_units": accessible_units,
                        "status": "scored",
                        "score": with_prefill["score"],
                        "stop_reason": with_prefill["stop_reason"],
                        "stable_rerun_match": with_prefill["stable_rerun_match"],
                        "accepted_prefix_word_count": with_prefill["accepted_prefix_word_count"],
                        "draft_word_count": with_prefill["draft_word_count"],
                        "draft_text": with_prefill["draft_text"],
                        "acceptance_text": with_prefill["acceptance_text"],
                        "heads_path": with_prefill["heads_path"],
                    }
                )
                if with_prefill["score"] > best_score:
                    best_score = with_prefill["score"]
                    best_snapshot = with_prefill
        scored_rows = [row for row in language_rows if row.get("status") == "scored"]
        scored_rows.sort(key=lambda row: float(row.get("score", float("-inf"))), reverse=True)
        overall_results["languages"][target_lang] = {
            "heads_path": str(execution.heads_path),
            "top_rows": scored_rows[:10],
            "row_count": len(language_rows),
            "scored_count": len(scored_rows),
            "stop_reason_counts": dict(stop_reason_counts),
            "provenance_summary": {
                "all_draft_units": summarize_target_units(all_draft_units),
                "accepted_draft_units": summarize_target_units(accepted_units),
            },
        }

    if best_snapshot is None:
        raise RuntimeError("Could not find any qualitative example with a non-empty accepted prefix.")
    overall_results["selected"] = {
        "target_lang": best_snapshot["target_lang"],
        "candidate_id": best_snapshot["candidate_id"],
        "family": best_snapshot["family"],
        "score": best_snapshot["score"],
        "source_text": best_snapshot["source_text"],
        "stop_reason": best_snapshot["stop_reason"],
        "heads_path": best_snapshot["heads_path"],
    }
    selected_language = str(best_snapshot["target_lang"])
    selected_summary = overall_results["languages"][selected_language]["provenance_summary"]
    provenance_summary = {
        "target_lang": selected_language,
        "snapshot_count": int(overall_results["languages"][selected_language]["scored_count"]),
        "all_draft_units": selected_summary["all_draft_units"],
        "accepted_draft_units": selected_summary["accepted_draft_units"],
        "stop_reason_counts": overall_results["languages"][selected_language]["stop_reason_counts"],
    }
    write_json(QUALITATIVE_RESULTS_PATH, overall_results)
    write_json(QUALITATIVE_BEST_PATH, best_snapshot)
    write_json(QUALITATIVE_PROVENANCE_PATH, provenance_summary)
    write_qualitative_figure(best_snapshot)
    write_qualitative_provenance_summary(provenance_summary)


def write_qualitative_figure(snapshot: dict[str, Any]) -> None:
    source_units = snapshot["source_units"]
    target_units = snapshot["target_units"]
    if not source_units or not target_units:
        raise RuntimeError("Qualitative figure requires non-empty source and target unit lists.")

    source_words = [unit["text"] for unit in source_units]
    target_words = [unit["text"] for unit in target_units]
    matrix = [unit["source_unit_weights"] for unit in target_units]
    n_src = len(source_words)
    n_tgt = len(target_words)
    max_weight = max(max(row) for row in matrix) or 1.0
    accessible_count = sum(1 for u in source_units if u["is_accessible"])

    argmax_per_row: list[int] = []
    for row in matrix:
        if max(row) > 0:
            argmax_per_row.append(max(range(len(row)), key=lambda i: row[i]))
        else:
            argmax_per_row.append(-1)

    heat_lo = (250, 247, 243)
    heat_hi = (178, 34, 34)

    def heat_strength(value: float) -> float:
        if value <= 0.0 or max_weight <= 0.0:
            return 0.0
        return math.pow(clamp01(value / max_weight), 0.45)

    prefix_text = snapshot["assistant_prefill"].strip() or snapshot["acceptance_text"].strip()
    draft_text = str(snapshot["draft_text"]).strip()
    draft_suffix_text = draft_text
    if prefix_text and draft_text.startswith(prefix_text):
        draft_suffix_text = draft_text[len(prefix_text):].lstrip()

    example_note = ""
    if snapshot.get("family") == "perspective_inversion":
        example_note = (
            r" In the selected example, English \emph{I miss you} becomes French "
            r"\emph{Tu me manques}, so the accepted target prefix already follows a "
            r"non-monotone source order that a simple token-count heuristic would not capture."
        )
    elif snapshot.get("family") == "perspective_clause_boundary":
        example_note = (
            r" Here the accepted prefix comes from the earlier source clause, while the "
            r"draft begins with the English object phrase \emph{those good old days} "
            r"before the French experiencer ending \emph{me manquent}, exposing a genuine "
            r"non-monotone continuation beyond the frontier."
        )
    else:
        example_note = (
            r" The green band is previously committed target text reused as causal context; "
            r"it is not itself the object of the gate. The first black stars in the orange "
            r"draft show that newly proposed words may still align to multiple source words "
            r"inside the current accessible span, and only the first red star beyond the "
            r"frontier triggers rejection."
        )

    # ---------- Layout (TikZ units; x=y=0.55cm) ----------
    TU_CM = 0.55

    cell_w = 1.15
    cell_h = 0.95
    left_label_w = 2.2

    heat_x0 = left_label_w
    heat_x1 = heat_x0 + n_src * cell_w
    prov_gap = 0.70
    prov_x0 = heat_x1 + prov_gap
    prov_w = 6.8
    prov_x1 = prov_x0 + prov_w

    heat_y0 = 2.20
    heat_y1 = heat_y0 + n_tgt * cell_h

    # Space above heatmap for the accessible/inaccessible band and rotated column headers.
    band_y = heat_y1 + 0.25
    col_header_top = band_y + 0.45

    panel_head_y = col_header_top + 2.40  # leave room for rotated column headers

    # Tag pills sit just above each ribbon box; tags use \tiny so ~0.25 TikZ unit.
    ribbon_height = 2.25
    prompt_y0 = panel_head_y + 0.70
    prompt_y1 = prompt_y0 + ribbon_height
    tag_y = prompt_y1 + 0.10

    subtitle_y = tag_y + 1.00
    title_y = subtitle_y + 0.70

    canvas_x0 = -0.25
    canvas_x1 = prov_x1 + 0.25

    lines: list[str] = []
    lines.append(r"\begin{figure*}[t]")
    lines.append(r"\centering")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tikzpicture}[x=0.55cm,y=0.55cm, every node/.style={font=\scriptsize}]")

    # -------- Title + subtitle --------
    lines.append(
        rf"\node[anchor=west, font=\bfseries\small] at ({canvas_x0 + 0.10:.2f},{title_y:.2f}) "
        rf"{{Word-level selective reconstruction on a live MT draft}};"
    )
    blocked_idx = snapshot.get("blocked_source_unit_index")
    subtitle_parts = [
        rf"target=\texttt{{{tex_escape(snapshot['target_lang'])}}}",
        rf"gate=\texttt{{{tex_escape(str(snapshot['stop_reason']))}}}",
    ]
    if isinstance(blocked_idx, int) and 0 <= blocked_idx < n_src:
        subtitle_parts.append(
            rf"blocked at source word {blocked_idx + 1} (\emph{{{tex_escape(source_words[blocked_idx])}}})"
        )
    subtitle_txt = r" \; \textbullet \; ".join(subtitle_parts)
    lines.append(
        rf"\node[anchor=west, text=black!60] at ({canvas_x0 + 0.10:.2f},{subtitle_y:.2f}) "
        rf"{{{subtitle_txt}}};"
    )

    # -------- Prompt ribbon --------
    ribbon_x0 = canvas_x0 + 0.10
    ribbon_x1 = canvas_x1 - 0.10
    ribbon_total = ribbon_x1 - ribbon_x0
    gap = 0.25
    raw_widths = {"sys": 3.8, "src": 8.8, "acc": 3.4, "draft": 6.4}
    raw_sum = sum(raw_widths.values()) + 3 * gap
    scale_w = (ribbon_total - 3 * gap) / (raw_sum - 3 * gap)
    widths = {k: v * scale_w for k, v in raw_widths.items()}

    # Body text uses the TU_CM factor so that text_width in cm tracks the TikZ geometry
    # (\resizebox scales both uniformly so proportions stay consistent).
    body_pad_x = 0.20  # TikZ units of horizontal padding inside the box
    body_top_offset = 0.22  # vertical offset from top of box to start of body text
    # Body text is set at the top of the box; wrapping happens downward.

    def segment_box(x_lo: float, x_hi: float, *, fill: str, draw: str,
                    tag: str, tag_color: str,
                    body: str, body_color: str) -> None:
        # External tag pill above the box.
        lines.append(
            rf"\node[anchor=south west, font=\bfseries\tiny, text={tag_color}] "
            rf"at ({x_lo + 0.05:.2f},{tag_y:.2f}) {{{tag}}};"
        )
        # Main box.
        lines.append(
            rf"\draw[rounded corners=1.5pt, fill={fill}, draw={draw}, line width=0.5pt] "
            rf"({x_lo:.2f},{prompt_y0:.2f}) rectangle ({x_hi:.2f},{prompt_y1:.2f});"
        )
        # Body text fills the box.
        tw_cm = max(0.3, (x_hi - x_lo - 2 * body_pad_x) * TU_CM)
        lines.append(
            rf"\node[anchor=north west, text={body_color}, align=left, text width={tw_cm:.2f}cm] "
            rf"at ({x_lo + body_pad_x:.2f},{prompt_y1 - body_top_offset:.2f}) {{{body}}};"
        )

    cursor = ribbon_x0
    segment_box(cursor, cursor + widths["sys"],
                fill="black!4", draw="black!30",
                tag="system", tag_color="black!65",
                body="Translate to French.", body_color="black!60")
    cursor += widths["sys"] + gap

    src_x0, src_x1 = cursor, cursor + widths["src"]
    # Tag pill for the source box.
    lines.append(
        rf"\node[anchor=south west, font=\bfseries\tiny, text=blue!55!black] "
        rf"at ({src_x0 + 0.05:.2f},{tag_y:.2f}) {{source}};"
    )
    lines.append(
        rf"\draw[rounded corners=1.5pt, fill=blue!6, draw=blue!35, line width=0.5pt] "
        rf"({src_x0:.2f},{prompt_y0:.2f}) rectangle ({src_x1:.2f},{prompt_y1:.2f});"
    )
    # Two inner columns for accessible | inaccessible, separated by the frontier.
    split_frac = accessible_count / max(1, n_src)
    src_inner_x0 = src_x0 + body_pad_x
    src_inner_x1 = src_x1 - body_pad_x
    frontier_ribbon_x = src_inner_x0 + (src_inner_x1 - src_inner_x0) * split_frac
    acc_tw_cm = max(0.3, (frontier_ribbon_x - src_inner_x0 - 0.15) * TU_CM)
    inacc_tw_cm = max(0.3, (src_inner_x1 - frontier_ribbon_x - 0.15) * TU_CM)
    acc_text = " ".join(tex_escape(u["text"]) for u in source_units if u["is_accessible"])
    inacc_text = " ".join(tex_escape(u["text"]) for u in source_units if not u["is_accessible"])
    lines.append(
        rf"\node[anchor=north west, text=blue!55!black, align=left, text width={acc_tw_cm:.2f}cm] "
        rf"at ({src_inner_x0:.2f},{prompt_y1 - body_top_offset:.2f}) {{{acc_text}}};"
    )
    lines.append(
        rf"\draw[draw=blue!60!black, dash pattern=on 2pt off 1pt, line width=0.7pt] "
        rf"({frontier_ribbon_x:.2f},{prompt_y0 + 0.10:.2f}) -- ({frontier_ribbon_x:.2f},{prompt_y1 - 0.10:.2f});"
    )
    lines.append(
        rf"\node[anchor=north west, text=blue!30!black, align=left, text width={inacc_tw_cm:.2f}cm] "
        rf"at ({frontier_ribbon_x + 0.12:.2f},{prompt_y1 - body_top_offset:.2f}) {{{inacc_text}}};"
    )
    cursor = src_x1 + gap

    segment_box(cursor, cursor + widths["acc"],
                fill="green!8", draw="green!50!black",
                tag="accepted target prefix", tag_color="green!40!black",
                body=tex_escape(prefix_text or "(empty)"), body_color="green!30!black")
    cursor += widths["acc"] + gap

    segment_box(cursor, cursor + widths["draft"],
                fill="orange!10", draw="orange!60!black",
                tag="draft", tag_color="orange!70!black",
                body=tex_escape(draft_suffix_text or "(empty)"),
                body_color="orange!70!black")

    # -------- Panel sub-headers --------
    lines.append(
        rf"\node[anchor=west, font=\bfseries\scriptsize] at ({canvas_x0 + 0.10:.2f},{panel_head_y:.2f}) "
        rf"{{Draft-to-source attention (word level)}};"
    )
    lines.append(
        rf"\node[anchor=west, font=\bfseries\scriptsize] at ({prov_x0:.2f},{panel_head_y:.2f}) "
        rf"{{Per-word provenance}};"
    )

    # -------- Column headers (source words, rotated 45) above the heatmap --------
    for col, word in enumerate(source_words):
        x = heat_x0 + col * cell_w + cell_w / 2.0
        is_acc = source_units[col]["is_accessible"]
        color = "blue!55!black" if is_acc else "blue!30!black"
        lines.append(
            rf"\node[anchor=south west, rotate=45, text={color}, font=\scriptsize] "
            rf"at ({x - 0.05:.2f},{col_header_top:.2f}) {{{tex_escape(word)}}};"
        )

    # -------- accessible / inaccessible underline band just above the heatmap --------
    acc_mid_x = heat_x0 + cell_w * accessible_count / 2.0
    inacc_mid_x = heat_x0 + cell_w * (accessible_count + (n_src - accessible_count) / 2.0)
    split_x = heat_x0 + accessible_count * cell_w
    lines.append(
        rf"\draw[draw=blue!55!black, line width=1.0pt] "
        rf"({heat_x0:.2f},{band_y:.2f}) -- ({split_x - 0.05:.2f},{band_y:.2f});"
    )
    lines.append(
        rf"\draw[draw=blue!30!black, line width=1.0pt, dash pattern=on 2pt off 1pt] "
        rf"({split_x + 0.05:.2f},{band_y:.2f}) -- ({heat_x1:.2f},{band_y:.2f});"
    )
    lines.append(
        rf"\node[anchor=south, font=\tiny, text=blue!55!black] "
        rf"at ({acc_mid_x:.2f},{band_y + 0.02:.2f}) {{accessible}};"
    )
    lines.append(
        rf"\node[anchor=south, font=\tiny, text=blue!30!black] "
        rf"at ({inacc_mid_x:.2f},{band_y + 0.02:.2f}) {{inaccessible}};"
    )

    # -------- Heatmap cells + argmax markers --------
    for row_idx, target_word in enumerate(target_words):
        y = heat_y0 + (n_tgt - 1 - row_idx) * cell_h
        lines.append(
            rf"\node[anchor=east, text=orange!70!black, font=\scriptsize] "
            rf"at ({heat_x0 - 0.10:.2f},{y + cell_h / 2.0:.2f}) {{{tex_escape(target_word)}}};"
        )
        for col_idx, value in enumerate(matrix[row_idx]):
            fill_rgb = rgb_mix(heat_lo, heat_hi, heat_strength(value))
            lines.append(
                rf"\filldraw[fill={tikz_color(fill_rgb)}, draw=black!12, line width=0.2pt] "
                rf"({heat_x0 + col_idx * cell_w:.2f},{y:.2f}) rectangle "
                rf"({heat_x0 + (col_idx + 1) * cell_w:.2f},{y + cell_h:.2f});"
            )
        amax = argmax_per_row[row_idx]
        if amax >= 0:
            ax = heat_x0 + amax * cell_w + cell_w / 2.0
            ay = y + cell_h / 2.0
            is_acc = source_units[amax]["is_accessible"]
            marker_color = "black!85" if is_acc else "red!70!black"
            lines.append(
                rf"\node[font=\bfseries, text={marker_color}] "
                rf"at ({ax:.2f},{ay:.2f}) {{$\bigstar$}};"
            )

    # -------- Frontier dashed line over heatmap --------
    if 0 < accessible_count < n_src:
        lines.append(
            rf"\draw[draw=blue!60!black, dash pattern=on 2pt off 1.2pt, line width=0.9pt] "
            rf"({split_x:.2f},{heat_y0 - 0.05:.2f}) -- ({split_x:.2f},{heat_y1 + 0.05:.2f});"
        )

    lines.append(
        rf"\draw[draw=black!45, line width=0.5pt] "
        rf"({heat_x0:.2f},{heat_y0:.2f}) rectangle ({heat_x1:.2f},{heat_y1:.2f});"
    )

    # -------- Provenance bars (normalized) --------
    prov_segments = [
        ("blue!55!black", "source_accessible"),
        ("blue!22", "source_inaccessible"),
        ("black!22", "non_source_prompt"),
        ("orange!55!black", "suffix"),
    ]
    for row_idx, unit in enumerate(target_units):
        y = heat_y0 + (n_tgt - 1 - row_idx) * cell_h
        bar_y0 = y + 0.17
        bar_y1 = y + cell_h - 0.17
        prov = unit["provenance"]
        total_mass = sum(max(0.0, prov[k]) for _, k in prov_segments) or 1.0
        running = prov_x0
        for color, key in prov_segments:
            mass = max(0.0, prov[key])
            width = prov_w * mass / total_mass
            if width > 0.005:
                lines.append(
                    rf"\filldraw[fill={color}, draw={color}] "
                    rf"({running:.2f},{bar_y0:.2f}) rectangle ({running + width:.2f},{bar_y1:.2f});"
                )
            running += width
        lines.append(
            rf"\draw[draw=black!40, line width=0.35pt] "
            rf"({prov_x0:.2f},{bar_y0:.2f}) rectangle ({prov_x1:.2f},{bar_y1:.2f});"
        )
        src_share = prov["source_accessible"] + prov["source_inaccessible"]
        lines.append(
            rf"\node[anchor=west, font=\tiny, text=black!75] "
            rf"at ({prov_x1 + 0.10:.2f},{(bar_y0 + bar_y1) / 2.0:.2f}) "
            rf"{{{src_share * 100:.0f}\% source}};"
        )

    # -------- Bottom legend: two rows --------
    # Row A (top): colorbar + argmax stars.
    # Row B (bottom): provenance swatches in a single horizontal row.
    row_a_y = 1.35
    row_b_y = 0.55
    cbar_half = 0.18

    cbar_x0 = canvas_x0 + 1.25  # leave room for "attention" label to the left
    cbar_x1 = cbar_x0 + 4.50
    n_grad = 48
    for idx in range(n_grad):
        t = idx / (n_grad - 1)
        fill_rgb = rgb_mix(heat_lo, heat_hi, t)
        gx0 = cbar_x0 + (cbar_x1 - cbar_x0) * idx / n_grad
        gx1 = cbar_x0 + (cbar_x1 - cbar_x0) * (idx + 1) / n_grad
        lines.append(
            rf"\filldraw[fill={tikz_color(fill_rgb)}, draw={tikz_color(fill_rgb)}] "
            rf"({gx0:.2f},{row_a_y - cbar_half:.2f}) rectangle ({gx1:.2f},{row_a_y + cbar_half:.2f});"
        )
    lines.append(
        rf"\draw[draw=black!40, line width=0.3pt] "
        rf"({cbar_x0:.2f},{row_a_y - cbar_half:.2f}) rectangle ({cbar_x1:.2f},{row_a_y + cbar_half:.2f});"
    )
    lines.append(
        rf"\node[anchor=east, font=\tiny, text=black!65] "
        rf"at ({cbar_x0 - 0.10:.2f},{row_a_y:.2f}) {{attention}};"
    )
    lines.append(
        rf"\node[anchor=north, font=\tiny, text=black!60] "
        rf"at ({cbar_x0:.2f},{row_a_y - cbar_half - 0.03:.2f}) {{0.00}};"
    )
    lines.append(
        rf"\node[anchor=north, font=\tiny, text=black!60] "
        rf"at ({cbar_x1:.2f},{row_a_y - cbar_half - 0.03:.2f}) {{{max_weight:.2f}}};"
    )

    star1_x = cbar_x1 + 1.30
    lines.append(
        rf"\node[font=\bfseries, text=black!85] at ({star1_x:.2f},{row_a_y:.2f}) {{$\bigstar$}};"
    )
    lines.append(
        rf"\node[anchor=west, font=\tiny, text=black!65] "
        rf"at ({star1_x + 0.28:.2f},{row_a_y:.2f}) {{argmax in accessible region}};"
    )
    star2_x = star1_x + 6.80
    lines.append(
        rf"\node[font=\bfseries, text=red!70!black] at ({star2_x:.2f},{row_a_y:.2f}) {{$\bigstar$}};"
    )
    lines.append(
        rf"\node[anchor=west, font=\tiny, text=black!65] "
        rf"at ({star2_x + 0.28:.2f},{row_a_y:.2f}) {{argmax beyond frontier $\Rightarrow$ gate fires}};"
    )

    # Row B: 4 provenance swatches laid out horizontally.
    sw_entries = [
        ("blue!55!black", "accessible source"),
        ("blue!22", "inaccessible source"),
        ("black!22", "non-source prompt"),
        ("orange!55!black", "suffix"),
    ]
    sw_w = 0.32
    sw_gap_text = 0.15
    entry_gap = 0.90
    entry_widths = [3.80, 4.10, 3.80, 1.80]
    total_entries_w = sum(entry_widths) + entry_gap * (len(sw_entries) - 1)
    cursor_x = canvas_x0 + (canvas_x1 - canvas_x0 - total_entries_w) / 2.0
    for (color, label), ew in zip(sw_entries, entry_widths):
        lines.append(
            rf"\filldraw[fill={color}, draw={color}] "
            rf"({cursor_x:.2f},{row_b_y - 0.17:.2f}) rectangle ({cursor_x + sw_w:.2f},{row_b_y + 0.17:.2f});"
        )
        lines.append(
            rf"\node[anchor=west, font=\tiny, text=black!65] "
            rf"at ({cursor_x + sw_w + sw_gap_text:.2f},{row_b_y:.2f}) {{{label}}};"
        )
        cursor_x += ew + entry_gap

    lines.append(r"\end{tikzpicture}%")
    lines.append(r"}")

    lines.append(
        r"\caption{\textbf{Word-level selective reconstruction on a live MT draft.} "
        r"\emph{Top ribbon:} the decoder-only prompt is split into the system instruction, "
        r"the live source (with the accessibility frontier shown in-line between the "
        r"committed accessible prefix and the still-inaccessible tail), the already "
        r"accepted target prefix reused from earlier streaming steps, and the draft currently "
        r"under inference. "
        r"\emph{Left panel:} reconstructed draft-to-source attention from the selected "
        r"AlignAtt heads, aggregated to the word level; rows are drafted words, columns "
        r"are source words split by the dashed frontier. A star marks the per-row argmax "
        r"(black when it lands in the accessible region, red when it lands beyond the "
        r"frontier, which is exactly what triggers the \textsc{source-frontier} gate). "
        r"\emph{Right panel:} for each drafted word the reconstructed attention mass is "
        r"partitioned into the four provenance classes; the numeric annotation is the "
        r"total source share (accessible $+$ inaccessible)."
        + example_note
        + r"}")
    lines.append(r"\label{fig:mt-selective-reconstruction}")
    lines.append(r"\end{figure*}")
    QUALITATIVE_FIG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def snapshot_prompt_kv_from_past_key_values(past_key_values) -> list[tuple[int, Any, Any, int]]:
    snapshot: list[tuple[int, Any, Any, int]] = []
    if past_key_values is None:
        return snapshot
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        for layer_idx, (key, value) in enumerate(zip(past_key_values.key_cache, past_key_values.value_cache)):
            seq_length = int(key.shape[2])
            snapshot.append((int(layer_idx), key.detach(), value.detach(), seq_length))
        return snapshot
    if isinstance(past_key_values, (list, tuple)):
        for layer_idx, layer_kv in enumerate(past_key_values):
            if not isinstance(layer_kv, (list, tuple)) or len(layer_kv) < 2:
                continue
            key, value = layer_kv[:2]
            seq_length = int(key.shape[2])
            snapshot.append((int(layer_idx), key.detach(), value.detach(), seq_length))
    return snapshot


def prompt_for_benchmark(spec: dict[str, Any], *, target_lang: str) -> Any:
    ensure_repo_imports()
    from cascade.translation_variants import ALIGNATT_PREFIX_TRANSLATION_VARIANT

    frontier = build_frontier_with_accessible_units(spec["source_text"], int(spec["accessible_units"]))
    return ALIGNATT_PREFIX_TRANSLATION_VARIANT.render_messages(
        source_lang="English",
        target_lang=target_lang,
        text=spec["source_text"],
        source_frontier=frontier,
        source_history=[],
        translation_history=[],
        is_partial=True,
        assistant_prefill=str(spec["assistant_prefill"]),
    )


class PaperTransformersReference:
    def __init__(self, *, heads_path: Path, attn_impl: str, seam_name: str, max_new_tokens: int) -> None:
        ensure_repo_imports()
        from cascade.mt.base import AlignAttDecoderPolicy, BaseMTBackend, load_alignatt_heads
        from cascade.runtime import gemma_model_name

        class _PromptOnlyBackend(BaseMTBackend):
            def load(self) -> None:
                raise NotImplementedError

            def translate(self, *, rendered_prompt, variant, is_partial, prompt_cache_state=None):
                raise NotImplementedError

        self._backend = _PromptOnlyBackend(model_name=gemma_model_name, runtime_config=type("Cfg", (), {
            "translation_alignatt_filter_width": 7,
            "translation_alignatt_border_margin": 0,
            "gemma_max_model_len": 1024,
            "partial_translation_min_new_tokens": 4,
            "partial_translation_token_budget_ratio": 1.0,
            "partial_translation_token_budget_buffer": 8,
            "partial_max_new_tokens": max_new_tokens,
            "max_new_tokens": max_new_tokens,
            "translation_min_new_tokens": 4,
            "translation_token_budget_ratio": 1.0,
            "translation_token_budget_buffer": 8,
            "translation_generation_margin": 8,
            "repetition_penalty": 1.0,
        })())
        self.model_name = gemma_model_name
        self.attn_impl = attn_impl
        self.seam_name = seam_name
        self.max_new_tokens = int(max_new_tokens)
        self.alignatt_heads = load_alignatt_heads(str(heads_path), top_k=8)
        self.policy = None
        self.model = None
        self.tokenizer = None
        self._AlignAttDecoderPolicy = AlignAttDecoderPolicy

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, local_files_only=True)
        self._backend.tokenizer = self.tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype="auto",
            device_map="cuda:0",
            attn_implementation=self.attn_impl,
        )
        self.model.eval()
        self.policy = self._AlignAttDecoderPolicy(tokenizer=self.tokenizer, runtime_config=self._backend.runtime_config)

    def render_prompt_package(self, rendered_prompt):
        return self._backend.render_prompt_package(rendered_prompt)

    def resolve_generation_stop_token_ids(self) -> tuple[int, ...]:
        return self._backend.resolve_generation_stop_token_ids()

    def compute_max_tokens(self, *, prompt_tokens: int, source_text: str, assistant_prefill: str) -> int:
        return self._backend.compute_max_tokens(
            prompt_tokens=prompt_tokens,
            source_text=source_text,
            is_partial=True,
            assistant_prefill=assistant_prefill,
        )

    def decode_candidate_text(self, *, generated_ids: list[int], assistant_prefill: str, variant) -> str:
        return self._backend.decode_candidate_text(
            generated_ids=generated_ids,
            assistant_prefill=assistant_prefill,
            variant=variant,
            is_partial=True,
        )

    def run_prompt(self, rendered_prompt, variant) -> dict[str, Any]:
        import torch
        from cascade.mt.base import (
            SelectedAttentionRecorder,
            SelectedLayerInputRecorder,
            extract_source_attention_rows_per_token,
            extract_source_attention_rows_per_token_from_fast_path,
        )

        prompt_package = self.render_prompt_package(rendered_prompt)
        source_map = prompt_package.source_map
        if source_map is None:
            raise RuntimeError("Benchmark prompt package did not produce a source map.")
        prompt_ids = list(prompt_package.prompt_token_ids)
        stop_ids = set(self.resolve_generation_stop_token_ids())
        max_new_tokens = self.compute_max_tokens(
            prompt_tokens=len(prompt_ids),
            source_text=rendered_prompt.source_text,
            assistant_prefill=rendered_prompt.assistant_prefill,
        )

        attention_recorder = None
        layer_input_recorder = None
        if self.seam_name == "transformers_eager":
            attention_recorder = SelectedAttentionRecorder(model=self.model, alignatt_heads=self.alignatt_heads)
        else:
            layer_input_recorder = SelectedLayerInputRecorder(model=self.model, alignatt_heads=self.alignatt_heads)

        input_ids = torch.tensor([prompt_ids], device=self.model.device, dtype=torch.long)
        with torch.no_grad():
            prompt_outputs = self.model(
                input_ids=input_ids,
                use_cache=True,
                return_dict=True,
                output_attentions=False,
            )
        past = prompt_outputs.past_key_values
        prompt_snapshot = snapshot_prompt_kv_from_past_key_values(past)
        next_token = int(torch.argmax(prompt_outputs.logits[0, -1]).item())

        generated_ids: list[int] = []
        captured_rows = 0
        row_counts: list[int] = []
        total_start = perf_counter()
        step_times_ms: list[float] = []
        stop_reason = "max_new_tokens"
        while len(generated_ids) < max_new_tokens:
            generated_ids.append(next_token)
            current_input = torch.tensor([[next_token]], device=self.model.device, dtype=torch.long)
            step_start = perf_counter()
            if self.seam_name == "transformers_eager":
                assert attention_recorder is not None
                with attention_recorder.capture() as captured_attn, torch.no_grad():
                    outputs = self.model(
                        input_ids=current_input,
                        past_key_values=past,
                        use_cache=True,
                        return_dict=True,
                        output_attentions=True,
                    )
                rows = extract_source_attention_rows_per_token(
                    layer_attentions_by_layer=captured_attn,
                    alignatt_heads=self.alignatt_heads,
                    source_positions=source_map.source_token_positions,
                )
            else:
                assert layer_input_recorder is not None
                with layer_input_recorder.capture() as captured_inputs, torch.no_grad():
                    outputs = self.model(
                        input_ids=current_input,
                        past_key_values=past,
                        use_cache=True,
                        return_dict=True,
                        output_attentions=False,
                    )
                rows, _ = extract_source_attention_rows_per_token_from_fast_path(
                    layer_inputs_by_layer=captured_inputs,
                    prompt_kv_snapshot=prompt_snapshot,
                    runtime_past_key_values=outputs.past_key_values,
                    alignatt_heads=self.alignatt_heads,
                    source_positions=source_map.source_token_positions,
                    accessible_source_token_count=source_map.accessible_source_token_count,
                )
            step_times_ms.append((perf_counter() - step_start) * 1000.0)
            row_counts.append(len(rows))
            if rows:
                captured_rows += len(rows)
            past = outputs.past_key_values
            next_token = int(torch.argmax(outputs.logits[0, -1]).item())
            if next_token in stop_ids:
                generated_ids.append(next_token)
                stop_reason = "stop_token"
                break

        special_ids = set(int(token_id) for token_id in getattr(self.tokenizer, "all_special_ids", []) or [])
        trimmed_ids = list(generated_ids)
        while trimmed_ids and trimmed_ids[-1] in special_ids:
            trimmed_ids.pop()
        total_ms = (perf_counter() - total_start) * 1000.0
        draft_text = self.decode_candidate_text(
            generated_ids=trimmed_ids,
            assistant_prefill=rendered_prompt.assistant_prefill,
            variant=variant,
        )
        return {
            "prompt_token_count": len(prompt_ids),
            "generated_token_count": len(trimmed_ids),
            "draft_text": draft_text,
            "stop_reason": stop_reason,
            "total_ms": total_ms,
            "median_step_ms": statistics.median(step_times_ms) if step_times_ms else 0.0,
            "max_step_ms": max(step_times_ms) if step_times_ms else 0.0,
            "row_counts": row_counts,
            "captured_rows": captured_rows,
        }


def run_vllm_benchmark_worker(specs: list[dict[str, Any]]) -> dict[str, Any]:
    ensure_repo_imports()
    from cascade.translation_variants import ALIGNATT_PREFIX_TRANSLATION_VARIANT

    execution = load_vllm_execution("German", max_new_tokens=20)
    backend = execution.backend
    variant = ALIGNATT_PREFIX_TRANSLATION_VARIANT
    results = []
    if specs:
        warmup_rendered = prompt_for_benchmark(specs[0], target_lang="German")
        backend.translate(rendered_prompt=warmup_rendered, variant=variant, is_partial=True)
    for repeat_index in range(BENCHMARK_REPEATS):
        for spec in specs:
            rendered = prompt_for_benchmark(spec, target_lang="German")
            result = backend.translate(rendered_prompt=rendered, variant=variant, is_partial=True)
            timings = result.timings_ms or {}
            generated_token_count = len(result.draft_generated_token_ids)
            total_ms = float(timings.get("total", 0.0))
            results.append(
                {
                    "id": spec["id"],
                    "repeat_index": repeat_index,
                    "length_bin": spec["length_bin"],
                    "assistant_prefill": spec["assistant_prefill"],
                    "prompt_num_tokens": result.prompt_num_tokens,
                    "generated_token_count": generated_token_count,
                    "draft_text": result.draft_text,
                    "acceptance_text": result.acceptance_text,
                    "stop_reason": result.stop_reason,
                    "total_ms": total_ms,
                    "per_generated_token_ms": total_ms / max(1, generated_token_count),
                    "generate_ms": float(timings.get("generate", 0.0)),
                    "prepare_observer_ms": float(timings.get("prepare_observer", 0.0)),
                    "fetch_observer_ms": float(timings.get("fetch_observer", 0.0)),
                    "reconstruct_ms": float(timings.get("reconstruct", 0.0)),
                }
            )
    return {"seam": "vllm_qk_fast", "results": results}


def run_transformers_benchmark_worker(seam: str, specs: list[dict[str, Any]]) -> dict[str, Any]:
    ensure_repo_imports()
    from cascade.translation_variants import ALIGNATT_PREFIX_TRANSLATION_VARIANT

    runner = PaperTransformersReference(
        heads_path=TARGET_LANGUAGE_TO_HEADS["German"],
        attn_impl="eager" if seam == "transformers_eager" else "sdpa",
        seam_name=seam,
        max_new_tokens=20,
    )
    runner.load()
    variant = ALIGNATT_PREFIX_TRANSLATION_VARIANT
    if specs:
        warmup_rendered = prompt_for_benchmark(specs[0], target_lang="German")
        runner.run_prompt(warmup_rendered, variant)
    results = []
    for repeat_index in range(BENCHMARK_REPEATS):
        for spec in specs:
            rendered = prompt_for_benchmark(spec, target_lang="German")
            prompt_result = runner.run_prompt(rendered, variant)
            total_ms = float(prompt_result["total_ms"])
            generated_token_count = int(prompt_result["generated_token_count"])
            results.append(
                {
                    "id": spec["id"],
                    "repeat_index": repeat_index,
                    "length_bin": spec["length_bin"],
                    "assistant_prefill": spec["assistant_prefill"],
                    "per_generated_token_ms": total_ms / max(1, generated_token_count),
                    **prompt_result,
                }
            )
    return {"seam": seam, "results": results}


def benchmark_worker_main() -> None:
    payload = json.loads(sys.stdin.read())
    seam = payload["seam"]
    specs = payload["specs"]
    if seam == "vllm_qk_fast":
        response = run_vllm_benchmark_worker(specs)
    else:
        response = run_transformers_benchmark_worker(seam, specs)
    sys.stdout.write(
        f"{BENCHMARK_JSON_BEGIN}{json.dumps(response)}{BENCHMARK_JSON_END}"
    )


def extract_benchmark_json(stdout_text: str) -> dict[str, Any]:
    start = stdout_text.rfind(BENCHMARK_JSON_BEGIN)
    end = stdout_text.rfind(BENCHMARK_JSON_END)
    if start < 0 or end < 0 or end < start:
        raise RuntimeError(
            "Benchmark worker did not return a framed JSON payload.\n"
            f"stdout tail:\n{stdout_text[-4000:]}"
        )
    payload = stdout_text[start + len(BENCHMARK_JSON_BEGIN):end]
    return json.loads(payload)


def dominant_value(values: list[Any]) -> Any:
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def summarize_benchmark_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_ms = [float(row["total_ms"]) for row in rows]
    per_token_ms = [float(row["per_generated_token_ms"]) for row in rows]
    by_length: dict[str, dict[str, float]] = {}
    grouped_by_length: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_by_spec: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_by_length[str(row["length_bin"])].append(row)
        grouped_by_spec[str(row["id"])].append(row)
    for length_bin, group in grouped_by_length.items():
        length_total_ms = [float(row["total_ms"]) for row in group]
        length_per_token_ms = [float(row["per_generated_token_ms"]) for row in group]
        by_length[length_bin] = {
            "median_total_ms": statistics.median(length_total_ms) if length_total_ms else 0.0,
            "median_per_generated_token_ms": statistics.median(length_per_token_ms) if length_per_token_ms else 0.0,
            "row_count": len(group),
        }
    by_spec = {
        spec_id: {
            "modal_draft_text": dominant_value([str(row["draft_text"]) for row in group]),
            "modal_stop_reason": dominant_value([str(row.get("stop_reason")) for row in group]),
            "median_total_ms": statistics.median([float(row["total_ms"]) for row in group]),
            "median_per_generated_token_ms": statistics.median(
                [float(row["per_generated_token_ms"]) for row in group]
            ),
        }
        for spec_id, group in grouped_by_spec.items()
    }
    return {
        "median_total_ms": statistics.median(total_ms) if total_ms else 0.0,
        "mean_total_ms": stable_mean(total_ms),
        "max_total_ms": max(total_ms) if total_ms else 0.0,
        "median_per_generated_token_ms": statistics.median(per_token_ms) if per_token_ms else 0.0,
        "mean_per_generated_token_ms": stable_mean(per_token_ms),
        "by_length_bin": by_length,
        "by_spec": by_spec,
        "rows": rows,
    }


def summarize_benchmark(results_by_seam: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"seams": {}, "divergences": []}
    all_spec_ids: set[str] = set()
    for seam, payload in results_by_seam.items():
        seam_summary = summarize_benchmark_rows(payload["results"])
        summary["seams"][seam] = seam_summary
        all_spec_ids.update(seam_summary["by_spec"].keys())

    for spec_id in sorted(all_spec_ids):
        drafts_by_seam: dict[str, Any] = {}
        stops_by_seam: dict[str, Any] = {}
        for seam, seam_summary in summary["seams"].items():
            by_spec = seam_summary["by_spec"]
            if spec_id not in by_spec:
                continue
            drafts_by_seam[seam] = by_spec[spec_id]["modal_draft_text"]
            stops_by_seam[seam] = by_spec[spec_id]["modal_stop_reason"]
        if len(set(drafts_by_seam.values())) > 1 or len(set(stops_by_seam.values())) > 1:
            summary["divergences"].append(
                {
                    "id": spec_id,
                    "drafts_by_seam": drafts_by_seam,
                    "stop_reasons_by_seam": stops_by_seam,
                }
            )
    return summary


def write_benchmark_figure(summary: dict[str, Any]) -> None:
    seam_order = [
        ("transformers_eager", "Transformers eager", "gray!45"),
        ("transformers_qk_fast", "Transformers SDPA - qk\\_fast", "orange!75!black"),
        ("vllm_qk_fast", "vLLM qk\\_fast", "blue!60!black"),
    ]
    medians = [summary["seams"][seam]["median_per_generated_token_ms"] for seam, _label, _color in seam_order]
    max_median = max(medians) if medians else 1.0
    scale = 3.9 / max(1.0, max_median)
    lines: list[str] = []
    lines.append(r"\begin{figure}[t]")
    lines.append(r"\centering")
    lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.append(r"\begin{tikzpicture}[x=1cm,y=1cm, every node/.style={font=\scriptsize}]")
    lines.append(r"\draw[rounded corners=2pt, fill=white, draw=black!20] (0,-0.2) rectangle (8.35,4.45);")
    lines.append(r"\node[anchor=west, font=\bfseries\small] at (0.25,4.00) {Paper-only MT capture benchmark};")
    lines.append(r"\node[anchor=west, text=black!60] at (0.25,3.58) {Median latency per generated token on a fixed 16-prompt text-only suite.};")
    for idx, (seam, label, color) in enumerate(seam_order):
        y = 2.80 - idx * 0.95
        width = summary["seams"][seam]["median_per_generated_token_ms"] * scale
        lines.append(rf"\node[anchor=west, text=black!80] at (0.35,{y + 0.22:.2f}) {{{label}}};")
        lines.append(rf"\filldraw[fill={color}, draw={color}] (2.95,{y:.2f}) rectangle ({2.95 + width:.2f},{y + 0.45:.2f});")
        lines.append(rf"\node[anchor=west, text=black!70] at ({3.15 + width:.2f},{y + 0.23:.2f}) {{{summary['seams'][seam]['median_per_generated_token_ms']:.1f} ms/token}};")
    lines.append(r"\node[anchor=west, text=black!55, align=left, text width=7.7cm] at (0.25,0.18) {16 prompts, 3 hot repeats per seam, greedy decode, fixed head set, subprocess isolation. Full per-prompt results and divergences are stored in the generated JSON artifact.};")
    lines.append(r"\end{tikzpicture}")
    lines.append(r"}")
    lines.append(
        r"\caption{\textbf{Inference-time comparison of MT capture seams.} "
        r"We compare a minimal Transformers eager reference, a Transformers SDPA - "
        r"qk-fast reference that reconstructs source rows from captured layer inputs, "
        r"and the engine-native vLLM qk-fast path used by the shipped backend. The "
        r"main plot reports the median latency per generated token on a fixed "
        r"16-prompt benchmark suite, "
        r"with one unmeasured warmup pass and repeated hot runs per seam.}")
    lines.append(r"\label{fig:mt-capture-speed}")
    lines.append(r"\end{figure}")
    BENCHMARK_FIG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmark_speed() -> None:
    seams = ["transformers_eager", "transformers_qk_fast", "vllm_qk_fast"]
    results_by_seam: dict[str, dict[str, Any]] = {}
    for seam in seams:
        payload = {"seam": seam, "specs": BENCHMARK_SUITE}
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker-benchmark"],
            input=json.dumps(payload).encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            cwd=str(REPO_ROOT),
        )
        if proc.stderr:
            sys.stderr.write(proc.stderr.decode("utf-8"))
        results_by_seam[seam] = extract_benchmark_json(
            proc.stdout.decode("utf-8", errors="replace")
        )
    summary = summarize_benchmark(results_by_seam)
    write_json(BENCHMARK_RESULTS_PATH, summary)
    write_benchmark_figure(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        choices=[
            "section3-figure",
            "mt-head-table",
            "mt-e2e-table",
            "low-config-table",
            "qualitative-search",
            "benchmark-speed",
        ],
    )
    parser.add_argument("--worker-benchmark", action="store_true")
    return parser.parse_args()


def main() -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    args = parse_args()
    if args.worker_benchmark:
        benchmark_worker_main()
        return
    if args.command == "section3-figure":
        build_section3_figure()
        return
    if args.command == "mt-head-table":
        build_mt_head_filtering_table()
        return
    if args.command == "mt-e2e-table":
        build_mt_e2e_head_ablation_table()
        return
    if args.command == "low-config-table":
        build_low_regime_config_snapshot_table()
        return
    if args.command == "qualitative-search":
        search_qualitative_example()
        return
    if args.command == "benchmark-speed":
        run_benchmark_speed()
        return
    raise SystemExit(
        "Choose one of: section3-figure, mt-head-table, mt-e2e-table, "
        "low-config-table, "
        "qualitative-search, benchmark-speed"
    )


if __name__ == "__main__":
    main()
