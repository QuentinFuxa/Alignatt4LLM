#!/usr/bin/env python3
"""Lean MiLMMT AlignAtt calibration sweep.

Runs only the three paper-defensible knobs selected for v1:

* chunk_ms
* translation_alignatt_border_margin
* translation_alignatt_top_k_heads

All other AlignAtt policy knobs stay fixed.  The script keeps the loaded ASR
and MiLMMT vLLM engines hot across candidates, scores every mini-run with
OmniSTEval without COMET, and optionally promotes the best candidates to the
full dev set.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cascade.runtime import LoadedModelBundle  # noqa: E402
from cascade.simulstream_processor import CascadeAlignAttProcessor  # noqa: E402
from run_simulstream_batch import run_batch_inference  # noqa: E402


DEFAULT_AUDIO_ROOT = Path("/home/dev-set/mcif-long-trans/audio")
DEFAULT_SEGMENTATION = Path("/home/dev-set/mcif-long-trans/audio-segments.yaml")
DEFAULT_REF_ROOT = Path("/home/dev-set/mcif-long-trans/ref")
DEFAULT_MINI_AUDIO = ("ccpXHNfaoy.wav", "DyXpuURBMP.wav", "rxrToXvRyM.wav")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "milmmt_alignatt_lean_sweep_enzh"


@dataclass(frozen=True)
class Candidate:
    chunk_ms: int
    border_margin: int
    top_k_heads: int

    @property
    def tag(self) -> str:
        return f"chunk{self.chunk_ms}_border{self.border_margin}_top{self.top_k_heads}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="zh", choices=("de", "it", "zh"))
    parser.add_argument("--source", default="en")
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT)
    parser.add_argument("--segmentation", type=Path, default=DEFAULT_SEGMENTATION)
    parser.add_argument("--ref-root", type=Path, default=DEFAULT_REF_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--mini-audios",
        nargs="+",
        default=list(DEFAULT_MINI_AUDIO),
        help="Mini-sweep wav filenames or paths.",
    )
    parser.add_argument("--chunk-ms", nargs="+", type=int, default=[750, 800, 850])
    parser.add_argument("--border-margin", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--top-k-heads", nargs="+", type=int, default=[8, 12, 16])
    parser.add_argument("--filter-width", type=int, default=7)
    parser.add_argument("--partial-max-new-tokens", type=int, default=16)
    parser.add_argument("--promote-full", type=int, default=2)
    parser.add_argument(
        "--full-input-dir",
        type=Path,
        default=DEFAULT_AUDIO_ROOT,
        help="Full dev-set audio directory used for promoted candidates.",
    )
    parser.add_argument(
        "--eval-python",
        type=Path,
        default=REPO_ROOT / ".venv-evaluation" / "bin" / "python",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse candidate directories that already contain evaluation.json.",
    )
    return parser.parse_args()


def resolve_audio_paths(audio_root: Path, audios: list[str]) -> list[str]:
    paths: list[str] = []
    for item in audios:
        path = Path(item)
        if not path.is_absolute():
            path = audio_root / path
        if not path.is_file():
            raise FileNotFoundError(f"Missing mini-sweep audio: {path}")
        paths.append(str(path))
    return paths


def build_processor_config(
    *,
    source: str,
    target: str,
    candidate: Candidate,
    filter_width: int,
    partial_max_new_tokens: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        source_lang_code=source,
        target_lang_code=target,
        chunk_ms=int(candidate.chunk_ms),
        speech_chunk_size=float(candidate.chunk_ms) / 1000.0,
        alignment_backend_name="qwen_forced",
        mt_backend_name="milmmt_vllm_alignatt",
        min_start_seconds=2.0,
        max_history_utterances=0,
        partial_max_new_tokens=int(partial_max_new_tokens),
        translation_alignatt_min_source_mass=0.0,
        translation_alignatt_border_margin=int(candidate.border_margin),
        translation_alignatt_inaccessible_ms=0.0,
        translation_alignatt_argmax_mass_threshold=0.0,
        translation_alignatt_top_k_heads=int(candidate.top_k_heads),
        translation_alignatt_filter_width=int(filter_width),
        paper_context_path=None,
        paper_context_mode="off",
    )


def install_hot_bundle_reuse() -> None:
    """Reuse loaded engines while allowing policy/head-count changes."""

    def _ensure_hot_bundle(cls, runtime_config):
        if cls._bundle is None:
            cls._bundle = LoadedModelBundle(runtime_config)
            cls._bundle.load()
            cls._bundle_signature = cls._bundle_key(runtime_config)
            return cls._bundle

        cls._bundle.config = runtime_config
        mt_backend = cls._bundle.mt_backend
        if mt_backend is not None:
            mt_backend.runtime_config = runtime_config
            mt_backend.refresh_alignatt_artifacts()
        alignment_backend = cls._bundle.alignment_backend
        if alignment_backend is not None and hasattr(alignment_backend, "runtime_config"):
            alignment_backend.runtime_config = runtime_config
        return cls._bundle

    CascadeAlignAttProcessor._ensure_bundle = classmethod(_ensure_hot_bundle)


def evaluate_output(
    *,
    eval_python: Path,
    output_dir: Path,
    segmentation: Path,
    target_reference: Path,
    source_reference: Path,
    target: str,
) -> dict[str, Any]:
    cmd = [
        str(eval_python),
        str(REPO_ROOT / "evaluate_cascade_outputs.py"),
        "--output-dir",
        str(output_dir),
        "--speech-segmentation",
        str(segmentation),
        "--target-reference",
        str(target_reference),
        "--source-reference",
        str(source_reference),
        "--target-lang-code",
        target,
        "--skip-comet",
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    return json.loads((output_dir / "evaluation.json").read_text(encoding="utf-8"))


def load_existing_evaluation(output_dir: Path) -> dict[str, Any] | None:
    path = output_dir / "evaluation.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def extract_empty_predictions(evaluation: dict[str, Any]) -> int | None:
    for line in evaluation.get("report_lines", []):
        if "Empty Predictions:" in str(line):
            return int(str(line).rsplit(":", 1)[1].strip())
    return None


def summarize_result(
    *,
    candidate: Candidate,
    output_dir: Path,
    evaluation: dict[str, Any],
    kind: str,
) -> dict[str, Any]:
    scores = evaluation.get("contract_scores", {})
    return {
        "kind": kind,
        **asdict(candidate),
        "tag": candidate.tag,
        "output_dir": str(output_dir),
        "bleu": scores.get("BLEU"),
        "chrf": scores.get("CHRF"),
        "longyaal_cu_ms": scores.get("LongYAAL CU"),
        "longyaal_ca_ms": scores.get("LongYAAL CA"),
        "empty_predictions": extract_empty_predictions(evaluation),
    }


def write_summaries(output_root: Path, rows: list[dict[str, Any]], name: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / f"{name}.json"
    tsv_path = output_root / f"{name}.tsv"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    columns = [
        "kind",
        "tag",
        "chunk_ms",
        "border_margin",
        "top_k_heads",
        "bleu",
        "chrf",
        "longyaal_cu_ms",
        "longyaal_ca_ms",
        "empty_predictions",
        "output_dir",
    ]
    with tsv_path.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(columns) + "\n")
        for row in rows:
            fh.write("\t".join("" if row.get(col) is None else str(row.get(col)) for col in columns) + "\n")


def is_valid_for_promotion(row: dict[str, Any]) -> bool:
    return (
        row.get("empty_predictions") == 0
        and row.get("longyaal_cu_ms") is not None
        and float(row["longyaal_cu_ms"]) < 2000.0
    )


def rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = [row for row in rows if is_valid_for_promotion(row)]
    return sorted(
        valid,
        key=lambda row: (
            float(row.get("chrf") or 0.0),
            float(row.get("bleu") or 0.0),
            -float(row.get("longyaal_cu_ms") or 999999.0),
        ),
        reverse=True,
    )


def ordered_candidates(args: argparse.Namespace) -> list[Candidate]:
    baseline = Candidate(chunk_ms=800, border_margin=0, top_k_heads=8)
    candidates = [
        Candidate(chunk_ms=chunk, border_margin=border, top_k_heads=top_k)
        for chunk, border, top_k in itertools.product(
            args.chunk_ms,
            args.border_margin,
            args.top_k_heads,
        )
    ]
    unique: list[Candidate] = []
    for candidate in [baseline, *candidates]:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def main() -> None:
    args = parse_args()
    install_hot_bundle_reuse()

    mini_inputs = resolve_audio_paths(args.audio_root, args.mini_audios)
    target_reference = args.ref_root / f"{args.target}.txt"
    source_reference = args.ref_root / f"{args.source}.txt"
    if not target_reference.is_file():
        raise FileNotFoundError(target_reference)
    if not source_reference.is_file():
        raise FileNotFoundError(source_reference)

    candidates = ordered_candidates(args)
    mini_rows: list[dict[str, Any]] = []

    print(
        f"Lean AlignAtt sweep target={args.target} candidates={len(candidates)} "
        f"mini_inputs={len(mini_inputs)} output_root={args.output_root}",
        flush=True,
    )

    for index, candidate in enumerate(candidates, start=1):
        output_dir = args.output_root / "mini" / candidate.tag
        print(f"\n[{index}/{len(candidates)}] mini {candidate.tag}", flush=True)
        evaluation = load_existing_evaluation(output_dir) if args.skip_existing else None
        if evaluation is None:
            config = build_processor_config(
                source=args.source,
                target=args.target,
                candidate=candidate,
                filter_width=args.filter_width,
                partial_max_new_tokens=args.partial_max_new_tokens,
            )
            run_batch_inference(
                processor_config=config,
                input_paths=mini_inputs,
                output_dir=str(output_dir),
                source_lang_code=args.source,
                target_lang_code=args.target,
                explicit_paper_context_path=None,
                paper_context_dir=None,
            )
            evaluation = evaluate_output(
                eval_python=args.eval_python,
                output_dir=output_dir,
                segmentation=args.segmentation,
                target_reference=target_reference,
                source_reference=source_reference,
                target=args.target,
            )
        row = summarize_result(
            candidate=candidate,
            output_dir=output_dir,
            evaluation=evaluation,
            kind="mini",
        )
        mini_rows.append(row)
        write_summaries(args.output_root, mini_rows, "mini_summary")
        print(
            "  BLEU={bleu:.4f} chrF={chrf:.4f} CU={cu:.1f} empty={empty}".format(
                bleu=float(row.get("bleu") or 0.0),
                chrf=float(row.get("chrf") or 0.0),
                cu=float(row.get("longyaal_cu_ms") or 0.0),
                empty=row.get("empty_predictions"),
            ),
            flush=True,
        )

    ranked = rank_rows(mini_rows)
    write_summaries(args.output_root, ranked, "mini_ranked")

    full_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(ranked[: max(0, int(args.promote_full))], start=1):
        candidate = Candidate(
            chunk_ms=int(row["chunk_ms"]),
            border_margin=int(row["border_margin"]),
            top_k_heads=int(row["top_k_heads"]),
        )
        output_dir = args.output_root / "full" / f"rank{rank}_{candidate.tag}"
        print(f"\n[promote {rank}] full {candidate.tag}", flush=True)
        evaluation = load_existing_evaluation(output_dir) if args.skip_existing else None
        if evaluation is None:
            config = build_processor_config(
                source=args.source,
                target=args.target,
                candidate=candidate,
                filter_width=args.filter_width,
                partial_max_new_tokens=args.partial_max_new_tokens,
            )
            run_batch_inference(
                processor_config=config,
                input_paths=[str(path) for path in sorted(args.full_input_dir.glob("*.wav"))],
                output_dir=str(output_dir),
                source_lang_code=args.source,
                target_lang_code=args.target,
                explicit_paper_context_path=None,
                paper_context_dir=None,
            )
            evaluation = evaluate_output(
                eval_python=args.eval_python,
                output_dir=output_dir,
                segmentation=args.segmentation,
                target_reference=target_reference,
                source_reference=source_reference,
                target=args.target,
            )
        full_row = summarize_result(
            candidate=candidate,
            output_dir=output_dir,
            evaluation=evaluation,
            kind="full",
        )
        full_rows.append(full_row)
        write_summaries(args.output_root, full_rows, "full_summary")

    print("\nDone. Summaries:", flush=True)
    print(f"  {args.output_root / 'mini_summary.tsv'}", flush=True)
    print(f"  {args.output_root / 'mini_ranked.tsv'}", flush=True)
    if full_rows:
        print(f"  {args.output_root / 'full_summary.tsv'}", flush=True)


if __name__ == "__main__":
    main()
