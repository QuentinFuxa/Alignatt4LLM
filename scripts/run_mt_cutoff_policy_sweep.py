#!/usr/bin/env python3
"""Run real MT acceptance-policy points for AlignAtt vs fixed cut-last-x."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from cascade.runtime import LoadedModelBundle  # noqa: E402
from cascade.simulstream_processor import CascadeAlignAttProcessor  # noqa: E402
from cascade.submission import get_submission_preset  # noqa: E402
from run_simulstream_batch import resolve_input_paths, run_batch_inference  # noqa: E402


def _float_tag(value: float) -> str:
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--inputs", nargs="+", help="Explicit wav/media paths.")
    inputs.add_argument("--input-dir", help="Input directory for a full subset.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target", default="de", choices=("de", "it", "zh"))
    parser.add_argument("--source", default="en")
    parser.add_argument("--base-preset", default="main_low_latency")
    parser.add_argument("--chunk-ms", type=int, default=850)
    parser.add_argument("--cutoffs", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7])
    parser.add_argument(
        "--alignatt-border-margins",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Optional AlignAtt border-margin variants. When omitted, the sweep "
            "uses the base preset's single AlignAtt setting."
        ),
    )
    parser.add_argument(
        "--alignatt-top-k-heads",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Optional AlignAtt head-count variants. When omitted, the sweep "
            "uses the base preset's configured head count."
        ),
    )
    parser.add_argument(
        "--alignatt-min-source-masses",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Optional AlignAtt accessible-source-mass gate variants. When "
            "omitted, the sweep uses the base preset's configured gate."
        ),
    )
    parser.add_argument("--alignatt-frontier-min-inaccessible-mass", type=float, default=0.0)
    parser.add_argument("--alignatt-max-inaccessible-source-mass", type=float, default=1.0)
    parser.add_argument("--alignatt-min-accessible-inaccessible-margin", type=float, default=-1.0)
    parser.add_argument("--skip-alignatt", action="store_true")
    parser.add_argument("--skip-cutoffs", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing policy output directories that already contain hypothesis.jsonl.",
    )
    parser.add_argument("--mt-vllm-gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--gemma-vllm-gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--mt-vllm-enable-speculative-decoding", action="store_true")
    parser.add_argument("--mt-vllm-speculative-assistant-model", default=None)
    parser.add_argument("--mt-vllm-num-speculative-tokens", type=int, default=4)
    return parser.parse_args()


def install_hot_bundle_reuse() -> None:
    """Reuse loaded ASR/MT engines across policy points."""

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
            if hasattr(mt_backend, "refresh_alignatt_artifacts"):
                mt_backend.refresh_alignatt_artifacts()
        alignment_backend = cls._bundle.alignment_backend
        if alignment_backend is not None and hasattr(alignment_backend, "runtime_config"):
            alignment_backend.runtime_config = runtime_config
        return cls._bundle

    CascadeAlignAttProcessor._ensure_bundle = classmethod(_ensure_hot_bundle)


def build_processor_config(
    *,
    base_preset_name: str,
    source: str,
    target: str,
    chunk_ms: int,
    policy: str,
    cutoff_units: int,
    alignatt_border_margin: int | None,
    alignatt_top_k_heads: int | None,
    alignatt_min_source_mass: float | None,
    alignatt_frontier_min_inaccessible_mass: float,
    alignatt_max_inaccessible_source_mass: float,
    alignatt_min_accessible_inaccessible_margin: float,
    mt_vllm_gpu_memory_utilization: float | None,
    gemma_vllm_gpu_memory_utilization: float | None,
    mt_vllm_enable_speculative_decoding: bool,
    mt_vllm_speculative_assistant_model: str | None,
    mt_vllm_num_speculative_tokens: int,
) -> SimpleNamespace:
    preset = get_submission_preset(base_preset_name)
    config = preset.build_speech_processor_config(
        source_lang_code=source,
        target_lang_code=target,
        paper_context_path=None,
    )
    config.chunk_ms = int(chunk_ms)
    config.speech_chunk_size = int(chunk_ms) / 1000.0
    config.translation_acceptance_policy = policy
    config.translation_static_cutoff_units = int(cutoff_units)
    if alignatt_border_margin is not None:
        config.translation_alignatt_border_margin = int(alignatt_border_margin)
    if alignatt_top_k_heads is not None:
        config.translation_alignatt_top_k_heads = int(alignatt_top_k_heads)
    if alignatt_min_source_mass is not None:
        config.translation_alignatt_min_source_mass = float(alignatt_min_source_mass)
    config.translation_alignatt_frontier_min_inaccessible_mass = float(
        alignatt_frontier_min_inaccessible_mass
    )
    config.translation_alignatt_max_inaccessible_source_mass = float(
        alignatt_max_inaccessible_source_mass
    )
    config.translation_alignatt_min_accessible_inaccessible_margin = float(
        alignatt_min_accessible_inaccessible_margin
    )
    if mt_vllm_gpu_memory_utilization is not None:
        config.mt_vllm_gpu_memory_utilization = float(mt_vllm_gpu_memory_utilization)
    if gemma_vllm_gpu_memory_utilization is not None:
        config.gemma_vllm_gpu_memory_utilization = float(gemma_vllm_gpu_memory_utilization)
    config.mt_vllm_enable_speculative_decoding = bool(mt_vllm_enable_speculative_decoding)
    config.mt_vllm_speculative_assistant_model = mt_vllm_speculative_assistant_model
    config.mt_vllm_num_speculative_tokens = int(mt_vllm_num_speculative_tokens)
    return config


def policy_points(args: argparse.Namespace) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    if not args.skip_alignatt:
        if (
            args.alignatt_border_margins is None
            and args.alignatt_top_k_heads is None
            and args.alignatt_min_source_masses is None
        ):
            points.append(
                {
                    "tag": "alignatt",
                    "policy": "alignatt",
                    "cutoff_units": 0,
                    "alignatt_border_margin": None,
                    "alignatt_top_k_heads": None,
                    "alignatt_min_source_mass": None,
                }
            )
        else:
            preset = get_submission_preset(args.base_preset)
            border_margins = (
                args.alignatt_border_margins
                if args.alignatt_border_margins is not None
                else [preset.translation_alignatt_border_margin]
            )
            top_k_heads = (
                args.alignatt_top_k_heads
                if args.alignatt_top_k_heads is not None
                else [8]
            )
            min_source_masses = (
                args.alignatt_min_source_masses
                if args.alignatt_min_source_masses is not None
                else [preset.translation_alignatt_min_source_mass]
            )
            for border_margin in sorted({int(value) for value in border_margins}):
                for top_k in sorted({int(value) for value in top_k_heads}):
                    for min_source_mass in sorted({float(value) for value in min_source_masses}):
                        mass_tag = f"_m{_float_tag(min_source_mass)}"
                        if min_source_mass == 0.0:
                            mass_tag = ""
                        points.append(
                            {
                                "tag": (
                                    f"alignatt_b{border_margin}_top{top_k}{mass_tag}"
                                ),
                                "policy": "alignatt",
                                "cutoff_units": 0,
                                "alignatt_border_margin": int(border_margin),
                                "alignatt_top_k_heads": int(top_k),
                                "alignatt_min_source_mass": float(min_source_mass),
                            }
                        )
    if not args.skip_cutoffs:
        for cutoff in sorted({max(0, int(value)) for value in args.cutoffs}):
            points.append(
                {
                    "tag": f"cut_last_{cutoff}",
                    "policy": "cut_last_target_units",
                    "cutoff_units": int(cutoff),
                    "alignatt_border_margin": None,
                    "alignatt_top_k_heads": None,
                    "alignatt_min_source_mass": None,
                }
            )
    return points


def main() -> None:
    args = parse_args()
    input_paths = resolve_input_paths(inputs=args.inputs, input_dir=args.input_dir)
    args.output_root.mkdir(parents=True, exist_ok=True)

    first_policy = "alignatt" if not args.skip_alignatt else "cut_last_target_units"
    first_cutoff = 0 if not args.skip_alignatt else int(sorted(args.cutoffs)[0])
    load_config = build_processor_config(
        base_preset_name=args.base_preset,
        source=args.source,
        target=args.target,
        chunk_ms=args.chunk_ms,
        policy=first_policy,
        cutoff_units=first_cutoff,
        alignatt_border_margin=None,
        alignatt_top_k_heads=None,
        alignatt_min_source_mass=None,
        alignatt_frontier_min_inaccessible_mass=args.alignatt_frontier_min_inaccessible_mass,
        alignatt_max_inaccessible_source_mass=args.alignatt_max_inaccessible_source_mass,
        alignatt_min_accessible_inaccessible_margin=args.alignatt_min_accessible_inaccessible_margin,
        mt_vllm_gpu_memory_utilization=args.mt_vllm_gpu_memory_utilization,
        gemma_vllm_gpu_memory_utilization=args.gemma_vllm_gpu_memory_utilization,
        mt_vllm_enable_speculative_decoding=args.mt_vllm_enable_speculative_decoding,
        mt_vllm_speculative_assistant_model=args.mt_vllm_speculative_assistant_model,
        mt_vllm_num_speculative_tokens=args.mt_vllm_num_speculative_tokens,
    )
    print(
        f"Loading hot bundle for {args.source}->{args.target}, "
        f"chunk_ms={args.chunk_ms}, inputs={len(input_paths)}",
        flush=True,
    )
    CascadeAlignAttProcessor.load_model(load_config)
    install_hot_bundle_reuse()

    rows: list[dict[str, Any]] = []
    for point in policy_points(args):
        tag = str(point["tag"])
        policy = str(point["policy"])
        cutoff_units = int(point["cutoff_units"])
        output_dir = args.output_root / tag
        hypothesis_path = output_dir / "hypothesis.jsonl"
        config = build_processor_config(
            base_preset_name=args.base_preset,
            source=args.source,
            target=args.target,
            chunk_ms=args.chunk_ms,
            policy=policy,
            cutoff_units=cutoff_units,
            alignatt_border_margin=point.get("alignatt_border_margin"),
            alignatt_top_k_heads=point.get("alignatt_top_k_heads"),
            alignatt_min_source_mass=point.get("alignatt_min_source_mass"),
            alignatt_frontier_min_inaccessible_mass=args.alignatt_frontier_min_inaccessible_mass,
            alignatt_max_inaccessible_source_mass=args.alignatt_max_inaccessible_source_mass,
            alignatt_min_accessible_inaccessible_margin=args.alignatt_min_accessible_inaccessible_margin,
            mt_vllm_gpu_memory_utilization=args.mt_vllm_gpu_memory_utilization,
            gemma_vllm_gpu_memory_utilization=args.gemma_vllm_gpu_memory_utilization,
            mt_vllm_enable_speculative_decoding=args.mt_vllm_enable_speculative_decoding,
            mt_vllm_speculative_assistant_model=args.mt_vllm_speculative_assistant_model,
            mt_vllm_num_speculative_tokens=args.mt_vllm_num_speculative_tokens,
        )
        if args.resume and hypothesis_path.exists():
            print(f"\n>>> {tag}: existing hypothesis found, reusing {output_dir}", flush=True)
            rows.append(
                {
                    "tag": tag,
                    "policy": policy,
                    "cutoff_units": int(cutoff_units),
                    "alignatt_border_margin": point.get("alignatt_border_margin"),
                    "alignatt_top_k_heads": point.get("alignatt_top_k_heads"),
                    "alignatt_min_source_mass": point.get("alignatt_min_source_mass"),
                    "alignatt_frontier_min_inaccessible_mass": args.alignatt_frontier_min_inaccessible_mass,
                    "alignatt_max_inaccessible_source_mass": args.alignatt_max_inaccessible_source_mass,
                    "alignatt_min_accessible_inaccessible_margin": args.alignatt_min_accessible_inaccessible_margin,
                    "output_dir": str(output_dir),
                    "num_inputs": len(input_paths),
                    "model_load_ms": None,
                    "resumed": True,
                    "mt_vllm_enable_speculative_decoding": args.mt_vllm_enable_speculative_decoding,
                    "mt_vllm_speculative_assistant_model": args.mt_vllm_speculative_assistant_model,
                    "mt_vllm_num_speculative_tokens": args.mt_vllm_num_speculative_tokens,
                }
            )
            continue

        print(
            f"\n>>> {tag}: policy={policy} cutoff={cutoff_units} "
            f"border={point.get('alignatt_border_margin')} "
            f"top_k={point.get('alignatt_top_k_heads')} "
            f"min_mass={point.get('alignatt_min_source_mass')} -> {output_dir}",
            flush=True,
        )
        result = run_batch_inference(
            processor_config=config,
            input_paths=input_paths,
            output_dir=str(output_dir),
            source_lang_code=args.source,
            target_lang_code=args.target,
            explicit_paper_context_path=None,
            paper_context_dir=None,
        )
        rows.append(
            {
                "tag": tag,
                "policy": policy,
                "cutoff_units": int(cutoff_units),
                "alignatt_border_margin": point.get("alignatt_border_margin"),
                "alignatt_top_k_heads": point.get("alignatt_top_k_heads"),
                "alignatt_min_source_mass": point.get("alignatt_min_source_mass"),
                "alignatt_frontier_min_inaccessible_mass": args.alignatt_frontier_min_inaccessible_mass,
                "alignatt_max_inaccessible_source_mass": args.alignatt_max_inaccessible_source_mass,
                "alignatt_min_accessible_inaccessible_margin": args.alignatt_min_accessible_inaccessible_margin,
                "output_dir": str(output_dir),
                "num_inputs": len(input_paths),
                "model_load_ms": result.get("model_load_ms"),
                "mt_vllm_enable_speculative_decoding": args.mt_vllm_enable_speculative_decoding,
                "mt_vllm_speculative_assistant_model": args.mt_vllm_speculative_assistant_model,
                "mt_vllm_num_speculative_tokens": args.mt_vllm_num_speculative_tokens,
            }
        )

    summary_path = args.output_root / "policy_points.json"
    summary_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWrote {summary_path}")
    print("Score each output with:")
    print(
        "  .venv-evaluation/bin/python evaluate_cascade_outputs.py "
        "--output-dir <output_dir>"
    )


if __name__ == "__main__":
    main()
