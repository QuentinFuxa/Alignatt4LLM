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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from alignatt4llm.runtime import (  # noqa: E402
    LoadedModelBundle,
    VALID_ALIGNATT_ACCEPTANCE_VARIANTS,
    VALID_ALIGNATT_ONLINE_NORMALIZATIONS,
    VALID_MT_BACKEND_NAMES,
)
from alignatt4llm.simulstream_processor import CascadeAlignAttProcessor  # noqa: E402
from alignatt4llm.presets import get_runtime_preset  # noqa: E402
from alignatt4llm.cli.batch import resolve_input_paths, run_batch_inference  # noqa: E402


def _float_tag(value: float) -> str:
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def point_value(point: dict[str, Any], key: str, default: Any) -> Any:
    value = point.get(key, default)
    return default if value is None else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--inputs", nargs="+", help="Explicit wav/media paths.")
    inputs.add_argument("--input-dir", help="Input directory for a full subset.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target", default="de", choices=("de", "it", "zh"))
    parser.add_argument("--source", default="en")
    parser.add_argument("--base-preset", default="gemma_low_latency")
    parser.add_argument(
        "--mt-backend-name",
        default="gemma_vllm_alignatt",
        choices=VALID_MT_BACKEND_NAMES,
        help="MT backend route to use for every policy point.",
    )
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
        "--alignatt-acceptance-variants",
        nargs="+",
        choices=VALID_ALIGNATT_ACCEPTANCE_VARIANTS,
        default=None,
        help=(
            "Optional AlignAtt acceptance variants. Use this to compare token "
            "frontier acceptance against clean unit-level source-mass variants "
            "without reloading the models."
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
    parser.add_argument("--alignatt-unit-consensus-min-head-ratio", type=float, default=0.60)
    parser.add_argument(
        "--alignatt-unit-consensus-min-head-ratios",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Optional unit-consensus head-ratio thresholds. This only affects "
            "`unit_consensus` AlignAtt acceptance points and is recorded in "
            "policy_points.json."
        ),
    )
    parser.add_argument("--alignatt-min-alignment-confidence", type=float, default=0.0)
    parser.add_argument(
        "--alignatt-min-alignment-confidences",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Optional alignment-confidence floors. This only affects "
            "`unit_conf` AlignAtt acceptance points and is recorded in "
            "policy_points.json."
        ),
    )
    parser.add_argument(
        "--alignatt-online-normalization",
        default="zscore",
        choices=VALID_ALIGNATT_ONLINE_NORMALIZATIONS,
    )
    parser.add_argument(
        "--alignatt-online-normalizations",
        nargs="+",
        choices=VALID_ALIGNATT_ONLINE_NORMALIZATIONS,
        default=None,
        help=(
            "Optional AlignAtt online normalization variants. `zscore` is the "
            "maintained default; `raw` is useful for ablation and is recorded "
            "in policy_points.json."
        ),
    )
    parser.add_argument("--alignatt-source-bearing-min-source-mass", type=float, default=0.005)
    parser.add_argument(
        "--alignatt-source-bearing-min-source-masses",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Optional source-bearing thresholds for `unit_mass_source_bearing` "
            "AlignAtt points. This sweeps an existing runtime knob and is "
            "recorded in policy_points.json."
        ),
    )
    parser.add_argument(
        "--alignatt-source-bearing-hard-inaccessible-cap",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--alignatt-source-bearing-hard-inaccessible-caps",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Optional hard inaccessible-source caps for source-bearing unit "
            "AlignAtt points."
        ),
    )
    parser.add_argument("--alignatt-inaccessible-ms", type=float, default=0.0)
    parser.add_argument(
        "--alignatt-inaccessible-ms-values",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Optional AlignAtt source-frontier holdback values. When provided, "
            "each value becomes its own AlignAtt policy point."
        ),
    )
    parser.add_argument("--alignatt-frontier-min-inaccessible-mass", type=float, default=0.0)
    parser.add_argument(
        "--alignatt-frontier-min-inaccessible-masses",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Optional soft-frontier future-mass values. When provided, each "
            "value becomes its own AlignAtt policy point."
        ),
    )
    parser.add_argument(
        "--alignatt-source-frontier-action",
        default="stop",
        choices=("stop", "trim_unrecovered"),
        help=(
            "Apply source-frontier hits as the historical token stop or as an "
            "unrecovered target-unit suffix trim after draft generation."
        ),
    )
    parser.add_argument("--alignatt-max-inaccessible-source-mass", type=float, default=1.0)
    parser.add_argument("--alignatt-max-non-source-prompt-mass", type=float, default=1.0)
    parser.add_argument(
        "--alignatt-max-non-source-prompt-masses",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Optional provenance diagnostic caps on non-source prompt mass. "
            "Values below 1.0 are guarded policy variants and are recorded in "
            "policy_points.json."
        ),
    )
    parser.add_argument(
        "--alignatt-min-accepted-accessible-source-mass",
        type=float,
        default=0.0,
        help=(
            "Clean source-mass floor over the accepted target prefix. The "
            "runtime trims to the longest target-unit prefix whose global and "
            "recent accessible-source mass exceed this value."
        ),
    )
    parser.add_argument(
        "--alignatt-min-accepted-accessible-source-masses",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Optional sweep values for accepted-prefix source-mass floors. "
            "This is a clean source-accessibility axis when other guards are "
            "disabled and is recorded in policy_points.json."
        ),
    )
    parser.add_argument(
        "--alignatt-accepted-accessible-source-mass-recent-units",
        type=int,
        default=2,
        help=(
            "Number of recent target stability units used by the accepted-"
            "prefix source-mass floor."
        ),
    )
    parser.add_argument("--alignatt-min-accessible-inaccessible-margin", type=float, default=-1.0)
    parser.add_argument("--alignatt-min-accessible-source-units", type=int, default=None)
    parser.add_argument(
        "--alignatt-min-accessible-source-units-mode",
        default="block",
        choices=("block", "target_unit_cap"),
    )
    parser.add_argument(
        "--alignatt-min-accessible-source-units-modes",
        nargs="+",
        default=None,
        choices=("block", "target_unit_cap"),
        help=(
            "Optional variants for the min-accessible-source-units guard. This "
            "sweeps an existing runtime knob so guarded diagnostics can test "
            "whether source-unit blocking is making AlignAtt too conservative."
        ),
    )
    parser.add_argument("--alignatt-max-source-regression", type=int, default=None)
    parser.add_argument("--alignatt-source-regression-min-inaccessible-mass", type=float, default=0.0)
    parser.add_argument(
        "--alignatt-source-regression-min-inaccessible-masses",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Optional AlignAtt sweep values for the future-bearing "
            "source-regression gate. When provided, each value becomes its own "
            "AlignAtt policy point and is recorded in policy_points.json."
        ),
    )
    parser.add_argument("--alignatt-source-regression-recent-tokens", type=int, default=0)
    parser.add_argument(
        "--alignatt-source-regression-reference-mode",
        default="max",
        choices=("max", "median_recent"),
    )
    parser.add_argument(
        "--alignatt-source-regression-activation-mode",
        default="always",
        choices=("always", "frontier_reached"),
    )
    parser.add_argument("--alignatt-source-regression-activation-slack-tokens", type=int, default=0)
    parser.add_argument("--alignatt-source-regression-patience-tokens", type=int, default=1)
    parser.add_argument(
        "--alignatt-source-regression-action",
        default="stop",
        choices=("stop", "trim_target_unit", "trim_unrecovered"),
        help=(
            "Apply source regression as the historical token-level stop or as "
            "a target-unit suffix trim after draft generation. trim_unrecovered "
            "keeps local regressions that recover later in the same draft."
        ),
    )
    parser.add_argument("--alignatt-token-argmax-frontier-gate", action="store_true")
    parser.add_argument("--alignatt-token-argmax-frontier-patience-tokens", type=int, default=1)
    parser.add_argument("--alignatt-source-lcp-stability", action="store_true")
    parser.add_argument("--alignatt-source-lcp-append-slack-units", type=int, default=0)
    parser.add_argument("--skip-alignatt", action="store_true")
    parser.add_argument("--skip-cutoffs", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing policy output directories that already contain hypothesis.jsonl.",
    )
    parser.add_argument("--mt-vllm-gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--gemma-vllm-gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--asr-gpu-memory-utilization", type=float, default=None)
    parser.add_argument(
        "--mt-vllm-enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run the MT vLLM engine eagerly (default: enabled). Required for "
            "trustworthy AlignAtt observer capture under vLLM 0.22 nightlies; "
            "cudagraph replay corrupts the captured q/k payload. Pass "
            "--no-mt-vllm-enforce-eager to re-enable CUDA graphs (corrupts "
            "observer capture; debugging only)."
        ),
    )
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
    mt_backend_name: str,
    policy: str,
    cutoff_units: int,
    alignatt_border_margin: int | None,
    alignatt_top_k_heads: int | None,
    alignatt_acceptance_variant: str | None,
    alignatt_min_source_mass: float | None,
    alignatt_unit_consensus_min_head_ratio: float,
    alignatt_min_alignment_confidence: float = 0.0,
    alignatt_online_normalization: str = "zscore",
    alignatt_source_bearing_min_source_mass: float,
    alignatt_source_bearing_hard_inaccessible_cap: float,
    alignatt_inaccessible_ms: float,
    alignatt_frontier_min_inaccessible_mass: float,
    alignatt_source_frontier_action: str,
    alignatt_max_inaccessible_source_mass: float,
    alignatt_max_non_source_prompt_mass: float = 1.0,
    alignatt_min_accepted_accessible_source_mass: float = 0.0,
    alignatt_accepted_accessible_source_mass_recent_units: int = 2,
    alignatt_min_accessible_inaccessible_margin: float,
    alignatt_min_accessible_source_units: int | None,
    alignatt_min_accessible_source_units_mode: str,
    alignatt_max_source_regression: int | None,
    alignatt_source_regression_min_inaccessible_mass: float,
    alignatt_source_regression_recent_tokens: int,
    alignatt_source_regression_reference_mode: str,
    alignatt_source_regression_activation_mode: str,
    alignatt_source_regression_activation_slack_tokens: int,
    alignatt_source_regression_patience_tokens: int,
    alignatt_source_regression_action: str,
    alignatt_token_argmax_frontier_gate: bool,
    alignatt_token_argmax_frontier_patience_tokens: int,
    alignatt_source_lcp_stability: bool,
    alignatt_source_lcp_append_slack_units: int,
    asr_gpu_memory_utilization: float | None,
    mt_vllm_gpu_memory_utilization: float | None,
    gemma_vllm_gpu_memory_utilization: float | None,
    mt_vllm_enforce_eager: bool = True,
    mt_vllm_enable_speculative_decoding: bool,
    mt_vllm_speculative_assistant_model: str | None,
    mt_vllm_num_speculative_tokens: int,
) -> SimpleNamespace:
    preset = get_runtime_preset(base_preset_name)
    config = preset.build_speech_processor_config(
        source_lang_code=source,
        target_lang_code=target,
        paper_context_path=None,
    )
    config.chunk_ms = int(chunk_ms)
    config.speech_chunk_size = int(chunk_ms) / 1000.0
    config.mt_backend_name = str(mt_backend_name)
    config.translation_acceptance_policy = policy
    config.translation_static_cutoff_units = int(cutoff_units)
    if alignatt_border_margin is not None:
        config.translation_alignatt_border_margin = int(alignatt_border_margin)
    if alignatt_top_k_heads is not None:
        config.translation_alignatt_top_k_heads = int(alignatt_top_k_heads)
    if alignatt_acceptance_variant is not None:
        config.translation_alignatt_acceptance_variant = str(alignatt_acceptance_variant)
    if alignatt_min_source_mass is not None:
        config.translation_alignatt_min_source_mass = float(alignatt_min_source_mass)
    config.translation_alignatt_unit_consensus_min_head_ratio = float(
        alignatt_unit_consensus_min_head_ratio
    )
    config.translation_alignatt_min_alignment_confidence = float(
        alignatt_min_alignment_confidence
    )
    config.translation_alignatt_online_normalization = str(alignatt_online_normalization)
    config.translation_alignatt_source_bearing_min_source_mass = float(
        alignatt_source_bearing_min_source_mass
    )
    config.translation_alignatt_source_bearing_hard_inaccessible_cap = float(
        alignatt_source_bearing_hard_inaccessible_cap
    )
    config.translation_alignatt_inaccessible_ms = float(alignatt_inaccessible_ms)
    config.translation_alignatt_frontier_min_inaccessible_mass = float(
        alignatt_frontier_min_inaccessible_mass
    )
    config.translation_alignatt_source_frontier_action = str(
        alignatt_source_frontier_action
    )
    config.translation_alignatt_max_inaccessible_source_mass = float(
        alignatt_max_inaccessible_source_mass
    )
    config.translation_alignatt_max_non_source_prompt_mass = float(
        alignatt_max_non_source_prompt_mass
    )
    config.translation_alignatt_min_accepted_accessible_source_mass = float(
        alignatt_min_accepted_accessible_source_mass
    )
    config.translation_alignatt_accepted_accessible_source_mass_recent_units = int(
        alignatt_accepted_accessible_source_mass_recent_units
    )
    config.translation_alignatt_min_accessible_inaccessible_margin = float(
        alignatt_min_accessible_inaccessible_margin
    )
    if alignatt_min_accessible_source_units is not None:
        config.translation_alignatt_min_accessible_source_units = int(
            alignatt_min_accessible_source_units
        )
    config.translation_alignatt_min_accessible_source_units_mode = str(
        alignatt_min_accessible_source_units_mode
    )
    if alignatt_max_source_regression is not None:
        config.translation_alignatt_max_source_regression = int(
            alignatt_max_source_regression
        )
    config.translation_alignatt_source_regression_min_inaccessible_mass = float(
        alignatt_source_regression_min_inaccessible_mass
    )
    config.translation_alignatt_source_regression_recent_tokens = int(
        alignatt_source_regression_recent_tokens
    )
    config.translation_alignatt_source_regression_reference_mode = str(
        alignatt_source_regression_reference_mode
    )
    config.translation_alignatt_source_regression_activation_mode = str(
        alignatt_source_regression_activation_mode
    )
    config.translation_alignatt_source_regression_activation_slack_tokens = int(
        alignatt_source_regression_activation_slack_tokens
    )
    config.translation_alignatt_source_regression_patience_tokens = int(
        alignatt_source_regression_patience_tokens
    )
    config.translation_alignatt_source_regression_action = str(
        alignatt_source_regression_action
    )
    config.translation_alignatt_token_argmax_frontier_gate = bool(
        alignatt_token_argmax_frontier_gate
    )
    config.translation_alignatt_token_argmax_frontier_patience_tokens = int(
        alignatt_token_argmax_frontier_patience_tokens
    )
    config.translation_alignatt_source_lcp_stability = bool(
        alignatt_source_lcp_stability
    )
    config.translation_alignatt_source_lcp_append_slack_units = int(
        alignatt_source_lcp_append_slack_units
    )
    if (
        config.translation_alignatt_source_lcp_append_slack_units > 0
        and not config.translation_alignatt_source_lcp_stability
    ):
        raise ValueError(
            "alignatt_source_lcp_append_slack_units requires "
            "alignatt_source_lcp_stability=True."
        )
    if mt_vllm_gpu_memory_utilization is not None:
        config.mt_vllm_gpu_memory_utilization = float(mt_vllm_gpu_memory_utilization)
    if gemma_vllm_gpu_memory_utilization is not None:
        config.gemma_vllm_gpu_memory_utilization = float(gemma_vllm_gpu_memory_utilization)
    if asr_gpu_memory_utilization is not None:
        config.asr_gpu_memory_utilization = float(asr_gpu_memory_utilization)
    config.mt_vllm_enforce_eager = bool(mt_vllm_enforce_eager)
    config.mt_vllm_enable_speculative_decoding = bool(mt_vllm_enable_speculative_decoding)
    config.mt_vllm_speculative_assistant_model = mt_vllm_speculative_assistant_model
    config.mt_vllm_num_speculative_tokens = int(mt_vllm_num_speculative_tokens)
    validate = getattr(config, "_validate", None)
    if callable(validate):
        validate()
    return config


def policy_points(args: argparse.Namespace) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    lcp_tag = ""
    if bool(getattr(args, "alignatt_source_lcp_stability", False)):
        lcp_tag = "_lcp"
    append_slack = int(getattr(args, "alignatt_source_lcp_append_slack_units", 0))
    if append_slack > 0:
        lcp_tag += f"_aslack{append_slack}"
    if not args.skip_alignatt:
        if (
            args.alignatt_border_margins is None
            and args.alignatt_top_k_heads is None
            and args.alignatt_acceptance_variants is None
            and args.alignatt_min_source_masses is None
            and args.alignatt_unit_consensus_min_head_ratios is None
            and getattr(args, "alignatt_min_alignment_confidences", None) is None
            and getattr(args, "alignatt_online_normalizations", None) is None
            and getattr(args, "alignatt_source_bearing_min_source_masses", None) is None
            and getattr(args, "alignatt_source_bearing_hard_inaccessible_caps", None) is None
            and getattr(args, "alignatt_max_non_source_prompt_masses", None) is None
            and getattr(args, "alignatt_min_accepted_accessible_source_masses", None) is None
            and args.alignatt_inaccessible_ms_values is None
            and getattr(args, "alignatt_min_accessible_source_units_modes", None) is None
            and args.alignatt_frontier_min_inaccessible_masses is None
            and args.alignatt_source_regression_min_inaccessible_masses is None
        ):
            preset = get_runtime_preset(args.base_preset)
            source_regression_action_tag = (
                ""
                if args.alignatt_source_regression_action == "stop"
                else f"_sr{args.alignatt_source_regression_action}"
            )
            source_frontier_action_tag = (
                ""
                if args.alignatt_source_frontier_action == "stop"
                else f"_sf{args.alignatt_source_frontier_action}"
            )
            points.append(
                {
                    "tag": (
                        f"alignatt{source_frontier_action_tag}"
                        f"{source_regression_action_tag}{lcp_tag}"
                    ),
                    "policy": "alignatt",
                    "cutoff_units": 0,
                    "alignatt_border_margin": int(
                        preset.translation_alignatt_border_margin
                    ),
                    "alignatt_top_k_heads": int(
                        preset.translation_alignatt_top_k_heads
                    ),
                    "alignatt_acceptance_variant": str(
                        getattr(preset, "translation_alignatt_acceptance_variant", "token")
                    ),
                    "alignatt_min_source_mass": float(
                        preset.translation_alignatt_min_source_mass
                    ),
                    "alignatt_unit_consensus_min_head_ratio": float(
                        args.alignatt_unit_consensus_min_head_ratio
                    ),
                    "alignatt_min_alignment_confidence": float(
                        getattr(args, "alignatt_min_alignment_confidence", 0.0)
                    ),
                    "alignatt_online_normalization": str(
                        getattr(args, "alignatt_online_normalization", "zscore")
                    ),
                    "alignatt_source_bearing_min_source_mass": float(
                        getattr(args, "alignatt_source_bearing_min_source_mass", 0.005)
                    ),
                    "alignatt_source_bearing_hard_inaccessible_cap": float(
                        getattr(
                            args,
                            "alignatt_source_bearing_hard_inaccessible_cap",
                            1.0,
                        )
                    ),
                    "alignatt_max_non_source_prompt_mass": float(
                        getattr(args, "alignatt_max_non_source_prompt_mass", 1.0)
                    ),
                    "alignatt_min_accepted_accessible_source_mass": float(
                        getattr(
                            args,
                            "alignatt_min_accepted_accessible_source_mass",
                            0.0,
                        )
                    ),
                    "alignatt_inaccessible_ms": float(args.alignatt_inaccessible_ms),
                    "alignatt_frontier_min_inaccessible_mass": float(
                        args.alignatt_frontier_min_inaccessible_mass
                    ),
                    "alignatt_source_frontier_action": (
                        args.alignatt_source_frontier_action
                    ),
                    "alignatt_min_accessible_source_units_mode": str(
                        getattr(args, "alignatt_min_accessible_source_units_mode", "block")
                    ),
                    "alignatt_source_regression_min_inaccessible_mass": (
                        args.alignatt_source_regression_min_inaccessible_mass
                    ),
                    "alignatt_source_regression_action": (
                        args.alignatt_source_regression_action
                    ),
                }
            )
        else:
            preset = get_runtime_preset(args.base_preset)
            border_margins = (
                args.alignatt_border_margins
                if args.alignatt_border_margins is not None
                else [preset.translation_alignatt_border_margin]
            )
            top_k_heads = (
                args.alignatt_top_k_heads
                if args.alignatt_top_k_heads is not None
                else [preset.translation_alignatt_top_k_heads]
            )
            acceptance_variants = (
                args.alignatt_acceptance_variants
                if args.alignatt_acceptance_variants is not None
                else [getattr(preset, "translation_alignatt_acceptance_variant", "token")]
            )
            min_source_masses = (
                args.alignatt_min_source_masses
                if args.alignatt_min_source_masses is not None
                else [preset.translation_alignatt_min_source_mass]
            )
            consensus_ratios = (
                args.alignatt_unit_consensus_min_head_ratios
                if args.alignatt_unit_consensus_min_head_ratios is not None
                else [args.alignatt_unit_consensus_min_head_ratio]
            )
            online_normalizations = (
                getattr(args, "alignatt_online_normalizations", None)
                if getattr(args, "alignatt_online_normalizations", None) is not None
                else [
                    getattr(
                        preset,
                        "translation_alignatt_online_normalization",
                        getattr(args, "alignatt_online_normalization", "zscore"),
                    )
                ]
            )
            source_bearing_min_source_masses = (
                getattr(args, "alignatt_source_bearing_min_source_masses", None)
                if getattr(args, "alignatt_source_bearing_min_source_masses", None)
                is not None
                else [getattr(args, "alignatt_source_bearing_min_source_mass", 0.005)]
            )
            source_bearing_hard_inaccessible_caps = (
                getattr(args, "alignatt_source_bearing_hard_inaccessible_caps", None)
                if getattr(args, "alignatt_source_bearing_hard_inaccessible_caps", None)
                is not None
                else [
                    getattr(
                        args,
                        "alignatt_source_bearing_hard_inaccessible_cap",
                        1.0,
                    )
                ]
            )
            non_source_prompt_masses = (
                getattr(args, "alignatt_max_non_source_prompt_masses", None)
                if getattr(args, "alignatt_max_non_source_prompt_masses", None)
                is not None
                else [getattr(args, "alignatt_max_non_source_prompt_mass", 1.0)]
            )
            accepted_prefix_source_masses = (
                getattr(args, "alignatt_min_accepted_accessible_source_masses", None)
                if getattr(args, "alignatt_min_accepted_accessible_source_masses", None)
                is not None
                else [
                    getattr(
                        args,
                        "alignatt_min_accepted_accessible_source_mass",
                        0.0,
                    )
                ]
            )
            inaccessible_ms_values = (
                args.alignatt_inaccessible_ms_values
                if args.alignatt_inaccessible_ms_values is not None
                else [args.alignatt_inaccessible_ms]
            )
            frontier_min_inaccessible_masses = (
                args.alignatt_frontier_min_inaccessible_masses
                if args.alignatt_frontier_min_inaccessible_masses is not None
                else [args.alignatt_frontier_min_inaccessible_mass]
            )
            source_unit_modes = (
                getattr(args, "alignatt_min_accessible_source_units_modes", None)
                if getattr(args, "alignatt_min_accessible_source_units_modes", None)
                is not None
                else [getattr(args, "alignatt_min_accessible_source_units_mode", "block")]
            )
            min_inaccessible_masses = (
                args.alignatt_source_regression_min_inaccessible_masses
                if args.alignatt_source_regression_min_inaccessible_masses is not None
                else [args.alignatt_source_regression_min_inaccessible_mass]
            )
            border_margin_values = sorted({int(value) for value in border_margins})
            top_k_values = sorted({int(value) for value in top_k_heads})
            variant_values = sorted({str(value) for value in acceptance_variants})
            min_source_mass_values = sorted(
                {float(value) for value in min_source_masses}
            )
            consensus_ratio_values = sorted(
                {float(value) for value in consensus_ratios}
            )
            normalization_values = sorted(
                {str(value) for value in online_normalizations}
            )
            source_bearing_mass_values = sorted(
                {float(value) for value in source_bearing_min_source_masses}
            )
            source_bearing_cap_values = sorted(
                {float(value) for value in source_bearing_hard_inaccessible_caps}
            )
            non_source_prompt_mass_values = sorted(
                {float(value) for value in non_source_prompt_masses}
            )
            accepted_prefix_source_mass_values = sorted(
                {float(value) for value in accepted_prefix_source_masses}
            )
            inaccessible_ms_value_list = sorted(
                {float(value) for value in inaccessible_ms_values}
            )
            frontier_mass_values = sorted(
                {float(value) for value in frontier_min_inaccessible_masses}
            )
            source_unit_mode_values = sorted({str(value) for value in source_unit_modes})
            min_inaccessible_mass_values = sorted(
                {float(value) for value in min_inaccessible_masses}
            )
            for border_margin in border_margin_values:
                for top_k in top_k_values:
                    for variant in variant_values:
                        variant_source_bearing_mass_values = (
                            source_bearing_mass_values
                            if variant == "unit_mass_source_bearing"
                            else [
                                float(
                                    getattr(
                                        args,
                                        "alignatt_source_bearing_min_source_mass",
                                        0.005,
                                    )
                                )
                            ]
                        )
                        variant_source_bearing_cap_values = (
                            source_bearing_cap_values
                            if variant == "unit_mass_source_bearing"
                            else [
                                float(
                                    getattr(
                                        args,
                                        "alignatt_source_bearing_hard_inaccessible_cap",
                                        1.0,
                                    )
                                )
                            ]
                        )
                        for min_source_mass in min_source_mass_values:
                            for consensus_ratio in consensus_ratio_values:
                                for online_normalization in normalization_values:
                                    for source_bearing_mass in variant_source_bearing_mass_values:
                                        for source_bearing_cap in variant_source_bearing_cap_values:
                                            for non_source_prompt_mass in non_source_prompt_mass_values:
                                                for accepted_prefix_source_mass in accepted_prefix_source_mass_values:
                                                    for inaccessible_ms in inaccessible_ms_value_list:
                                                        for frontier_mass in frontier_mass_values:
                                                            for source_unit_mode in source_unit_mode_values:
                                                                for min_inaccessible_mass in min_inaccessible_mass_values:
                                                                    variant_tag = (
                                                                        ""
                                                                        if variant == "token"
                                                                        else f"_{variant}"
                                                                    )
                                                                    consensus_tag = (
                                                                        f"_consensus{_float_tag(consensus_ratio)}"
                                                                        if variant == "unit_consensus"
                                                                        else ""
                                                                    )
                                                                    norm_tag = (
                                                                        ""
                                                                        if online_normalization == "zscore"
                                                                        else f"_{online_normalization}"
                                                                    )
                                                                    source_bearing_tag = (
                                                                        f"_sb{_float_tag(source_bearing_mass)}"
                                                                        if variant == "unit_mass_source_bearing"
                                                                        else ""
                                                                    )
                                                                    source_bearing_cap_tag = (
                                                                        f"_sbcap{_float_tag(source_bearing_cap)}"
                                                                        if variant == "unit_mass_source_bearing"
                                                                        and source_bearing_cap < 1.0
                                                                        else ""
                                                                    )
                                                                    non_source_prompt_tag = (
                                                                        f"_nsp{_float_tag(non_source_prompt_mass)}"
                                                                        if non_source_prompt_mass < 1.0
                                                                        else ""
                                                                    )
                                                                    accepted_prefix_mass_tag = (
                                                                        f"_apm{_float_tag(accepted_prefix_source_mass)}"
                                                                        if accepted_prefix_source_mass > 0.0
                                                                        else ""
                                                                    )
                                                                    mass_tag = f"_m{_float_tag(min_source_mass)}"
                                                                    if min_source_mass == 0.0:
                                                                        mass_tag = ""
                                                                    inacc_tag = (
                                                                        f"_inacc{_float_tag(inaccessible_ms)}"
                                                                        if inaccessible_ms > 0.0
                                                                        else ""
                                                                    )
                                                                    frontier_tag = (
                                                                        f"_frontier{_float_tag(frontier_mass)}"
                                                                        if frontier_mass > 0.0
                                                                        else ""
                                                                    )
                                                                    source_unit_mode_tag = (
                                                                        "_srcunitcap"
                                                                        if source_unit_mode == "target_unit_cap"
                                                                        else ""
                                                                    )
                                                                    if source_unit_mode not in {
                                                                        "block",
                                                                        "target_unit_cap",
                                                                    }:
                                                                        source_unit_mode_tag = (
                                                                            f"_srcunit{source_unit_mode}"
                                                                        )
                                                                    future_tag = (
                                                                        f"_future{_float_tag(min_inaccessible_mass)}"
                                                                        if min_inaccessible_mass > 0.0
                                                                        else ""
                                                                    )
                                                                    source_regression_action_tag = (
                                                                        ""
                                                                        if args.alignatt_source_regression_action
                                                                        == "stop"
                                                                        else (
                                                                            f"_sr{args.alignatt_source_regression_action}"
                                                                        )
                                                                    )
                                                                    source_frontier_action_tag = (
                                                                        ""
                                                                        if args.alignatt_source_frontier_action
                                                                        == "stop"
                                                                        else (
                                                                            f"_sf{args.alignatt_source_frontier_action}"
                                                                        )
                                                                    )
                                                                    points.append(
                                                                        {
                                                                            "tag": (
                                                                                f"alignatt_b{border_margin}_top{top_k}"
                                                                                f"{variant_tag}{consensus_tag}{norm_tag}"
                                                                                f"{source_bearing_tag}{source_bearing_cap_tag}"
                                                                                f"{non_source_prompt_tag}"
                                                                                f"{accepted_prefix_mass_tag}"
                                                                                f"{mass_tag}{inacc_tag}"
                                                                                f"{frontier_tag}{source_frontier_action_tag}"
                                                                                f"{source_unit_mode_tag}"
                                                                                f"{future_tag}"
                                                                                f"{source_regression_action_tag}{lcp_tag}"
                                                                            ),
                                                                            "policy": "alignatt",
                                                                            "cutoff_units": 0,
                                                                            "alignatt_border_margin": int(border_margin),
                                                                            "alignatt_top_k_heads": int(top_k),
                                                                            "alignatt_acceptance_variant": variant,
                                                                            "alignatt_min_source_mass": float(min_source_mass),
                                                                            "alignatt_unit_consensus_min_head_ratio": float(
                                                                                consensus_ratio
                                                                            ),
                                                                            "alignatt_online_normalization": online_normalization,
                                                                            "alignatt_source_bearing_min_source_mass": float(
                                                                                source_bearing_mass
                                                                            ),
                                                                            "alignatt_source_bearing_hard_inaccessible_cap": float(
                                                                                source_bearing_cap
                                                                            ),
                                                                            "alignatt_max_non_source_prompt_mass": float(
                                                                                non_source_prompt_mass
                                                                            ),
                                                                            "alignatt_min_accepted_accessible_source_mass": float(
                                                                                accepted_prefix_source_mass
                                                                            ),
                                                                            "alignatt_inaccessible_ms": float(inaccessible_ms),
                                                                            "alignatt_frontier_min_inaccessible_mass": float(
                                                                                frontier_mass
                                                                            ),
                                                                            "alignatt_source_frontier_action": (
                                                                                args.alignatt_source_frontier_action
                                                                            ),
                                                                            "alignatt_min_accessible_source_units_mode": (
                                                                                source_unit_mode
                                                                            ),
                                                                            "alignatt_source_regression_min_inaccessible_mass": float(
                                                                                min_inaccessible_mass
                                                                            ),
                                                                            "alignatt_source_regression_action": (
                                                                                args.alignatt_source_regression_action
                                                                            ),
                                                                        }
                                                                    )
    confidence_axis = (
        args.alignatt_min_alignment_confidences
        if getattr(args, "alignatt_min_alignment_confidences", None) is not None
        else [getattr(args, "alignatt_min_alignment_confidence", 0.0)]
    )
    confidence_values = sorted({float(value) for value in confidence_axis})
    expanded_points: list[dict[str, Any]] = []
    for point in points:
        if (
            point.get("policy") == "alignatt"
            and point.get("alignatt_acceptance_variant") == "unit_conf"
        ):
            for confidence in confidence_values:
                conf_point = dict(point)
                conf_point["alignatt_min_alignment_confidence"] = float(confidence)
                if confidence > 0.0:
                    conf_point["tag"] = f"{point['tag']}_conf{_float_tag(confidence)}"
                expanded_points.append(conf_point)
        else:
            if point.get("policy") == "alignatt":
                point.setdefault(
                    "alignatt_min_alignment_confidence",
                    float(getattr(args, "alignatt_min_alignment_confidence", 0.0)),
                )
            expanded_points.append(point)
    points = expanded_points
    if not args.skip_cutoffs:
        for cutoff in sorted({max(0, int(value)) for value in args.cutoffs}):
            points.append(
                {
                    "tag": f"cut_last_{cutoff}",
                    "policy": "cut_last_target_units",
                    "cutoff_units": int(cutoff),
                    "alignatt_border_margin": None,
                    "alignatt_top_k_heads": None,
                    "alignatt_acceptance_variant": None,
                    "alignatt_min_source_mass": None,
                    "alignatt_unit_consensus_min_head_ratio": None,
                    "alignatt_min_alignment_confidence": None,
                    "alignatt_online_normalization": None,
                    "alignatt_source_bearing_min_source_mass": None,
                    "alignatt_source_bearing_hard_inaccessible_cap": None,
                    "alignatt_max_non_source_prompt_mass": None,
                    "alignatt_min_accepted_accessible_source_mass": None,
                    "alignatt_min_accessible_source_units_mode": None,
                }
            )
    return points


def policy_point_record(
    *,
    args: argparse.Namespace,
    point: dict[str, Any],
    output_dir: Path,
    input_count: int,
    model_load_ms: float | None,
    resumed: bool = False,
) -> dict[str, Any]:
    default_unit_consensus_ratio = getattr(
        args, "alignatt_unit_consensus_min_head_ratio", 0.60
    )
    default_online_normalization = getattr(
        args, "alignatt_online_normalization", "zscore"
    )
    default_source_bearing_min_source_mass = getattr(
        args, "alignatt_source_bearing_min_source_mass", 0.005
    )
    default_source_bearing_hard_inaccessible_cap = getattr(
        args, "alignatt_source_bearing_hard_inaccessible_cap", 1.0
    )
    default_max_non_source_prompt_mass = getattr(
        args, "alignatt_max_non_source_prompt_mass", 1.0
    )
    default_min_accepted_accessible_source_mass = getattr(
        args, "alignatt_min_accepted_accessible_source_mass", 0.0
    )
    record = {
        "tag": str(point["tag"]),
        "policy": str(point["policy"]),
        "mt_backend_name": args.mt_backend_name,
        "cutoff_units": int(point["cutoff_units"]),
        "alignatt_border_margin": point.get("alignatt_border_margin"),
        "alignatt_top_k_heads": point.get("alignatt_top_k_heads"),
        "alignatt_acceptance_variant": point.get("alignatt_acceptance_variant"),
        "alignatt_min_source_mass": point.get("alignatt_min_source_mass"),
        "alignatt_min_alignment_confidence": point.get(
            "alignatt_min_alignment_confidence",
            getattr(args, "alignatt_min_alignment_confidence", 0.0),
        ),
        "alignatt_unit_consensus_min_head_ratio": point.get(
            "alignatt_unit_consensus_min_head_ratio",
            default_unit_consensus_ratio,
        ),
        "alignatt_online_normalization": point.get(
            "alignatt_online_normalization",
            default_online_normalization,
        ),
        "alignatt_source_bearing_min_source_mass": point.get(
            "alignatt_source_bearing_min_source_mass",
            default_source_bearing_min_source_mass,
        ),
        "alignatt_source_bearing_hard_inaccessible_cap": point.get(
            "alignatt_source_bearing_hard_inaccessible_cap",
            default_source_bearing_hard_inaccessible_cap,
        ),
        "alignatt_max_non_source_prompt_mass": point.get(
            "alignatt_max_non_source_prompt_mass",
            default_max_non_source_prompt_mass,
        ),
        "alignatt_min_accepted_accessible_source_mass": point.get(
            "alignatt_min_accepted_accessible_source_mass",
            default_min_accepted_accessible_source_mass,
        ),
        "alignatt_accepted_accessible_source_mass_recent_units": (
            getattr(args, "alignatt_accepted_accessible_source_mass_recent_units", 2)
        ),
        "alignatt_inaccessible_ms": point.get(
            "alignatt_inaccessible_ms",
            args.alignatt_inaccessible_ms,
        ),
        "alignatt_frontier_min_inaccessible_mass": point.get(
            "alignatt_frontier_min_inaccessible_mass",
            args.alignatt_frontier_min_inaccessible_mass,
        ),
        "alignatt_source_frontier_action": point.get(
            "alignatt_source_frontier_action",
            args.alignatt_source_frontier_action,
        ),
        "alignatt_max_inaccessible_source_mass": args.alignatt_max_inaccessible_source_mass,
        "alignatt_min_accessible_inaccessible_margin": args.alignatt_min_accessible_inaccessible_margin,
        "alignatt_min_accessible_source_units": args.alignatt_min_accessible_source_units,
        "alignatt_min_accessible_source_units_mode": point.get(
            "alignatt_min_accessible_source_units_mode",
            args.alignatt_min_accessible_source_units_mode,
        ),
        "alignatt_max_source_regression": args.alignatt_max_source_regression,
        "alignatt_source_regression_min_inaccessible_mass": point.get(
            "alignatt_source_regression_min_inaccessible_mass",
            args.alignatt_source_regression_min_inaccessible_mass,
        ),
        "alignatt_source_regression_recent_tokens": args.alignatt_source_regression_recent_tokens,
        "alignatt_source_regression_reference_mode": args.alignatt_source_regression_reference_mode,
        "alignatt_source_regression_activation_mode": args.alignatt_source_regression_activation_mode,
        "alignatt_source_regression_activation_slack_tokens": args.alignatt_source_regression_activation_slack_tokens,
        "alignatt_source_regression_patience_tokens": args.alignatt_source_regression_patience_tokens,
        "alignatt_source_regression_action": point.get(
            "alignatt_source_regression_action",
            args.alignatt_source_regression_action,
        ),
        "alignatt_token_argmax_frontier_gate": args.alignatt_token_argmax_frontier_gate,
        "alignatt_token_argmax_frontier_patience_tokens": args.alignatt_token_argmax_frontier_patience_tokens,
        "alignatt_source_lcp_stability": args.alignatt_source_lcp_stability,
        "alignatt_source_lcp_append_slack_units": args.alignatt_source_lcp_append_slack_units,
        "output_dir": str(output_dir),
        "num_inputs": int(input_count),
        "model_load_ms": model_load_ms,
        "asr_gpu_memory_utilization": args.asr_gpu_memory_utilization,
        "mt_vllm_enforce_eager": args.mt_vllm_enforce_eager,
        "mt_vllm_enable_speculative_decoding": args.mt_vllm_enable_speculative_decoding,
        "mt_vllm_speculative_assistant_model": args.mt_vllm_speculative_assistant_model,
        "mt_vllm_num_speculative_tokens": args.mt_vllm_num_speculative_tokens,
    }
    if resumed:
        record["resumed"] = True
    return record


def main() -> None:
    args = parse_args()
    input_paths = resolve_input_paths(inputs=args.inputs, input_dir=args.input_dir)
    args.output_root.mkdir(parents=True, exist_ok=True)
    points = policy_points(args)
    if not points:
        raise ValueError("No policy points selected for the sweep.")

    first_point = points[0]
    first_policy = str(first_point["policy"])
    first_cutoff = int(first_point["cutoff_units"])
    load_top_k_heads = None
    alignatt_top_k_values = [
        int(point["alignatt_top_k_heads"])
        for point in points
        if point.get("policy") == "alignatt"
        and point.get("alignatt_top_k_heads") is not None
    ]
    if alignatt_top_k_values:
        load_top_k_heads = max(alignatt_top_k_values)
    load_config = build_processor_config(
        base_preset_name=args.base_preset,
        source=args.source,
        target=args.target,
        chunk_ms=args.chunk_ms,
        mt_backend_name=args.mt_backend_name,
        policy=first_policy,
        cutoff_units=first_cutoff,
        alignatt_border_margin=first_point.get("alignatt_border_margin"),
        alignatt_top_k_heads=load_top_k_heads,
        alignatt_acceptance_variant=first_point.get("alignatt_acceptance_variant"),
        alignatt_min_source_mass=first_point.get("alignatt_min_source_mass"),
        alignatt_unit_consensus_min_head_ratio=point_value(
            first_point,
            "alignatt_unit_consensus_min_head_ratio",
            args.alignatt_unit_consensus_min_head_ratio,
        ),
        alignatt_online_normalization=point_value(
            first_point,
            "alignatt_online_normalization",
            args.alignatt_online_normalization,
        ),
        alignatt_source_bearing_min_source_mass=point_value(
            first_point,
            "alignatt_source_bearing_min_source_mass",
            args.alignatt_source_bearing_min_source_mass,
        ),
        alignatt_source_bearing_hard_inaccessible_cap=point_value(
            first_point,
            "alignatt_source_bearing_hard_inaccessible_cap",
            args.alignatt_source_bearing_hard_inaccessible_cap,
        ),
        alignatt_inaccessible_ms=point_value(
            first_point,
            "alignatt_inaccessible_ms",
            args.alignatt_inaccessible_ms,
        ),
        alignatt_frontier_min_inaccessible_mass=point_value(
            first_point,
            "alignatt_frontier_min_inaccessible_mass",
            args.alignatt_frontier_min_inaccessible_mass,
        ),
        alignatt_source_frontier_action=point_value(
            first_point,
            "alignatt_source_frontier_action",
            args.alignatt_source_frontier_action,
        ),
        alignatt_max_inaccessible_source_mass=args.alignatt_max_inaccessible_source_mass,
        alignatt_max_non_source_prompt_mass=point_value(
            first_point,
            "alignatt_max_non_source_prompt_mass",
            args.alignatt_max_non_source_prompt_mass,
        ),
        alignatt_min_accepted_accessible_source_mass=point_value(
            first_point,
            "alignatt_min_accepted_accessible_source_mass",
            args.alignatt_min_accepted_accessible_source_mass,
        ),
        alignatt_accepted_accessible_source_mass_recent_units=(
            args.alignatt_accepted_accessible_source_mass_recent_units
        ),
        alignatt_min_accessible_inaccessible_margin=args.alignatt_min_accessible_inaccessible_margin,
        alignatt_min_accessible_source_units=args.alignatt_min_accessible_source_units,
        alignatt_min_accessible_source_units_mode=(
            first_point.get("alignatt_min_accessible_source_units_mode")
            or args.alignatt_min_accessible_source_units_mode
        ),
        alignatt_max_source_regression=args.alignatt_max_source_regression,
        alignatt_source_regression_min_inaccessible_mass=point_value(
            first_point,
            "alignatt_source_regression_min_inaccessible_mass",
            args.alignatt_source_regression_min_inaccessible_mass,
        ),
        alignatt_source_regression_recent_tokens=args.alignatt_source_regression_recent_tokens,
        alignatt_source_regression_reference_mode=args.alignatt_source_regression_reference_mode,
        alignatt_source_regression_activation_mode=args.alignatt_source_regression_activation_mode,
        alignatt_source_regression_activation_slack_tokens=args.alignatt_source_regression_activation_slack_tokens,
        alignatt_source_regression_patience_tokens=args.alignatt_source_regression_patience_tokens,
        alignatt_source_regression_action=point_value(
            first_point,
            "alignatt_source_regression_action",
            args.alignatt_source_regression_action,
        ),
        alignatt_token_argmax_frontier_gate=args.alignatt_token_argmax_frontier_gate,
        alignatt_token_argmax_frontier_patience_tokens=args.alignatt_token_argmax_frontier_patience_tokens,
        alignatt_source_lcp_stability=args.alignatt_source_lcp_stability,
        alignatt_source_lcp_append_slack_units=args.alignatt_source_lcp_append_slack_units,
        asr_gpu_memory_utilization=args.asr_gpu_memory_utilization,
        mt_vllm_gpu_memory_utilization=args.mt_vllm_gpu_memory_utilization,
        gemma_vllm_gpu_memory_utilization=args.gemma_vllm_gpu_memory_utilization,
        mt_vllm_enforce_eager=args.mt_vllm_enforce_eager,
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
    for point in points:
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
            mt_backend_name=args.mt_backend_name,
            policy=policy,
            cutoff_units=cutoff_units,
            alignatt_border_margin=point.get("alignatt_border_margin"),
            alignatt_top_k_heads=point.get("alignatt_top_k_heads"),
            alignatt_acceptance_variant=point.get("alignatt_acceptance_variant"),
            alignatt_min_source_mass=point.get("alignatt_min_source_mass"),
            alignatt_unit_consensus_min_head_ratio=point_value(
                point,
                "alignatt_unit_consensus_min_head_ratio",
                args.alignatt_unit_consensus_min_head_ratio,
            ),
            alignatt_min_alignment_confidence=point_value(
                point,
                "alignatt_min_alignment_confidence",
                getattr(args, "alignatt_min_alignment_confidence", 0.0),
            ),
            alignatt_online_normalization=point_value(
                point,
                "alignatt_online_normalization",
                args.alignatt_online_normalization,
            ),
            alignatt_source_bearing_min_source_mass=point_value(
                point,
                "alignatt_source_bearing_min_source_mass",
                args.alignatt_source_bearing_min_source_mass,
            ),
            alignatt_source_bearing_hard_inaccessible_cap=point_value(
                point,
                "alignatt_source_bearing_hard_inaccessible_cap",
                args.alignatt_source_bearing_hard_inaccessible_cap,
            ),
            alignatt_inaccessible_ms=point_value(
                point,
                "alignatt_inaccessible_ms",
                args.alignatt_inaccessible_ms,
            ),
            alignatt_frontier_min_inaccessible_mass=point_value(
                point,
                "alignatt_frontier_min_inaccessible_mass",
                args.alignatt_frontier_min_inaccessible_mass,
            ),
            alignatt_source_frontier_action=point_value(
                point,
                "alignatt_source_frontier_action",
                args.alignatt_source_frontier_action,
            ),
            alignatt_max_inaccessible_source_mass=args.alignatt_max_inaccessible_source_mass,
            alignatt_max_non_source_prompt_mass=point_value(
                point,
                "alignatt_max_non_source_prompt_mass",
                args.alignatt_max_non_source_prompt_mass,
            ),
            alignatt_min_accepted_accessible_source_mass=point_value(
                point,
                "alignatt_min_accepted_accessible_source_mass",
                args.alignatt_min_accepted_accessible_source_mass,
            ),
            alignatt_accepted_accessible_source_mass_recent_units=(
                args.alignatt_accepted_accessible_source_mass_recent_units
            ),
            alignatt_min_accessible_inaccessible_margin=args.alignatt_min_accessible_inaccessible_margin,
            alignatt_min_accessible_source_units=args.alignatt_min_accessible_source_units,
            alignatt_min_accessible_source_units_mode=(
                point.get("alignatt_min_accessible_source_units_mode")
                or args.alignatt_min_accessible_source_units_mode
            ),
            alignatt_max_source_regression=args.alignatt_max_source_regression,
            alignatt_source_regression_min_inaccessible_mass=point_value(
                point,
                "alignatt_source_regression_min_inaccessible_mass",
                args.alignatt_source_regression_min_inaccessible_mass,
            ),
            alignatt_source_regression_recent_tokens=args.alignatt_source_regression_recent_tokens,
            alignatt_source_regression_reference_mode=args.alignatt_source_regression_reference_mode,
            alignatt_source_regression_activation_mode=args.alignatt_source_regression_activation_mode,
            alignatt_source_regression_activation_slack_tokens=args.alignatt_source_regression_activation_slack_tokens,
            alignatt_source_regression_patience_tokens=args.alignatt_source_regression_patience_tokens,
            alignatt_source_regression_action=point_value(
                point,
                "alignatt_source_regression_action",
                args.alignatt_source_regression_action,
            ),
            alignatt_token_argmax_frontier_gate=args.alignatt_token_argmax_frontier_gate,
            alignatt_token_argmax_frontier_patience_tokens=args.alignatt_token_argmax_frontier_patience_tokens,
            alignatt_source_lcp_stability=args.alignatt_source_lcp_stability,
            alignatt_source_lcp_append_slack_units=args.alignatt_source_lcp_append_slack_units,
            asr_gpu_memory_utilization=args.asr_gpu_memory_utilization,
            mt_vllm_gpu_memory_utilization=args.mt_vllm_gpu_memory_utilization,
            gemma_vllm_gpu_memory_utilization=args.gemma_vllm_gpu_memory_utilization,
            mt_vllm_enforce_eager=args.mt_vllm_enforce_eager,
            mt_vllm_enable_speculative_decoding=args.mt_vllm_enable_speculative_decoding,
            mt_vllm_speculative_assistant_model=args.mt_vllm_speculative_assistant_model,
            mt_vllm_num_speculative_tokens=args.mt_vllm_num_speculative_tokens,
        )
        if args.resume and hypothesis_path.exists():
            print(f"\n>>> {tag}: existing hypothesis found, reusing {output_dir}", flush=True)
            rows.append(
                policy_point_record(
                    args=args,
                    point=point,
                    output_dir=output_dir,
                    input_count=len(input_paths),
                    model_load_ms=None,
                    resumed=True,
                )
            )
            continue

        print(
            f"\n>>> {tag}: policy={policy} cutoff={cutoff_units} "
            f"border={point.get('alignatt_border_margin')} "
            f"top_k={point.get('alignatt_top_k_heads')} "
            f"variant={point.get('alignatt_acceptance_variant')} "
            f"min_mass={point.get('alignatt_min_source_mass')} "
            f"unit_consensus_ratio={point.get('alignatt_unit_consensus_min_head_ratio')} "
            f"min_alignment_confidence={point.get('alignatt_min_alignment_confidence')} "
            f"normalization={point.get('alignatt_online_normalization')} "
            f"source_bearing_mass={point.get('alignatt_source_bearing_min_source_mass')} "
            f"source_bearing_cap={point.get('alignatt_source_bearing_hard_inaccessible_cap')} "
            f"non_source_cap={point.get('alignatt_max_non_source_prompt_mass')} "
            f"inacc_ms={point.get('alignatt_inaccessible_ms', args.alignatt_inaccessible_ms)} "
            f"frontier_mass={point.get('alignatt_frontier_min_inaccessible_mass', args.alignatt_frontier_min_inaccessible_mass)} "
            f"source_frontier_action={point.get('alignatt_source_frontier_action', args.alignatt_source_frontier_action)} "
            f"source_unit_mode={point.get('alignatt_min_accessible_source_units_mode', args.alignatt_min_accessible_source_units_mode)} "
            f"regression_future_mass={point.get('alignatt_source_regression_min_inaccessible_mass')} "
            f"lcp={args.alignatt_source_lcp_stability} "
            f"lcp_slack={args.alignatt_source_lcp_append_slack_units} "
            f"-> {output_dir}",
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
            policy_point_record(
                args=args,
                point=point,
                output_dir=output_dir,
                input_count=len(input_paths),
                model_load_ms=result.get("model_load_ms"),
            )
        )

    summary_path = args.output_root / "policy_points.json"
    summary_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWrote {summary_path}")
    print("Score each output with:")
    print(
        "  .venv-evaluation/bin/alignatt-eval "
        "--output-dir <output_dir>"
    )


if __name__ == "__main__":
    main()
