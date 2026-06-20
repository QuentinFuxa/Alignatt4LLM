"""Single-prompt and curated-set MT backend probe harness.

The harness runs each prompt inside an isolated child subprocess so repeated
prompt probes do not pollute the live vLLM allocator state.

Single-prompt shape (backward compatible):

    alignatt-mt-parity \\
        --source-text "Hi I'm Si Yuan from Fudan University." \\
        --output tmp/mt_parity_smoke.json \\
        --is-final

Curated prompt set:

    alignatt-mt-parity \\
        --prompt-set data/smoke/mt_parity_set.json \\
        --output tmp/mt_parity_curated.json

Each entry of the JSON prompt-set file is a mapping like:

    {
      "name": "partial_no_prefill",
      "source_text": "Hi I'm Si Yuan from Fudan University and I",
      "is_final": false,
      "accepted_target_prefill": "",
      "min_source_mass": 0.0,
      "current_audio_ms": 10000.0,
      "inaccessible_ms": 0.0,
      "source_history": [],
      "translation_history": []
    }

Fields other than ``name`` and ``source_text`` default to the top-level CLI
defaults. ``min_source_mass`` is applied to
``CascadeRuntimeConfig.translation_alignatt_min_source_mass`` for that
prompt only.

``gemma_vllm_alignatt`` is the stable Gemma baseline. ``milmmt_vllm_alignatt``
is the active MiLMMT improvement route.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from alignatt4llm.mt.base import MTBackendResult, build_mt_backend
from alignatt4llm.runtime import (
    CascadeRuntimeConfig,
    LANGUAGE_CODE_TO_NAME,
    VALID_MT_BACKEND_NAMES,
    mt_model_name_for_backend,
)
from alignatt4llm.source_frontier import build_source_accessibility_frontier
from alignatt4llm.translation_variants import RenderedTranslationPrompt, TRANSLATION_VARIANTS


_BACKEND_ALIASES = {
    "vllm": "gemma_vllm_alignatt",
    "gemma": "gemma_vllm_alignatt",
    "gemma_vllm_alignatt": "gemma_vllm_alignatt",
    "milmmt": "milmmt_vllm_alignatt",
    "milmmt_vllm_alignatt": "milmmt_vllm_alignatt",
}


@dataclass
class PromptSpec:
    name: str
    source_text: str
    is_final: bool = False
    accepted_target_prefill: str = ""
    min_source_mass: float = 0.0
    current_audio_ms: float = 10_000.0
    inaccessible_ms: float = 0.0
    source_history: list[str] = field(default_factory=list)
    translation_history: list[str] = field(default_factory=list)

    @staticmethod
    def from_dict(raw: dict[str, Any], *, index: int) -> "PromptSpec":
        if "source_text" not in raw:
            raise ValueError(f"Prompt spec #{index} is missing required 'source_text'")
        return PromptSpec(
            name=str(raw.get("name", f"prompt_{index}")),
            source_text=str(raw["source_text"]),
            is_final=bool(raw.get("is_final", False)),
            accepted_target_prefill=str(raw.get("accepted_target_prefill", "")),
            min_source_mass=float(raw.get("min_source_mass", 0.0)),
            current_audio_ms=float(raw.get("current_audio_ms", 10_000.0)),
            inaccessible_ms=float(raw.get("inaccessible_ms", 0.0)),
            source_history=list(raw.get("source_history", []) or []),
            translation_history=list(raw.get("translation_history", []) or []),
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-text", default=None)
    parser.add_argument("--source-lang-code", default="en")
    parser.add_argument("--target-lang-code", default="de")
    parser.add_argument("--accepted-target-prefill", default="")
    parser.add_argument("--is-final", action="store_true")
    parser.add_argument("--current-audio-ms", type=float, default=10_000.0)
    parser.add_argument("--inaccessible-ms", type=float, default=0.0)
    parser.add_argument("--min-source-mass", type=float, default=0.0)
    parser.add_argument("--frontier-min-inaccessible-mass", type=float, default=0.0)
    parser.add_argument("--max-inaccessible-source-mass", type=float, default=1.0)
    parser.add_argument("--min-accessible-inaccessible-margin", type=float, default=-1.0)
    parser.add_argument("--source-history", nargs="*", default=[])
    parser.add_argument("--translation-history", nargs="*", default=[])
    parser.add_argument(
        "--prompt-set",
        default=None,
        help="Path to a JSON file listing prompt specs. Overrides the single-prompt flags.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["vllm"],
        help="Backend selection. Only the shipped alias 'vllm' is supported.",
    )
    parser.add_argument("--mt-vllm-enable-speculative-decoding", action="store_true")
    parser.add_argument("--mt-vllm-speculative-assistant-model", default=None)
    parser.add_argument(
        "--mt-vllm-num-speculative-tokens",
        "--mt-vllm-speculative-num-tokens",
        dest="mt_vllm_num_speculative_tokens",
        type=int,
        default=4,
    )
    parser.add_argument("--output", required=True)
    # Worker-mode private flags.
    parser.add_argument("--_worker-backend-name", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-output", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-prompt-set", default=None, help=argparse.SUPPRESS)
    return parser


def _resolve_backend_name(alias: str) -> str:
    name = _BACKEND_ALIASES.get(alias.lower())
    if name is None:
        raise ValueError(
            f"Unknown backend alias {alias!r}. Valid: {sorted(_BACKEND_ALIASES.keys())}"
        )
    if name not in VALID_MT_BACKEND_NAMES:
        raise ValueError(f"Unsupported mt_backend_name: {name!r}")
    return name


def _prompt_set_from_args(args: argparse.Namespace) -> list[PromptSpec]:
    if args.prompt_set:
        raw = json.loads(Path(args.prompt_set).read_text(encoding="utf-8"))
        if not isinstance(raw, list) or not raw:
            raise ValueError("--prompt-set JSON must be a non-empty list of prompt specs.")
        return [PromptSpec.from_dict(entry, index=i) for i, entry in enumerate(raw)]
    if not args.source_text:
        raise SystemExit("Either --source-text or --prompt-set must be provided.")
    return [
        PromptSpec(
            name="prompt_0",
            source_text=args.source_text,
            is_final=bool(args.is_final),
            accepted_target_prefill=args.accepted_target_prefill,
            min_source_mass=float(args.min_source_mass),
            current_audio_ms=float(args.current_audio_ms),
            inaccessible_ms=float(args.inaccessible_ms),
            source_history=list(args.source_history),
            translation_history=list(args.translation_history),
        )
    ]


def _summarize_result(result: MTBackendResult) -> dict[str, Any]:
    alignatt_md = result.alignatt_metadata or {}
    provenance = alignatt_md.get("provenance_per_draft_token") or []
    provenance_summary: dict[str, Any] | None
    if provenance:
        n = len(provenance)
        avg_accessible = sum(p.get("source_accessible", 0.0) for p in provenance) / n
        avg_inaccessible = sum(p.get("source_inaccessible", 0.0) for p in provenance) / n
        avg_non_source = sum(p.get("non_source_prompt", 0.0) for p in provenance) / n
        avg_suffix = sum(p.get("suffix", 0.0) for p in provenance) / n
        provenance_summary = {
            "per_token_count": n,
            "mean_source_accessible": avg_accessible,
            "mean_source_inaccessible": avg_inaccessible,
            "mean_non_source_prompt": avg_non_source,
            "mean_suffix": avg_suffix,
        }
    else:
        provenance_summary = None
    return {
        "draft_text": result.draft_text,
        "acceptance_text": result.acceptance_text,
        "generated_token_count": len(result.draft_generated_token_ids),
        "accepted_token_count": len(result.accepted_generated_token_ids),
        "draft_semantic_token_count": len(result.draft_token_ids),
        "accepted_semantic_token_count": len(result.accepted_token_ids),
        "num_cached_tokens": result.num_cached_tokens,
        "prompt_num_tokens": result.prompt_num_tokens,
        "stop_reason": result.stop_reason,
        "blocked_source_local_position": alignatt_md.get("blocked_source_local_position"),
        "blocked_source_unit_index": alignatt_md.get("blocked_source_unit_index"),
        "unsafe_reason": alignatt_md.get("unsafe_reason"),
        "alignatt_not_implemented": alignatt_md.get("alignatt_not_implemented", False),
        "provenance_summary": provenance_summary,
        "aligned_source_local_positions": alignatt_md.get("aligned_source_local_positions"),
        "observer_diagnostics": alignatt_md.get("observer_diagnostics"),
        "observer_raw_token_count": alignatt_md.get("observer_raw_token_count"),
        "observer_operating_token_count": alignatt_md.get("observer_operating_token_count"),
        "observer_debug": alignatt_md.get("observer_debug"),
        "timings_ms": result.timings_ms or {},
    }


def _build_rendered_prompt(
    spec: PromptSpec, runtime_config: CascadeRuntimeConfig
) -> tuple[RenderedTranslationPrompt, dict[str, Any]]:
    variant = TRANSLATION_VARIANTS[runtime_config.translation_variant_id]
    frontier = build_source_accessibility_frontier(
        spec.source_text.strip(),
        word_timestamps_ms=None,
        current_audio_ms=float(spec.current_audio_ms),
        inaccessible_ms=float(spec.inaccessible_ms),
        is_final=bool(spec.is_final),
    )
    rendered = variant.render_messages(
        source_lang=runtime_config.source_lang,
        target_lang=runtime_config.target_lang,
        text=spec.source_text.strip(),
        source_frontier=frontier,
        source_history=list(spec.source_history),
        translation_history=list(spec.translation_history),
        is_partial=not spec.is_final,
        assistant_prefill=spec.accepted_target_prefill,
    )
    prompt_metadata = {
        "name": spec.name,
        "source_text": spec.source_text.strip(),
        "source_lang": runtime_config.source_lang,
        "target_lang": runtime_config.target_lang,
        "is_partial": not spec.is_final,
        "assistant_prefill": spec.accepted_target_prefill,
        "min_source_mass": float(spec.min_source_mass),
        "variant_id": runtime_config.translation_variant_id,
        "accessible_unit_count": frontier.accessible_unit_count,
        "total_unit_count": len(frontier.units),
        "message_count": len(rendered.messages),
        "continue_final_message": rendered.continue_final_message,
    }
    return rendered, prompt_metadata


def _run_worker(args: argparse.Namespace) -> None:
    backend_name = _resolve_backend_name(args._worker_backend_name)
    source_lang = LANGUAGE_CODE_TO_NAME.get(args.source_lang_code, "English")
    target_lang = LANGUAGE_CODE_TO_NAME.get(args.target_lang_code, "German")
    runtime_config = CascadeRuntimeConfig(
        source_lang=source_lang,
        target_lang=target_lang,
        mt_backend_name=backend_name,
        translation_alignatt_frontier_min_inaccessible_mass=args.frontier_min_inaccessible_mass,
        translation_alignatt_max_inaccessible_source_mass=args.max_inaccessible_source_mass,
        translation_alignatt_min_accessible_inaccessible_margin=args.min_accessible_inaccessible_margin,
        mt_vllm_enable_speculative_decoding=args.mt_vllm_enable_speculative_decoding,
        mt_vllm_speculative_assistant_model=args.mt_vllm_speculative_assistant_model,
        mt_vllm_num_speculative_tokens=args.mt_vllm_num_speculative_tokens,
    )

    specs_raw = json.loads(Path(args._worker_prompt_set).read_text(encoding="utf-8"))
    specs = [PromptSpec.from_dict(entry, index=i) for i, entry in enumerate(specs_raw)]

    # The MT vLLM backend bakes max_decode_tokens into the observer buffers at
    # load() time; translation_alignatt_min_source_mass is read fresh each
    # translate(), so we override it per-prompt via apply_overrides.
    load_start = time.perf_counter()
    backend = build_mt_backend(
        model_name=mt_model_name_for_backend(backend_name),
        runtime_config=runtime_config,
    )
    backend.load()
    load_ms = (time.perf_counter() - load_start) * 1000.0

    per_prompt_results: list[dict[str, Any]] = []
    variant = TRANSLATION_VARIANTS[runtime_config.translation_variant_id]
    for spec in specs:
        runtime_config.apply_overrides(
            translation_alignatt_min_source_mass=float(spec.min_source_mass)
        )
        # Reset any per-request cache state before the next prompt so each
        # case is isolated the same way it would be across runtime sessions.
        backend.reset_caches()
        rendered, prompt_metadata = _build_rendered_prompt(spec, runtime_config)
        translate_start = time.perf_counter()
        result = backend.translate(
            rendered_prompt=rendered,
            variant=variant,
            is_partial=not spec.is_final,
        )
        translate_ms = (time.perf_counter() - translate_start) * 1000.0
        summary = _summarize_result(result)
        summary["translate_wallclock_ms"] = translate_ms
        per_prompt_results.append(
            {
                "prompt": prompt_metadata,
                "result": summary,
            }
        )

    out_path = Path(args._worker_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "backend_name": backend_name,
                "load_ms": load_ms,
                "prompts": per_prompt_results,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _run_worker_subprocess(
    backend_name: str,
    args: argparse.Namespace,
    prompt_set_path: Path,
    output_path: Path,
) -> None:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--source-lang-code",
        args.source_lang_code,
        "--target-lang-code",
        args.target_lang_code,
        "--backends",
        backend_name,
        "--output",
        str(output_path),  # consumed as the worker output path
        "--_worker-backend-name",
        backend_name,
        "--_worker-output",
        str(output_path),
        "--_worker-prompt-set",
        str(prompt_set_path),
        "--frontier-min-inaccessible-mass",
        str(args.frontier_min_inaccessible_mass),
        "--max-inaccessible-source-mass",
        str(args.max_inaccessible_source_mass),
        "--min-accessible-inaccessible-margin",
        str(args.min_accessible_inaccessible_margin),
    ]
    if args.mt_vllm_enable_speculative_decoding:
        cmd.append("--mt-vllm-enable-speculative-decoding")
    if args.mt_vllm_speculative_assistant_model:
        cmd.extend(
            [
                "--mt-vllm-speculative-assistant-model",
                str(args.mt_vllm_speculative_assistant_model),
            ]
        )
    cmd.extend(
        [
            "--mt-vllm-num-speculative-tokens",
            str(args.mt_vllm_num_speculative_tokens),
        ]
    )
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Worker process for backend {backend_name!r} failed with exit code "
            f"{completed.returncode}. See subprocess output above."
        )


def _build_agreement(
    *, tf: dict[str, Any], vl: dict[str, Any], provenance_tol: float
) -> dict[str, Any]:
    tf_res = tf["result"]
    vl_res = vl["result"]
    tf_prov = tf_res.get("provenance_summary") or {}
    vl_prov = vl_res.get("provenance_summary") or {}
    mean_accessible_delta = None
    if tf_prov and vl_prov:
        mean_accessible_delta = abs(
            float(tf_prov.get("mean_source_accessible", 0.0))
            - float(vl_prov.get("mean_source_accessible", 0.0))
        )
    return {
        "draft_text_equal": tf_res["draft_text"] == vl_res["draft_text"],
        "acceptance_text_equal": tf_res["acceptance_text"] == vl_res["acceptance_text"],
        "stop_reason_equal": tf_res["stop_reason"] == vl_res["stop_reason"],
        "blocked_position_equal": (
            tf_res.get("blocked_source_local_position")
            == vl_res.get("blocked_source_local_position")
        ),
        "blocked_unit_equal": (
            tf_res.get("blocked_source_unit_index")
            == vl_res.get("blocked_source_unit_index")
        ),
        "generated_token_count_delta": (
            vl_res["generated_token_count"] - tf_res["generated_token_count"]
        ),
        "accepted_token_count_delta": (
            vl_res["accepted_token_count"] - tf_res["accepted_token_count"]
        ),
        "mean_source_accessible_abs_delta": mean_accessible_delta,
        "mean_source_accessible_within_tol": (
            mean_accessible_delta is not None and mean_accessible_delta <= provenance_tol
        ),
    }


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args._worker_backend_name is not None:
        if not args._worker_output or not args._worker_prompt_set:
            raise SystemExit(
                "Worker mode requires --_worker-backend-name, --_worker-output, "
                "--_worker-prompt-set."
            )
        _run_worker(args)
        return

    specs = _prompt_set_from_args(args)
    backend_names = [_resolve_backend_name(alias) for alias in args.backends]
    if not backend_names:
        raise SystemExit("At least one backend must be requested.")
    if len(backend_names) != len(set(backend_names)):
        raise SystemExit("Backend selection must not repeat the same backend.")

    per_backend: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="mt_parity_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        # Persist the prompt set as JSON so both workers process exactly the
        # same specs even when the orchestrator was invoked with single-prompt
        # CLI flags.
        prompt_set_file = tmpdir_path / "prompt_set.json"
        prompt_set_file.write_text(
            json.dumps(
                [
                    {
                        "name": s.name,
                        "source_text": s.source_text,
                        "is_final": s.is_final,
                        "accepted_target_prefill": s.accepted_target_prefill,
                        "min_source_mass": s.min_source_mass,
                        "current_audio_ms": s.current_audio_ms,
                        "inaccessible_ms": s.inaccessible_ms,
                        "source_history": s.source_history,
                        "translation_history": s.translation_history,
                    }
                    for s in specs
                ],
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        for backend_name in backend_names:
            print(
                f"[mt-parity] running {backend_name} over {len(specs)} prompt(s) ...",
                flush=True,
            )
            worker_out = tmpdir_path / f"{backend_name}.json"
            _run_worker_subprocess(backend_name, args, prompt_set_file, worker_out)
            per_backend[backend_name] = json.loads(worker_out.read_text(encoding="utf-8"))

    bundle: dict[str, Any] = {
        "prompt_count": len(specs),
        "backends": per_backend,
    }
    if len(backend_names) == 2:
        a, b = backend_names
        per_prompt_agreement: list[dict[str, Any]] = []
        tf_prompts = per_backend[a]["prompts"]
        vl_prompts = per_backend[b]["prompts"]
        provenance_tol = 0.05  # 5% absolute delta on mean_source_accessible
        for tf_entry, vl_entry in zip(tf_prompts, vl_prompts):
            per_prompt_agreement.append(
                {
                    "name": tf_entry["prompt"]["name"],
                    **_build_agreement(tf=tf_entry, vl=vl_entry, provenance_tol=provenance_tol),
                }
            )
        bundle["agreement_per_prompt"] = per_prompt_agreement
        bundle["agreement_overall"] = {
            "total": len(per_prompt_agreement),
            "decision_equal_count": sum(
                1
                for row in per_prompt_agreement
                if row["acceptance_text_equal"]
                and row["stop_reason_equal"]
                and row["blocked_position_equal"]
            ),
            "draft_equal_count": sum(
                1 for row in per_prompt_agreement if row["draft_text_equal"]
            ),
            "provenance_within_tol_count": sum(
                1 for row in per_prompt_agreement if row["mean_source_accessible_within_tol"]
            ),
            "provenance_tol": provenance_tol,
        }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[mt-parity] wrote {out_path}")


if __name__ == "__main__":
    main()
