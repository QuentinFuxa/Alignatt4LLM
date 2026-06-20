#!/usr/bin/env python3
"""Score MT head-set filtering on held-out word alignments.

This script measures the MT-side gain from keeping the shipped top-k AlignAtt
heads instead of uniformly averaging all Gemma query heads. The score mirrors
the paper's per-head Translation Score (TS), but evaluates the *set-averaged*
attention row that the online policy actually consumes:

1. average full self-attention rows over a candidate head set,
2. take the full-sequence argmax for each aligned target token,
3. count it as correct only when it lands on a gold aligned source token.

Outputs are written as JSON so the paper can render a small ablation table
without re-running the model.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import json
import sys
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, modeling_utils


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.alignatt_heads.detect_translation_heads import (  # noqa: E402
    build_translation_prompt,
    coerce_alignment_rows,
    project_char_span_to_token_indices,
)


DEFAULT_DIRECTIONS = ("en-de", "en-zh", "en-it")
DEFAULT_TOP_K = 8
DEFAULT_OUTPUT_JSON = REPO_ROOT / "paper" / "generated" / "mt_head_filtering_ablation.json"
DEFAULT_HEAD_DIR = REPO_ROOT / "data" / "alignatt_heads"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directions",
        nargs="+",
        default=list(DEFAULT_DIRECTIONS),
        help="Language directions to score, e.g. en-de en-zh en-it.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of retained filtered heads to score (default: 8).",
    )
    parser.add_argument(
        "--output-json",
        default=str(DEFAULT_OUTPUT_JSON),
        help="Where to write the cached paper results JSON.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model name or local snapshot path override.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
        help="Model dtype for eager scoring.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device / device_map target (default: cuda:0).",
    )
    parser.add_argument(
        "--disable-cuda-warmup",
        action="store_true",
        help="Disable transformers allocator warmup during model load.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress every N usable pairs (default: 50).",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_model_name(directions: list[str], model_override: str | None) -> str:
    if model_override:
        return model_override
    for direction in directions:
        payload = load_head_payload(direction)
        candidate = str(payload.get("model", "")).strip()
        if candidate and Path(candidate).exists():
            return candidate
    return "google/gemma-4-E4B-it"


def load_head_payload(direction: str) -> dict[str, Any]:
    return load_json(
        DEFAULT_HEAD_DIR / f"translation_heads_google_gemma-4-E4B-it_{direction}.json"
    )


def load_alignment_rows(direction: str) -> list[dict[str, Any]]:
    rows = load_json(DEFAULT_HEAD_DIR / f"word_alignments_{direction}.json")
    return [row for row in coerce_alignment_rows(rows) if row.get("alignments")]


def build_head_sets(direction: str, top_k: int) -> dict[str, Any]:
    payload = load_head_payload(direction)
    num_layers = int(payload["num_layers"])
    num_heads = int(payload["num_heads"])
    top_heads = [
        (int(entry["layer"]), int(entry["head"]))
        for entry in payload.get("token_alignment_heads", [])[:top_k]
    ]
    all_heads = [
        (layer, head)
        for layer in range(num_layers)
        for head in range(num_heads)
    ]

    def group_by_layer(heads: list[tuple[int, int]]) -> dict[int, list[int]]:
        grouped: dict[int, list[int]] = defaultdict(list)
        for layer, head in heads:
            grouped[layer].append(head)
        return dict(grouped)

    return {
        "payload": payload,
        "top_k_count": len(top_heads),
        "all_head_count": len(all_heads),
        "head_sets": {
            "filtered_topk": {
                "display_name": f"Filtered top-{len(top_heads)}",
                "heads": top_heads,
                "heads_by_layer": group_by_layer(top_heads),
            },
            "all_heads": {
                "display_name": f"All {len(all_heads)} heads",
                "heads": all_heads,
                "heads_by_layer": group_by_layer(all_heads),
            },
        },
    }


def build_valid_alignment_targets(
    *,
    row: dict[str, Any],
    direction: str,
    tokenizer,
    model_name: str,
) -> tuple[torch.Tensor | None, list[set[int]], list[int], list[int], dict[str, int]]:
    source_text = row["source_text"]
    target_text = row["target_text"]

    prompt_text = build_translation_prompt(
        model_name,
        direction,
        source_text,
        tokenizer=tokenizer,
    )
    full_text = prompt_text + target_text

    prompt_enc = tokenizer(
        prompt_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    full_enc = tokenizer(
        full_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_enc = tokenizer(
        full_text,
        return_tensors="pt",
        add_special_tokens=False,
    )

    prompt_offsets = [tuple(map(int, off)) for off in prompt_enc["offset_mapping"]]
    full_offsets = [tuple(map(int, off)) for off in full_enc["offset_mapping"]]

    source_char_start = prompt_text.rfind(source_text)
    if source_char_start < 0:
        return None, [], [], [], {}
    source_char_end = source_char_start + len(source_text)
    target_char_start = len(prompt_text)
    target_char_end = len(full_text)

    source_token_positions = project_char_span_to_token_indices(
        prompt_offsets,
        source_char_start,
        source_char_end,
    )
    target_token_positions_global = project_char_span_to_token_indices(
        full_offsets,
        target_char_start,
        target_char_end,
    )
    if not source_token_positions or not target_token_positions_global:
        return None, [], [], [], {}

    source_offsets = [
        (
            prompt_offsets[idx][0] - source_char_start,
            prompt_offsets[idx][1] - source_char_start,
        )
        for idx in source_token_positions
    ]
    target_offsets = [
        (
            full_offsets[idx][0] - target_char_start,
            full_offsets[idx][1] - target_char_start,
        )
        for idx in target_token_positions_global
    ]

    valid_source_global_by_target_local: dict[int, set[int]] = {}
    for alignment in row["alignments"]:
        src_word_span = row["source_words"][alignment["source_start"]:alignment["source_end"]]
        tgt_word_span = row["target_words"][alignment["target_start"]:alignment["target_end"]]
        if not src_word_span or not tgt_word_span:
            continue

        src_token_local = project_char_span_to_token_indices(
            source_offsets,
            int(src_word_span[0]["start_char"]),
            int(src_word_span[-1]["end_char"]),
        )
        tgt_token_local = project_char_span_to_token_indices(
            target_offsets,
            int(tgt_word_span[0]["start_char"]),
            int(tgt_word_span[-1]["end_char"]),
        )
        if not src_token_local or not tgt_token_local:
            continue

        src_global_set = {source_token_positions[idx] for idx in src_token_local}
        for tgt_local_idx in tgt_token_local:
            valid_source_global_by_target_local.setdefault(tgt_local_idx, set()).update(src_global_set)

    if not valid_source_global_by_target_local:
        return None, [], [], [], {}

    valid_target_locals = sorted(valid_source_global_by_target_local)
    valid_source_global_by_target = [
        valid_source_global_by_target_local[tgt_local_idx]
        for tgt_local_idx in valid_target_locals
    ]
    tgt_token_positions = [
        target_token_positions_global[idx]
        for idx in valid_target_locals
    ]
    metadata = {
        "source_prompt_tokens": len(source_token_positions),
        "target_tokens": len(target_token_positions_global),
    }
    return (
        input_enc["input_ids"],
        valid_source_global_by_target,
        tgt_token_positions,
        source_token_positions,
        metadata,
    )


def score_direction(
    *,
    direction: str,
    model_name: str,
    model,
    tokenizer,
    top_k: int,
    device: str,
    progress_every: int,
) -> dict[str, Any]:
    rows = load_alignment_rows(direction)
    head_sets = build_head_sets(direction, top_k)
    counters: dict[str, dict[str, dict[str, float]]] = {
        name: {
            "full_sequence": {"correct": 0.0, "total": 0.0},
            "source_prompt_only": {"correct": 0.0, "total": 0.0},
        }
        for name in head_sets["head_sets"]
    }

    attempted_rows = 0
    used_pairs = 0
    used_target_tokens = 0
    source_prompt_tokens = 0
    total_target_tokens = 0

    for row_idx, row in enumerate(rows):
        attempted_rows += 1
        (
            input_ids,
            valid_source_global_by_target,
            tgt_token_positions,
            source_token_positions,
            metadata,
        ) = build_valid_alignment_targets(
            row=row,
            direction=direction,
            tokenizer=tokenizer,
            model_name=model_name,
        )
        if input_ids is None or not tgt_token_positions:
            continue

        input_ids = input_ids.to(device)
        with torch.inference_mode():
            outputs = model(input_ids=input_ids, output_attentions=True)

        per_set_sum: dict[str, torch.Tensor | None] = {
            name: None
            for name in head_sets["head_sets"]
        }

        for layer_idx, attn in enumerate(outputs.attentions):
            if attn is None:
                continue
            layer_rows = attn[0, :, tgt_token_positions, :]
            for set_name, set_payload in head_sets["head_sets"].items():
                selected_heads = set_payload["heads_by_layer"].get(layer_idx)
                if not selected_heads:
                    continue
                partial_sum = layer_rows[selected_heads].sum(dim=0)
                if per_set_sum[set_name] is None:
                    per_set_sum[set_name] = partial_sum
                else:
                    per_set_sum[set_name] = per_set_sum[set_name] + partial_sum

        for set_name, set_payload in head_sets["head_sets"].items():
            numerator = per_set_sum[set_name]
            if numerator is None:
                continue
            averaged_rows = numerator / float(len(set_payload["heads"]))
            full_argmax_positions = averaged_rows.argmax(dim=-1).detach().cpu().tolist()
            full_correct = sum(
                1
                for target_idx, global_pos in enumerate(full_argmax_positions)
                if global_pos in valid_source_global_by_target[target_idx]
            )
            counters[set_name]["full_sequence"]["correct"] += float(full_correct)
            counters[set_name]["full_sequence"]["total"] += float(len(valid_source_global_by_target))

            source_only_rows = averaged_rows[:, source_token_positions]
            source_local_argmax_positions = source_only_rows.argmax(dim=-1).detach().cpu().tolist()
            source_global_argmax_positions = [
                int(source_token_positions[int(local_idx)])
                for local_idx in source_local_argmax_positions
            ]
            source_only_correct = sum(
                1
                for target_idx, global_pos in enumerate(source_global_argmax_positions)
                if global_pos in valid_source_global_by_target[target_idx]
            )
            counters[set_name]["source_prompt_only"]["correct"] += float(source_only_correct)
            counters[set_name]["source_prompt_only"]["total"] += float(len(valid_source_global_by_target))

        used_pairs += 1
        used_target_tokens += len(valid_source_global_by_target)
        source_prompt_tokens += int(metadata["source_prompt_tokens"])
        total_target_tokens += int(metadata["target_tokens"])

        if used_pairs == 1 or (progress_every > 0 and used_pairs % progress_every == 0):
            filtered_total = counters["filtered_topk"]["full_sequence"]["total"]
            filtered_score = (
                counters["filtered_topk"]["full_sequence"]["correct"] / filtered_total
                if filtered_total > 0
                else 0.0
            )
            all_total = counters["all_heads"]["full_sequence"]["total"]
            all_score = (
                counters["all_heads"]["full_sequence"]["correct"] / all_total
                if all_total > 0
                else 0.0
            )
            filtered_source_total = counters["filtered_topk"]["source_prompt_only"]["total"]
            filtered_source_score = (
                counters["filtered_topk"]["source_prompt_only"]["correct"] / filtered_source_total
                if filtered_source_total > 0
                else 0.0
            )
            all_source_total = counters["all_heads"]["source_prompt_only"]["total"]
            all_source_score = (
                counters["all_heads"]["source_prompt_only"]["correct"] / all_source_total
                if all_source_total > 0
                else 0.0
            )
            print(
                f"[{direction}] used {used_pairs}/{len(rows)} pairs "
                f"(row={row_idx + 1}, aligned_tokens={used_target_tokens}) "
                f"full_topk={filtered_score:.4f} full_all={all_score:.4f} "
                f"src_topk={filtered_source_score:.4f} src_all={all_source_score:.4f}",
                flush=True,
            )

        del outputs
        torch.cuda.empty_cache()

    set_scores = {}
    for set_name, metric_counts in counters.items():
        set_scores[set_name] = {}
        for metric_name, counts in metric_counts.items():
            total = float(counts["total"])
            score = float(counts["correct"]) / total if total > 0 else 0.0
            set_scores[set_name][metric_name] = {
                "score": score,
                "score_points": 100.0 * score,
                "correct": int(round(float(counts["correct"]))),
                "total": int(round(total)),
            }

    deltas = {}
    for metric_name in ("full_sequence", "source_prompt_only"):
        filtered_score = set_scores["filtered_topk"][metric_name]["score"]
        all_score = set_scores["all_heads"][metric_name]["score"]
        delta = filtered_score - all_score
        deltas[metric_name] = {
            "score": delta,
            "score_points": 100.0 * delta,
            "relative_gain_vs_all_percent": (
                (100.0 * delta / all_score) if all_score > 0.0 else None
            ),
        }

    return {
        "direction": direction,
        "attempted_rows": attempted_rows,
        "used_pairs": used_pairs,
        "used_target_tokens": used_target_tokens,
        "mean_source_prompt_tokens": (
            float(source_prompt_tokens) / float(used_pairs) if used_pairs > 0 else 0.0
        ),
        "mean_target_tokens": (
            float(total_target_tokens) / float(used_pairs) if used_pairs > 0 else 0.0
        ),
        "top_k": int(head_sets["top_k_count"]),
        "all_head_count": int(head_sets["all_head_count"]),
        "score_variants": {
            "full_sequence": {
                "score_name": "head_set_ts_argmax_full_sequence",
                "score_description": (
                    "Average the full attention rows over the candidate head set, "
                    "take the full-sequence argmax for each aligned target token, "
                    "and count it correct only when the argmax lands on a gold "
                    "aligned source token."
                ),
            },
            "source_prompt_only": {
                "score_name": "head_set_ts_argmax_source_prompt_only",
                "score_description": (
                    "Average the full attention rows over the candidate head set, "
                    "restrict the argmax to the source-prompt token positions, and "
                    "count it correct only when that source-side argmax lands on a "
                    "gold aligned source token."
                ),
            },
        },
        "filtered_topk": set_scores["filtered_topk"],
        "all_heads": set_scores["all_heads"],
        "delta": deltas,
    }


def load_model_and_tokenizer(
    *,
    model_name: str,
    dtype_str: str,
    device: str,
    disable_cuda_warmup: bool,
):
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    load_kwargs: dict[str, Any] = {
        "torch_dtype": dtype_map[dtype_str],
        "trust_remote_code": True,
        "attn_implementation": "eager",
        "low_cpu_mem_usage": True,
        "local_files_only": True,
        "device_map": device,
    }

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=True,
    )

    original_caching_allocator_warmup = None
    if disable_cuda_warmup and hasattr(modeling_utils, "caching_allocator_warmup"):
        original_caching_allocator_warmup = modeling_utils.caching_allocator_warmup
        modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    finally:
        if original_caching_allocator_warmup is not None:
            modeling_utils.caching_allocator_warmup = original_caching_allocator_warmup
    model.eval()
    return model, tokenizer


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    directions = list(args.directions)
    model_name = resolve_model_name(directions, args.model)
    print(f"Loading model for MT head-set scoring: {model_name}", flush=True)
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        dtype_str=args.dtype,
        device=args.device,
        disable_cuda_warmup=args.disable_cuda_warmup,
    )

    direction_results: dict[str, Any] = {}
    for direction in directions:
        print(f"\nScoring {direction} ...", flush=True)
        direction_results[direction] = score_direction(
            direction=direction,
            model_name=model_name,
            model=model,
            tokenizer=tokenizer,
            top_k=int(args.top_k),
            device=args.device,
            progress_every=int(args.progress_every),
        )

    del model
    torch.cuda.empty_cache()

    summary_rows = []
    for direction in directions:
        result = direction_results[direction]
        summary_rows.append(
            {
                "direction": direction,
                "full_top_k_score": result["filtered_topk"]["full_sequence"]["score"],
                "full_all_heads_score": result["all_heads"]["full_sequence"]["score"],
                "full_delta_score": result["delta"]["full_sequence"]["score"],
                "full_delta_points": result["delta"]["full_sequence"]["score_points"],
                "source_top_k_score": result["filtered_topk"]["source_prompt_only"]["score"],
                "source_all_heads_score": result["all_heads"]["source_prompt_only"]["score"],
                "source_delta_score": result["delta"]["source_prompt_only"]["score"],
                "source_delta_points": result["delta"]["source_prompt_only"]["score_points"],
                "used_pairs": result["used_pairs"],
                "used_target_tokens": result["used_target_tokens"],
            }
        )

    payload = {
        "model": model_name,
        "directions": direction_results,
        "summary_rows": summary_rows,
        "top_k": int(args.top_k),
        "score_names": {
            "full_sequence": "head_set_ts_argmax_full_sequence",
            "source_prompt_only": "head_set_ts_argmax_source_prompt_only",
        },
        "score_units": "fraction",
        "score_point_scale": 100.0,
        "notes": [
            "Filtered top-k uses the same per-direction retained MT heads shipped in the runtime payloads.",
            "All-head averaging uniformly averages all 42x8 Gemma query heads before taking the full-sequence argmax.",
            "Scores are measured on the held-out word-aligned MT calibration sets under the exact runtime prompt layout.",
            "The source_prompt_only variant restricts the argmax to the prompt positions occupied by the source span.",
        ],
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"\nSaved MT head-set ablation results to {output_path}", flush=True)
    print("\nFiltered top-k minus all-head averaging:", flush=True)
    for row in summary_rows:
        print(
            f"  {row['direction']}: "
            f"full={row['full_delta_points']:+.2f} "
            f"(topk={100.0 * row['full_top_k_score']:.2f}, all={100.0 * row['full_all_heads_score']:.2f})  "
            f"source-only={row['source_delta_points']:+.2f} "
            f"(topk={100.0 * row['source_top_k_score']:.2f}, all={100.0 * row['source_all_heads_score']:.2f})",
            flush=True,
        )


if __name__ == "__main__":
    main()
