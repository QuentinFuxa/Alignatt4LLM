#!/usr/bin/env python3
"""Materialise a Phase 4 AlignAtt head-set regime on disk.

This CLI constructs one of the three head-set regimes described in the
historical plan notes now archived under ``docs/archive/`` and writes it as a
``translation_heads_*.json`` file that ``cascade_mt_backend.load_alignatt_heads``
can consume directly. Pair it with ``run_cascade_baseline.py --target-lang ... ``
and a runtime override of ``translation_alignatt_heads_path`` to drive head-set
comparison sweeps once a GPU is available.

Regimes
-------
``per_direction``
    Copy the top-k heads of a single direction (sanity check / identity).

``shared_kernel``
    Intersect the top-k heads across several directions and rank by mean
    ``ts``. Produces the small multilingual kernel conjectured in the archived
    plan notes.

``multilingual_union``
    Union the top-k heads across directions, rank by mean ``ts``, optionally
    truncate to ``--max-heads`` so the runtime head budget stays comparable to
    the per-direction baseline.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from cascade_mt_backend import (
    load_alignatt_heads,
    load_alignatt_heads_by_direction,
    multilingual_union_alignatt_heads,
    shared_kernel_alignatt_heads,
    write_alignatt_heads_file,
)


DEFAULT_DIRECTION_PATHS = {
    "en-de": "assets/attention_heads/translation_heads_google_gemma-4-E4B-it_en-de.json",
    "en-it": "assets/attention_heads/translation_heads_google_gemma-4-E4B-it_en-it.json",
    "en-zh": "assets/attention_heads/translation_heads_google_gemma-4-E4B-it_en-zh.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--regime",
        required=True,
        choices=["per_direction", "shared_kernel", "multilingual_union"],
        help="Which head-set regime to materialise.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination JSON file.",
    )
    parser.add_argument(
        "--top-k",
        default=8,
        type=int,
        help="Top-k heads to load from each direction before combining.",
    )
    parser.add_argument(
        "--max-heads",
        default=None,
        type=int,
        help=(
            "For multilingual_union: cap the output to this many heads. "
            "Use it to keep the runtime head budget comparable to top-k."
        ),
    )
    parser.add_argument(
        "--direction",
        default=None,
        help=(
            "For per_direction: which direction tag (e.g. en-de) to copy. "
            "Must be a key of --direction-path or DEFAULT_DIRECTION_PATHS."
        ),
    )
    parser.add_argument(
        "--direction-path",
        action="append",
        default=[],
        metavar="DIRECTION=PATH",
        help=(
            "Override or add direction->path entries, e.g. "
            "--direction-path en-de=/some/path.json. Can be repeated."
        ),
    )
    return parser.parse_args()


def resolve_direction_paths(overrides: list[str]) -> dict[str, str]:
    resolved = dict(DEFAULT_DIRECTION_PATHS)
    for entry in overrides:
        if "=" not in entry:
            raise SystemExit(f"--direction-path expects KEY=PATH, got {entry!r}")
        key, path = entry.split("=", 1)
        resolved[key.strip()] = path.strip()
    return resolved


def _report_head_set_cost(heads, regime_label: str) -> None:
    """Print cost-aware summary: head count, distinct layers, layer list."""
    layers = sorted({h.layer for h in heads})
    print(f"  {regime_label}: {len(heads)} heads across {len(layers)} layers")
    print(f"  layers touched: {layers}")
    for h in heads:
        print(f"    L{h.layer:>2} H{h.head:>2}  ts={h.ts:.3f}")


def main() -> None:
    args = parse_args()
    direction_paths = resolve_direction_paths(args.direction_path)

    if args.regime == "per_direction":
        if args.direction is None:
            raise SystemExit("--direction is required for --regime per_direction")
        if args.direction not in direction_paths:
            raise SystemExit(
                f"Unknown direction {args.direction!r}; known: {sorted(direction_paths)}"
            )
        heads = load_alignatt_heads(direction_paths[args.direction], top_k=args.top_k)
        written = write_alignatt_heads_file(
            heads,
            args.output,
            direction=args.direction,
            extra_metadata={
                "regime": "per_direction",
                "top_k": args.top_k,
                "source_direction": args.direction,
            },
        )
        print(f"Wrote {len(heads)} heads for {args.direction} to {written}")
        _report_head_set_cost(heads, f"per_direction({args.direction})")
        return

    heads_by_direction = load_alignatt_heads_by_direction(
        direction_paths, top_k=args.top_k
    )

    if args.regime == "shared_kernel":
        heads = shared_kernel_alignatt_heads(heads_by_direction)
        written = write_alignatt_heads_file(
            heads,
            args.output,
            direction="shared_kernel",
            extra_metadata={
                "regime": "shared_kernel",
                "top_k": args.top_k,
                "source_directions": sorted(heads_by_direction),
            },
        )
        print(
            f"Wrote shared kernel of {len(heads)} heads "
            f"(from {sorted(heads_by_direction)} @ top_k={args.top_k}) to {written}"
        )
        _report_head_set_cost(heads, "shared_kernel")
        return

    heads = multilingual_union_alignatt_heads(
        heads_by_direction, max_heads=args.max_heads
    )
    written = write_alignatt_heads_file(
        heads,
        args.output,
        direction="multilingual_union",
        extra_metadata={
            "regime": "multilingual_union",
            "top_k": args.top_k,
            "max_heads": args.max_heads,
            "source_directions": sorted(heads_by_direction),
        },
    )
    print(
        f"Wrote multilingual union of {len(heads)} heads "
        f"(from {sorted(heads_by_direction)} @ top_k={args.top_k}, "
        f"max_heads={args.max_heads}) to {written}"
    )
    _report_head_set_cost(heads, "multilingual_union")


if __name__ == "__main__":
    main()
