#!/usr/bin/env python3
"""Offline backbone screening: translate line-aligned EN segments with one model.

Self-contained (vllm + transformers only): meant to run on a throwaway GPU box,
one invocation per model so each vLLM engine gets a fresh process. Greedy
decoding everywhere — matching the streaming pipeline's deployment mode — and
the comparison is anchored by a gemma-4-E4B-it control arm under the same
protocol, so only deltas vs control are interpreted.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

PROMPT_STYLES = ("instruct_user", "hymt", "milmmt_raw")


def build_prompts(lines: list[str], style: str, tokenizer) -> list[str]:
    if style == "milmmt_raw":
        return [
            "Translate this from English to Chinese (Simplified):\n"
            f"English: {text}\nChinese (Simplified):"
            for text in lines
        ]
    if style == "hymt":
        def messages(text: str) -> list[dict[str, str]]:
            return [
                {
                    "role": "user",
                    "content": (
                        "Translate the following segment into Chinese, "
                        f"without additional explanation.\n\n{text}"
                    ),
                }
            ]
    else:
        def messages(text: str) -> list[dict[str, str]]:
            return [
                {
                    "role": "user",
                    "content": (
                        "Translate the following English text into Chinese "
                        "(Simplified). Output only the translation, with no "
                        f"additional explanation.\n\n{text}"
                    ),
                }
            ]
    return [
        tokenizer.apply_chat_template(
            messages(text), tokenize=False, add_generation_prompt=True
        )
        for text in lines
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prompt-style", choices=PROMPT_STYLES, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="Translate only the first N lines (smoke).")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    args = parser.parse_args()

    lines = [line.rstrip("\n") for line in args.source.read_text(encoding="utf-8").splitlines()]
    if args.limit:
        lines = lines[: args.limit]

    from vllm import LLM, SamplingParams

    started = time.time()
    llm = LLM(
        model=args.model_id,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    tokenizer = llm.get_tokenizer()
    prompts = build_prompts(lines, args.prompt_style, tokenizer)
    outputs = llm.generate(
        prompts, SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    )
    translations = [
        (output.outputs[0].text if output.outputs else "").strip().replace("\n", " ")
        for output in outputs
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(text + "\n" for text in translations), encoding="utf-8"
    )
    sidecar = {
        "model_id": args.model_id,
        "prompt_style": args.prompt_style,
        "n_lines": len(lines),
        "empty_outputs": sum(1 for text in translations if not text),
        "elapsed_s": round(time.time() - started, 1),
        "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(sidecar, ensure_ascii=False))


if __name__ == "__main__":
    main()
