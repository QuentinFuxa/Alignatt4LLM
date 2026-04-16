#!/usr/bin/env python
"""Standalone Gemma ASR smoke test using AutoModelForMultimodalLM.

This intentionally follows the official multimodal usage pattern:

- AutoProcessor
- AutoModelForMultimodalLM
- audio block before text
- processor.apply_chat_template(...).to(model.device)
- model.generate(...)
- processor.decode(...)
- processor.parse_response(...)

It is meant to answer one narrow question:

    "If we use Gemma exactly like the official multimodal recipe, what
    transcript do we actually get on our audios?"
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor


DEFAULT_PROMPT = (
    "Transcribe the following speech segment in its original language. "
    "Follow these specific instructions for formatting the answer:\n"
    "* Only output the transcription, with no newlines.\n"
    "* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, "
    "and write 3 instead of three."
)


def _resolve_default_model() -> str:
    local_candidates = [
        "/home/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/83df0a889143b1dbfc61b591bbc639540fd9ce4c",
        str(
            Path.home()
            / ".cache"
            / "huggingface"
            / "hub"
            / "models--google--gemma-4-E4B-it"
            / "snapshots"
            / "83df0a889143b1dbfc61b591bbc639540fd9ce4c"
        ),
    ]
    for candidate in local_candidates:
        if Path(candidate).exists():
            return candidate
    return "google/gemma-4-E4B-it"


def _build_messages(audio_ref: str, prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_ref},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _materialize_audio_ref(audio_ref: str) -> str:
    if not audio_ref.startswith(("http://", "https://")):
        return str(Path(audio_ref).resolve())

    parsed = urlparse(audio_ref)
    suffix = Path(parsed.path).suffix or ".bin"
    cache_dir = Path("tmp/standalone_audio_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_name = hashlib.sha1(audio_ref.encode("utf-8")).hexdigest() + suffix
    cache_path = cache_dir / cache_name
    if not cache_path.exists():
        response = requests.get(audio_ref, timeout=120)
        response.raise_for_status()
        cache_path.write_bytes(response.content)
    return str(cache_path.resolve())


def _run_one(
    *,
    processor,
    model,
    audio_ref: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> dict[str, Any]:
    messages = _build_messages(audio_ref, prompt)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)
    input_len = int(inputs["input_ids"].shape[-1])

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    parsed = None
    parse_error = None
    try:
        parsed = processor.parse_response(response)
    except Exception as exc:  # noqa: BLE001
        parse_error = f"{type(exc).__name__}: {exc}"

    return {
        "audio": audio_ref,
        "raw_response": response,
        "parsed_response": parsed,
        "parse_error": parse_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=_resolve_default_model(),
        help="Model id or local snapshot path.",
    )
    parser.add_argument(
        "--audio",
        action="append",
        required=True,
        help="Audio URL or local WAV path. Repeat for multiple inputs.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"[info] loading processor from: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    print(f"[info] loading model from: {args.model}")
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model,
        dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

    results: list[dict[str, Any]] = []
    for audio in args.audio:
        audio_ref = _materialize_audio_ref(audio)
        print(f"\n=== audio: {audio_ref}")
        result = _run_one(
            processor=processor,
            model=model,
            audio_ref=audio_ref,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        result["original_audio_arg"] = audio
        results.append(result)
        print("[raw_response]")
        print(result["raw_response"])
        print("[parsed_response]")
        print(json.dumps(result["parsed_response"], ensure_ascii=False, indent=2))
        if result["parse_error"] is not None:
            print("[parse_error]")
            print(result["parse_error"])

    payload = {
        "model": args.model,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
        },
        "prompt": args.prompt,
        "results": results,
    }
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[info] wrote results to {output_path}")


if __name__ == "__main__":
    main()
