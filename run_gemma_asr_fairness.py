#!/usr/bin/env python
"""Fair Gemma free-run ASR benchmark — Phase 2 of PLAN.md.

Answers exactly: "was the previous Gemma ASR evaluation fair?"

Runs a controlled matrix on one short clip and saves every variant's
exact output + WER/CER vs a trusted reference. Variables under control:

1. decode path            : ``model.generate()`` vs the cascade's manual
                            greedy decode (the path the alignment probe
                            uses). These should agree token-for-token at
                            ``temperature=0``; any divergence is a bug
                            and must be reported, not papered over.
2. prompt ordering        : audio-block-before-text vs text-before-audio.
3. prompt wording         : the cookbook ``"in its original language"``
                            phrasing vs the more explicit English
                            transcription wording previously used.
4. input cast policy      : single ``.to(device)`` (cookbook) vs an
                            additional explicit cast of float tensors to
                            ``model.dtype`` (the older path that ran a
                            bf16 quantization pass over mel features).

This script does not call the alignment probe — no attention extraction,
no head ranking. It reports transcripts only, so the verdict on
free-run ASR is independent of the alignment story. Produces a JSON
file with every variant's metadata + transcript + WER/CER.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from run_alignment_single_audio import load_wav


PROMPT_WORDINGS: dict[str, str] = {
    "cookbook_original_language": (
        "Transcribe the following speech segment in its original language. "
        "Follow these specific instructions for formatting the answer:\n"
        "* Only output the transcription, with no newlines.\n"
        "* When transcribing numbers, write the digits, i.e. write 1.7 "
        "and not one point seven, and write 3 instead of three."
    ),
    "explicit_english": (
        "Transcribe the following English speech segment into English text. "
        "Output only the transcription, no newlines, no commentary."
    ),
}


@dataclass
class FairnessVariant:
    decode_path: str        # "generate" | "manual_greedy"
    prompt_order: str       # "audio_first" | "text_first"
    prompt_wording: str     # key into PROMPT_WORDINGS
    input_cast: str         # "to_device" | "to_device_and_dtype"


def _build_messages(
    *, audio: np.ndarray, prompt_order: str, prompt_wording: str
) -> list[dict]:
    text_block = {"type": "text", "text": PROMPT_WORDINGS[prompt_wording]}
    audio_block = {"type": "audio", "audio": np.asarray(audio, dtype=np.float32)}
    if prompt_order == "audio_first":
        content = [audio_block, text_block]
    elif prompt_order == "text_first":
        content = [text_block, audio_block]
    else:
        raise ValueError(f"Unknown prompt_order: {prompt_order!r}")
    return [{"role": "user", "content": content}]


def _cast_inputs(inputs, *, model, policy: str):
    inputs = inputs.to(model.device)
    if policy == "to_device":
        return inputs
    if policy == "to_device_and_dtype":
        # Older path: explicitly cast every float tensor to ``model.dtype``.
        # Reported (in the implementation notes) to introduce a bf16
        # quantization pass over mel features before the audio tower.
        target_dtype = model.dtype
        for key in list(inputs.keys()):
            value = inputs[key]
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                inputs[key] = value.to(target_dtype)
        return inputs
    raise ValueError(f"Unknown input_cast policy: {policy!r}")


def _resolve_stop_token_ids(*, tokenizer, model) -> tuple[int, ...]:
    stops: set[int] = set()
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        stops.add(int(eos))
    config_eos = getattr(model.config, "eos_token_id", None)
    if isinstance(config_eos, (list, tuple)):
        stops.update(int(t) for t in config_eos)
    elif isinstance(config_eos, int):
        stops.add(int(config_eos))
    end_of_turn = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if isinstance(end_of_turn, int) and end_of_turn >= 0:
        stops.add(end_of_turn)
    return tuple(sorted(stops))


def _decode_generate(
    *, model, processor, tokenizer, inputs: dict, max_new_tokens: int
) -> str:
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    prompt_len = int(inputs["input_ids"].shape[1])
    new_ids = out[0, prompt_len:].tolist()
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def _decode_manual_greedy(
    *, model, tokenizer, inputs: dict, max_new_tokens: int
) -> str:
    """Reproduces the cascade probe's per-step argmax decode at T=0.

    Should agree token-for-token with ``model.generate(do_sample=False)``;
    any divergence is a bug worth reporting, not a quality metric.
    """
    eos_token_ids = set(_resolve_stop_token_ids(tokenizer=tokenizer, model=model))
    generated_ids: list[int] = []

    model_kwargs = {k: v for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**model_kwargs, use_cache=True, return_dict=True)
        past_key_values = outputs.past_key_values
        for _ in range(max_new_tokens):
            logits = outputs.logits[0, -1, :].float()
            next_token_id = int(logits.argmax().item())
            if next_token_id in eos_token_ids:
                break
            generated_ids.append(next_token_id)
            step_input_ids = torch.tensor(
                [[next_token_id]], device=model.device
            )
            outputs = model(
                input_ids=step_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def _normalize_for_metric(text: str) -> list[str]:
    import string

    table = str.maketrans("", "", string.punctuation + "“”‘’")
    return text.lower().translate(table).split()


def _levenshtein(ref: Sequence, hyp: Sequence) -> int:
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        curr = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, start=1):
            cost = 0 if r == h else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def _wer(reference: str, hypothesis: str) -> float:
    ref_tokens = _normalize_for_metric(reference)
    hyp_tokens = _normalize_for_metric(hypothesis)
    if not ref_tokens:
        return float("nan")
    return _levenshtein(ref_tokens, hyp_tokens) / len(ref_tokens)


def _cer(reference: str, hypothesis: str) -> float:
    ref_chars = list("".join(_normalize_for_metric(reference)))
    hyp_chars = list("".join(_normalize_for_metric(hypothesis)))
    if not ref_chars:
        return float("nan")
    return _levenshtein(ref_chars, hyp_chars) / len(ref_chars)


def run_variant(
    variant: FairnessVariant,
    *,
    audio: np.ndarray,
    model,
    processor,
    tokenizer,
    max_new_tokens: int,
) -> dict:
    messages = _build_messages(
        audio=audio,
        prompt_order=variant.prompt_order,
        prompt_wording=variant.prompt_wording,
    )
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = _cast_inputs(inputs, model=model, policy=variant.input_cast)
    inputs_dict = dict(inputs)

    if variant.decode_path == "generate":
        text = _decode_generate(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            inputs=inputs_dict,
            max_new_tokens=max_new_tokens,
        )
    elif variant.decode_path == "manual_greedy":
        text = _decode_manual_greedy(
            model=model,
            tokenizer=tokenizer,
            inputs=inputs_dict,
            max_new_tokens=max_new_tokens,
        )
    else:
        raise ValueError(f"Unknown decode_path: {variant.decode_path!r}")

    return {"variant": asdict(variant), "transcript": text}


def cmd_run(args: argparse.Namespace) -> None:
    audio, sr = load_wav(args.wav)
    if sr != 16000:
        raise RuntimeError("expected 16 kHz audio after load_wav")
    duration_s = len(audio) / sr
    print(f"[info] {args.wav}: duration={duration_s:.2f}s")

    reference_text = Path(args.reference).read_text(encoding="utf-8").strip()
    print(f"[info] reference ({len(reference_text)} chars): {reference_text[:120]}...")

    # Load Gemma once; reuse for every variant.
    from gemma_alignment_probe import GemmaAttentionAlignmentBackend
    from qwen3asr_gemma_cascade_core import config, gemma_model_name

    backend = GemmaAttentionAlignmentBackend(
        model_name=gemma_model_name,
        runtime_config=config,
        audio_heads_path=None,  # ASR fairness — no alignment heads needed
        audio_heads_top_k=0,
    )
    backend.load()
    model = backend.model
    processor = backend.processor
    tokenizer = backend.tokenizer

    variants: list[FairnessVariant] = []
    if args.full_matrix:
        for decode_path, prompt_order, wording, cast in itertools.product(
            ["generate", "manual_greedy"],
            ["audio_first", "text_first"],
            list(PROMPT_WORDINGS.keys()),
            ["to_device", "to_device_and_dtype"],
        ):
            variants.append(
                FairnessVariant(
                    decode_path=decode_path,
                    prompt_order=prompt_order,
                    prompt_wording=wording,
                    input_cast=cast,
                )
            )
    else:
        # Minimum useful set: decode-path equivalence on the cookbook
        # configuration, plus the two single-axis ablations from the
        # implementation notes (prompt order, prompt wording).
        cookbook = ("audio_first", "cookbook_original_language", "to_device")
        variants = [
            FairnessVariant("generate", *cookbook),
            FairnessVariant("manual_greedy", *cookbook),
            FairnessVariant("generate", "text_first", "cookbook_original_language", "to_device"),
            FairnessVariant("generate", "audio_first", "explicit_english", "to_device"),
            FairnessVariant("generate", "audio_first", "cookbook_original_language", "to_device_and_dtype"),
        ]

    rows: list[dict] = []
    for variant in variants:
        print(f"\n[run] {variant}")
        try:
            row = run_variant(
                variant,
                audio=audio,
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                max_new_tokens=int(args.max_new_tokens),
            )
        except Exception as exc:  # noqa: BLE001 — capture per-variant
            row = {"variant": asdict(variant), "error": f"{type(exc).__name__}: {exc}"}
            print(f"  ! error: {row['error']}")
            rows.append(row)
            continue
        row["wer"] = _wer(reference_text, row["transcript"])
        row["cer"] = _cer(reference_text, row["transcript"])
        print(f"  WER={row['wer']:.3f}  CER={row['cer']:.3f}")
        print(f"  text: {row['transcript'][:140]}...")
        rows.append(row)

    payload = {
        "wav_path": str(args.wav),
        "audio_duration_s": float(duration_s),
        "reference": reference_text,
        "model": gemma_model_name,
        "max_new_tokens": int(args.max_new_tokens),
        "variants": rows,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[saved] {out_path}")

    # Decode-path equivalence check: at T=0, generate() and manual greedy
    # should agree token-for-token under matched prompt/cast settings.
    print("\n[decode-path equivalence under matched prompt+cast]")
    by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        if "transcript" not in row:
            continue
        v = row["variant"]
        key = (v["prompt_order"], v["prompt_wording"], v["input_cast"])
        by_key.setdefault(key, {})[v["decode_path"]] = row["transcript"]
    for key, mapping in by_key.items():
        if len(mapping) < 2:
            continue
        agree = mapping["generate"].strip() == mapping["manual_greedy"].strip()
        print(f"  order={key[0]:11s} wording={key[1]:32s} cast={key[2]:22s}  agree={agree}")


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wav", required=True)
    parser.add_argument(
        "--reference",
        required=True,
        help="Path to a .txt file containing the trusted reference transcript.",
    )
    parser.add_argument(
        "--output",
        default="tmp/alignment_research/gemma_asr_fairness.json",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--full-matrix",
        action="store_true",
        help="Run all 2x2x2x2 = 16 variants instead of the 5-variant minimum.",
    )
    parser.set_defaults(func=cmd_run)
    return parser


def main(argv=None) -> None:
    args = build_cli().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
