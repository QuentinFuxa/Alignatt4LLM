#!/usr/bin/env python
"""Gemma ASR fairness benchmark — controlled ablation of the discrepancy.

Root-causes the gap between the old fairness harness (WER ~0.83, hallucinated)
and the standalone multimodal test (WER ~0.09–0.26, mostly correct).

Candidate root causes under test:

1. attn_implementation  : "eager" (old path via backend) vs default
2. audio_input_format   : numpy array (old path) vs file path string (standalone)
3. decode_policy        : greedy (old path) vs sampled (standalone)

The model is loaded ONCE. ``attn_implementation`` is toggled at runtime via
the config attribute (the same mechanism GemmaAttentionAlignmentBackend uses).
"""

from __future__ import annotations

import argparse
import json
import string
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor


PROMPT = (
    "Transcribe the following speech segment in its original language. "
    "Follow these specific instructions for formatting the answer:\n"
    "* Only output the transcription, with no newlines.\n"
    "* When transcribing numbers, write the digits, i.e. write 1.7 "
    "and not one point seven, and write 3 instead of three."
)


@dataclass
class Variant:
    attn_impl: str        # "eager" | "default"
    audio_format: str     # "filepath" | "numpy"
    decode_policy: str    # "greedy" | "sampled"


def _resolve_model() -> str:
    candidates = [
        "/home/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/83df0a889143b1dbfc61b591bbc639540fd9ce4c",
        str(
            Path.home()
            / ".cache/huggingface/hub/models--google--gemma-4-E4B-it"
            / "snapshots/83df0a889143b1dbfc61b591bbc639540fd9ce4c"
        ),
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return "google/gemma-4-E4B-it"


def _load_wav_numpy(path: str) -> np.ndarray:
    with wave.open(path, "rb") as wav:
        sr = wav.getframerate()
        width = wav.getsampwidth()
        ch = wav.getnchannels()
        raw = wav.readframes(wav.getnframes())
    if width != 2:
        raise ValueError("Only 16-bit PCM WAV supported.")
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1)
    if sr != 16000:
        duration = len(audio) / sr
        new_length = int(duration * 16000)
        old_times = np.linspace(0.0, duration, num=len(audio), endpoint=False)
        new_times = np.linspace(0.0, duration, num=new_length, endpoint=False)
        audio = np.interp(new_times, old_times, audio).astype(np.float32)
    return audio


def _build_messages(wav_path: str, *, audio_format: str) -> list[dict]:
    if audio_format == "filepath":
        audio_ref = str(Path(wav_path).resolve())
    elif audio_format == "numpy":
        audio_ref = _load_wav_numpy(wav_path)
    else:
        raise ValueError(f"Unknown audio_format: {audio_format!r}")
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_ref},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]


def _set_attn_impl(model, impl: str) -> list[tuple[object, str | None]]:
    """Toggle attn_implementation on all config objects, return originals."""
    saved = []
    seen = set()
    for obj in (
        model,
        getattr(model, "model", None),
        getattr(getattr(model, "model", None), "language_model", None),
    ):
        config = getattr(obj, "config", None)
        if config is None or id(config) in seen:
            continue
        seen.add(id(config))
        original = getattr(config, "_attn_implementation", None)
        saved.append((config, original))
        if impl == "eager":
            config._attn_implementation = "eager"
        else:
            if original is not None:
                config._attn_implementation = original
    return saved


def _restore_attn_impl(saved: list[tuple[object, str | None]]) -> None:
    for config, original in saved:
        if original is not None:
            config._attn_implementation = original


def _run_variant(
    *,
    model,
    processor,
    wav_path: str,
    variant: Variant,
    max_new_tokens: int,
) -> dict:
    messages = _build_messages(wav_path, audio_format=variant.audio_format)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)
    input_len = int(inputs["input_ids"].shape[-1])

    saved = _set_attn_impl(model, variant.attn_impl)
    try:
        if variant.decode_policy == "greedy":
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        elif variant.decode_policy == "sampled":
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                top_k=64,
            )
        else:
            raise ValueError(f"Unknown decode_policy: {variant.decode_policy!r}")
    finally:
        _restore_attn_impl(saved)

    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    parsed = None
    try:
        parsed = processor.parse_response(response)
    except Exception:
        pass

    transcript = parsed["content"] if isinstance(parsed, dict) and "content" in parsed else response
    # strip trailing turn marker
    for marker in ("<turn|>", "<end_of_turn>"):
        transcript = transcript.replace(marker, "")
    transcript = transcript.strip()

    return {
        "variant": asdict(variant),
        "raw_response": response,
        "parsed_response": parsed,
        "transcript": transcript,
    }


# ── scoring ──────────────────────────────────────────────────────────


def _normalize(text: str) -> list[str]:
    table = str.maketrans("", "", string.punctuation + "\u201c\u201d\u2018\u2019")
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
    ref = _normalize(reference)
    hyp = _normalize(hypothesis)
    return _levenshtein(ref, hyp) / max(1, len(ref))


def _cer(reference: str, hypothesis: str) -> float:
    ref = list("".join(_normalize(reference)))
    hyp = list("".join(_normalize(hypothesis)))
    return _levenshtein(ref, hyp) / max(1, len(ref))


# ── main ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wav", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", default="tmp/alignment_research/gemma_asr_fairness_ablation.json")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_path = _resolve_model()
    reference = Path(args.reference).read_text(encoding="utf-8").strip()

    print(f"[info] loading model from: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_path, dtype="auto", device_map="auto", trust_remote_code=True
    )

    # Ablation matrix: 2 attn × 2 audio × 2 decode = 8 variants
    variants = [
        Variant(attn_impl=a, audio_format=f, decode_policy=d)
        for a in ("default", "eager")
        for f in ("filepath", "numpy")
        for d in ("greedy", "sampled")
    ]

    rows = []
    for v in variants:
        print(f"\n[run] {v}")
        try:
            row = _run_variant(
                model=model,
                processor=processor,
                wav_path=args.wav,
                variant=v,
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as exc:
            row = {"variant": asdict(v), "error": f"{type(exc).__name__}: {exc}"}
            print(f"  ! error: {row['error']}")
            rows.append(row)
            continue
        row["wer"] = _wer(reference, row["transcript"])
        row["cer"] = _cer(reference, row["transcript"])
        print(f"  WER={row['wer']:.3f}  CER={row['cer']:.3f}")
        print(f"  text: {row['transcript'][:160]}...")
        rows.append(row)

    payload = {
        "wav_path": str(args.wav),
        "reference": reference,
        "model": model_path,
        "device": str(model.device),
        "dtype": str(model.dtype),
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "normalization": "lowercase, strip punctuation (incl curly quotes), split on whitespace",
        "variants": rows,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[saved] {out}")

    # Summary table
    print("\n=== ABLATION SUMMARY ===")
    print(f"{'attn':>8s} {'audio':>10s} {'decode':>8s}  {'WER':>6s}  {'CER':>6s}  transcript[:80]")
    for row in rows:
        v = row["variant"]
        wer = row.get("wer", float("nan"))
        cer = row.get("cer", float("nan"))
        text = row.get("transcript", row.get("error", ""))[:80]
        print(f"{v['attn_impl']:>8s} {v['audio_format']:>10s} {v['decode_policy']:>8s}  {wer:6.3f}  {cer:6.3f}  {text}")


if __name__ == "__main__":
    main()
