#!/usr/bin/env python3
"""Capture Gemma E4B ASR traces on MCIF without AlignAtt.

This is deliberately a decoder-only ASR probe:

1. feed Gemma E4B the latest audio window through vLLM;
2. force a bounded suffix of the append-only stable transcript as decoder
   prefix tokens;
3. commit only the whole-word LCP between consecutive full hypotheses;
4. write final transcripts and corpus WER/CER when MCIF references exist.

No Q/K observer, attention heads, timestamp reconstruction, or AlignAtt commit
policy is used here.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import string
import sys
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alignatt4llm.runtime import gemma_model_name  # noqa: E402
from tools.research.alignment_single_audio import load_wav  # noqa: E402


PUNCT_TABLE = str.maketrans("", "", string.punctuation + "“”‘’")
NONSPACE_RE = re.compile(r"\S+")
GEMMA_ASR_INSTRUCTION = (
    "Provide a verbatim, word-for-word transcription of the audio. "
    "Only output the transcription, with no newlines."
)
GEMMA_AUDIO_TOKEN_ID = 258881
GEMMA_AUDIO_MS_PER_TOKEN = 40.0
GEMMA_AUDIO_MAX_SECONDS = 30.0


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def normalize_spaces(text: str) -> str:
    return " ".join(str(text).replace("\n", " ").split())


def join_text(left: str, right: str) -> str:
    left = str(left).strip()
    right = str(right).strip()
    if not left:
        return right
    if not right:
        return left
    if right[0] in string.punctuation:
        return f"{left}{right}"
    return f"{left} {right}"


def word_safe_prefix(text: str, max_chars: int) -> str:
    end = 0
    for match in NONSPACE_RE.finditer(text):
        if match.end() > int(max_chars):
            break
        end = match.end()
    return text[:end].rstrip()


def lcp_chars(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    idx = 0
    while idx < limit and left[idx] == right[idx]:
        idx += 1
    return idx


def whole_word_lcp(left: str, right: str) -> str:
    return word_safe_prefix(right, lcp_chars(left, right))


def split_words(text: str) -> list[str]:
    return [match.group(0) for match in NONSPACE_RE.finditer(str(text))]


def suffix_text_with_token_budget(tokenizer: Any, text: str, budget: int) -> tuple[str, list[int]]:
    text = str(text).strip()
    budget = int(budget)
    if not text or budget <= 0:
        return "", []

    words = list(NONSPACE_RE.finditer(text))
    if not words:
        return "", []

    lo = 0
    hi = len(words) - 1
    best_text = text[words[-1].start() :].strip()
    best_ids = [
        int(t)
        for t in tokenizer.encode(best_text, add_special_tokens=False)
    ]

    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[words[mid].start() :].strip()
        token_ids = [
            int(t)
            for t in tokenizer.encode(candidate, add_special_tokens=False)
        ]
        if len(token_ids) <= budget:
            best_text = candidate
            best_ids = token_ids
            hi = mid - 1
        else:
            lo = mid + 1

    if len(best_ids) > budget:
        best_ids = best_ids[-budget:]
        best_text = tokenizer.decode(best_ids, skip_special_tokens=True).strip()
    return best_text, best_ids


def normalize_for_error_rate(text: str) -> list[str]:
    return str(text).lower().translate(PUNCT_TABLE).split()


def levenshtein_distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    previous = list(range(len(hyp) + 1))
    for i, ref_item in enumerate(ref, start=1):
        current = [i] + [0] * len(hyp)
        for j, hyp_item in enumerate(hyp, start=1):
            cost = 0 if ref_item == hyp_item else 1
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            )
        previous = current
    return previous[-1]


def error_rates(reference_text: str, hypothesis_text: str) -> dict[str, Any]:
    ref_words = normalize_for_error_rate(reference_text)
    hyp_words = normalize_for_error_rate(hypothesis_text)
    ref_chars = list("".join(ref_words))
    hyp_chars = list("".join(hyp_words))
    word_errors = levenshtein_distance(ref_words, hyp_words)
    char_errors = levenshtein_distance(ref_chars, hyp_chars)
    return {
        "wer": float(word_errors / max(1, len(ref_words))),
        "cer": float(char_errors / max(1, len(ref_chars))),
        "word_errors": int(word_errors),
        "char_errors": int(char_errors),
        "reference_word_count": int(len(ref_words)),
        "hypothesis_word_count": int(len(hyp_words)),
        "reference_char_count": int(len(ref_chars)),
        "hypothesis_char_count": int(len(hyp_chars)),
    }


def load_unique_wav_names(segmentation_path: Path) -> list[str]:
    rows = yaml.safe_load(segmentation_path.read_text(encoding="utf-8")) or []
    seen: set[str] = set()
    ordered: list[str] = []
    for row in rows:
        wav_name = Path(str(row["wav"])).name
        if wav_name in seen:
            continue
        seen.add(wav_name)
        ordered.append(wav_name)
    return ordered


def iter_wavs(args: argparse.Namespace) -> list[Path]:
    if args.wavs:
        wavs = [Path(path) for path in args.wavs]
    else:
        wav_root = Path(args.wav_dir)
        wavs = [
            wav_root / wav_name
            for wav_name in load_unique_wav_names(Path(args.segmentation))
        ]
    if args.limit is not None:
        wavs = wavs[: int(args.limit)]
    missing = [str(path) for path in wavs if not path.exists()]
    if missing:
        raise SystemExit("Missing wavs:\n- " + "\n- ".join(missing))
    return wavs


def load_references_by_wav(
    *,
    segmentation_path: Path,
    reference_path: Path,
) -> dict[str, str]:
    if not segmentation_path.exists() or not reference_path.exists():
        return {}
    segments = yaml.safe_load(segmentation_path.read_text(encoding="utf-8")) or []
    references = reference_path.read_text(encoding="utf-8").splitlines()
    if len(segments) != len(references):
        raise ValueError(
            f"Reference mismatch: {segmentation_path} has {len(segments)} rows, "
            f"{reference_path} has {len(references)} lines."
        )
    grouped: dict[str, list[str]] = {}
    for segment, reference in zip(segments, references):
        wav_name = Path(str(segment["wav"])).name
        grouped.setdefault(wav_name, []).append(str(reference).strip())
    return {
        wav_name: normalize_spaces(" ".join(part for part in parts if part))
        for wav_name, parts in grouped.items()
    }


def lexical_word_rows(text: str, *, emitted_at_s: float | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for match in NONSPACE_RE.finditer(str(text)):
        rows.append(
            {
                "text": match.group(0),
                "start_time": None,
                "end_time": None,
                "emitted_at_s": emitted_at_s,
            }
        )
    return rows


class GemmaE4BASRLocalAgreement:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.model_name = str(args.model)
        self.max_model_len = int(args.max_model_len)
        self.prompt_budget_reserve_tokens = int(args.prompt_budget_reserve_tokens)
        self.max_new_tokens = int(args.max_new_tokens)
        self.processor = None
        self.tokenizer = None
        self.llm = None
        self.user_prompt_token_ids: list[int] = []
        self.audio_placeholder_token_count = 0
        self.audio_ms_per_token = GEMMA_AUDIO_MS_PER_TOKEN
        self.audio_seq_length = int(round(GEMMA_AUDIO_MAX_SECONDS * 1000.0 / self.audio_ms_per_token))
        self.max_audio_seconds = GEMMA_AUDIO_MAX_SECONDS

    def load(self) -> None:
        from transformers import AutoProcessor
        from vllm import LLM

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.tokenizer = self.processor.tokenizer
        self.audio_ms_per_token = float(
            getattr(self.processor, "audio_ms_per_token", self.audio_ms_per_token)
            or self.audio_ms_per_token
        )
        self.audio_seq_length = int(
            getattr(self.processor, "audio_seq_length", self.audio_seq_length)
            or self.audio_seq_length
        )
        self.max_audio_seconds = (
            float(self.audio_seq_length) * float(self.audio_ms_per_token) / 1000.0
        )
        if self.args.max_window_seconds is not None:
            self.max_audio_seconds = min(
                self.max_audio_seconds,
                float(self.args.max_window_seconds),
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": np.asarray([], dtype=np.float32)},
                    {"type": "text", "text": GEMMA_ASR_INSTRUCTION},
                ],
            }
        ]
        prompt_text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        self.user_prompt_token_ids = [
            int(token_id)
            for token_id in self.tokenizer.encode(prompt_text, add_special_tokens=False)
        ]
        self.audio_placeholder_token_count = sum(
            1
            for token_id in self.user_prompt_token_ids
            if int(token_id) == GEMMA_AUDIO_TOKEN_ID
        )
        if self.audio_placeholder_token_count <= 0:
            raise RuntimeError("Could not detect Gemma audio placeholder tokens.")

        compilation_config = {}
        if self.args.cudagraph_mode:
            compilation_config["cudagraph_mode"] = str(self.args.cudagraph_mode)
        if self.args.compilation_mode:
            compilation_config["mode"] = str(self.args.compilation_mode)
        if self.args.compile_cache_dir:
            compilation_config["cache_dir"] = str(
                Path(self.args.compile_cache_dir).expanduser().resolve()
            )

        self.llm = LLM(
            model=self.model_name,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=self.max_model_len,
            gpu_memory_utilization=float(self.args.gpu_memory_utilization),
            allowed_local_media_path=str(REPO_ROOT.resolve()),
            enforce_eager=bool(self.args.enforce_eager),
            enable_prefix_caching=bool(self.args.enable_prefix_caching),
            distributed_executor_backend=str(self.args.executor_backend),
            compilation_config=compilation_config or None,
        )

    def prefix_budget(self, *, audio_sample_count: int, sample_rate: int) -> int:
        duration_ms = float(audio_sample_count) * 1000.0 / float(sample_rate)
        audio_tokens = min(
            int(self.audio_seq_length),
            max(1, int(math.ceil(duration_ms / float(self.audio_ms_per_token)))),
        )
        non_audio_prompt_tokens = (
            len(self.user_prompt_token_ids) - int(self.audio_placeholder_token_count)
        )
        budget = (
            int(self.max_model_len)
            - int(self.prompt_budget_reserve_tokens)
            - int(non_audio_prompt_tokens)
            - int(audio_tokens)
            - int(self.max_new_tokens)
        )
        budget = max(0, int(budget))
        return min(budget, int(self.args.max_forced_prefix_tokens))

    def generate(
        self,
        *,
        audio: np.ndarray,
        sample_rate: int,
        forced_prefix_text: str,
        forced_prefix_token_ids: Sequence[int],
    ) -> dict[str, Any]:
        from vllm import SamplingParams

        if self.llm is None:
            raise RuntimeError("Gemma local-agreement ASR is not loaded.")

        prompt_token_ids = list(self.user_prompt_token_ids) + [
            int(token_id) for token_id in forced_prefix_token_ids
        ]
        audio = np.ascontiguousarray(np.asarray(audio, dtype=np.float32))
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=int(self.max_new_tokens),
            repetition_penalty=float(self.args.repetition_penalty),
            skip_special_tokens=True,
        )
        start = perf_counter()
        outputs = self.llm.generate(
            [
                {
                    "prompt_token_ids": prompt_token_ids,
                    "multi_modal_data": {"audio": [audio]},
                }
            ],
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        generate_s = perf_counter() - start
        if not outputs or not outputs[0].outputs:
            raise RuntimeError("vLLM Gemma E4B ASR produced no completion.")
        completion = outputs[0].outputs[0]
        generated_token_ids = [int(t) for t in completion.token_ids]
        generated_text = normalize_spaces(str(completion.text))
        return {
            "generated_text": generated_text,
            "generated_token_ids": generated_token_ids,
            "generated_token_count": int(len(generated_token_ids)),
            "forced_prefix_text": str(forced_prefix_text),
            "forced_prefix_token_count": int(len(forced_prefix_token_ids)),
            "prompt_token_count_local": int(len(prompt_token_ids)),
            "finish_reason": getattr(completion, "finish_reason", None),
            "generate_s": float(generate_s),
        }

    def capture_wav(self, wav_path: Path) -> dict[str, Any]:
        audio, sample_rate = load_wav(str(wav_path))
        chunk_size = max(1, int(sample_rate * float(self.args.chunk_ms) / 1000.0))
        max_window_samples = max(1, int(sample_rate * float(self.max_audio_seconds)))
        audio_duration_s = float(len(audio)) / float(sample_rate)

        rows: list[dict[str, Any]] = []
        committed_text = ""
        previous_hypothesis_text = ""
        effective_audio_processed_s = 0.0
        total_generate_s = 0.0
        processing_start = perf_counter()

        for chunk_idx, stop_sample in enumerate(
            range(chunk_size, len(audio) + chunk_size, chunk_size),
            start=1,
        ):
            stop_sample = min(stop_sample, len(audio))
            audio_processed_s = float(stop_sample) / float(sample_rate)
            is_final_chunk = stop_sample >= len(audio)
            if audio_processed_s < float(self.args.min_start_seconds) and not is_final_chunk:
                continue
            if self.args.max_chunks is not None and len(rows) >= int(self.args.max_chunks):
                break

            window_start_sample = max(0, stop_sample - max_window_samples)
            audio_window = audio[window_start_sample:stop_sample]
            prefix_budget = self.prefix_budget(
                audio_sample_count=len(audio_window),
                sample_rate=sample_rate,
            )
            forced_prefix_text, forced_prefix_token_ids = suffix_text_with_token_budget(
                self.tokenizer,
                committed_text,
                prefix_budget,
            )

            generated = self.generate(
                audio=audio_window,
                sample_rate=sample_rate,
                forced_prefix_text=forced_prefix_text,
                forced_prefix_token_ids=forced_prefix_token_ids,
            )
            total_generate_s += float(generated["generate_s"])
            effective_audio_processed_s += float(len(audio_window)) / float(sample_rate)

            continuation_text = str(generated["generated_text"])
            hypothesis_text = join_text(committed_text, continuation_text)
            append_only_conflict = not hypothesis_text.startswith(committed_text)

            if is_final_chunk:
                next_committed_text = hypothesis_text.strip()
                stable_prefix_text = next_committed_text
            elif previous_hypothesis_text:
                stable_prefix_text = whole_word_lcp(
                    previous_hypothesis_text,
                    hypothesis_text,
                )
                if not stable_prefix_text.startswith(committed_text):
                    stable_prefix_text = committed_text
                    append_only_conflict = True
                next_committed_text = stable_prefix_text
            else:
                stable_prefix_text = committed_text
                next_committed_text = committed_text

            old_word_count = len(split_words(committed_text))
            new_words = split_words(next_committed_text)[old_word_count:]
            did_commit = bool(new_words)
            committed_text = next_committed_text
            previous_hypothesis_text = hypothesis_text
            wallclock_s = perf_counter() - processing_start

            row = {
                "chunk_idx": int(chunk_idx),
                "audio_processed_s": float(audio_processed_s),
                "audio_window_start_s": float(window_start_sample) / float(sample_rate),
                "audio_window_duration_s": float(len(audio_window)) / float(sample_rate),
                "wallclock_s": float(wallclock_s),
                "is_final_chunk": bool(is_final_chunk),
                "hypothesis_text": hypothesis_text,
                "committed_text": committed_text,
                "stable_prefix_text": stable_prefix_text,
                "did_commit_segment": did_commit,
                "new_committed_words": [
                    {
                        "text": word,
                        "start_time": None,
                        "end_time": None,
                        "emitted_at_s": float(audio_processed_s),
                    }
                    for word in new_words
                ],
                "words": lexical_word_rows(hypothesis_text),
                "append_only_conflict": bool(append_only_conflict),
                "alignment_count_match": False,
                **generated,
            }
            rows.append(row)
            if self.args.checkpoint_every_chunk:
                partial_path = Path(self.args.output_dir) / "captures" / f"{wav_path.stem}.partial"
                write_json(
                    partial_path,
                    self.build_capture_payload(
                        wav_path=wav_path,
                        rows=rows,
                        audio_duration_s=audio_duration_s,
                        processing_s=wallclock_s,
                        effective_audio_processed_s=effective_audio_processed_s,
                        total_generate_s=total_generate_s,
                        final_text=committed_text or hypothesis_text,
                    ),
                )

            if is_final_chunk:
                break

        if not rows:
            raise RuntimeError(f"Capture produced no rows for {wav_path}.")

        processing_s = float(rows[-1]["wallclock_s"])
        final_text = str(rows[-1]["committed_text"] or rows[-1]["hypothesis_text"]).strip()
        return self.build_capture_payload(
            wav_path=wav_path,
            rows=rows,
            audio_duration_s=audio_duration_s,
            processing_s=processing_s,
            effective_audio_processed_s=effective_audio_processed_s,
            total_generate_s=total_generate_s,
            final_text=final_text,
        )

    def build_capture_payload(
        self,
        *,
        wav_path: Path,
        rows: list[dict[str, Any]],
        audio_duration_s: float,
        processing_s: float,
        effective_audio_processed_s: float,
        total_generate_s: float,
        final_text: str,
    ) -> dict[str, Any]:
        return {
            "schema_version": "gemma_e4b_local_agreement_capture_v1",
            "strategy": {
                "backend": "gemma_e4b_vllm_greedy",
                "decode_mode": "sliding_audio_window_with_forced_text_suffix",
                "stable_prefix_rule": "whole_word_lcp_between_consecutive_hypotheses",
                "commit_rule": "append_only_local_agreement",
                "timestamp_rule": "none",
                "uses_alignatt": False,
            },
            "wav_path": str(wav_path),
            "wav_name": wav_path.name,
            "model": self.model_name,
            "chunk_ms": int(self.args.chunk_ms),
            "min_start_seconds": float(self.args.min_start_seconds),
            "max_audio_window_seconds": float(self.max_audio_seconds),
            "max_new_tokens": int(self.max_new_tokens),
            "max_model_len": int(self.max_model_len),
            "max_forced_prefix_tokens": int(self.args.max_forced_prefix_tokens),
            "audio_duration_s": float(audio_duration_s),
            "effective_audio_processed_s": float(effective_audio_processed_s),
            "processing_s": float(processing_s),
            "generate_s": float(total_generate_s),
            "rtf_wallclock": (
                0.0 if audio_duration_s <= 0.0 else float(processing_s / audio_duration_s)
            ),
            "rtf_effective_window_audio": (
                0.0
                if effective_audio_processed_s <= 0.0
                else float(processing_s / effective_audio_processed_s)
            ),
            "final_text": str(final_text).strip(),
            "final_words": lexical_word_rows(final_text),
            "local_agreement_final_text": str(final_text).strip(),
            "local_agreement_committed_words": lexical_word_rows(final_text),
            "chunks": rows,
        }


def add_metrics(
    *,
    capture: dict[str, Any],
    references_by_wav: dict[str, str],
) -> dict[str, Any] | None:
    wav_name = str(capture.get("wav_name") or "")
    reference = references_by_wav.get(wav_name)
    if reference is None:
        return None
    metrics = error_rates(reference, str(capture.get("final_text") or ""))
    capture["reference_text"] = reference
    capture["metrics"] = metrics
    capture["wer"] = float(metrics["wer"])
    capture["cer"] = float(metrics["cer"])
    return metrics


def build_manifest(
    *,
    args: argparse.Namespace,
    captures_dir: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    metric_rows = [row for row in rows if row.get("metrics")]
    ref_words = sum(float(row["metrics"]["reference_word_count"]) for row in metric_rows)
    weighted_wer = (
        None
        if ref_words <= 0.0
        else float(
            sum(
                float(row["metrics"]["wer"])
                * float(row["metrics"]["reference_word_count"])
                for row in metric_rows
            )
            / ref_words
        )
    )
    weighted_cer_den = sum(
        float(row["metrics"]["reference_char_count"]) for row in metric_rows
    )
    weighted_cer = (
        None
        if weighted_cer_den <= 0.0
        else float(
            sum(
                float(row["metrics"]["cer"])
                * float(row["metrics"]["reference_char_count"])
                for row in metric_rows
            )
            / weighted_cer_den
        )
    )
    return {
        "schema_version": "gemma_e4b_local_agreement_capture_manifest_v1",
        "wav_count": len(rows),
        "metric_wav_count": len(metric_rows),
        "chunk_ms": int(args.chunk_ms),
        "min_start_seconds": float(args.min_start_seconds),
        "segmentation": str(args.segmentation),
        "reference": str(args.reference),
        "wav_dir": str(args.wav_dir),
        "captures_dir": str(captures_dir),
        "weighted_wer": weighted_wer,
        "weighted_cer": weighted_cer,
        "mean_wer": (
            None
            if not metric_rows
            else float(sum(float(row["metrics"]["wer"]) for row in metric_rows) / len(metric_rows))
        ),
        "mean_cer": (
            None
            if not metric_rows
            else float(sum(float(row["metrics"]["cer"]) for row in metric_rows) / len(metric_rows))
        ),
        "mean_rtf_wallclock": (
            None if not rows else float(sum(float(row["rtf_wallclock"]) for row in rows) / len(rows))
        ),
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=gemma_model_name)
    parser.add_argument("--wav-dir", type=Path, default=REPO_ROOT / "data/devset/audio")
    parser.add_argument("--segmentation", type=Path, default=REPO_ROOT / "data/devset/audio-segments.yaml")
    parser.add_argument("--reference", type=Path, default=REPO_ROOT / "data/devset/ref/en.txt")
    parser.add_argument("--wavs", nargs="*", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--chunk-ms", type=int, default=800)
    parser.add_argument("--min-start-seconds", type=float, default=2.0)
    parser.add_argument("--max-window-seconds", type=float, default=None)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--prompt-budget-reserve-tokens", type=int, default=32)
    parser.add_argument("--max-forced-prefix-tokens", type=int, default=192)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--executor-backend", default="mp")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--cudagraph-mode", default="full")
    parser.add_argument("--compilation-mode", default=None)
    parser.add_argument("--compile-cache-dir", default=None)
    parser.add_argument(
        "--checkpoint-every-chunk",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wavs = iter_wavs(args)
    references_by_wav = load_references_by_wav(
        segmentation_path=Path(args.segmentation),
        reference_path=Path(args.reference),
    )
    captures_dir = Path(args.output_dir) / "captures"
    captures_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[load] Gemma E4B ASR model={args.model} wavs={len(wavs)} "
        f"output={args.output_dir}",
        flush=True,
    )
    asr = GemmaE4BASRLocalAgreement(args)
    asr.load()
    print(
        f"[load] ready audio_cap={asr.max_audio_seconds:.1f}s "
        f"audio_seq_length={asr.audio_seq_length} "
        f"audio_ms_per_token={asr.audio_ms_per_token}",
        flush=True,
    )

    manifest_rows: list[dict[str, Any]] = []
    for idx, wav_path in enumerate(wavs, start=1):
        out_path = captures_dir / f"{wav_path.stem}.json"
        if args.resume and out_path.exists():
            capture = json.loads(out_path.read_text(encoding="utf-8"))
            print(f"[skip] ({idx}/{len(wavs)}) {wav_path.name}", flush=True)
        else:
            print(f"[capture] ({idx}/{len(wavs)}) {wav_path.name}", flush=True)
            capture = asr.capture_wav(wav_path)
            add_metrics(capture=capture, references_by_wav=references_by_wav)
            write_json(out_path, capture)
            partial_path = captures_dir / f"{wav_path.stem}.partial"
            if partial_path.exists():
                partial_path.unlink()

        metrics = dict(capture.get("metrics") or {})
        row = {
            "wav_name": wav_path.name,
            "wav_path": str(wav_path),
            "chunk_row_count": len(list(capture.get("chunks") or [])),
            "audio_duration_s": float(capture["audio_duration_s"]),
            "effective_audio_processed_s": float(capture["effective_audio_processed_s"]),
            "processing_s": float(capture["processing_s"]),
            "generate_s": float(capture.get("generate_s", 0.0)),
            "rtf_wallclock": float(capture["rtf_wallclock"]),
            "rtf_effective_window_audio": float(capture["rtf_effective_window_audio"]),
            "final_word_count": len(split_words(str(capture.get("final_text") or ""))),
            "append_only_conflict_chunk_count": int(
                sum(bool(row.get("append_only_conflict")) for row in capture.get("chunks", []))
            ),
            "metrics": metrics,
        }
        manifest_rows.append(row)
        metric_suffix = ""
        if metrics:
            metric_suffix = (
                f" wer={100.0 * float(metrics['wer']):.2f}%"
                f" cer={100.0 * float(metrics['cer']):.2f}%"
            )
        print(
            f"    chunks={row['chunk_row_count']} "
            f"rtf_wall={row['rtf_wallclock']:.2f}"
            f"{metric_suffix}",
            flush=True,
        )

        manifest = build_manifest(args=args, captures_dir=captures_dir, rows=manifest_rows)
        write_json(Path(args.output_dir) / "manifest.json", manifest)

    print(f"[done] wrote {Path(args.output_dir) / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
