#!/usr/bin/env python3
"""Detect translation alignment heads with Gemma-style causal decoders.

Implements the algorithm from "Token Alignment Heads: Unveiling Attention's
Role in LLM Multilingual Translation" (ICLR 2026 submission).

Pipeline:
  1. Load a parallel corpus for a chosen translation direction.
  2. Use GPT-5-mini to annotate word-level alignments for each pair.
  3. Filter to high-quality anchor alignments.
  4. Run the translation model, collect attention maps.
  5. Score every attention head with the paper's Translation Score (TS).
  6. Output:
     - word_alignments_<direction>.json   (GPT-5-mini word mappings)
     - translation_heads_<model>_<direction>.json  (head scores & ranked heads)

Usage:
  # Full pipeline (align + detect) for Czech -> English:
  python detect_translation_heads.py --direction cs-en \
    --src-path /path/to/corpus.cs --tgt-path /path/to/corpus.en \
    --dataset-name czeng2.0

  # Alignment only:
  python detect_translation_heads.py --step align --direction cs-en \
    --src-path /path/to/corpus.cs --tgt-path /path/to/corpus.en

  # Detection only (reuse existing alignments):
  python detect_translation_heads.py --step detect --direction cs-en \
    --alignment-file word_alignments_cs-en.json

  # Different model / language pair:
  python detect_translation_heads.py --direction en-de \
    --model google/gemma-4-E4B-it
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_MT_MODEL = "google/gemma-4-E4B-it"
DEFAULT_ALIGNMENT_MODEL = "gpt-5-mini"
DEFAULT_DIRECTION = "cs-en"
PAPER_THRESHOLD = 0.1
MILMMT_EXPECTED_NUM_LAYERS = 34
MILMMT_EXPECTED_NUM_HEADS = 8

HYMT_PROMPTS = {
    "en-zh": "将以下文本翻译为中文，注意只需要输出翻译后的结果，不要额外解释：\n\n",
    "en-de": "Translate the following text into German, please only output the translated result without additional explanation:\n\n",
    "en-it": "Translate the following text into Italian, please only output the translated result without additional explanation:\n\n",
    "en-fr": "Translate the following text into French, please only output the translated result without additional explanation:\n\n",
    "cs-en": "Translate the following text into English, please only output the translated result without additional explanation:\n\n",
    "en-cs": "Translate the following text into Czech, please only output the translated result without additional explanation:\n\n",
}
HYMT_PLACEHOLDER = "<｜hy_place▁holder▁no▁0｜>"

LANGUAGE_NAMES = {
    "en": "English",
    "zh": "Chinese",
    "de": "German",
    "it": "Italian",
    "fr": "French",
    "cs": "Czech",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
}

# No corpus is baked into the repo on purpose: the detector should run on
# challenge-legal parallel text chosen explicitly by the caller, not silently
# on stale or evaluation data.
DATASET_PATHS: dict[str, list[dict[str, str]]] = {}

CORPUS_HINTS = {
    "cs-en": (
        "For IWSLT 2026 Czech->English, pass challenge-legal parallel Czech-English "
        "text explicitly. Prefer CzEng 2.0 as the primary corpus, optionally "
        "supplemented with Europarl and VoxPopuli cs->en translated data. "
        "Keep the official 2026 dev set for validation rather than head discovery."
    ),
}

LANGUAGE_STOPWORDS = {
    "en": {
        "a", "an", "the", "of", "to", "for", "and", "or", "is", "are", "be", "am",
        "in", "on", "at", "with", "by", "from", "that", "this", "these", "those",
        "we", "our", "you", "your", "it", "its", "they", "their", "he", "she",
        "i", "me", "my", "mine", "us", "them", "his", "her", "hers",
        "as", "if", "then", "than", "into", "onto", "over", "under", "through",
        "which", "who", "whom", "what", "when", "where", "why", "how", "not",
        "no", "do", "does", "did", "done", "have", "has", "had", "having",
        "will", "would", "should", "could", "may", "might", "can", "must",
        "also", "very", "much", "more", "most", "such", "only", "just",
        "there", "here", "because", "while", "during", "about",
    },
    "cs": {
        "a", "aby", "ale", "ani", "bez", "by", "byl", "byla", "byli", "bylo",
        "byly", "být", "co", "do", "ho", "i", "ja", "jak", "je", "jeho", "její",
        "jejich", "jen", "jsme", "jsem", "jste", "jsou", "k", "kde", "kdo",
        "která", "které", "který", "má", "mají", "máme", "mi", "mne", "mě", "na",
        "nad", "ne", "nebo", "něco", "některé", "některý", "o", "od", "pak", "po",
        "pod", "pro", "při", "s", "se", "si", "své", "ta", "tak", "tato", "te",
        "ten", "tento", "to", "tohle", "tom", "tu", "tuto", "ty", "u", "už", "v",
        "ve", "vy", "z", "za", "ze", "že",
    },
    "de": {
        "aber", "als", "am", "an", "auch", "auf", "aus", "bei", "bin", "bis",
        "bist", "da", "damit", "das", "dass", "dein", "deine", "dem", "den",
        "der", "des", "die", "dies", "diese", "dieser", "doch", "du", "durch",
        "ein", "eine", "einer", "eines", "er", "es", "für", "hat", "habe",
        "haben", "hier", "ich", "ihr", "ihre", "im", "in", "ist", "ja", "mit",
        "nach", "nicht", "noch", "nur", "oder", "sein", "seine", "sich", "sie",
        "so", "um", "und", "uns", "unter", "vom", "von", "vor", "war", "was",
        "weil", "wenn", "wer", "wie", "wir", "wird", "wo", "zu", "zum", "zur",
    },
    "it": {
        "a", "ad", "al", "alla", "allo", "anche", "che", "chi", "con", "da",
        "dal", "dalla", "dello", "dei", "del", "della", "di", "e", "era", "eri",
        "fra", "gli", "ha", "hai", "hanno", "ho", "i", "il", "in", "io", "la",
        "le", "lei", "lo", "loro", "ma", "mi", "nei", "nel", "nella", "noi",
        "non", "o", "per", "più", "poi", "se", "sei", "si", "sia", "sono",
        "su", "sul", "sulla", "tra", "tu", "un", "una", "uno", "vi",
    },
}

LANGUAGE_WEAK_ANCHORS = {
    "en": {
        "another", "aren't", "briefly", "current", "different", "each", "first",
        "find", "hi", "i'll", "i'm", "introduce", "other", "question",
        "questions", "second", "see", "some", "specific", "today", "two",
        "usually", "we'll", "will",
    },
    "cs": {
        "ano", "asi", "dnes", "jen", "jistě", "možná", "například", "nějaký",
        "nějaké", "nějakou", "někdo", "některé", "některý", "otázka", "otázky",
        "podobně", "potom", "prostě", "první", "příklad", "spíš", "také",
        "teď", "tedy", "trochu", "už", "vlastně",
    },
    "de": {
        "aktuell", "also", "andere", "beispiel", "einige", "erste", "erstens",
        "frage", "fragen", "heute", "kurz", "meistens", "noch", "spezifisch",
        "weitere", "zweite", "zweitens",
    },
    "it": {
        "alcuni", "altra", "altre", "altro", "brevemente", "domanda", "domande",
        "oggi", "primo", "secondo", "solito", "specifico",
    },
    "zh": {
        "的", "了", "是", "在", "有", "和", "但", "将", "另", "托", "通常",
        "可以", "一个", "一些", "每个", "所有", "首先", "因此", "当前", "其中",
    },
}
HAN_RE = re.compile(r"[\u4e00-\u9fff]")
ASCII_ALNUM_RE = re.compile(r"[A-Za-z0-9]")
GREEK_RE = re.compile(r"[α-ωΑ-ΩλΛ]")

# ---------------------------------------------------------------------------
# JSON Schema for GPT Structured Output
# ---------------------------------------------------------------------------

JOINT_ALIGNMENT_SCHEMA = {
    "name": "word_alignment",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "source_words": {
                "type": "array",
                "items": {"type": "string"},
            },
            "target_words": {
                "type": "array",
                "items": {"type": "string"},
            },
            "alignments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "source_start": {"type": "integer", "minimum": 0},
                        "source_end": {"type": "integer", "minimum": 0},
                        "target_start": {"type": "integer", "minimum": 0},
                        "target_end": {"type": "integer", "minimum": 0},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                    "required": [
                        "source_start", "source_end",
                        "target_start", "target_end",
                        "confidence",
                    ],
                },
            },
        },
        "required": ["source_words", "target_words", "alignments"],
    },
}

JOINT_ALIGNMENT_SYSTEM_PROMPT = """You annotate bilingual word alignments in one step.

Return JSON only.
- Copy exact source and target words from the original sentences.
- Do not normalize, rewrite, or explain.
- Do not emit spaces or newline characters as tokens.
- Keep punctuation and quote marks as separate tokens when possible.
- Never add or remove quote marks to make punctuation look balanced.
- Copy only quote marks that literally appear in the sentence.
- For Chinese, segment into lexical words and punctuation.
- Use half-open spans: start inclusive, end exclusive.
- Example: a single aligned word at index 3 must be written as start=3, end=4.
- The token lists must exactly cover all non-whitespace characters from each sentence.
- Only include confident semantic alignments between contiguous spans.
- Provide a confidence score for each alignment (0.0 to 1.0).
- Words may be left unaligned.
"""


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text.strip())


def get_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    key_file = Path.home() / ".openai_api_key"
    if key_file.exists():
        return key_file.read_text().strip()
    raise RuntimeError(
        "OpenAI API key not found. Set OPENAI_API_KEY env var or create ~/.openai_api_key"
    )


# ---------------------------------------------------------------------------
# Step 1: Load parallel corpus
# ---------------------------------------------------------------------------

def load_parallel_corpus(
    direction: str,
    src_path: str | None = None,
    tgt_path: str | None = None,
    dataset_name: str | None = None,
    max_pairs: int = 0,
) -> list[dict]:
    """Load parallel sentence pairs.

    If src_path/tgt_path are given, use those. Otherwise, look up default
    dataset paths for the given direction.
    """
    if src_path and tgt_path:
        sources = [{"name": dataset_name or "custom", "src": src_path, "tgt": tgt_path}]
    elif direction in DATASET_PATHS:
        sources = DATASET_PATHS[direction]
    else:
        hint = CORPUS_HINTS.get(direction)
        suffix = f" {hint}" if hint else ""
        raise ValueError(
            f"No default dataset for direction {direction!r}. "
            f"Provide --src-path and --tgt-path explicitly.{suffix}"
        )

    pairs: list[dict] = []
    for source in sources:
        src_p = Path(source["src"])
        tgt_p = Path(source["tgt"])
        if not src_p.is_absolute():
            src_p = REPO_ROOT / src_p
        if not tgt_p.is_absolute():
            tgt_p = REPO_ROOT / tgt_p
        if not src_p.exists():
            raise FileNotFoundError(f"Source file not found: {src_p}")
        if not tgt_p.exists():
            raise FileNotFoundError(f"Target file not found: {tgt_p}")

        src_lines = src_p.read_text(encoding="utf-8").splitlines()
        tgt_lines = tgt_p.read_text(encoding="utf-8").splitlines()
        if len(src_lines) != len(tgt_lines):
            raise ValueError(
                f"Line count mismatch in {source['name']}: "
                f"{len(src_lines)} src vs {len(tgt_lines)} tgt"
            )
        for i, (src, tgt) in enumerate(zip(src_lines, tgt_lines)):
            src_text = normalize_text(src)
            tgt_text = normalize_text(tgt)
            if not src_text or not tgt_text:
                continue
            pairs.append({
                "pair_id": len(pairs),
                "dataset": source["name"],
                "line_idx": i,
                "direction": direction,
                "source_text": src_text,
                "target_text": tgt_text,
            })
    if max_pairs > 0:
        pairs = pairs[:max_pairs]
    print(f"Loaded {len(pairs)} parallel pairs for {direction}", flush=True)
    return pairs


def load_alignment_rows(path: str | Path) -> list[dict]:
    """Load alignment rows from either JSONL or a JSON array."""
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON list in {path}")
        return payload
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Step 2: GPT-5-mini word alignment annotation
# ---------------------------------------------------------------------------

def call_openai_response(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    schema: dict,
    max_completion_tokens: int = 16384,
    timeout_s: float = 120.0,
) -> dict:
    url = "https://api.openai.com/v1/responses"
    body = {
        "model": model,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema["name"],
                "schema": schema["schema"],
                "strict": True,
            },
        },
        "reasoning": {"effort": "low"},
        "max_output_tokens": max_completion_tokens,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def extract_response_json(response: dict) -> dict:
    status = response.get("status", "unknown")
    if status != "completed":
        raise ValueError(
            f"API response status={status}: "
            f"{response.get('incomplete_details') or response.get('error') or 'unknown reason'}"
        )
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") != "output_text":
                continue
            text = content.get("text", "")
            if text.strip():
                return json.loads(text)
    raise ValueError("No JSON output found in API response")


def locate_words_in_text(text: str, words: list[str]) -> list[dict]:
    """Find the character positions of each word in the original text.

    Returns list of {text, start_char, end_char} dicts.
    """
    result = []
    search_from = 0
    for word in words:
        idx = text.find(word, search_from)
        if idx < 0:
            # Try case-insensitive or relaxed search
            idx = text.lower().find(word.lower(), search_from)
        if idx < 0:
            # Fallback: search from the beginning
            idx = text.find(word)
        if idx < 0:
            idx = text.lower().find(word.lower())
        if idx < 0:
            # Last resort: put at current position
            idx = search_from
        result.append({
            "text": word,
            "start_char": idx,
            "end_char": idx + len(word),
        })
        search_from = idx + len(word)
    return result


def coerce_alignment_rows(rows: list[dict]) -> list[dict]:
    """Normalize either raw JSONL rows or clean JSON rows to the raw row schema."""
    normalized = []
    for row in rows:
        source_words = row.get("source_words", [])
        target_words = row.get("target_words", [])
        if source_words and isinstance(source_words[0], dict):
            normalized.append(row)
            continue

        source_spans = locate_words_in_text(row["source_text"], list(source_words))
        target_spans = locate_words_in_text(row["target_text"], list(target_words))
        alignments = []
        for alignment in row.get("alignments", []):
            if "source_start" in alignment:
                ss = int(alignment["source_start"])
                se = int(alignment["source_end"])
                ts = int(alignment["target_start"])
                te = int(alignment["target_end"])
                conf = float(alignment.get("confidence", 1.0))
                src_text = alignment.get(
                    "source_text",
                    " ".join(word["text"] for word in source_spans[ss:se]),
                )
                tgt_text = alignment.get(
                    "target_text",
                    " ".join(word["text"] for word in target_spans[ts:te]),
                )
            else:
                ss, se = map(int, alignment["source_span"])
                ts, te = map(int, alignment["target_span"])
                conf = float(alignment.get("confidence", 1.0))
                src_text = str(alignment.get("source_words", "")).strip()
                tgt_text = str(alignment.get("target_words", "")).strip()
            alignments.append(
                {
                    "source_start": ss,
                    "source_end": se,
                    "target_start": ts,
                    "target_end": te,
                    "confidence": conf,
                    "source_text": src_text,
                    "target_text": tgt_text,
                }
            )

        normalized.append(
            {
                **row,
                "source_words": source_spans,
                "target_words": target_spans,
                "alignments": alignments,
            }
        )
    return normalized


def annotate_pair(
    pair: dict,
    api_key: str,
    model: str,
) -> dict:
    """Call GPT-5-mini to get word tokenization + alignment for one pair."""
    direction = pair["direction"]
    user_prompt = (
        "Return exact source_words, exact target_words, and confident alignments for "
        "this JSON object.\n"
        "Copy words exactly from the JSON strings below.\n"
        "Do not repair unmatched quotes or punctuation.\n\n"
        + json.dumps({
            "direction": direction,
            "source_sentence": pair["source_text"],
            "target_sentence": pair["target_text"],
        }, ensure_ascii=False)
        + "\n"
    )

    response = call_openai_response(
        api_key=api_key,
        model=model,
        system_prompt=JOINT_ALIGNMENT_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        schema=JOINT_ALIGNMENT_SCHEMA,
    )
    payload = extract_response_json(response)

    source_words_raw = payload.get("source_words", [])
    target_words_raw = payload.get("target_words", [])
    alignments_raw = payload.get("alignments", [])

    # Locate words in original text
    source_spans = locate_words_in_text(pair["source_text"], source_words_raw)
    target_spans = locate_words_in_text(pair["target_text"], target_words_raw)

    # Validate and enrich alignments
    valid_alignments = []
    for aln in alignments_raw:
        ss, se = int(aln["source_start"]), int(aln["source_end"])
        ts, te = int(aln["target_start"]), int(aln["target_end"])
        conf = float(aln["confidence"])

        # Fix inclusive indexing (LLM sometimes returns inclusive end)
        if se <= ss:
            se = ss + 1
        if te <= ts:
            te = ts + 1

        if ss < 0 or se > len(source_spans) or ts < 0 or te > len(target_spans):
            continue
        if conf < 0.5:
            continue

        valid_alignments.append({
            "source_start": ss,
            "source_end": se,
            "target_start": ts,
            "target_end": te,
            "confidence": conf,
            "source_text": " ".join(w["text"] for w in source_spans[ss:se]),
            "target_text": " ".join(w["text"] for w in target_spans[ts:te]),
        })

    return {
        **pair,
        "llm_model": model,
        "source_words": source_spans,
        "target_words": target_spans,
        "alignments": valid_alignments,
    }


def _annotate_one(pair: dict, api_key: str, model: str) -> dict | None:
    """Annotate a single pair with retries. Returns row or None on failure."""
    pid = int(pair["pair_id"])
    for attempt in range(1, 5):
        try:
            return annotate_pair(pair, api_key, model)
        except Exception as exc:
            print(f"  pair_id={pid} attempt={attempt} failed: {exc}", flush=True)
            time.sleep(min(10.0, 1.5 * attempt))
    print(f"  SKIP pair_id={pid} after 4 attempts", flush=True)
    return None


def annotate_corpus(
    pairs: list[dict],
    api_key: str,
    model: str,
    output_path: Path,
    workers: int = 20,
) -> list[dict]:
    """Annotate all pairs with word alignments using concurrent workers.

    Supports resume: skips pair_ids already in the output file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load already-completed pairs
    completed_ids: set[int] = set()
    if output_path.exists():
        with output_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    completed_ids.add(int(row["pair_id"]))
        print(f"Resuming: {len(completed_ids)} pairs already annotated", flush=True)

    pending = [p for p in pairs if int(p["pair_id"]) not in completed_ids]
    if not pending:
        print("All pairs already annotated.", flush=True)
    else:
        print(f"Annotating {len(pending)} remaining pairs with {workers} workers...", flush=True)
        write_lock = threading.Lock()
        done_count = [len(completed_ids)]

        def _process(pair):
            row = _annotate_one(pair, api_key, model)
            if row is not None:
                line = json.dumps(row, ensure_ascii=False) + "\n"
                with write_lock:
                    with output_path.open("a", encoding="utf-8") as out_f:
                        out_f.write(line)
                    done_count[0] += 1
                    if done_count[0] % 50 == 0 or done_count[0] == 1:
                        print(
                            f"  Progress: {done_count[0]}/{len(pairs)} "
                            f"(links={len(row['alignments'])})",
                            flush=True,
                        )
            return row

        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(_process, pending))

    print(f"Annotation complete: {done_count[0] if pending else len(completed_ids)} total", flush=True)

    # Reload all rows in order
    all_rows = []
    with output_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_rows.append(json.loads(line))
    return all_rows


# ---------------------------------------------------------------------------
# Step 3: Filter alignments to high-quality anchors
# ---------------------------------------------------------------------------

def _normalize_anchor_token(token: str) -> str:
    return token.strip().strip("\"'`\u201c\u201d\u2018\u2019()[]{}<>.,:;!?").lower()


def _normalized_anchor_tokens(text: str) -> list[str]:
    return [
        token
        for token in (_normalize_anchor_token(part) for part in text.split())
        if token
    ]


def _has_anchor_signal(text: str) -> bool:
    upper_chars = [ch for ch in text if ch.isupper()]
    return (
        any(ch.isdigit() for ch in text)
        or "-" in text
        or GREEK_RE.search(text) is not None
        or len(upper_chars) >= 2
        or any(ch.isupper() for ch in text[1:] if ch.isalpha())
    )


def _is_punct_only(text: str) -> bool:
    text = text.strip()
    if not text:
        return True
    return not (ASCII_ALNUM_RE.search(text) or HAN_RE.search(text) or GREEK_RE.search(text))


def _span_is_function_only(text: str, lang_code: str) -> bool:
    stopwords = LANGUAGE_STOPWORDS.get(lang_code, set())
    if not stopwords:
        return False
    parts = _normalized_anchor_tokens(text)
    if not parts:
        return True
    if any(_has_anchor_signal(t) for t in parts):
        return False
    return all(t in stopwords for t in parts)


def _span_is_weak_anchor(text: str, lang_code: str) -> bool:
    weak_anchors = LANGUAGE_WEAK_ANCHORS.get(lang_code, set())
    stopwords = LANGUAGE_STOPWORDS.get(lang_code, set())
    parts = _normalized_anchor_tokens(text)
    if not parts:
        return True
    if any(_has_anchor_signal(t) for t in parts):
        return False
    if weak_anchors:
        return all(t in weak_anchors or t in stopwords for t in parts)
    return bool(stopwords) and all(t in stopwords for t in parts)


def keep_anchor_alignment(alignment: dict, direction: str) -> bool:
    """Filter a single alignment to decide if it's a reliable anchor.

    Uses strict anchor-based filtering: only keeps named entities, technical
    terms, numbers, and other high-signal alignments that serve as reliable
    position anchors for scoring attention heads.
    """
    src_lang, tgt_lang = direction.split("-", 1)
    src_text = alignment.get("source_text", "").strip()
    tgt_text = alignment.get("target_text", "").strip()
    if not src_text or not tgt_text:
        return False

    # Identical source=target with anchor signals: always keep (e.g. "SimulST")
    if src_text == tgt_text and _has_anchor_signal(src_text):
        return True

    # Reject pure punctuation
    if _is_punct_only(src_text) or _is_punct_only(tgt_text):
        return False

    # Span length limits
    ss, se = alignment["source_start"], alignment["source_end"]
    ts, te = alignment["target_start"], alignment["target_end"]
    if (se - ss) > 3 or (te - ts) > 4:
        return False

    # Reject function words
    if _span_is_function_only(src_text, src_lang):
        return False

    # Reject weak anchors
    if _span_is_weak_anchor(src_text, src_lang):
        return False
    if _span_is_weak_anchor(tgt_text, tgt_lang):
        return False

    # Short token checks
    norm_src = _normalize_anchor_token(src_text)
    norm_tgt = _normalize_anchor_token(tgt_text)
    if len(norm_tgt) <= 1 and not _has_anchor_signal(src_text):
        return False
    if len(norm_src) <= 1 and not (
        _has_anchor_signal(src_text) or HAN_RE.search(tgt_text)
    ):
        return False

    return True


def filter_alignments(rows: list[dict], direction: str) -> list[dict]:
    """Filter alignment rows to keep only high-quality anchor alignments."""
    filtered_rows = []
    for row in rows:
        kept = [
            a for a in row.get("alignments", [])
            if keep_anchor_alignment(a, direction)
        ]
        filtered_rows.append({
            **row,
            "raw_alignment_count": len(row.get("alignments", [])),
            "filtered_alignment_count": len(kept),
            "alignments": kept,
        })

    total_raw = sum(r["raw_alignment_count"] for r in filtered_rows)
    total_kept = sum(r["filtered_alignment_count"] for r in filtered_rows)
    rows_with_alignments = sum(1 for r in filtered_rows if r["filtered_alignment_count"] > 0)
    print(
        f"Filtered alignments: {total_kept}/{total_raw} kept "
        f"({rows_with_alignments}/{len(filtered_rows)} pairs have anchors)",
        flush=True,
    )
    return filtered_rows


# ---------------------------------------------------------------------------
# Step 4: Detect translation heads
# ---------------------------------------------------------------------------

def project_char_span_to_token_indices(
    offsets: list[tuple[int, int]],
    start_char: int,
    end_char: int,
) -> list[int]:
    """Map a character span [start_char, end_char) to token indices."""
    indices = []
    for idx, (tok_start, tok_end) in enumerate(offsets):
        if tok_end <= start_char:
            continue
        if tok_start >= end_char:
            break
        if tok_start < end_char and tok_end > start_char:
            indices.append(idx)
    return indices


def build_translation_prompt(
    model_name: str,
    direction: str,
    source_text: str,
    tokenizer=None,
) -> str:
    """Build the prompt text used by the translation model."""
    lowered = model_name.lower()
    if "milmmt" in lowered:
        src_lang, tgt_lang = direction.split("-", 1)
        src_name = LANGUAGE_NAMES.get(src_lang, src_lang)
        tgt_name = LANGUAGE_NAMES.get(tgt_lang, tgt_lang)
        if src_name == "Chinese":
            src_name = "Chinese (Simplified)"
        if tgt_name == "Chinese":
            tgt_name = "Chinese (Simplified)"
        return (
            f"Translate this from {src_name} to {tgt_name}:\n"
            f"{src_name}: {source_text}\n"
            f"{tgt_name}:"
        )

    if "qwen" in lowered:
        src_lang, tgt_lang = direction.split("-", 1)
        src_name = LANGUAGE_NAMES.get(src_lang, src_lang)
        tgt_name = LANGUAGE_NAMES.get(tgt_lang, tgt_lang)
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a professional {src_name} to {tgt_name} translator. "
                    f"Output {tgt_name} only."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Translate the following {src_name} text into {tgt_name}:\n\n"
                    f"{source_text}"
                ),
            },
        ]
        if tokenizer is None:
            raise ValueError("tokenizer required for Qwen model")
        # Disable Qwen3 chain-of-thought so the forced target follows the prompt
        # directly (no <think> block); harmless for Qwen2-class templates.
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    if "gemma" in lowered:
        src_lang, tgt_lang = direction.split("-", 1)
        src_name = LANGUAGE_NAMES.get(src_lang, src_lang)
        tgt_name = LANGUAGE_NAMES.get(tgt_lang, tgt_lang)
        messages = [
            {
                "role": "user",
                "content": (
                    f"Translate the following {src_name} segment into {tgt_name} as a faithful written translation. "
                    "Output only the translation without explanation.\n\n"
                    f"{source_text}"
                ),
            },
        ]
        if tokenizer is None:
            raise ValueError("tokenizer required for Gemma model")
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # Default: legacy text-only prompt
    if direction not in HYMT_PROMPTS:
        raise ValueError(
            f"No prompt template for direction {direction!r} and model {model_name!r}. "
            f"Available: {sorted(HYMT_PROMPTS)}"
        )
    return HYMT_PROMPTS[direction] + source_text + HYMT_PLACEHOLDER


def compute_paper_translation_scores(
    full_argmax_by_head: list[list[int]],
    valid_source_global_by_target: list[set[int]],
) -> list[float]:
    """Compute the Translation Score (TS) as defined in the paper.

    TS_h = g_h / m, where:
      g_h = number of target positions where argmax(attention over FULL sequence)
            falls on a valid aligned source token position
      m = total number of target positions with valid alignment

    Key: the argmax is over the ENTIRE attention vector (all positions),
    not restricted to source positions only. This ensures only heads that
    truly attend to source tokens (rather than BOS/prompt tokens) score high.
    """
    if not valid_source_global_by_target:
        return [0.0 for _ in full_argmax_by_head]
    m = len(valid_source_global_by_target)
    scores = []
    for head_argmax in full_argmax_by_head:
        g_h = sum(
            1
            for target_idx, global_pos in enumerate(head_argmax)
            if global_pos in valid_source_global_by_target[target_idx]
        )
        scores.append(g_h / m)
    return scores


def detect_heads(
    *,
    model_name: str,
    direction: str,
    filtered_rows: list[dict],
    dtype_str: str = "bfloat16",
    device: str = "cuda:0",
    disable_cuda_warmup: bool = False,
    max_gpu_memory: str | None = None,
) -> dict:
    """Run the translation model and score attention heads.

    Returns the full result dict (same format as the paper).
    """
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, modeling_utils

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[dtype_str]

    rows = [r for r in filtered_rows if r.get("alignments")]
    if not rows:
        raise RuntimeError("No filtered alignments with valid anchors to score")

    print(f"Loading model {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    original_caching_allocator_warmup = None
    if disable_cuda_warmup and hasattr(modeling_utils, "caching_allocator_warmup"):
        original_caching_allocator_warmup = modeling_utils.caching_allocator_warmup
        modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
    load_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "attn_implementation": "eager",
        "low_cpu_mem_usage": True,
    }
    if max_gpu_memory:
        load_kwargs["device_map"] = "auto"
        load_kwargs["max_memory"] = {
            0: max_gpu_memory,
            "cpu": "256GiB",
        }
    else:
        load_kwargs["device_map"] = device

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs,
        )
    finally:
        if original_caching_allocator_warmup is not None:
            modeling_utils.caching_allocator_warmup = original_caching_allocator_warmup
    model.eval()

    config = model.config
    text_config = getattr(config, "text_config", config)
    num_layers = int(text_config.num_hidden_layers)
    num_heads = int(text_config.num_attention_heads)
    print(f"Model: {num_layers} layers x {num_heads} heads", flush=True)
    if "milmmt" in model_name.lower() and (
        num_layers != MILMMT_EXPECTED_NUM_LAYERS
        or num_heads != MILMMT_EXPECTED_NUM_HEADS
    ):
        raise RuntimeError(
            "MiLMMT head detection expected "
            f"{MILMMT_EXPECTED_NUM_LAYERS} layers x {MILMMT_EXPECTED_NUM_HEADS} heads, "
            f"got {num_layers} x {num_heads}."
        )

    score_sum = np.zeros((num_layers, num_heads), dtype=np.float64)
    score_count = np.zeros((num_layers, num_heads), dtype=np.int64)
    used_pairs = 0
    used_target_tokens = 0

    for row_idx, row in enumerate(rows):
        source_text = row["source_text"]
        target_text = row["target_text"]

        prompt_text = build_translation_prompt(
            model_name, direction, source_text, tokenizer=tokenizer,
        )
        full_text = prompt_text + target_text

        prompt_enc = tokenizer(
            prompt_text, add_special_tokens=False, return_offsets_mapping=True,
        )
        full_enc = tokenizer(
            full_text, add_special_tokens=False, return_offsets_mapping=True,
        )
        input_enc = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)

        prompt_offsets = [tuple(map(int, off)) for off in prompt_enc["offset_mapping"]]
        full_offsets = [tuple(map(int, off)) for off in full_enc["offset_mapping"]]

        # Find source and target regions
        source_char_start = prompt_text.rfind(source_text)
        if source_char_start < 0:
            continue
        source_char_end = source_char_start + len(source_text)
        target_char_start = len(prompt_text)
        target_char_end = len(full_text)

        source_token_positions = project_char_span_to_token_indices(
            prompt_offsets, source_char_start, source_char_end,
        )
        target_token_positions_global = project_char_span_to_token_indices(
            full_offsets, target_char_start, target_char_end,
        )
        if not source_token_positions or not target_token_positions_global:
            continue

        # Build local offset maps for source and target
        source_offsets = [
            (prompt_offsets[idx][0] - source_char_start,
             prompt_offsets[idx][1] - source_char_start)
            for idx in source_token_positions
        ]
        target_offsets = [
            (full_offsets[idx][0] - target_char_start,
             full_offsets[idx][1] - target_char_start)
            for idx in target_token_positions_global
        ]

        # Build alignment ground truth: target_local -> set of GLOBAL source token positions
        # We use global positions because the paper checks full-sequence argmax
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
            # Convert local source indices to global sequence positions
            src_global_set = set(source_token_positions[i] for i in src_token_local)
            for tgt_local_idx in tgt_token_local:
                valid_source_global_by_target_local.setdefault(
                    tgt_local_idx, set()
                ).update(src_global_set)

        if not valid_source_global_by_target_local:
            continue

        # Forward pass with attention output
        input_ids = input_enc["input_ids"].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_attentions=True)

        # Score each head using FULL-SEQUENCE argmax (paper's exact algorithm)
        valid_target_locals = sorted(valid_source_global_by_target_local)
        valid_source_global_by_target = [
            valid_source_global_by_target_local[tgt_local_idx]
            for tgt_local_idx in valid_target_locals
        ]
        tgt_token_positions = [
            target_token_positions_global[i] for i in valid_target_locals
        ]

        for layer_idx, attn in enumerate(outputs.attentions):
            if attn is None:
                continue
            # attn shape: (1, num_heads, seq_len, seq_len)
            # Full-sequence argmax: argmax over ALL positions, not just source
            tgt_attn = attn[0, :, tgt_token_positions, :]  # (num_heads, n_tgt, seq_len)
            full_argmax = tgt_attn.argmax(dim=-1).detach().cpu().tolist()
            ts_values = compute_paper_translation_scores(
                full_argmax_by_head=full_argmax,
                valid_source_global_by_target=valid_source_global_by_target,
            )
            score_sum[layer_idx, :] += np.asarray(ts_values, dtype=np.float64)
            score_count[layer_idx, :] += 1

        used_pairs += 1
        used_target_tokens += len(valid_target_locals)

        if (row_idx + 1) % 25 == 0 or row_idx == 0:
            print(
                f"  Scored {row_idx + 1}/{len(rows)} pairs "
                f"(used={used_pairs}, tokens={used_target_tokens})",
                flush=True,
            )

        # Free GPU memory
        del outputs
        torch.cuda.empty_cache()

    print(
        f"Scoring complete: {used_pairs} pairs, {used_target_tokens} target tokens",
        flush=True,
    )

    # Average scores
    with np.errstate(divide="ignore", invalid="ignore"):
        score_matrix = np.divide(
            score_sum, score_count,
            out=np.zeros_like(score_sum),
            where=score_count > 0,
        )

    # Rank all heads
    all_heads = []
    for layer in range(num_layers):
        for head in range(num_heads):
            all_heads.append({
                "layer": layer,
                "head": head,
                "ts": round(float(score_matrix[layer, head]), 6),
                "count": int(score_count[layer, head]),
            })
    all_heads.sort(key=lambda h: h["ts"], reverse=True)

    # Identify token alignment heads (TS > threshold)
    token_alignment_heads = [h for h in all_heads if h["ts"] > PAPER_THRESHOLD]

    return {
        "model": model_name,
        "direction": direction,
        "datasets": sorted({row.get("dataset", "unknown") for row in rows}),
        "num_layers": num_layers,
        "num_heads": num_heads,
        "used_pairs": used_pairs,
        "used_target_tokens": used_target_tokens,
        "paper_threshold": PAPER_THRESHOLD,
        "score_name": "paper_translation_score_argmax",
        "token_alignment_heads": token_alignment_heads,
        "all_heads_ranked": all_heads[:20],
        "ts_matrix": score_matrix.round(6).tolist(),
    }


def _top_head_map(result: dict, top_k: int) -> dict[tuple[int, int], float]:
    return {
        (int(head["layer"]), int(head["head"])): float(head["ts"])
        for head in result.get("token_alignment_heads", [])[:top_k]
    }


def run_head_stability_checks(
    *,
    model_name: str,
    direction: str,
    filtered_rows: list[dict],
    full_result: dict,
    split_count: int,
    split_size: int,
    top_k: int,
    dtype_str: str,
    device: str,
    seed: int,
    disable_cuda_warmup: bool = False,
    max_gpu_memory: str | None = None,
) -> list[dict]:
    rows = [row for row in filtered_rows if row.get("alignments")]
    if split_count <= 0 or not rows:
        return []

    rng = random.Random(seed)
    actual_split_size = split_size or max(1, len(rows) // 2)
    actual_split_size = min(actual_split_size, len(rows))
    full_top = _top_head_map(full_result, top_k)
    checks: list[dict] = []

    for split_idx in range(split_count):
        sample = rng.sample(rows, actual_split_size)
        split_result = detect_heads(
            model_name=model_name,
            direction=direction,
            filtered_rows=sample,
            dtype_str=dtype_str,
            device=device,
            disable_cuda_warmup=disable_cuda_warmup,
            max_gpu_memory=max_gpu_memory,
        )
        split_top = _top_head_map(split_result, top_k)
        overlap = sorted(set(full_top) & set(split_top))
        max_delta = (
            max(abs(split_top[key] - full_top[key]) for key in overlap)
            if overlap
            else None
        )
        checks.append(
            {
                "split_index": split_idx,
                "sample_size": actual_split_size,
                "used_pairs": split_result["used_pairs"],
                "used_target_tokens": split_result["used_target_tokens"],
                "overlap_with_full_top_k": len(overlap),
                "common_top_heads": [
                    {
                        "layer": key[0],
                        "head": key[1],
                        "full_ts": round(full_top[key], 6),
                        "split_ts": round(split_top[key], 6),
                        "abs_delta": round(abs(split_top[key] - full_top[key]), 6),
                    }
                    for key in overlap
                ],
                "max_abs_ts_delta_vs_full": (
                    round(max_delta, 6) if max_delta is not None else None
                ),
                "stable_vs_full": (
                    len(overlap) == min(top_k, len(full_top), len(split_top))
                    and max_delta is not None
                    and max_delta <= 0.03
                ),
                "top_heads": split_result.get("token_alignment_heads", [])[:top_k],
            }
        )

    return checks


def summarize_head_promotion_gate(
    result: dict,
    *,
    top_k: int = 8,
    min_ts: float = 0.1,
    required_stability_checks: int = 2,
    max_ts_delta: float = 0.03,
) -> dict:
    heads = list(result.get("token_alignment_heads", []))
    required_heads = min(max(0, top_k), len(heads))
    heads_above_threshold = sum(
        1 for head in heads[:required_heads] if float(head.get("ts", 0.0)) > min_ts
    )
    stability_checks = list(result.get("stability_checks", []))
    stable_checks = sum(1 for check in stability_checks if check.get("stable_vs_full"))
    max_observed_delta = max(
        (
            float(check["max_abs_ts_delta_vs_full"])
            for check in stability_checks
            if check.get("max_abs_ts_delta_vs_full") is not None
        ),
        default=None,
    )
    stability_ok = (
        required_stability_checks <= 0
        or (
            len(stability_checks) >= required_stability_checks
            and stable_checks == len(stability_checks)
            and (
                max_observed_delta is None
                or max_observed_delta <= max_ts_delta
            )
        )
    )
    return {
        "required_top_heads": required_heads,
        "heads_above_ts_threshold": heads_above_threshold,
        "min_ts_threshold": min_ts,
        "required_stability_checks": required_stability_checks,
        "stable_checks": stable_checks,
        "max_ts_delta_threshold": max_ts_delta,
        "max_abs_ts_delta_observed": (
            round(max_observed_delta, 6) if max_observed_delta is not None else None
        ),
        "eligible_for_promotion": (
            required_heads > 0
            and heads_above_threshold >= required_heads
            and stability_ok
        ),
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_word_alignments_json(rows: list[dict], output_path: Path) -> None:
    """Save a clean JSON with all word mappings."""
    clean = []
    for row in rows:
        if not row.get("alignments"):
            continue
        clean.append({
            "pair_id": row["pair_id"],
            "dataset": row.get("dataset"),
            "source_text": row["source_text"],
            "target_text": row["target_text"],
            "direction": row["direction"],
            "alignment_model": row.get("llm_model"),
            "source_words": [w["text"] for w in row["source_words"]],
            "target_words": [w["text"] for w in row["target_words"]],
            "alignments": [
                {
                    "source_words": a["source_text"],
                    "target_words": a["target_text"],
                    "source_span": [a["source_start"], a["source_end"]],
                    "target_span": [a["target_start"], a["target_end"]],
                    "confidence": a["confidence"],
                }
                for a in row["alignments"]
            ],
        })
    output_path.write_text(
        json.dumps(clean, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved word alignments: {output_path} ({len(clean)} pairs)", flush=True)


def save_heads_json(result: dict, output_path: Path) -> None:
    if "milmmt" in str(result.get("model", "")).lower():
        invalid = [
            {"layer": int(head["layer"]), "head": int(head["head"])}
            for head in result.get("token_alignment_heads", [])
            if not (
                0 <= int(head["layer"]) < MILMMT_EXPECTED_NUM_LAYERS
                and 0 <= int(head["head"]) < MILMMT_EXPECTED_NUM_HEADS
            )
        ]
        if invalid:
            raise ValueError(f"Invalid MiLMMT head indices: {invalid}")
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved translation heads: {output_path}", flush=True)

    # Print summary
    heads = result.get("token_alignment_heads", [])
    print(f"\n{'='*60}")
    print(f"TOKEN ALIGNMENT HEADS (TS > {result['paper_threshold']})")
    print(f"Model: {result['model']}")
    print(f"Direction: {result['direction']}")
    print(f"Scored {result['used_pairs']} pairs, {result['used_target_tokens']} target tokens")
    print(f"Found {len(heads)} token alignment heads out of "
          f"{result['num_layers']}x{result['num_heads']}="
          f"{result['num_layers']*result['num_heads']} total")
    print(f"{'='*60}")
    for i, h in enumerate(heads[:20]):
        print(f"  #{i+1:2d}  Layer {h['layer']:2d}  Head {h['head']:2d}  TS={h['ts']:.4f}")
    print()


def make_safe_model_name(model_name: str) -> str:
    path = Path(model_name)
    if path.exists():
        model_dir = path.parent.parent.name if len(path.parents) >= 2 else path.parent.name
        if model_dir.startswith("models--"):
            model_dir = model_dir[len("models--"):]
        return model_dir.replace("--", "_").replace("/", "_").replace(".", "_")
    return model_name.replace("/", "_").replace(".", "_")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Detect token alignment heads in Gemma-style translation decoders",
    )
    parser.add_argument(
        "--step", choices=["align", "detect", "all"], default="all",
        help="Which step to run (default: all)",
    )
    parser.add_argument(
        "--direction", default=DEFAULT_DIRECTION,
        help=f"Translation direction, e.g. cs-en (default: {DEFAULT_DIRECTION})",
    )
    parser.add_argument(
        "--model", "--mt-model", dest="mt_model", default=DEFAULT_MT_MODEL,
        help=f"Translation model to analyze (default: {DEFAULT_MT_MODEL})",
    )
    parser.add_argument(
        "--alignment-model", default=DEFAULT_ALIGNMENT_MODEL,
        help=f"OpenAI model for word alignment (default: {DEFAULT_ALIGNMENT_MODEL})",
    )
    parser.add_argument(
        "--src-path", default=None,
        help="Custom source text file (one sentence per line)",
    )
    parser.add_argument(
        "--tgt-path", default=None,
        help="Custom target text file (one sentence per line)",
    )
    parser.add_argument(
        "--dataset-name", default=None,
        help="Optional dataset label stored in outputs (e.g. czeng2.0, europarl, voxpopuli)",
    )
    parser.add_argument(
        "--alignment-file", default=None,
        help="Pre-existing alignment JSONL file to use (skip annotation step)",
    )
    parser.add_argument(
        "--max-pairs", type=int, default=0,
        help="Max number of pairs to process (0 = all)",
    )
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--device", default="cuda:0",
    )
    parser.add_argument(
        "--disable-cuda-warmup",
        action="store_true",
        help=(
            "Disable transformers' CUDA allocator warmup during model load. "
            "Useful when another stack already occupies part of GPU memory."
        ),
    )
    parser.add_argument(
        "--max-gpu-memory",
        default=None,
        help=(
            "When set, load the model with device_map=auto and cap GPU 0 memory "
            "to this value, offloading the remaining weights to CPU."
        ),
    )
    parser.add_argument(
        "--workers", type=int, default=20,
        help="Number of concurrent API workers for alignment (default: 20)",
    )
    parser.add_argument(
        "--stability-splits", type=int, default=0,
        help="Number of random split checks to run after the full detection pass",
    )
    parser.add_argument(
        "--stability-split-size", type=int, default=0,
        help="Aligned pairs per stability split (0 = half of the usable rows)",
    )
    parser.add_argument(
        "--top-k-heads", type=int, default=8,
        help="Number of top heads to compare for stability (default: 8)",
    )
    parser.add_argument(
        "--stability-seed", type=int, default=13,
        help="Random seed used for stability split sampling (default: 13)",
    )
    parser.add_argument(
        "--promotion-min-ts", type=float, default=0.1,
        help="Minimum TS score required for each top head promotion slot (default: 0.1)",
    )
    parser.add_argument(
        "--promotion-required-stability-checks", type=int, default=2,
        help="Minimum number of stable split checks required for promotion (default: 2)",
    )
    parser.add_argument(
        "--promotion-max-ts-delta", type=float, default=0.03,
        help="Largest allowed TS delta vs full-run top heads for promotion (default: 0.03)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: same as this script)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_model = make_safe_model_name(args.mt_model)
    align_jsonl = output_dir / f"raw_alignments_{args.direction}.jsonl"
    align_json_out = output_dir / f"word_alignments_{args.direction}.json"
    heads_json_out = output_dir / f"translation_heads_{safe_model}_{args.direction}.json"

    # -- Step: Align --
    if args.step in ("align", "all"):
        if args.alignment_file:
            print(f"Loading pre-existing alignments from {args.alignment_file}")
            rows = coerce_alignment_rows(load_alignment_rows(args.alignment_file))
        else:
            pairs = load_parallel_corpus(
                args.direction,
                src_path=args.src_path,
                tgt_path=args.tgt_path,
                dataset_name=args.dataset_name,
                max_pairs=args.max_pairs,
            )
            api_key = get_api_key()
            rows = annotate_corpus(
                pairs, api_key, args.alignment_model, align_jsonl,
                workers=args.workers,
            )

        # Filter
        filtered_rows = filter_alignments(rows, args.direction)

        # Save clean word alignment JSON
        save_word_alignments_json(filtered_rows, align_json_out)

        if args.step == "align":
            print("Alignment step complete.")
            return

    # -- Step: Detect --
    if args.step in ("detect", "all"):
        if args.step == "detect":
            # Load from file
            if args.alignment_file:
                src = Path(args.alignment_file)
            else:
                src = align_jsonl
            if not src.exists():
                raise FileNotFoundError(
                    f"Alignment file not found: {src}\n"
                    f"Run --step align first, or provide --alignment-file"
                )
            rows = coerce_alignment_rows(load_alignment_rows(src))
            filtered_rows = filter_alignments(rows, args.direction)

        result = detect_heads(
            model_name=args.mt_model,
            direction=args.direction,
            filtered_rows=filtered_rows,
            dtype_str=args.dtype,
            device=args.device,
            disable_cuda_warmup=args.disable_cuda_warmup,
            max_gpu_memory=args.max_gpu_memory,
        )
        if args.stability_splits > 0:
            print(
                f"Running {args.stability_splits} split-stability checks "
                f"(sample_size={args.stability_split_size or 'half'}, top_k={args.top_k_heads})...",
                flush=True,
            )
            result["stability_checks"] = run_head_stability_checks(
                model_name=args.mt_model,
                direction=args.direction,
                filtered_rows=filtered_rows,
                full_result=result,
                split_count=args.stability_splits,
                split_size=args.stability_split_size,
                top_k=args.top_k_heads,
                dtype_str=args.dtype,
                device=args.device,
                seed=args.stability_seed,
                disable_cuda_warmup=args.disable_cuda_warmup,
                max_gpu_memory=args.max_gpu_memory,
            )
            deltas = [
                check["max_abs_ts_delta_vs_full"]
                for check in result["stability_checks"]
                if check["max_abs_ts_delta_vs_full"] is not None
            ]
            result["stability_summary"] = {
                "top_k": args.top_k_heads,
                "split_count": args.stability_splits,
                "all_checks_stable_vs_full": all(
                    check["stable_vs_full"] for check in result["stability_checks"]
                ),
                "max_abs_ts_delta_vs_full": round(max(deltas), 6) if deltas else None,
            }
        result["promotion_gate"] = summarize_head_promotion_gate(
            result,
            top_k=args.top_k_heads,
            min_ts=args.promotion_min_ts,
            required_stability_checks=args.promotion_required_stability_checks,
            max_ts_delta=args.promotion_max_ts_delta,
        )
        save_heads_json(result, heads_json_out)

    print("Done.")


if __name__ == "__main__":
    main()
