Use the PDF in this directory as the algorithm reference, but treat the runnable entry point as:

`assets/attention_heads/detect_translation_heads.py`

Current repo assumptions:

- We are Gemma-first for the stable MT baseline; MiLMMT text-head files are
  experimental and must be selected explicitly.
- Word alignment annotation should use `gpt-5-mini`.
- The detector should be rerunnable on arbitrary language pairs by passing explicit `--src-path` and `--tgt-path`.
- For IWSLT 2026 Czech->English, use challenge-legal Czech-English parallel text for head discovery and keep the official dev set for validation.

Quick sanity check for the OpenAI key:

```bash
python assets/attention_heads/test_openai_token.py
```

Typical Czech->English run:

```bash
python assets/attention_heads/detect_translation_heads.py \
  --direction cs-en \
  --model google/gemma-4-E4B-it \
  --src-path /path/to/czeng.cs \
  --tgt-path /path/to/czeng.en \
  --dataset-name czeng2.0
```

Expected outputs:

- `raw_alignments_cs-en.jsonl`
- `word_alignments_cs-en.json`
- `translation_heads_google_gemma-4-E4B-it_cs-en.json`
