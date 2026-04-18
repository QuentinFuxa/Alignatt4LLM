# `docs/reference/`

Upstream model cards and reference implementations used during the design of this repo. Not our code — kept here for convenience and attribution.

## Contents

- `Qwen3_aligner.md` — upstream Qwen3-ASR + Qwen3-ForcedAligner model card (Apache-2.0, from Qwen HF repo).
- `alignatt_markdown.md` — AlignAtt paper writeup / reference notes.
- `alignatt_whipser.py` — reference implementation of AlignAtt for Whisper, used as a design anchor for the decoding policy.
- `simul_streaming_whisper.py` — simultaneous-streaming Whisper reference harness.
- `whisper_online_main.py` — Whisper-online reference entry point.

The AlignAtt policy in `cascade/mt/base.py::AlignAttDecoderPolicy` was adapted from `alignatt_whipser.py`; see `docs/archive/ALIGNATT_LLM.md` for the adaptation rationale.
