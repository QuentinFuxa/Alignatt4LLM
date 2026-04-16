from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parent
    / "assets"
    / "attention_heads"
    / "detect_translation_heads.py"
)
SPEC = spec_from_file_location("detect_translation_heads", MODULE_PATH)
MODULE = module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_keep_anchor_alignment_rejects_czech_function_word_pair():
    alignment = {
        "source_text": "a",
        "target_text": "and",
        "source_start": 0,
        "source_end": 1,
        "target_start": 0,
        "target_end": 1,
    }

    assert not MODULE.keep_anchor_alignment(alignment, "cs-en")


def test_keep_anchor_alignment_keeps_czech_content_anchor():
    alignment = {
        "source_text": "Praha",
        "target_text": "Prague",
        "source_start": 0,
        "source_end": 1,
        "target_start": 0,
        "target_end": 1,
    }

    assert MODULE.keep_anchor_alignment(alignment, "cs-en")
