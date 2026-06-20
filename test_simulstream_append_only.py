from __future__ import annotations

import json

from simulstream.server.message_processor import merge_incremental_outputs
from simulstream.server.speech_processors.incremental_output import IncrementalOutput


def test_incremental_output_serializes_only_append_payload():
    output = IncrementalOutput(["A"], "A", ["old"], "old")

    payload = json.loads(output.strings_to_json())

    assert payload == {"new": "A"}
    assert not hasattr(output, "deleted_tokens")
    assert not hasattr(output, "deleted_string")


def test_merge_incremental_outputs_stays_append_only():
    merged = merge_incremental_outputs(
        [
            IncrementalOutput(["A"], "A"),
            IncrementalOutput(["B"], "B"),
        ],
        lambda tokens: "".join(tokens),
    )

    assert merged.new_tokens == ["A", "B"]
    assert json.loads(merged.strings_to_json()) == {"new": "AB"}
    assert not hasattr(merged, "deleted_tokens")
    assert not hasattr(merged, "deleted_string")
