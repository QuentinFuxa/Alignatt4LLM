"""Small IncrementalOutput stand-in for offline batch inference."""

from __future__ import annotations


class IncrementalOutput:
    def __init__(
        self,
        new_tokens,
        new_string,
        *_ignored_legacy_args,
        **_ignored_legacy_kwargs,
    ):
        self.new_tokens = list(new_tokens or [])
        self.new_string = str(new_string or "")

    def strings_to_json(self) -> str:
        import json

        return json.dumps(
            {
                "new": self.new_string,
            }
        )
