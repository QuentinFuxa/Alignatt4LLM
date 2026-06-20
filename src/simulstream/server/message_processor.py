"""Small subset of SimulStream message processor APIs used by patches."""

from __future__ import annotations

import logging

from simulstream.server.speech_processors.incremental_output import IncrementalOutput


METRICS_LOGGER = logging.getLogger("simulstream.metrics")


class MessageProcessor:
    pass


def merge_incremental_outputs(outputs, tokens_to_string):
    new_tokens = []
    for output in outputs:
        if output is None:
            continue
        new_tokens.extend(getattr(output, "new_tokens", []) or [])
    return IncrementalOutput(new_tokens, tokens_to_string(new_tokens), [], "")
