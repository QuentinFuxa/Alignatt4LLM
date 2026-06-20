"""Small subset of SimulStream speech processor APIs used by the batch runner."""

from __future__ import annotations


SAMPLE_RATE = 16_000


class SpeechProcessor:
    def __init__(self, config):
        self.config = config
