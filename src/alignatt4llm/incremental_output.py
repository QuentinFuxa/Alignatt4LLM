from __future__ import annotations

import json
import time

from simulstream.server import message_processor as simulstream_message_processor
from simulstream.server.speech_processors.incremental_output import IncrementalOutput


def _patch_simulstream_append_only() -> None:
    if getattr(IncrementalOutput, "_cascade_append_only_patched", False):
        return

    def _strings_to_json(self: IncrementalOutput) -> str:
        return json.dumps({"new": self.new_string})

    def _process_speech(self, speech_data: bytes):
        self.client_buffer += speech_data
        buffer_len_seconds = len(self.client_buffer) / 2 / self.sample_rate
        if buffer_len_seconds < self.speech_processor.speech_chunk_size:
            return None

        self.processed_audio_seconds += buffer_len_seconds
        start_time = time.time()
        incremental_output = self._run_speech_processor()
        end_time = time.time()
        simulstream_message_processor.METRICS_LOGGER.info(
            json.dumps(
                {
                    "id": self.client_id,
                    "total_audio_processed": self.processed_audio_seconds,
                    "computation_time": end_time - start_time,
                    "generated_tokens": incremental_output.new_tokens,
                }
            )
        )
        return incremental_output

    def _end_of_stream(self):
        outputs = []
        start_time = time.time()
        if self.client_buffer:
            self.processed_audio_seconds += len(self.client_buffer) / 2 / self.sample_rate
            outputs.append(self._run_speech_processor())

        outputs.append(self.speech_processor.end_of_stream())
        incremental_output = simulstream_message_processor.merge_incremental_outputs(
            outputs,
            self.speech_processor.tokens_to_string,
        )
        end_time = time.time()
        simulstream_message_processor.METRICS_LOGGER.info(
            json.dumps(
                {
                    "id": self.client_id,
                    "total_audio_processed": self.processed_audio_seconds,
                    "computation_time": end_time - start_time,
                    "generated_tokens": incremental_output.new_tokens,
                }
            )
        )
        self.clear()
        return incremental_output

    IncrementalOutput.strings_to_json = _strings_to_json
    IncrementalOutput._cascade_append_only_patched = True
    simulstream_message_processor.MessageProcessor.process_speech = _process_speech
    simulstream_message_processor.MessageProcessor.end_of_stream = _end_of_stream


_patch_simulstream_append_only()


def empty_incremental_output() -> IncrementalOutput:
    return IncrementalOutput([], "", [], "")


def append_only_incremental_output(
    *,
    new_tokens: list[str],
    new_string: str,
) -> IncrementalOutput:
    return IncrementalOutput(list(new_tokens), new_string, [], "")
