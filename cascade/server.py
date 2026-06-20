from __future__ import annotations

import asyncio
import json
import logging
from typing import Awaitable, Callable

import websockets
from websockets.asyncio.server import ServerConnection, serve

from simulstream.metrics.logger import METRICS_LOGGER, setup_metrics_logger
from simulstream.server.message_processor import MessageProcessor
from simulstream.server.websocket_server import SpeechProcessorPool


LOGGER = logging.getLogger("cascade_server")


class CascadeMessageProcessor(MessageProcessor):
    """Message processor with per-stream paper-context support."""

    def process_metadata(self, metadata: dict):
        super().process_metadata(metadata)
        if "paper_context_path" in metadata:
            setter = getattr(self.speech_processor, "set_paper_context_path", None)
            if callable(setter):
                setter(metadata.get("paper_context_path"))


def connection_handler_factory(
    speech_processor_pool: SpeechProcessorPool,
) -> Callable[[ServerConnection], Awaitable[None]]:
    async def handle_connection(websocket: ServerConnection) -> None:
        loop = asyncio.get_running_loop()
        client_id = id(websocket)
        LOGGER.info("Client %s connected", client_id)
        try:
            async with speech_processor_pool.acquire() as speech_processor:
                message_processor = CascadeMessageProcessor(client_id, speech_processor)
                try:
                    async for message in websocket:
                        if isinstance(message, bytes):
                            incremental_output = await loop.run_in_executor(
                                None, message_processor.process_speech, message
                            )
                            if incremental_output is not None:
                                await websocket.send(incremental_output.strings_to_json())
                            continue

                        if not isinstance(message, str):
                            continue

                        try:
                            data = json.loads(message)
                            if "end_of_stream" in data:
                                incremental_output = await loop.run_in_executor(
                                    None, message_processor.end_of_stream
                                )
                                await websocket.send(incremental_output.strings_to_json())
                                await websocket.send(json.dumps({"end_of_processing": True}))
                            else:
                                message_processor.process_metadata(data)
                        except Exception as exc:
                            LOGGER.error(
                                "Invalid string message from client %s: %s (%s)",
                                client_id,
                                message,
                                exc,
                            )
                except websockets.exceptions.ConnectionClosed:
                    LOGGER.info("Client %s disconnected", client_id)
                finally:
                    message_processor.clear()
        except TimeoutError:
            LOGGER.error("Timeout waiting for a new processor for client %s", client_id)
            await websocket.close()

    return handle_connection


async def serve_cascade_processor(
    *,
    speech_processor_config,
    hostname: str,
    port: int,
    pool_size: int,
    acquire_timeout: int,
    metrics_log_file: str,
) -> None:
    setup_metrics_logger(type("MetricsConfig", (), {
        "enabled": True,
        "filename": metrics_log_file,
    })())

    LOGGER.info("Loading speech processor pool (size=%s)", pool_size)
    speech_processors_pool = SpeechProcessorPool(
        speech_processor_config=speech_processor_config,
        size=pool_size,
        acquire_timeout=acquire_timeout,
    )
    METRICS_LOGGER.info(json.dumps({"pool_size": pool_size}))

    LOGGER.info("Serving websocket server at %s:%s", hostname, port)
    async with serve(
        connection_handler_factory(speech_processors_pool),
        hostname,
        port,
        ping_timeout=None,
    ) as server:
        await server.serve_forever()
