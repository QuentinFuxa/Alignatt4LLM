from __future__ import annotations

from pathlib import Path
import wave

import numpy as np

from simulstream.server.speech_processors import SAMPLE_RATE


SUPPORTED_MEDIA_SUFFIXES = (
    ".wav",
    ".mp4",
    ".m4a",
    ".mp3",
    ".flac",
    ".ogg",
    ".opus",
)


def _resample_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == SAMPLE_RATE:
        return audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32)
    duration = len(audio) / float(sample_rate)
    old_times = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    new_length = int(round(duration * SAMPLE_RATE))
    new_times = np.linspace(0.0, duration, num=new_length, endpoint=False)
    return np.interp(new_times, old_times, audio).astype(np.float32)


def _load_wav_pcm16(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        num_channels = wav_file.getnchannels()
        raw = wav_file.readframes(wav_file.getnframes())
    if sample_width != 2:
        raise ValueError("Only 16-bit PCM WAV is supported by the fast path.")
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if num_channels > 1:
        audio = audio.reshape(-1, num_channels).mean(axis=1)
    return _resample_audio(audio, sample_rate)


def _load_media_with_pyav(path: Path) -> np.ndarray:
    import av

    container = av.open(str(path))
    try:
        audio_stream = next((stream for stream in container.streams if stream.type == "audio"), None)
        if audio_stream is None:
            raise ValueError(f"No audio stream found in {path}.")
        resampler = av.audio.resampler.AudioResampler(
            format="s16",
            layout="mono",
            rate=SAMPLE_RATE,
        )
        pcm_chunks: list[np.ndarray] = []
        for frame in container.decode(audio=audio_stream.index):
            for resampled_frame in resampler.resample(frame):
                pcm = resampled_frame.to_ndarray()
                if pcm.ndim == 2:
                    pcm = pcm[0]
                pcm_chunks.append(np.asarray(pcm, dtype=np.int16))
        for resampled_frame in resampler.resample(None):
            pcm = resampled_frame.to_ndarray()
            if pcm.ndim == 2:
                pcm = pcm[0]
            pcm_chunks.append(np.asarray(pcm, dtype=np.int16))
    finally:
        container.close()
    if not pcm_chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(pcm_chunks).astype(np.float32) / 32768.0


def load_audio_mono_16khz(path: str | Path) -> np.ndarray:
    media_path = Path(path)
    if not media_path.exists():
        raise FileNotFoundError(f"Missing input media: {media_path}")
    if media_path.suffix.lower() == ".wav":
        try:
            return _load_wav_pcm16(media_path)
        except (ValueError, wave.Error):
            # Fall back to container decoding for non-PCM or unusual WAV layouts.
            pass
    return _load_media_with_pyav(media_path)


def discover_input_media_paths(input_dir: str | Path) -> list[str]:
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Missing input directory: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Expected an input directory, got: {root}")
    media_paths = sorted(
        str(path)
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_MEDIA_SUFFIXES
    )
    if not media_paths:
        supported = ", ".join(SUPPORTED_MEDIA_SUFFIXES)
        raise FileNotFoundError(
            f"No supported media files found in {root}. Supported suffixes: {supported}"
        )
    return media_paths
