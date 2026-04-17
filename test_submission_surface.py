from __future__ import annotations

from pathlib import Path
import wave

import numpy as np
import pytest


def test_discover_input_media_paths_accepts_wav_and_mp4(tmp_path: Path):
    from cascade_audio import discover_input_media_paths

    (tmp_path / "a.wav").write_bytes(b"")
    (tmp_path / "b.mp4").write_bytes(b"")
    (tmp_path / "ignore.txt").write_text("x", encoding="utf-8")

    assert discover_input_media_paths(tmp_path) == [
        str(tmp_path / "a.wav"),
        str(tmp_path / "b.mp4"),
    ]


def test_load_audio_mono_16khz_resamples_pcm16_wav(tmp_path: Path):
    from cascade_audio import load_audio_mono_16khz

    wav_path = tmp_path / "stereo8k.wav"
    sample_rate = 8_000
    duration_s = 0.25
    t = np.linspace(0.0, duration_s, int(sample_rate * duration_s), endpoint=False)
    left = 0.5 * np.sin(2 * np.pi * 220 * t)
    right = 0.5 * np.sin(2 * np.pi * 440 * t)
    stereo = np.stack([left, right], axis=1)
    pcm16 = np.clip(stereo * 32767.0, -32768, 32767).astype(np.int16)

    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm16.tobytes())

    audio = load_audio_mono_16khz(wav_path)
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert len(audio) == int(round(duration_s * 16_000))
    assert np.max(np.abs(audio)) <= 1.0


def test_submission_preset_freezes_vllm_submission_config():
    from cascade_submission import get_submission_preset

    preset = get_submission_preset("main_high_latency")
    config = preset.build_speech_processor_config(
        source_lang_code="en",
        target_lang_code="de",
    )

    assert config.alignment_backend_name == "qwen_forced"
    assert config.mt_backend_name == "gemma_vllm_alignatt"
    assert config.chunk_ms == 700
    assert config.min_start_seconds == 2.0
    assert config.max_history_utterances == 1
    assert config.partial_max_new_tokens == 16
    assert config.partial_followup_max_new_tokens == 8
    assert config.mt_vllm_cudagraph_mode == "full"
    assert config.mt_vllm_enable_prefix_caching is False


def test_resolve_paper_context_path_for_input_matches_stem(tmp_path: Path):
    from run_simulstream_batch import resolve_paper_context_path_for_input

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    artifact_path = artifact_dir / "talk123.json"
    artifact_path.write_text("{}", encoding="utf-8")

    resolved = resolve_paper_context_path_for_input(
        "talk123.mp4",
        paper_context_dir=str(artifact_dir),
    )
    assert resolved == str(artifact_path)

    with pytest.raises(FileNotFoundError):
        resolve_paper_context_path_for_input(
            "missing.wav",
            paper_context_dir=str(artifact_dir),
        )
