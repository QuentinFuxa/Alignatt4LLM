from __future__ import annotations

import json

from cascade_artifacts import MANIFEST_FILENAME, STREAM_UPDATES_FILENAME, final_asr_filename, final_translation_filename


def _write_backend_bundle(
    root,
    *,
    backend_id: str,
    final_asr: str,
    final_translation: str,
    stream_updates: list[dict],
    model_load_ms: float,
    total_wallclock_s: float,
    rtf: float,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": "cascade_v1",
                "kind": "inference",
                "wav_path": "tmp/alignatt_smoke18.wav",
                "audio_duration_ms": 2400.0,
                "source_language_code": "en",
                "target_language_code": "de",
                "runtime_config": {
                    "alignment_backend_name": backend_id,
                },
                "run_provenance": {
                    "alignment_backend_name": backend_id,
                    "model_load_ms": model_load_ms,
                    "total_wallclock_s": total_wallclock_s,
                    "rtf": rtf,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / STREAM_UPDATES_FILENAME).write_text(
        "".join(json.dumps(row) + "\n" for row in stream_updates),
        encoding="utf-8",
    )
    (root / final_asr_filename("en")).write_text(final_asr + "\n", encoding="utf-8")
    (root / final_translation_filename("de")).write_text(
        final_translation + "\n",
        encoding="utf-8",
    )


def test_build_comparison_report_consolidates_backend_metrics(tmp_path):
    from run_simulstream_compare import build_comparison_report

    reference_path = tmp_path / "reference.txt"
    reference_path.write_text("hello brave world\n", encoding="utf-8")

    qwen_dir = tmp_path / "qwen_forced"
    _write_backend_bundle(
        qwen_dir,
        backend_id="qwen_forced",
        final_asr="hello brave world",
        final_translation="Hallo Welt",
        stream_updates=[
            {
                "update_idx": 0,
                "audio_processed_ms": 1000.0,
                "wallclock_elapsed_ms": 400.0,
                "asr_text": "hello",
                "translation_text": "Hallo",
                "new_words": ["Hallo"],
                "is_eos": False,
            },
            {
                "update_idx": 1,
                "audio_processed_ms": 2000.0,
                "wallclock_elapsed_ms": 800.0,
                "asr_text": "hello brave world",
                "translation_text": "Hallo Welt",
                "new_words": ["Welt"],
                "is_eos": False,
            },
            {
                "update_idx": 2,
                "audio_processed_ms": 2400.0,
                "wallclock_elapsed_ms": 1000.0,
                "asr_text": "hello brave world",
                "translation_text": "Hallo Welt",
                "new_words": [],
                "is_eos": True,
            },
        ],
        model_load_ms=1200.0,
        total_wallclock_s=1.0,
        rtf=0.4167,
    )

    gemma_dir = tmp_path / "gemma_onepass_qk_fast"
    _write_backend_bundle(
        gemma_dir,
        backend_id="gemma_onepass_qk_fast",
        final_asr="hello world",
        final_translation="Hallo Welt",
        stream_updates=[
            {
                "update_idx": 0,
                "audio_processed_ms": 1200.0,
                "wallclock_elapsed_ms": 500.0,
                "asr_text": "hello",
                "translation_text": "Hallo da",
                "new_words": ["Hallo", "da"],
                "is_eos": False,
            },
            {
                "update_idx": 1,
                "audio_processed_ms": 1800.0,
                "wallclock_elapsed_ms": 900.0,
                "asr_text": "hello world",
                "translation_text": "Hallo Welt",
                "new_words": ["Welt"],
                "is_eos": False,
            },
            {
                "update_idx": 2,
                "audio_processed_ms": 2400.0,
                "wallclock_elapsed_ms": 1200.0,
                "asr_text": "hello world",
                "translation_text": "Hallo Welt",
                "new_words": [],
                "is_eos": True,
            },
        ],
        model_load_ms=3400.0,
        total_wallclock_s=1.2,
        rtf=0.5,
    )

    report = build_comparison_report(
        wav_path="tmp/alignatt_smoke18.wav",
        reference_path=str(reference_path),
        backend_artifact_dirs={
            "qwen_forced": qwen_dir,
            "gemma_onepass_qk_fast": gemma_dir,
        },
    )

    assert set(report["backend_ids"]) == {"qwen_forced", "gemma_onepass_qk_fast"}
    by_backend = {entry["backend_id"]: entry for entry in report["backends"]}

    assert by_backend["qwen_forced"]["has_terminal_eos_update"] is True
    assert by_backend["qwen_forced"]["wer"] == 0.0
    assert by_backend["qwen_forced"]["first_nonempty_emission_audio_s"] == 1.0
    assert by_backend["qwen_forced"]["model_load_ms"] == 1200.0

    assert by_backend["gemma_onepass_qk_fast"]["has_terminal_eos_update"] is True
    assert by_backend["gemma_onepass_qk_fast"]["wer"] > 0.0
    assert by_backend["gemma_onepass_qk_fast"]["first_nonempty_emission_wallclock_s"] == 0.5
    assert by_backend["gemma_onepass_qk_fast"]["revision_updates"] == 1
    assert by_backend["gemma_onepass_qk_fast"]["suppressed_units"] == 1
