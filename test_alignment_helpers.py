"""Pure-Python invariants for the Gemma audio alignment helpers.

These tests do not load any model; they lock the data-flow rules that
make the rest of the pipeline correct. We want:

- ``detect_audio_span`` actually finds a contiguous audio-token span
- ``audio_position_to_end_seconds`` respects the known 40 ms calibration
- ``_enforce_monotone`` is idempotent and non-decreasing
- ``split_text_into_word_spans`` matches Qwen's punctuation-stripping
  word-unit convention so the downstream cascade treats the two
  backends identically
- ``aggregate_token_timings_to_words`` turns per-token end-times into
  per-word end-times without introducing non-monotone regressions
"""

from __future__ import annotations


def test_detect_audio_span_returns_contiguous_range():
    from gemma_alignment_probe import detect_audio_span

    ids = [1, 2, 3, 258881, 258881, 258881, 7]
    span = detect_audio_span(ids, audio_token_id=258881, audio_ms_per_token=40.0)
    assert span is not None
    assert span.prompt_start == 3
    assert span.prompt_end == 6
    assert span.length == 3
    assert span.ms_per_token == 40.0


def test_detect_audio_span_returns_none_when_absent():
    from gemma_alignment_probe import detect_audio_span

    assert detect_audio_span([1, 2, 3], audio_token_id=258881, audio_ms_per_token=40.0) is None


def test_audio_position_uses_40ms_calibration():
    from gemma_alignment_probe import audio_position_to_end_seconds

    assert audio_position_to_end_seconds(0, ms_per_token=40.0, audio_duration_s=30.0) == 0.04
    assert audio_position_to_end_seconds(24, ms_per_token=40.0, audio_duration_s=30.0) == 1.0
    # Clamps to audio duration.
    assert audio_position_to_end_seconds(1000, ms_per_token=40.0, audio_duration_s=3.5) == 3.5
    assert audio_position_to_end_seconds(None, ms_per_token=40.0, audio_duration_s=1.0) is None


def test_enforce_monotone_projects_to_running_max():
    from gemma_alignment_probe import _enforce_monotone

    assert _enforce_monotone([0.1, 0.2, 0.15, 0.3, 0.25]) == [0.1, 0.2, 0.2, 0.3, 0.3]
    assert _enforce_monotone([0.0, None, 0.3, None, 0.2]) == [0.0, None, 0.3, None, 0.3]


def test_split_text_into_word_spans_strips_trailing_punctuation():
    from gemma_alignment_probe import split_text_into_word_spans

    spans = split_text_into_word_spans("Hello, world!")
    assert [span[2] for span in spans] == ["Hello", "world"]

    spans2 = split_text_into_word_spans("(hello)")
    assert [span[2] for span in spans2] == ["hello"]


def test_aggregate_token_timings_preserves_monotonicity():
    _TOKEN_TABLE = {
        10: "Hel",
        11: "lo",
        12: ",",
        13: " wor",
        14: "ld",
        15: "!",
    }

    class StubTokenizer:
        def decode(self, ids, skip_special_tokens=False):
            return "".join(_TOKEN_TABLE[i] for i in ids)

    generated_ids = [10, 11, 12, 13, 14, 15]
    # Per-token end times (monotone).
    token_ends = [0.12, 0.20, 0.20, 0.50, 0.80, 0.80]

    from gemma_alignment_probe import aggregate_token_timings_to_words

    words = aggregate_token_timings_to_words(
        "Hello, world!",
        generated_ids=generated_ids,
        tokenizer=StubTokenizer(),
        token_end_times_s=token_ends,
        audio_duration_s=1.0,
    )
    # Two words: "Hello", "world"
    assert [w.text for w in words] == ["Hello", "world"]
    # Each word's end time is monotone non-decreasing.
    assert words[0].end_time <= words[1].end_time
    assert words[0].end_time <= 0.20 + 1e-9
    assert words[1].end_time <= 0.80 + 1e-9


def test_monotonicity_score_rewards_forward_progress():
    from gemma_alignment_probe import monotonicity_score

    assert monotonicity_score([0, 1, 2, 3]) == 1.0
    assert monotonicity_score([0, 0, 0, 0]) == 1.0
    # One backward jump across three transitions -> 2/3.
    assert abs(monotonicity_score([0, 1, 0, 2]) - (2.0 / 3.0)) < 1e-9
    assert monotonicity_score([None, None]) == 0.0


def test_alignment_result_observer_tokens_roundtrip():
    from alignment_backend import (
        AlignAttObserverToken,
        AlignAttProvenanceBreakdown,
        AlignmentResult,
        WordAlignment,
    )
    from run_alignment_single_audio import deserialize_alignment, serialize_alignment

    result = AlignmentResult(
        text="hello",
        words=(WordAlignment(text="hello", start_time=0.1, end_time=0.2),),
        audio_duration_s=0.5,
        observer_tokens=(
            AlignAttObserverToken(
                token_id=42,
                token_str="hello",
                aligned_source_position=3,
                source_accessible_mass=0.75,
                blocked_source_local_position=4,
                blocked_source_unit_index=2,
                provenance=AlignAttProvenanceBreakdown(
                    source_accessible=0.75,
                    source_inaccessible=0.05,
                    non_source_prompt=0.10,
                    suffix=0.10,
                ),
            ),
        ),
        diagnostics={"backend": "gemma_onepass_qk_fast"},
    )

    roundtrip = deserialize_alignment(serialize_alignment(result))

    assert roundtrip.text == "hello"
    assert len(roundtrip.observer_tokens) == 1
    assert roundtrip.observer_tokens[0].token_id == 42
    assert roundtrip.observer_tokens[0].aligned_source_position == 3
    assert roundtrip.observer_tokens[0].source_accessible_mass == 0.75
    assert roundtrip.observer_tokens[0].provenance is not None
    assert roundtrip.observer_tokens[0].provenance.source_inaccessible == 0.05


def test_reconstruct_vllm_audio_attention_rows_matches_raw_qk_scores():
    import numpy as np
    from cascade_mt_backend import AlignAttHead
    from gemma_vllm_alignment_backend import reconstruct_vllm_audio_attention_rows

    rows, diagnostics = reconstruct_vllm_audio_attention_rows(
        {
            "layer_captures": {
                4: {
                    "selected_heads": [7],
                    "prompt_audio_k": np.asarray(
                        [[[1.0, 0.0], [0.0, 1.0]]],
                        dtype=np.float32,
                    ),
                    "decode_q": np.asarray(
                        [[[2.0, 5.0]]],
                        dtype=np.float32,
                    ),
                    "scaling": 1.0,
                }
            }
        },
        alignatt_heads=(AlignAttHead(layer=4, head=7, ts=0.0),),
    )

    assert diagnostics["effective_head_count"] == 1
    assert diagnostics["generated_token_count"] == 1
    assert len(rows) == 1
    assert tuple(rows[0].shape) == (1, 2)
    assert rows[0][0].tolist() == [2.0, 5.0]


def test_reconstruct_vllm_audio_attention_rows_reports_prompt_only_capture_gap():
    import numpy as np
    from cascade_mt_backend import AlignAttHead
    from gemma_vllm_alignment_backend import reconstruct_vllm_audio_attention_rows

    rows, diagnostics = reconstruct_vllm_audio_attention_rows(
        {
            "layer_captures": {
                4: {
                    "selected_heads": [7],
                    "prompt_audio_k": np.asarray(
                        [[[1.0, 0.0], [0.0, 1.0]]],
                        dtype=np.float32,
                    ),
                    "decode_q": None,
                    "scaling": 1.0,
                }
            }
        },
        alignatt_heads=(AlignAttHead(layer=4, head=7, ts=0.0),),
    )

    assert rows == []
    assert diagnostics["captured_layer_count"] == 1
    assert diagnostics["effective_head_count"] == 0
    assert diagnostics["generated_token_count"] == 0
    assert diagnostics["audio_span_length"] == 2
    assert diagnostics["missing_heads"] == [{"layer": 4, "head": 7}]


def test_prompt_observer_cache_entry_requires_complete_prompt_capture():
    import numpy as np
    from gemma_vllm_alignment_backend import _build_prompt_observer_cache_entry

    assert (
        _build_prompt_observer_cache_entry(
            {
                "prompt_length": 10,
                "audio_prompt_positions": [6, 7],
                "layer_captures": {
                    0: {
                        "prompt_audio_k": None,
                        "prompt_audio_missing_positions": [0, 1],
                    }
                },
            }
        )
        is None
    )

    entry = _build_prompt_observer_cache_entry(
        {
            "prompt_length": 10,
            "audio_prompt_positions": [6, 7],
            "layer_captures": {
                0: {
                    "prompt_audio_k": np.asarray([[[1.0], [2.0]]], dtype=np.float32),
                    "prompt_audio_missing_positions": [],
                }
            },
        }
    )

    assert entry is not None
    assert entry.prompt_length == 10
    assert entry.audio_prompt_positions == (6, 7)
    assert tuple(entry.layer_prompt_audio_k[0].shape) == (1, 2, 1)


def test_prompt_observer_cache_hydrates_missing_prompt_keys():
    import numpy as np
    from gemma_vllm_alignment_backend import (
        _build_prompt_observer_cache_entry,
        _hydrate_capture_payload_from_prompt_observer_cache,
    )

    cache_entry = _build_prompt_observer_cache_entry(
        {
            "prompt_length": 10,
            "audio_prompt_positions": [6, 7],
            "layer_captures": {
                0: {
                    "prompt_audio_k": np.asarray(
                        [[[1.0, 0.0], [0.0, 1.0]]],
                        dtype=np.float32,
                    ),
                    "prompt_audio_missing_positions": [],
                }
            },
        }
    )
    assert cache_entry is not None

    hydrated, diagnostics = _hydrate_capture_payload_from_prompt_observer_cache(
        {
            "prompt_length": 10,
            "audio_prompt_positions": [6, 7],
            "layer_captures": {
                0: {
                    "selected_heads": [7],
                    "prompt_audio_k": None,
                    "prompt_audio_missing_positions": [0, 1],
                    "decode_q": np.asarray([[[2.0, 5.0]]], dtype=np.float32),
                    "scaling": 1.0,
                    "head_dim": 2,
                }
            },
            "debug": {
                "layer_stats": {
                    "0": {
                        "selected_head_count": 1,
                        "prompt_audio_capture_count": 0,
                        "decode_q_count": 1,
                        "missing_prompt_audio_count": 2,
                    }
                }
            },
        },
        cache_entry=cache_entry,
    )

    assert hydrated is not None
    assert diagnostics == {
        "hit": True,
        "restored_layer_count": 1,
        "complete_after_restore": True,
    }
    assert tuple(hydrated["layer_captures"][0]["prompt_audio_k"].shape) == (1, 2, 2)
    assert hydrated["layer_captures"][0]["prompt_audio_missing_positions"] == []
    assert hydrated["debug"]["layer_stats"]["0"]["prompt_audio_capture_count"] == 2
    assert hydrated["debug"]["layer_stats"]["0"]["missing_prompt_audio_count"] == 0


def test_prompt_observer_cache_ignores_signature_mismatch():
    import numpy as np
    from gemma_vllm_alignment_backend import (
        _build_prompt_observer_cache_entry,
        _hydrate_capture_payload_from_prompt_observer_cache,
    )

    cache_entry = _build_prompt_observer_cache_entry(
        {
            "prompt_length": 10,
            "audio_prompt_positions": [6, 7],
            "layer_captures": {
                0: {
                    "prompt_audio_k": np.asarray([[[1.0], [2.0]]], dtype=np.float32),
                    "prompt_audio_missing_positions": [],
                }
            },
        }
    )
    assert cache_entry is not None

    hydrated, diagnostics = _hydrate_capture_payload_from_prompt_observer_cache(
        {
            "prompt_length": 11,
            "audio_prompt_positions": [6, 7],
            "layer_captures": {
                0: {
                    "prompt_audio_k": None,
                    "prompt_audio_missing_positions": [0, 1],
                }
            },
        },
        cache_entry=cache_entry,
    )

    assert hydrated is not None
    assert diagnostics == {
        "hit": False,
        "restored_layer_count": 0,
        "complete_after_restore": False,
    }
    assert hydrated["layer_captures"][0]["prompt_audio_k"] is None


def test_fetch_audio_qk_observer_payload_preserves_debug_state_and_clears_model():
    import numpy as np
    from gemma_vllm_alignment_backend import (
        _fetch_audio_qk_observer_from_model,
        _install_audio_qk_observer_on_model,
    )

    class FakeAttention:
        def __init__(self):
            self.forward = lambda *args, **kwargs: None
            self.scaling = 0.5
            self.head_dim = 2

    class FakeLayer:
        def __init__(self):
            self.self_attn = FakeAttention()

    class FakeInnerModel:
        def __init__(self):
            self.layers = [FakeLayer()]

    class FakeLanguageModel:
        def __init__(self):
            self.model = FakeInnerModel()

    class FakeModel:
        def __init__(self):
            self.language_model = FakeLanguageModel()

    model = FakeModel()
    install = _install_audio_qk_observer_on_model(
        model,
        selected_heads=[{"layer": 0, "head": 1}],
        audio_prompt_positions=[10, 11],
        prompt_length=12,
        patch_mode="postload_instance",
    )

    assert install == {
        "layer_count": 1,
        "audio_prompt_length": 2,
        "patch_mode": "postload_instance",
    }

    state = model._alignatt_audio_qk_state
    state["debug"] = {
        "forward_call_count": 7,
        "prompt_forward_call_count": 2,
        "decode_forward_call_count": 5,
        "positions_sample": [[0, 1, 2], [12]],
        "prompt_positions_sample": [[0, 1, 2]],
        "decode_positions_sample": [[12], [13]],
    }
    layer_state = state["layer_captures"][0]
    layer_state["prompt_audio_k_by_position"][10] = np.asarray(
        [[1.0, 0.0]],
        dtype=np.float32,
    )
    layer_state["prompt_audio_k_by_position"][11] = np.asarray(
        [[0.0, 1.0]],
        dtype=np.float32,
    )
    layer_state["decode_q"].append(
        np.asarray([[3.0, 4.0]], dtype=np.float32)
    )

    payload = _fetch_audio_qk_observer_from_model(model)

    assert payload is not None
    assert payload["debug"]["forward_call_count"] == 7
    assert payload["debug"]["prompt_forward_call_count"] == 2
    assert payload["debug"]["decode_forward_call_count"] == 5
    assert payload["debug"]["positions_sample"] == [[0, 1, 2], [12]]
    assert payload["debug"]["layer_stats"] == {
        "0": {
            "selected_head_count": 1,
            "prompt_audio_capture_count": 2,
            "decode_q_count": 1,
            "missing_prompt_audio_count": 0,
        }
    }
    assert tuple(payload["layer_captures"][0]["prompt_audio_k"].shape) == (1, 2, 2)
    assert tuple(payload["layer_captures"][0]["decode_q"].shape) == (1, 1, 2)
    assert model._alignatt_audio_qk_state is None
    assert model.language_model.model.layers[0].self_attn._alignatt_audio_qk_state is None
    assert model.language_model.model.layers[0].self_attn._alignatt_audio_qk_layer_idx is None


def test_tensor_buffer_observer_roundtrip_preserves_prompt_and_decode_buffers():
    import torch
    from gemma_vllm_alignment_backend import (
        _configure_audio_qk_tensor_observer_on_model,
        _fetch_audio_qk_tensor_observer_from_model,
        _prepare_audio_qk_tensor_observer_on_model,
    )

    class FakeProj:
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.zeros(1))

    class FakeAttention:
        def __init__(self):
            self.qkv_proj = FakeProj()
            self.num_heads = 4
            self.num_kv_heads = 2
            self.head_dim = 2
            self.scaling = 0.5

    class FakeLayer:
        def __init__(self):
            self.self_attn = FakeAttention()

    class FakeInnerModel:
        def __init__(self):
            self.layers = [FakeLayer()]

    class FakeLanguageModel:
        def __init__(self):
            self.model = FakeInnerModel()

    class FakeModel:
        def __init__(self):
            self.language_model = FakeLanguageModel()

    model = FakeModel()
    configure = _configure_audio_qk_tensor_observer_on_model(
        model,
        selected_heads=[{"layer": 0, "head": 1}],
        max_audio_tokens=4,
        max_decode_tokens=3,
    )
    assert configure == {
        "layer_count": 1,
        "max_audio_tokens": 4,
        "max_decode_tokens": 3,
        "storage_mode": "tensor_buffers",
    }

    prepare = _prepare_audio_qk_tensor_observer_on_model(
        model,
        prompt_length=10,
        audio_prompt_start=6,
        audio_prompt_length=4,
    )
    assert prepare == {
        "prompt_length": 10,
        "audio_prompt_start": 6,
        "audio_prompt_length": 4,
        "storage_mode": "tensor_buffers",
    }

    state = model._alignatt_audio_qk_state
    assert state == {
        "storage_mode": "tensor_buffers",
        "layer_indices": (0,),
    }
    observer = model.language_model.model.layers[0].self_attn._alignatt_audio_qk_tensor_observer
    assert set(dict(observer.named_buffers())) >= {
        "prompt_length_tensor",
        "audio_prompt_start_tensor",
        "audio_prompt_length_tensor",
        "forward_call_count_tensor",
        "prompt_audio_k_buffer",
        "decode_q_buffer",
    }
    assert observer.prompt_length_tensor.item() == 10
    assert observer.audio_prompt_start_tensor.item() == 6
    assert observer.audio_prompt_length_tensor.item() == 4
    observer.forward_call_count_tensor.fill_(9)
    observer.prompt_forward_call_count_tensor.fill_(3)
    observer.decode_forward_call_count_tensor.fill_(6)
    observer.prompt_audio_k_buffer[0, :4, :] = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]],
        dtype=torch.float32,
    )
    observer.prompt_written_buffer[:4] = True
    observer.decode_q_buffer[:2, 0, :] = torch.tensor(
        [[3.0, 4.0], [5.0, 6.0]],
        dtype=torch.float32,
    )
    observer.decode_written_buffer[:2] = True

    payload = _fetch_audio_qk_tensor_observer_from_model(model)

    assert payload is not None
    assert payload["prompt_length"] == 10
    assert payload["audio_prompt_positions"] == [6, 7, 8, 9]
    assert payload["debug"]["storage_mode"] == "tensor_buffers"
    assert payload["debug"]["forward_call_count"] == 9
    assert payload["debug"]["decode_forward_call_count"] == 6
    assert payload["debug"]["layer_stats"] == {
        "0": {
            "selected_head_count": 1,
            "prompt_audio_capture_count": 4,
            "decode_q_count": 2,
            "missing_prompt_audio_count": 0,
        }
    }
    assert tuple(payload["layer_captures"][0]["prompt_audio_k"].shape) == (1, 4, 2)
    assert tuple(payload["layer_captures"][0]["decode_q"].shape) == (2, 1, 2)


def test_tensor_buffer_observer_configuration_reuses_existing_module_when_compatible():
    import torch
    from gemma_vllm_alignment_backend import _configure_audio_qk_tensor_observer_on_model

    class FakeProj:
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.zeros(1))

    class FakeAttention:
        def __init__(self):
            self.qkv_proj = FakeProj()
            self.num_heads = 4
            self.num_kv_heads = 2
            self.head_dim = 2
            self.scaling = 0.5

    class FakeLayer:
        def __init__(self):
            self.self_attn = FakeAttention()

    class FakeInnerModel:
        def __init__(self):
            self.layers = [FakeLayer()]

    class FakeLanguageModel:
        def __init__(self):
            self.model = FakeInnerModel()

    class FakeModel:
        def __init__(self):
            self.language_model = FakeLanguageModel()

    model = FakeModel()
    _configure_audio_qk_tensor_observer_on_model(
        model,
        selected_heads=[{"layer": 0, "head": 1}],
        max_audio_tokens=4,
        max_decode_tokens=3,
    )
    observer = model.language_model.model.layers[0].self_attn._alignatt_audio_qk_tensor_observer

    _configure_audio_qk_tensor_observer_on_model(
        model,
        selected_heads=[{"layer": 0, "head": 1}],
        max_audio_tokens=4,
        max_decode_tokens=3,
    )
    reused = model.language_model.model.layers[0].self_attn._alignatt_audio_qk_tensor_observer

    assert reused is observer


def test_tensor_buffer_capture_ignores_out_of_span_positions_without_edge_corruption():
    import torch
    from gemma_vllm_alignment_backend import (
        _capture_audio_qk_into_tensor_buffers,
        _configure_audio_qk_tensor_observer_on_model,
        _prepare_audio_qk_tensor_observer_on_model,
    )

    class FakeProj:
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.zeros(1))

    class FakeAttention:
        def __init__(self):
            self.qkv_proj = FakeProj()
            self.num_heads = 4
            self.num_kv_heads = 2
            self.head_dim = 2
            self.scaling = 1.0

    class FakeLayer:
        def __init__(self):
            self.self_attn = FakeAttention()

    class FakeInnerModel:
        def __init__(self):
            self.layers = [FakeLayer()]

    class FakeLanguageModel:
        def __init__(self):
            self.model = FakeInnerModel()

    class FakeModel:
        def __init__(self):
            self.language_model = FakeLanguageModel()

    model = FakeModel()
    _configure_audio_qk_tensor_observer_on_model(
        model,
        selected_heads=[{"layer": 0, "head": 0}],
        max_audio_tokens=4,
        max_decode_tokens=2,
    )
    _prepare_audio_qk_tensor_observer_on_model(
        model,
        prompt_length=10,
        audio_prompt_start=6,
        audio_prompt_length=4,
    )

    attn = model.language_model.model.layers[0].self_attn
    positions = torch.tensor([5, 6, 8, 10, 12], dtype=torch.int64)
    q = torch.tensor(
        [
            [90.0, 91.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [10.0, 11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [20.0, 21.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [30.0, 31.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [99.0, 98.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    k = torch.tensor(
        [
            [70.0, 71.0, 0.0, 0.0],
            [1.0, 2.0, 0.0, 0.0],
            [3.0, 4.0, 0.0, 0.0],
            [80.0, 81.0, 0.0, 0.0],
            [88.0, 89.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    _capture_audio_qk_into_tensor_buffers(attn, positions, q, k)

    observer = attn._alignatt_audio_qk_tensor_observer
    assert observer.forward_call_count_tensor.item() == 1
    assert observer.prompt_forward_call_count_tensor.item() == 1
    assert observer.decode_forward_call_count_tensor.item() == 1
    assert observer.prompt_written_buffer.tolist() == [True, False, True, False]
    assert observer.decode_written_buffer.tolist() == [True, False]
    assert observer.prompt_audio_k_buffer[0, 0, :].tolist() == [1.0, 2.0]
    assert observer.prompt_audio_k_buffer[0, 2, :].tolist() == [3.0, 4.0]
    assert observer.decode_q_buffer[0, 0, :].tolist() == [30.0, 31.0]
    assert observer.decode_q_buffer[1, 0, :].tolist() == [0.0, 0.0]


def test_gemma_vllm_backend_builds_explicit_compilation_config():
    from pathlib import Path
    from types import SimpleNamespace

    from gemma_vllm_alignment_backend import GemmaVLLMAttentionAlignmentBackend

    backend = GemmaVLLMAttentionAlignmentBackend(
        model_name="stub-model",
        runtime_config=SimpleNamespace(
            gemma_vllm_compilation_mode="vllm_compile",
            gemma_vllm_cudagraph_mode="none",
            gemma_vllm_compile_cache_dir="tmp/test-vllm-cache",
            gemma_vllm_disable_compile_cache=True,
        ),
    )

    config = backend._build_compilation_config()

    assert config == {
        "mode": "vllm_compile",
        "cudagraph_mode": "none",
        "cache_dir": str(Path("tmp/test-vllm-cache").resolve()),
        "inductor_compile_config": {"force_disable_caches": True},
    }


def test_gemma_vllm_backend_builds_no_compilation_config_by_default():
    from types import SimpleNamespace

    from gemma_vllm_alignment_backend import GemmaVLLMAttentionAlignmentBackend

    backend = GemmaVLLMAttentionAlignmentBackend(
        model_name="stub-model",
        runtime_config=SimpleNamespace(),
    )

    assert backend._build_compilation_config() is None


def test_gemma_vllm_backend_disables_prefix_caching_by_default_for_observer_path():
    from types import SimpleNamespace

    from gemma_vllm_alignment_backend import GemmaVLLMAttentionAlignmentBackend

    backend = GemmaVLLMAttentionAlignmentBackend(
        model_name="stub-model",
        runtime_config=SimpleNamespace(),
    )

    assert backend.enable_prefix_caching is False


def test_gemma_vllm_backend_allows_explicit_prefix_caching_override():
    from types import SimpleNamespace

    from gemma_vllm_alignment_backend import GemmaVLLMAttentionAlignmentBackend

    backend = GemmaVLLMAttentionAlignmentBackend(
        model_name="stub-model",
        runtime_config=SimpleNamespace(gemma_vllm_enable_prefix_caching=True),
    )

    assert backend.enable_prefix_caching is True


def test_gemma_vllm_inspect_cli_accepts_compilation_overrides():
    from run_alignment_single_audio import build_cli

    parser = build_cli()
    args = parser.parse_args(
        [
            "gemma_vllm_inspect",
            "--wav",
            "tmp/alignatt_smoke18.wav",
            "--output",
            "tmp/out.json",
            "--vllm-executor-backend",
            "uni",
            "--vllm-patch-mode",
            "preload_class",
            "--no-vllm-enforce-eager",
            "--no-vllm-enable-prefix-caching",
            "--vllm-compilation-mode",
            "vllm_compile",
            "--vllm-cudagraph-mode",
            "none",
            "--vllm-compile-cache-dir",
            "tmp/cache",
            "--vllm-disable-compile-cache",
        ]
    )

    assert args.cmd == "gemma_vllm_inspect"
    assert args.vllm_executor_backend == "uni"
    assert args.vllm_patch_mode == "preload_class"
    assert args.vllm_enforce_eager is False
    assert args.vllm_enable_prefix_caching is False
    assert args.vllm_compilation_mode == "vllm_compile"
    assert args.vllm_cudagraph_mode == "none"
    assert args.vllm_compile_cache_dir == "tmp/cache"
    assert args.vllm_disable_compile_cache is True


def test_audio_too_long_raises_with_explicit_error():
    """Long-audio guard must fail loudly (PLAN.md Phase 5)."""
    import numpy as np
    from gemma_alignment_probe import (
        GemmaAttentionAlignmentBackend,
        GemmaAudioTooLongError,
    )

    # Build a backend without loading the model; only the guard is exercised.
    backend = GemmaAttentionAlignmentBackend.__new__(GemmaAttentionAlignmentBackend)
    backend.max_audio_seconds = 30.0
    audio = np.zeros(31 * 16000, dtype=np.float32)  # 31 s

    raised = False
    try:
        backend._enforce_audio_cap(audio, sample_rate=16000)
    except GemmaAudioTooLongError as exc:
        raised = True
        assert "31" in str(exc) and "30" in str(exc)
    assert raised, "audio past cap must raise GemmaAudioTooLongError, not silently truncate"

    # In-cap audio returns the duration without raising.
    short = np.zeros(5 * 16000, dtype=np.float32)
    assert abs(backend._enforce_audio_cap(short, sample_rate=16000) - 5.0) < 1e-6


def test_qk_fast_prefix_slicing_preserves_audio_features():
    import torch
    from gemma_alignment_probe import GemmaAttentionAlignmentBackend

    inputs = {
        "input_ids": torch.tensor([[11, 12, 13, 14]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        "audio_features": torch.randn(1, 80, 16),
        "non_tensor": "keep",
    }
    sliced = GemmaAttentionAlignmentBackend._slice_inputs_to_prefix(inputs, 2)

    assert sliced["input_ids"].tolist() == [[11, 12]]
    assert sliced["attention_mask"].tolist() == [[1, 1]]
    assert tuple(sliced["audio_features"].shape) == (1, 80, 16)
    assert sliced["non_tensor"] == "keep"


def test_gemma_onepass_backend_uses_public_runtime_id():
    from gemma_alignment_probe import GemmaAttentionAlignmentBackend

    assert GemmaAttentionAlignmentBackend.name == "gemma_onepass_qk_fast"


def test_gemma_vllm_backend_reset_caches_clears_prompt_observer_cache():
    from types import SimpleNamespace

    from gemma_vllm_alignment_backend import (
        GemmaVLLMAttentionAlignmentBackend,
        _PromptObserverCacheEntry,
    )

    backend = GemmaVLLMAttentionAlignmentBackend(
        model_name="stub-model",
        runtime_config=SimpleNamespace(),
    )
    backend._prompt_observer_cache["test"] = _PromptObserverCacheEntry(
        prompt_length=10,
        audio_prompt_positions=(6, 7),
        layer_prompt_audio_k={},
    )
    backend._last_generated_token_ids = [1, 2, 3]
    assert len(backend._prompt_observer_cache) == 1
    assert backend._last_generated_token_ids is not None

    backend.reset_caches()

    assert len(backend._prompt_observer_cache) == 0
    assert backend._last_generated_token_ids is None


def test_gemma_vllm_backend_decode_drift_reports_divergence():
    from types import SimpleNamespace

    from gemma_vllm_alignment_backend import GemmaVLLMAttentionAlignmentBackend

    backend = GemmaVLLMAttentionAlignmentBackend(
        model_name="stub-model",
        runtime_config=SimpleNamespace(),
    )
    # First call: no previous run.
    assert backend._compute_decode_drift([10, 20, 30]) is None

    # Identical second run.
    backend._last_generated_token_ids = [10, 20, 30]
    drift = backend._compute_decode_drift([10, 20, 30])
    assert drift is not None
    assert drift["identical"] is True
    assert drift["first_divergence_index"] is None

    # Divergent run.
    drift = backend._compute_decode_drift([10, 20, 99])
    assert drift is not None
    assert drift["identical"] is False
    assert drift["first_divergence_index"] == 2

    # Length change.
    drift = backend._compute_decode_drift([10, 20, 30, 40])
    assert drift is not None
    assert drift["identical"] is False
    assert drift["first_divergence_index"] == 3
    assert drift["prev_token_count"] == 3
    assert drift["current_token_count"] == 4


def test_gemma_vllm_inspect_cli_accepts_repeat_flag():
    from run_alignment_single_audio import build_cli

    parser = build_cli()
    args = parser.parse_args(
        [
            "gemma_vllm_inspect",
            "--wav", "tmp/alignatt_smoke18.wav",
            "--output", "tmp/out.json",
            "--repeat", "3",
        ]
    )
    assert args.repeat == 3


def test_seam_comparison_cli_parses_correctly():
    from run_alignment_single_audio import build_cli

    parser = build_cli()
    args = parser.parse_args(
        [
            "seam_comparison",
            "--wav", "tmp/alignatt_smoke18.wav",
            "--heads-path", "assets/heads.json",
            "--top-k", "4",
            "--output-dir", "tmp/seam_out",
        ]
    )
    assert args.cmd == "seam_comparison"
    assert args.wav == "tmp/alignatt_smoke18.wav"
    assert args.heads_path == "assets/heads.json"
    assert args.top_k == 4
    assert args.output_dir == "tmp/seam_out"


def test_gemma_vllm_backend_is_valid_alignment_backend_name():
    from cascade_runtime import (
        CascadeRuntimeConfig,
        STABLE_ALIGNMENT_BACKEND_NAMES,
        VALID_ALIGNMENT_BACKEND_NAMES,
    )

    assert "gemma_vllm_qk_fast" in VALID_ALIGNMENT_BACKEND_NAMES
    assert "gemma_vllm_qk_fast" not in STABLE_ALIGNMENT_BACKEND_NAMES
    # Config accepts the experimental name without raising.
    config = CascadeRuntimeConfig(alignment_backend_name="gemma_vllm_qk_fast")
    assert config.alignment_backend_name == "gemma_vllm_qk_fast"


def test_gemma_vllm_backend_runtime_config_defaults_match_validated_seam():
    from cascade_runtime import CascadeRuntimeConfig

    config = CascadeRuntimeConfig(alignment_backend_name="gemma_vllm_qk_fast")
    # Defaults must match the validated cudagraph=full seam (PLAN.md).
    assert config.gemma_vllm_enforce_eager is False
    assert config.gemma_vllm_enable_prefix_caching is False
    assert config.gemma_vllm_cudagraph_mode == "full"


def test_build_alignment_backend_dispatches_to_gemma_vllm():
    """Runtime dispatcher must produce a correctly-configured vLLM backend
    when alignment_backend_name='gemma_vllm_qk_fast'. Does not call load()."""
    from cascade_runtime import CascadeRuntimeConfig, build_alignment_backend
    from gemma_vllm_alignment_backend import GemmaVLLMAttentionAlignmentBackend

    config = CascadeRuntimeConfig(alignment_backend_name="gemma_vllm_qk_fast")
    backend = build_alignment_backend(config)
    assert isinstance(backend, GemmaVLLMAttentionAlignmentBackend)
    assert backend.name == "gemma_vllm_qk_fast"
    assert backend.enforce_eager is False
    assert backend.enable_prefix_caching is False
    assert backend.cudagraph_mode == "full"
    assert backend.worker_mode == "custom_tensor"


def _make_session_for_streaming_prefix_tests():
    """Build a CascadeSession without touching any model.

    LoadedModelBundle is instantiable from config alone; CascadeSession only
    reads ``bundle.config`` in its __init__. That is enough to exercise the
    pure-Python streaming-prefix helpers.
    """
    from cascade_runtime import CascadeRuntimeConfig, CascadeSession, LoadedModelBundle

    config = CascadeRuntimeConfig(
        alignment_backend_name="gemma_vllm_qk_fast",
        asr_streaming_prefix_enabled=True,
        asr_streaming_rollback_words=2,
        asr_streaming_unfixed_chunks=2,
    )
    return CascadeSession(LoadedModelBundle(config))


def _make_words(pairs):
    from alignment_backend import WordAlignment

    return tuple(
        WordAlignment(text=txt, start_time=start, end_time=end)
        for txt, start, end in pairs
    )


def test_streaming_prefix_rejected_for_non_vllm_backend():
    """PLAN step 2: fast-fail validation must reject asr_streaming_prefix_enabled
    with a backend other than gemma_vllm_qk_fast. Today this is caught only
    later via NotImplementedError deep in the backend call."""
    import pytest

    from cascade_runtime import CascadeRuntimeConfig

    for name in ("qwen_forced", "gemma_onepass_qk_fast"):
        with pytest.raises(ValueError, match="asr_streaming_prefix_enabled"):
            CascadeRuntimeConfig(
                alignment_backend_name=name,
                asr_streaming_prefix_enabled=True,
            )

    # Setting the flag via apply_overrides must also revalidate.
    config = CascadeRuntimeConfig(alignment_backend_name="qwen_forced")
    with pytest.raises(ValueError, match="asr_streaming_prefix_enabled"):
        config.apply_overrides(asr_streaming_prefix_enabled=True)


def test_gemma_vllm_force_generate_api_rejected_for_non_vllm_backend():
    """The ablation knob should not silently succeed with the wrong backend."""
    import pytest

    from cascade_runtime import CascadeRuntimeConfig

    with pytest.raises(ValueError, match="gemma_vllm_force_generate_api"):
        CascadeRuntimeConfig(
            alignment_backend_name="qwen_forced",
            gemma_vllm_force_generate_api=True,
        )


def test_gemma_vllm_force_generate_api_default_is_off():
    from cascade_runtime import CascadeRuntimeConfig

    config = CascadeRuntimeConfig(alignment_backend_name="gemma_vllm_qk_fast")
    assert config.gemma_vllm_force_generate_api is False


def test_compute_streaming_prefix_preserves_word_count_invariant():
    """Core invariant: for the prefix returned by _compute_streaming_prefix,
    ``len(remove_punctuation(prefix_text).split()) == len(prefix_words)``.
    If this fails, find_end_time in the sentence-commit path returns None
    and the streaming branch breaks silently at the runtime level."""
    from cascade_runtime import remove_punctuation

    session = _make_session_for_streaming_prefix_tests()
    session._asr_streaming_last_text = "Hello, world how are you"
    session._asr_streaming_last_words = _make_words(
        [("Hello", 0.0, 0.3), ("world", 0.4, 0.7),
         ("how", 0.8, 1.0), ("are", 1.1, 1.2), ("you", 1.3, 1.5)]
    )
    session._asr_streaming_chunk_count = 5  # >= unfixed_chunks

    prefix_text, kept_words = session._compute_streaming_prefix(None)
    assert len(kept_words) == 3  # 5 - rollback(2)
    assert len(remove_punctuation(prefix_text).split()) == len(kept_words)
    assert [w.text for w in kept_words] == ["Hello", "world", "how"]


def test_compute_streaming_prefix_preserves_trailing_punctuation():
    """A period immediately after the last kept word must be carried in the
    prefix so sentence-terminal signals are preserved. After stripping
    punctuation the word count still matches."""
    from cascade_runtime import remove_punctuation

    session = _make_session_for_streaming_prefix_tests()
    session._asr_streaming_last_text = "Hello world. How are you today"
    session._asr_streaming_last_words = _make_words(
        [("Hello", 0.0, 0.3), ("world", 0.4, 0.7),
         ("How", 0.9, 1.0), ("are", 1.1, 1.2),
         ("you", 1.3, 1.4), ("today", 1.5, 1.8)]
    )
    session._asr_streaming_chunk_count = 5
    session.config.asr_streaming_rollback_words = 4  # keep "Hello world"

    prefix_text, kept_words = session._compute_streaming_prefix(None)
    assert prefix_text == "Hello world."
    assert [w.text for w in kept_words] == ["Hello", "world"]
    assert len(remove_punctuation(prefix_text).split()) == len(kept_words)


def test_compute_streaming_prefix_handles_repeated_words():
    """Repeated words are resolved positionally via the cursor-based find,
    not by identity. This was the concrete bug that token-level rollback hit."""
    from cascade_runtime import remove_punctuation

    session = _make_session_for_streaming_prefix_tests()
    session._asr_streaming_last_text = "the the cat sat on the mat"
    session._asr_streaming_last_words = _make_words(
        [("the", 0.0, 0.1), ("the", 0.2, 0.3),
         ("cat", 0.4, 0.6), ("sat", 0.7, 0.9),
         ("on", 1.0, 1.1), ("the", 1.2, 1.3), ("mat", 1.4, 1.7)]
    )
    session._asr_streaming_chunk_count = 5
    session.config.asr_streaming_rollback_words = 2  # keep first 5 words

    prefix_text, kept_words = session._compute_streaming_prefix(None)
    assert len(kept_words) == 5
    assert [w.text for w in kept_words] == ["the", "the", "cat", "sat", "on"]
    assert prefix_text == "the the cat sat on"
    assert len(remove_punctuation(prefix_text).split()) == len(kept_words)


def test_compute_streaming_prefix_returns_empty_before_unfixed_chunks():
    """The first ``asr_streaming_unfixed_chunks`` chunks must run without a
    prefix so the model sees enough context to anchor the hypothesis."""
    session = _make_session_for_streaming_prefix_tests()
    session._asr_streaming_last_text = "Hello world"
    session._asr_streaming_last_words = _make_words(
        [("Hello", 0.0, 0.3), ("world", 0.4, 0.7)]
    )
    session._asr_streaming_chunk_count = 1  # < unfixed_chunks (2)

    prefix_text, kept_words = session._compute_streaming_prefix(None)
    assert prefix_text == ""
    assert kept_words == ()


def test_compute_streaming_prefix_returns_empty_when_rollback_exceeds_words():
    """Rollback larger than the kept word count is the degenerate case; the
    session must fall back to cold decoding instead of emitting an empty
    prefix that still triggers the generate-API path."""
    session = _make_session_for_streaming_prefix_tests()
    session._asr_streaming_last_text = "Hello world"
    session._asr_streaming_last_words = _make_words(
        [("Hello", 0.0, 0.3), ("world", 0.4, 0.7)]
    )
    session._asr_streaming_chunk_count = 3
    session.config.asr_streaming_rollback_words = 5  # > len(last_words)

    prefix_text, kept_words = session._compute_streaming_prefix(None)
    assert prefix_text == ""
    assert kept_words == ()


def test_streaming_state_resets_on_session_clear():
    """session.clear() must wipe streaming state so the next utterance
    starts cold. This is what guarantees the streaming branch cannot leak
    prefix text across speech_id boundaries."""
    session = _make_session_for_streaming_prefix_tests()
    session._asr_streaming_last_text = "something"
    session._asr_streaming_last_words = _make_words([("something", 0.0, 0.3)])
    session._asr_streaming_chunk_count = 7

    session.clear()

    assert session._asr_streaming_last_text == ""
    assert session._asr_streaming_last_words == ()
    assert session._asr_streaming_chunk_count == 0


def test_streaming_state_resets_on_explicit_reset():
    """The session-internal reset helper clears the Python-side state even
    when reset_backend=False; this is the branch taken on in-process
    sentence commits."""
    session = _make_session_for_streaming_prefix_tests()
    session._asr_streaming_last_text = "prior"
    session._asr_streaming_last_words = _make_words([("prior", 0.0, 0.3)])
    session._asr_streaming_chunk_count = 4

    session._reset_asr_streaming_state(reset_backend=False)

    assert session._asr_streaming_last_text == ""
    assert session._asr_streaming_last_words == ()
    assert session._asr_streaming_chunk_count == 0


def test_split_text_at_word_boundary_preserves_word_count_invariant():
    """split_text_at_word_boundary must preserve the invariant that the
    cascade's find_end_time / commit path relies on."""
    from cascade_runtime import remove_punctuation, split_text_at_word_boundary

    text = "Hello, world. How are you today?"
    # total words: Hello world How are you today = 6
    committed, remainder = split_text_at_word_boundary(text, 3)
    assert len(remove_punctuation(committed).split()) == 3
    assert len(remove_punctuation(remainder).split()) == 3
    # Trailing punctuation of the 3rd word carries into the committed part.
    # "Hello, world. How" is the first 3 words with their attached commas/dots.
    assert committed == "Hello, world. How"
    assert remainder == "are you today?"


def test_split_text_at_word_boundary_zero_words_returns_only_remainder():
    from cascade_runtime import split_text_at_word_boundary

    committed, remainder = split_text_at_word_boundary("hello world", 0)
    assert committed == ""
    assert remainder == "hello world"


def test_split_text_at_word_boundary_beyond_total_returns_full_text():
    from cascade_runtime import remove_punctuation, split_text_at_word_boundary

    committed, remainder = split_text_at_word_boundary("one two", 5)
    assert committed == "one two"
    assert remainder == ""
    # Word count should match what the caller asked for the committed side if
    # it exists; otherwise it returns the whole text (graceful fallback).
    assert len(remove_punctuation(committed).split()) == 2


def test_asr_commit_mode_defaults_to_punctuation_lcp_for_qwen_path():
    # On the Qwen3-ASR + Gemma vLLM MT submission path, punctuation_lcp is the
    # default because alignatt_frontier's word-by-word commits give MT
    # fragmented context and cost ~11 BLEU / ~0.3 COMET on en→de (measured on
    # ccpXHNfaoy.wav at the same SHA + same EOS-flush path). alignatt_frontier
    # stays available as an explicit opt-in for Gemma-ASR paths.
    from cascade_runtime import CascadeRuntimeConfig

    config = CascadeRuntimeConfig(alignment_backend_name="qwen_forced")
    assert config.asr_commit_mode == "punctuation_lcp"
    # The margin knob is still exposed and keeps its default even when the
    # current commit mode doesn't use it — flipping mode at session time must
    # not require also passing a margin.
    assert config.asr_alignatt_frontier_margin_ms == 500.0


def test_asr_commit_mode_alignatt_frontier_is_opt_in():
    from cascade_runtime import CascadeRuntimeConfig

    config = CascadeRuntimeConfig(
        alignment_backend_name="qwen_forced",
        asr_commit_mode="alignatt_frontier",
    )
    assert config.asr_commit_mode == "alignatt_frontier"


def test_asr_commit_mode_rejects_unknown_value():
    import pytest

    from cascade_runtime import CascadeRuntimeConfig

    with pytest.raises(ValueError, match="asr_commit_mode"):
        CascadeRuntimeConfig(
            alignment_backend_name="qwen_forced",
            asr_commit_mode="stability_window",
        )


def _session_with_audio_and_hypothesis(
    *,
    audio_duration_s: float,
    words: "tuple",
    text: str,
    margin_ms: float = 500.0,
    commit_mode: str = "alignatt_frontier",
):
    """Build a CascadeSession primed with synthetic state for commit tests."""
    import numpy as np

    from cascade_runtime import (
        CascadeRuntimeConfig,
        CascadeSession,
        LoadedModelBundle,
    )
    from cascade_source_frontier import normalize_word_timestamps_ms

    SAMPLE_RATE = 16000
    config = CascadeRuntimeConfig(
        alignment_backend_name="qwen_forced",
        asr_commit_mode=commit_mode,
        asr_alignatt_frontier_margin_ms=margin_ms,
    )
    bundle = LoadedModelBundle(config)
    session = CascadeSession(bundle)
    # Fake audio of the right duration so len(audio)/SAMPLE_RATE matches
    # the synthetic word timings used by the test.
    n_samples = int(audio_duration_s * SAMPLE_RATE)
    session.state.source = np.zeros(n_samples, dtype=np.float32)
    # The commit path reads the LCP of the last two hypotheses; provide an
    # earlier identical hypothesis so the LCP = text.
    session.state.asr_hypotheses = [text, text]
    session.state.partial_word_timestamps_ms = normalize_word_timestamps_ms(words)
    return session, words


def test_alignatt_frontier_commits_only_words_past_margin():
    """With margin=500ms and audio_frontier=2.0s, only words with
    end_time <= 1.5s are safe. The cascade must commit exactly those."""
    from alignment_backend import AlignmentResult, WordAlignment
    from cascade_runtime import CascadeSession

    words = (
        WordAlignment("Hello", 0.0, 0.4),
        WordAlignment("world", 0.5, 0.9),
        WordAlignment("how", 1.0, 1.4),   # 1.4 <= 1.5, safe
        WordAlignment("are", 1.6, 1.9),   # 1.9 > 1.5, unsafe
        WordAlignment("you", 2.0, 2.0),   # unsafe
    )
    session, _ = _session_with_audio_and_hypothesis(
        audio_duration_s=2.0,
        words=words,
        text="Hello world how are you",
        margin_ms=500.0,
    )
    result = AlignmentResult(
        text="Hello world how are you",
        words=words,
        audio_duration_s=2.0,
    )

    ret = session._try_commit_alignatt_frontier(
        asr_hypo="Hello world how are you",
        result=result,
        lcp_text="Hello world how are you",
        audio=session.state.source,
    )
    assert ret is None  # successful commit path, no abort

    # 3 words committed: Hello, world, how
    assert session.state.utt_sources[1:] == ["Hello world how"]
    assert session.state.asr_hypotheses == ["are you"]
    # utt_timestamps advanced to the end_time of the last committed word (1.4s).
    SAMPLE_RATE = 16000
    expected = int(1.4 * SAMPLE_RATE)
    assert session.state.utt_timestamps[-1] == expected


def test_alignatt_frontier_does_not_commit_when_no_word_is_safe():
    """Every word's end_time is past the frontier - margin boundary.
    No commit should fire; the hypothesis stays as a partial."""
    from alignment_backend import AlignmentResult, WordAlignment
    from cascade_runtime import CascadeSession

    words = (
        WordAlignment("too", 1.2, 1.6),
        WordAlignment("close", 1.7, 1.9),
    )
    session, _ = _session_with_audio_and_hypothesis(
        audio_duration_s=2.0,
        words=words,
        text="too close",
        margin_ms=500.0,
    )
    result = AlignmentResult(text="too close", words=words, audio_duration_s=2.0)

    ret = session._try_commit_alignatt_frontier(
        asr_hypo="too close",
        result=result,
        lcp_text="too close",
        audio=session.state.source,
    )
    assert ret is None  # no abort
    # No commit occurred.
    assert session.state.utt_sources == [""]
    assert session.state.utt_timestamps == [0]


def test_alignatt_frontier_no_commit_without_punctuation_is_unblocked():
    """The commit mechanism must not depend on a sentence-terminal period.
    A long hypothesis with no punctuation whose end_times are all safe still
    commits — exactly the case that killed the streaming branch on Gemma."""
    from alignment_backend import AlignmentResult, WordAlignment
    from cascade_runtime import CascadeSession

    words = tuple(
        WordAlignment(f"word{i}", i * 0.3, i * 0.3 + 0.25)
        for i in range(10)
    )
    text = " ".join(w.text for w in words)  # no punctuation anywhere
    # audio_frontier = 5.0s, margin 500ms -> safe up to end_time 4.5s
    # word 14 has end_time 14 * 0.3 + 0.25 = ... lets compute which ones are safe:
    # word i end_time = 0.3*i + 0.25; <= 4.5 -> i <= (4.5 - 0.25) / 0.3 = 14.1666
    # but we only have 10 words, so all 10 are safe.
    session, _ = _session_with_audio_and_hypothesis(
        audio_duration_s=5.0,
        words=words,
        text=text,
        margin_ms=500.0,
    )
    result = AlignmentResult(text=text, words=words, audio_duration_s=5.0)

    ret = session._try_commit_alignatt_frontier(
        asr_hypo=text, result=result, lcp_text=text, audio=session.state.source,
    )
    assert ret is None
    assert session.state.utt_sources[1:] == [text]
    assert session.state.asr_hypotheses == [""]


def test_alignatt_frontier_returns_none_when_lcp_is_empty():
    """If two consecutive hypotheses share no prefix, we must not commit."""
    from alignment_backend import AlignmentResult, WordAlignment
    from cascade_runtime import CascadeSession

    words = (WordAlignment("only", 0.0, 0.4),)
    session, _ = _session_with_audio_and_hypothesis(
        audio_duration_s=2.0,
        words=words,
        text="only",
        margin_ms=500.0,
    )
    result = AlignmentResult(text="only", words=words, audio_duration_s=2.0)

    ret = session._try_commit_alignatt_frontier(
        asr_hypo="only", result=result, lcp_text="", audio=session.state.source,
    )
    assert ret is None
    assert session.state.utt_sources == [""]


def _run_all() -> None:
    failures = []
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except Exception as exc:  # pragma: no cover - surface every failure
                failures.append((name, exc))
            else:
                print(f"ok  {name}")
    if failures:
        print("")
        for name, exc in failures:
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    _run_all()
