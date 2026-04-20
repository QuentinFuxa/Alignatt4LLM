"""Experimental vLLM-native Gemma audio AlignAtt backend.

This backend is intentionally narrow in scope:

- single-request, single-audio diagnostic harness only
- ASR-side Gemma transcript generation runs through vLLM
- AlignAtt observer rows are reconstructed from Q/K captured inside the
  vLLM worker on selected heads

It does *not* change the stable SimulStream runtime surface yet. The goal
is to validate the engine-native seam proposed in ``PLAN.md`` before
promoting anything into the canonical streaming path.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import hashlib
from pathlib import Path
from time import perf_counter
from functools import partial
from types import MethodType, SimpleNamespace
from typing import Any, Sequence
import os

import numpy as np
import torch

from cascade.alignment.base import (
    AlignAttObserverToken,
    AlignmentBackend,
    AlignmentResult,
    WordAlignment,
)
from cascade.alignment.gemma_alignatt_stream import AlignAttStepRaw
from cascade.mt.base import AlignAttHead, compute_alignatt_source_argmaxes
from cascade.alignment.gemma_transformers_asr_backend import (
    AudioSpan,
    GEMMA_AUDIO_MAX_SECONDS_DEFAULT,
    GEMMA_AUDIO_MS_PER_TOKEN_DEFAULT,
    GEMMA_AUDIO_TOKEN_ID_DEFAULT,
    GemmaAudioTooLongError,
    _apply_word_end_offset,
    _enforce_monotone,
    aggregate_token_timings_to_words,
    audio_position_to_end_seconds,
    detect_audio_span,
    load_audio_alignment_heads,
    monotonicity_score,
)


# The experimental backend uses worker callables via ``collective_rpc`` to
# install and fetch compact observer state. vLLM gates callable transport
# behind this explicit local-only switch.
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
_ALIGNATT_TENSOR_OBSERVER_BOOTSTRAP_ENV = "ALIGNATT_VLLM_TENSOR_OBSERVER_BOOTSTRAP"

# Single verbatim-transcription instruction shared by the non-streaming
# full-utterance path and the streaming AlignAtt path. We deliberately
# removed the four-regime instruction matrix: regime-specific wording
# could leak into the transcript and then recirculate through the forced
# prefix, which is the exact desync AlignAtt streaming must avoid.
GEMMA_ASR_INSTRUCTION = (
    "Provide a verbatim, word-for-word transcription of the audio. "
    "Only output the transcription, with no newlines."
)
GEMMA_VLLM_SAMPLING_MODE_SHIPPING = "shipping"
GEMMA_VLLM_SAMPLING_MODE_HF_MODEL_CARD = "hf_model_card"
GEMMA_VLLM_SAMPLING_MODES = (
    GEMMA_VLLM_SAMPLING_MODE_SHIPPING,
    GEMMA_VLLM_SAMPLING_MODE_HF_MODEL_CARD,
)


@dataclass(frozen=True)
class _GemmaASRPromptLayout:
    user_only_token_ids: tuple[int, ...]
    prefix_token_ids: tuple[int, ...]
    prompt_token_ids: tuple[int, ...]
    audio_span: AudioSpan
    audio_placeholder_token_count: int

    @property
    def audio_token_count(self) -> int:
        return int(self.audio_span.length)

    @property
    def non_audio_prompt_tokens(self) -> int:
        return int(len(self.user_only_token_ids) - self.audio_placeholder_token_count)

    @property
    def prompt_token_count(self) -> int:
        return int(
            self.non_audio_prompt_tokens
            + len(self.prefix_token_ids)
            + self.audio_token_count
        )


def build_gemma_vllm_sampling_params(
    *,
    runtime_config: SimpleNamespace,
    max_new_tokens: int,
):
    from vllm import SamplingParams

    sampling_mode = str(
        getattr(
            runtime_config,
            "gemma_vllm_sampling_mode",
            GEMMA_VLLM_SAMPLING_MODE_SHIPPING,
        )
        or GEMMA_VLLM_SAMPLING_MODE_SHIPPING
    )
    if sampling_mode == GEMMA_VLLM_SAMPLING_MODE_SHIPPING:
        return SamplingParams(
            temperature=0.0,
            max_tokens=int(max_new_tokens),
            repetition_penalty=float(
                getattr(runtime_config, "repetition_penalty", 1.0)
            ),
            skip_special_tokens=True,
        ), {
            "sampling_mode": sampling_mode,
            "temperature": 0.0,
            "top_p": None,
            "top_k": None,
            "repetition_penalty": float(
                getattr(runtime_config, "repetition_penalty", 1.0)
            ),
        }
    if sampling_mode == GEMMA_VLLM_SAMPLING_MODE_HF_MODEL_CARD:
        # Match the public Gemma 4 model-card recommendation for ASR probes.
        return SamplingParams(
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            max_tokens=int(max_new_tokens),
            repetition_penalty=1.0,
            skip_special_tokens=True,
        ), {
            "sampling_mode": sampling_mode,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
            "repetition_penalty": 1.0,
        }
    raise ValueError(
        f"Unknown Gemma vLLM sampling mode {sampling_mode!r}; expected one of "
        f"{GEMMA_VLLM_SAMPLING_MODES}."
    )


def _resolve_vllm_gemma_decoder_layers(model) -> Sequence[object]:
    candidates = (
        getattr(getattr(getattr(model, "language_model", None), "model", None), "layers", None),
        getattr(getattr(model, "model", None), "layers", None),
        getattr(model, "layers", None),
    )
    for candidate in candidates:
        if candidate is not None:
            return candidate
    raise RuntimeError(
        "Could not locate Gemma decoder layers inside the loaded vLLM model."
    )


def _encode_tensor_observer_bootstrap(
    *,
    selected_heads: Sequence[dict[str, int]],
    max_audio_tokens: int,
    max_decode_tokens: int,
) -> str:
    return json.dumps(
        {
            "selected_heads": [
                {"layer": int(head["layer"]), "head": int(head["head"])}
                for head in selected_heads
            ],
            "max_audio_tokens": int(max_audio_tokens),
            "max_decode_tokens": int(max_decode_tokens),
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def _decode_tensor_observer_bootstrap_from_env() -> dict[str, Any] | None:
    raw = os.environ.get(_ALIGNATT_TENSOR_OBSERVER_BOOTSTRAP_ENV)
    if not raw:
        return None
    payload = json.loads(raw)
    return {
        "selected_heads": [
            {"layer": int(head["layer"]), "head": int(head["head"])}
            for head in payload.get("selected_heads", [])
        ],
        "max_audio_tokens": int(payload["max_audio_tokens"]),
        "max_decode_tokens": int(payload["max_decode_tokens"]),
    }


class _AudioQKTensorObserver(torch.nn.Module):
    """Per-layer tensor observer state carried by explicit PyTorch buffers.

    The compile-safe path should expose real module buffers to Dynamo/Inductor
    instead of mutating tensors hidden inside ad hoc Python dicts.
    """

    def __init__(
        self,
        *,
        selected_heads: Sequence[int],
        selected_kv_heads: Sequence[int],
        max_audio_tokens: int,
        max_decode_tokens: int,
        head_dim: int,
        scaling: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.selected_heads = tuple(int(head) for head in selected_heads)
        self.scaling = float(scaling)
        self.head_dim = int(head_dim)
        self.register_buffer(
            "selected_heads_tensor",
            torch.tensor(list(self.selected_heads), device=device, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "selected_kv_heads_tensor",
            torch.tensor(list(selected_kv_heads), device=device, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "prompt_length_tensor",
            torch.zeros((), device=device, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "audio_prompt_start_tensor",
            torch.zeros((), device=device, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "audio_prompt_length_tensor",
            torch.zeros((), device=device, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "forward_call_count_tensor",
            torch.zeros((), device=device, dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "prompt_forward_call_count_tensor",
            torch.zeros((), device=device, dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "decode_forward_call_count_tensor",
            torch.zeros((), device=device, dtype=torch.int32),
            persistent=False,
        )

        head_count = len(self.selected_heads)
        self.register_buffer(
            "prompt_audio_k_buffer",
            torch.zeros(
                (head_count, int(max_audio_tokens), int(head_dim)),
                device=device,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "prompt_audio_k_scratch",
            torch.zeros(
                (head_count, int(max_audio_tokens), int(head_dim)),
                device=device,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "prompt_written_buffer",
            torch.zeros((int(max_audio_tokens),), device=device, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "prompt_written_scratch",
            torch.zeros((int(max_audio_tokens),), device=device, dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "decode_q_buffer",
            torch.zeros(
                (int(max_decode_tokens), head_count, int(head_dim)),
                device=device,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "decode_q_scratch",
            torch.zeros(
                (int(max_decode_tokens), head_count, int(head_dim)),
                device=device,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "decode_written_buffer",
            torch.zeros((int(max_decode_tokens),), device=device, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "decode_written_scratch",
            torch.zeros((int(max_decode_tokens),), device=device, dtype=torch.int32),
            persistent=False,
        )

    def prepare(
        self,
        *,
        prompt_length: int,
        audio_prompt_start: int,
        audio_prompt_length: int,
    ) -> None:
        max_audio_tokens = int(self.prompt_audio_k_buffer.shape[1])
        if int(audio_prompt_length) > max_audio_tokens:
            raise ValueError(
                f"audio_prompt_length={audio_prompt_length} exceeds configured "
                f"max_audio_tokens={max_audio_tokens}"
            )
        self.prompt_length_tensor.fill_(int(prompt_length))
        self.audio_prompt_start_tensor.fill_(int(audio_prompt_start))
        self.audio_prompt_length_tensor.fill_(int(audio_prompt_length))
        self.forward_call_count_tensor.zero_()
        self.prompt_forward_call_count_tensor.zero_()
        self.decode_forward_call_count_tensor.zero_()
        self.prompt_audio_k_buffer.zero_()
        self.prompt_audio_k_scratch.zero_()
        self.prompt_written_buffer.zero_()
        self.prompt_written_scratch.zero_()
        self.decode_q_buffer.zero_()
        self.decode_q_scratch.zero_()
        self.decode_written_buffer.zero_()
        self.decode_written_scratch.zero_()


@dataclass
class _PromptObserverCacheEntry:
    prompt_length: int
    audio_prompt_positions: tuple[int, ...]
    layer_prompt_audio_k: dict[int, np.ndarray]


def _get_audio_qk_tensor_observer(attn_module) -> _AudioQKTensorObserver | None:
    observer = getattr(attn_module, "_alignatt_audio_qk_tensor_observer", None)
    if observer is None:
        return None
    if not isinstance(observer, _AudioQKTensorObserver):
        raise TypeError(
            "Expected _alignatt_audio_qk_tensor_observer to be an "
            "_AudioQKTensorObserver instance."
        )
    return observer


def _resolve_tensor_observer_bindings(
    model,
) -> list[tuple[int, _AudioQKTensorObserver]]:
    state = getattr(model, "_alignatt_audio_qk_state", None)
    if state is None or state.get("storage_mode") != "tensor_buffers":
        return []

    layers = _resolve_vllm_gemma_decoder_layers(model)
    bindings: list[tuple[int, _AudioQKTensorObserver]] = []
    for layer_idx in state.get("layer_indices", ()):
        attn = getattr(layers[int(layer_idx)], "self_attn", None)
        if attn is None:
            raise RuntimeError(f"Layer {layer_idx} has no self_attn module.")
        observer = _get_audio_qk_tensor_observer(attn)
        if observer is None:
            raise RuntimeError(
                f"Layer {layer_idx} is missing the configured tensor observer."
            )
        bindings.append((int(layer_idx), observer))
    return bindings


def _make_patched_gemma4_attention_forward():
    def _patched_forward(self, positions, hidden_states, **kwargs):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        q = self.q_norm(q)
        q = q.flatten(-2, -1)

        if not self.is_kv_shared_layer:
            k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
            k = self.k_norm(k)
            k = k.flatten(-2, -1)
            q, k = self.rotary_emb(positions, q, k)

            v = v.unflatten(-1, (self.num_kv_heads, self.head_dim))
            v = self.v_norm(v)
            v = v.flatten(-2, -1)
        else:
            q = self.rotary_emb(positions, q, k)[0]

        _capture_audio_qk_from_attention_module(self, positions, q, k)

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    return _patched_forward


def _make_tensor_buffer_gemma4_attention_forward():
    def _patched_forward(self, positions, hidden_states, **kwargs):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        q = self.q_norm(q)
        q = q.flatten(-2, -1)

        if not self.is_kv_shared_layer:
            k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
            k = self.k_norm(k)
            k = k.flatten(-2, -1)
            q, k = self.rotary_emb(positions, q, k)

            v = v.unflatten(-1, (self.num_kv_heads, self.head_dim))
            v = self.v_norm(v)
            v = v.flatten(-2, -1)
        else:
            q = self.rotary_emb(positions, q, k)[0]

        _capture_audio_qk_into_tensor_buffers(self, positions, q, k)

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    return _patched_forward


def install_global_gemma4_attention_patch() -> None:
    from vllm.model_executor.models.gemma4 import Gemma4Attention

    if not hasattr(Gemma4Attention, "_alignatt_audio_qk_original_forward"):
        Gemma4Attention._alignatt_audio_qk_original_forward = Gemma4Attention.forward
        Gemma4Attention.forward = _make_patched_gemma4_attention_forward()


def install_global_gemma4_attention_tensor_patch() -> None:
    from vllm.model_executor.models.gemma4 import Gemma4Attention

    if not hasattr(Gemma4Attention, "_alignatt_audio_qk_tensor_original_forward"):
        Gemma4Attention._alignatt_audio_qk_tensor_original_forward = Gemma4Attention.forward
        Gemma4Attention.forward = _make_tensor_buffer_gemma4_attention_forward()


def _capture_audio_qk_from_attention_module(attn_module, positions, q, k) -> None:
    state = getattr(attn_module, "_alignatt_audio_qk_state", None)
    layer_idx = getattr(attn_module, "_alignatt_audio_qk_layer_idx", None)
    if state is None or layer_idx is None:
        return

    layer_state = state["layer_captures"].get(int(layer_idx))
    if layer_state is None:
        return

    from cascade.mt.base import map_attention_head_to_key_value_head

    positions_list = [int(pos) for pos in positions.reshape(-1).detach().cpu().tolist()]
    if not positions_list:
        return

    debug_state = state.setdefault(
        "debug",
        {
            "forward_call_count": 0,
            "prompt_forward_call_count": 0,
            "decode_forward_call_count": 0,
            "positions_sample": [],
            "prompt_positions_sample": [],
            "decode_positions_sample": [],
        },
    )
    debug_state["forward_call_count"] += 1
    if len(debug_state["positions_sample"]) < 6:
        debug_state["positions_sample"].append(list(positions_list[:16]))

    q_heads = q.reshape(-1, attn_module.num_heads, attn_module.head_dim)
    k_heads = k.reshape(-1, attn_module.num_kv_heads, attn_module.head_dim)
    prompt_length = int(state["prompt_length"])
    audio_prompt_position_set = state["audio_prompt_position_set"]
    selected_heads = list(layer_state["selected_heads"])
    prompt_positions_seen = [pos for pos in positions_list if pos < prompt_length]
    decode_positions_seen = [pos for pos in positions_list if pos >= prompt_length]
    if prompt_positions_seen:
        debug_state["prompt_forward_call_count"] += 1
        if len(debug_state["prompt_positions_sample"]) < 6:
            debug_state["prompt_positions_sample"].append(
                list(prompt_positions_seen[:16])
            )
    if decode_positions_seen:
        debug_state["decode_forward_call_count"] += 1
        if len(debug_state["decode_positions_sample"]) < 6:
            debug_state["decode_positions_sample"].append(
                list(decode_positions_seen[:16])
            )

    for local_idx, absolute_position in enumerate(positions_list):
        if absolute_position >= prompt_length:
            continue
        if absolute_position not in audio_prompt_position_set:
            continue
        if absolute_position in layer_state["prompt_audio_k_by_position"]:
            continue
        per_head_key_rows: list[np.ndarray] = []
        for head_index in selected_heads:
            kv_head_index = map_attention_head_to_key_value_head(
                int(head_index),
                num_attention_heads=int(attn_module.num_heads),
                num_key_value_heads=int(attn_module.num_kv_heads),
            )
            per_head_key_rows.append(
                k_heads[local_idx, kv_head_index, :].detach().float().cpu().numpy()
            )
        layer_state["prompt_audio_k_by_position"][int(absolute_position)] = np.stack(
            per_head_key_rows,
            axis=0,
        )

    for local_idx, absolute_position in enumerate(positions_list):
        if absolute_position < prompt_length:
            continue
        layer_state["decode_q"].append(
            q_heads[local_idx, selected_heads, :].detach().float().cpu().numpy()
        )


def _capture_audio_qk_into_tensor_buffers(attn_module, positions, q, k) -> None:
    observer = _get_audio_qk_tensor_observer(attn_module)
    if observer is None:
        return

    positions_flat = positions.reshape(-1).to(dtype=torch.int64)
    if positions_flat.numel() == 0:
        return

    q_heads = q.reshape(-1, attn_module.num_heads, attn_module.head_dim)
    k_heads = k.reshape(-1, attn_module.num_kv_heads, attn_module.head_dim)
    selected_q = torch.index_select(
        q_heads,
        dim=1,
        index=observer.selected_heads_tensor,
    ).to(dtype=torch.float32)
    selected_k = torch.index_select(
        k_heads,
        dim=1,
        index=observer.selected_kv_heads_tensor,
    ).to(dtype=torch.float32)

    prompt_length = observer.prompt_length_tensor
    audio_prompt_start = observer.audio_prompt_start_tensor
    audio_prompt_length = observer.audio_prompt_length_tensor
    max_decode_tokens = int(observer.decode_q_buffer.shape[0])

    observer.forward_call_count_tensor.add_(1)

    prompt_offsets = positions_flat - audio_prompt_start
    prompt_mask = (
        (positions_flat < prompt_length)
        & (prompt_offsets >= 0)
        & (prompt_offsets < audio_prompt_length)
    )
    prompt_mask_i32 = prompt_mask.any().to(dtype=torch.int32)
    observer.prompt_forward_call_count_tensor.add_(prompt_mask_i32)
    max_audio_tokens = observer.prompt_audio_k_buffer.shape[1]
    prompt_offsets_clamped = prompt_offsets.clamp(min=0, max=max_audio_tokens - 1)
    prompt_values = selected_k.transpose(0, 1)
    prompt_mask_f32 = prompt_mask.to(dtype=prompt_values.dtype).view(1, -1, 1)
    prompt_index = prompt_offsets_clamped.view(1, -1, 1).expand(
        prompt_values.shape[0],
        -1,
        prompt_values.shape[2],
    )
    prompt_scratch = observer.prompt_audio_k_scratch
    prompt_scratch.zero_()
    prompt_scratch.scatter_add_(1, prompt_index, prompt_values * prompt_mask_f32)
    prompt_written_scratch = observer.prompt_written_scratch
    prompt_written_scratch.zero_()
    prompt_written_scratch.scatter_reduce_(
        0,
        prompt_offsets_clamped,
        prompt_mask.to(dtype=prompt_written_scratch.dtype),
        reduce="amax",
        include_self=False,
    )
    prompt_write_mask = prompt_written_scratch.to(dtype=torch.bool)
    observer.prompt_audio_k_buffer.copy_(
        torch.where(
            prompt_write_mask.view(1, -1, 1),
            prompt_scratch,
            observer.prompt_audio_k_buffer,
        )
    )
    observer.prompt_written_buffer.logical_or_(prompt_write_mask)

    decode_offsets = positions_flat - prompt_length
    decode_mask = (
        (positions_flat >= prompt_length)
        & (decode_offsets >= 0)
        & (decode_offsets < max_decode_tokens)
    )
    decode_mask_i32 = decode_mask.any().to(dtype=torch.int32)
    observer.decode_forward_call_count_tensor.add_(decode_mask_i32)
    decode_offsets_clamped = decode_offsets.clamp(min=0, max=max_decode_tokens - 1)
    decode_mask_f32 = decode_mask.to(dtype=selected_q.dtype).view(-1, 1, 1)
    decode_index = decode_offsets_clamped.view(-1, 1, 1).expand(
        -1,
        selected_q.shape[1],
        selected_q.shape[2],
    )
    decode_scratch = observer.decode_q_scratch
    decode_scratch.zero_()
    decode_scratch.scatter_add_(0, decode_index, selected_q * decode_mask_f32)
    decode_written_scratch = observer.decode_written_scratch
    decode_written_scratch.zero_()
    decode_written_scratch.scatter_reduce_(
        0,
        decode_offsets_clamped,
        decode_mask.to(dtype=decode_written_scratch.dtype),
        reduce="amax",
        include_self=False,
    )
    decode_write_mask = decode_written_scratch.to(dtype=torch.bool)
    observer.decode_q_buffer.copy_(
        torch.where(
            decode_write_mask.view(-1, 1, 1),
            decode_scratch,
            observer.decode_q_buffer,
        )
    )
    observer.decode_written_buffer.logical_or_(decode_write_mask)


def _install_audio_qk_observer_on_model(
    model,
    *,
    selected_heads: Sequence[dict[str, int]],
    audio_prompt_positions: Sequence[int],
    prompt_length: int,
    patch_mode: str,
) -> dict[str, Any]:
    layers = _resolve_vllm_gemma_decoder_layers(model)

    heads_by_layer: dict[int, list[int]] = {}
    for head in selected_heads:
        heads_by_layer.setdefault(int(head["layer"]), []).append(int(head["head"]))

    state = {
        "prompt_length": int(prompt_length),
        "audio_prompt_positions": tuple(int(pos) for pos in audio_prompt_positions),
        "audio_prompt_position_set": {
            int(pos) for pos in audio_prompt_positions
        },
        "layer_captures": {},
    }

    for layer_idx, layer_head_indices in heads_by_layer.items():
        layer = layers[int(layer_idx)]
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            raise RuntimeError(f"Layer {layer_idx} has no self_attn module.")

        state["layer_captures"][int(layer_idx)] = {
            "selected_heads": list(layer_head_indices),
            "prompt_audio_k_by_position": {},
            "decode_q": [],
            "scaling": float(getattr(attn, "scaling", 1.0)),
            "head_dim": int(getattr(attn, "head_dim", 0)),
        }

        if patch_mode == "postload_instance":
            if not hasattr(attn, "_alignatt_audio_qk_original_forward"):
                attn._alignatt_audio_qk_original_forward = attn.forward
                attn.forward = MethodType(_make_patched_gemma4_attention_forward(), attn)
        elif patch_mode != "preload_class":
            raise ValueError(f"Unknown vLLM patch_mode: {patch_mode!r}")

        attn._alignatt_audio_qk_state = state
        attn._alignatt_audio_qk_layer_idx = int(layer_idx)

    model._alignatt_audio_qk_state = state
    return {
        "layer_count": len(heads_by_layer),
        "audio_prompt_length": len(audio_prompt_positions),
        "patch_mode": str(patch_mode),
    }


def _configure_audio_qk_tensor_observer_on_model(
    model,
    *,
    selected_heads: Sequence[dict[str, int]],
    max_audio_tokens: int,
    max_decode_tokens: int,
) -> dict[str, Any]:
    from cascade.mt.base import map_attention_head_to_key_value_head

    layers = _resolve_vllm_gemma_decoder_layers(model)
    heads_by_layer: dict[int, list[int]] = {}
    for head in selected_heads:
        heads_by_layer.setdefault(int(head["layer"]), []).append(int(head["head"]))

    if not heads_by_layer:
        raise ValueError("selected_heads must not be empty for tensor-buffer observer")

    layer_indices: list[int] = []

    for layer_idx, layer_head_indices in heads_by_layer.items():
        layer = layers[int(layer_idx)]
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            raise RuntimeError(f"Layer {layer_idx} has no self_attn module.")
        device = attn.qkv_proj.weight.device

        selected_kv_heads = [
            map_attention_head_to_key_value_head(
                int(head_index),
                num_attention_heads=int(attn.num_heads),
                num_key_value_heads=int(attn.num_kv_heads),
            )
            for head_index in layer_head_indices
        ]
        observer = _get_audio_qk_tensor_observer(attn)
        if observer is None or any(
            (
                tuple(observer.selected_heads) != tuple(int(head) for head in layer_head_indices),
                tuple(int(head) for head in observer.selected_kv_heads_tensor.tolist())
                != tuple(int(head) for head in selected_kv_heads),
                int(observer.prompt_audio_k_buffer.shape[1]) != int(max_audio_tokens),
                int(observer.decode_q_buffer.shape[0]) != int(max_decode_tokens),
                int(observer.head_dim) != int(attn.head_dim),
            )
        ):
            observer = _AudioQKTensorObserver(
                selected_heads=list(layer_head_indices),
                selected_kv_heads=selected_kv_heads,
                max_audio_tokens=int(max_audio_tokens),
                max_decode_tokens=int(max_decode_tokens),
                head_dim=int(attn.head_dim),
                scaling=float(getattr(attn, "scaling", 1.0)),
                device=device,
            )
        attn._alignatt_audio_qk_tensor_observer = observer
        attn._alignatt_audio_qk_state = None
        attn._alignatt_audio_qk_layer_idx = None
        layer_indices.append(int(layer_idx))

    model._alignatt_audio_qk_state = {
        "storage_mode": "tensor_buffers",
        "layer_indices": tuple(sorted(layer_indices)),
    }
    return {
        "layer_count": len(heads_by_layer),
        "max_audio_tokens": int(max_audio_tokens),
        "max_decode_tokens": int(max_decode_tokens),
        "storage_mode": "tensor_buffers",
    }


def _prepare_audio_qk_tensor_observer_on_model(
    model,
    *,
    prompt_length: int,
    audio_prompt_start: int,
    audio_prompt_length: int,
) -> dict[str, Any]:
    state = getattr(model, "_alignatt_audio_qk_state", None)
    if state is None or state.get("storage_mode") != "tensor_buffers":
        raise RuntimeError("Tensor-buffer audio observer is not configured on the model.")

    for _layer_idx, observer in _resolve_tensor_observer_bindings(model):
        observer.prepare(
            prompt_length=int(prompt_length),
            audio_prompt_start=int(audio_prompt_start),
            audio_prompt_length=int(audio_prompt_length),
        )

    return {
        "prompt_length": int(prompt_length),
        "audio_prompt_start": int(audio_prompt_start),
        "audio_prompt_length": int(audio_prompt_length),
        "storage_mode": "tensor_buffers",
    }


def _fetch_audio_qk_observer_from_model(model) -> dict[str, Any] | None:
    state = getattr(model, "_alignatt_audio_qk_state", None)
    if state is None:
        return None

    payload: dict[str, Any] = {
        "prompt_length": int(state["prompt_length"]),
        "audio_prompt_positions": list(state["audio_prompt_positions"]),
        "layer_captures": {},
        "debug": {
            "forward_call_count": int(state.get("debug", {}).get("forward_call_count", 0)),
            "prompt_forward_call_count": int(
                state.get("debug", {}).get("prompt_forward_call_count", 0)
            ),
            "decode_forward_call_count": int(
                state.get("debug", {}).get("decode_forward_call_count", 0)
            ),
            "positions_sample": [
                list(sample)
                for sample in state.get("debug", {}).get("positions_sample", [])
            ],
            "prompt_positions_sample": [
                list(sample)
                for sample in state.get("debug", {}).get("prompt_positions_sample", [])
            ],
            "decode_positions_sample": [
                list(sample)
                for sample in state.get("debug", {}).get("decode_positions_sample", [])
            ],
            "layer_stats": {},
        },
    }
    for layer_idx, layer_state in state["layer_captures"].items():
        ordered_positions = list(state["audio_prompt_positions"])
        prompt_audio_missing_positions = [
            int(pos)
            for pos in ordered_positions
            if pos not in layer_state["prompt_audio_k_by_position"]
        ]
        prompt_audio_k = None
        if not prompt_audio_missing_positions and ordered_positions:
            prompt_audio_k = np.stack(
                [
                    layer_state["prompt_audio_k_by_position"][int(pos)]
                    for pos in ordered_positions
                ],
                axis=1,
            )
        decode_q = np.stack(layer_state["decode_q"], axis=0) if layer_state["decode_q"] else None
        payload["layer_captures"][int(layer_idx)] = {
            "selected_heads": list(layer_state["selected_heads"]),
            "prompt_audio_k": prompt_audio_k,
            "prompt_audio_missing_positions": prompt_audio_missing_positions,
            "decode_q": decode_q,
            "scaling": float(layer_state["scaling"]),
            "head_dim": int(layer_state["head_dim"]),
        }
        payload["debug"]["layer_stats"][str(int(layer_idx))] = {
            "selected_head_count": len(layer_state["selected_heads"]),
            "prompt_audio_capture_count": len(layer_state["prompt_audio_k_by_position"]),
            "decode_q_count": len(layer_state["decode_q"]),
            "missing_prompt_audio_count": len(prompt_audio_missing_positions),
        }

    for layer in _resolve_vllm_gemma_decoder_layers(model):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        if hasattr(attn, "_alignatt_audio_qk_state"):
            attn._alignatt_audio_qk_state = None
        if hasattr(attn, "_alignatt_audio_qk_layer_idx"):
            attn._alignatt_audio_qk_layer_idx = None
    model._alignatt_audio_qk_state = None
    return payload


def _fetch_audio_qk_tensor_observer_from_model(model) -> dict[str, Any] | None:
    bindings = _resolve_tensor_observer_bindings(model)
    if not bindings:
        return None

    first_observer = bindings[0][1]
    audio_prompt_length = int(first_observer.audio_prompt_length_tensor.item())
    payload: dict[str, Any] = {
        "prompt_length": int(first_observer.prompt_length_tensor.item()),
        "audio_prompt_positions": list(
            range(
                int(first_observer.audio_prompt_start_tensor.item()),
                int(first_observer.audio_prompt_start_tensor.item()) + audio_prompt_length,
            )
        ),
        "layer_captures": {},
        "debug": {
            "storage_mode": "tensor_buffers",
            "forward_call_count": 0,
            "prompt_forward_call_count": 0,
            "decode_forward_call_count": 0,
            "positions_sample": [],
            "prompt_positions_sample": [],
            "decode_positions_sample": [],
            "layer_stats": {},
        },
    }
    for layer_idx, observer in bindings:
        payload["debug"]["forward_call_count"] += int(observer.forward_call_count_tensor.item())
        payload["debug"]["prompt_forward_call_count"] += int(
            observer.prompt_forward_call_count_tensor.item()
        )
        payload["debug"]["decode_forward_call_count"] += int(
            observer.decode_forward_call_count_tensor.item()
        )

        prompt_written = observer.prompt_written_buffer[:audio_prompt_length]
        prompt_written_cpu = prompt_written.detach().cpu()
        missing_prompt_audio = [
            int(local_idx)
            for local_idx, present in enumerate(prompt_written_cpu.tolist())
            if not present
        ]
        prompt_audio_k = None
        if not missing_prompt_audio and audio_prompt_length > 0:
            prompt_audio_k = (
                observer.prompt_audio_k_buffer[:, :audio_prompt_length, :]
                .detach()
                .float()
                .cpu()
                .numpy()
            )

        decode_written_cpu = observer.decode_written_buffer.detach().cpu().tolist()
        decode_q_count = 0
        for present in decode_written_cpu:
            if not present:
                break
            decode_q_count += 1
        decode_q = None
        if decode_q_count > 0:
            decode_q = (
                observer.decode_q_buffer[:decode_q_count]
                .detach()
                .float()
                .cpu()
                .numpy()
            )

        payload["layer_captures"][int(layer_idx)] = {
            "selected_heads": list(observer.selected_heads),
            "prompt_audio_k": prompt_audio_k,
            "prompt_audio_missing_positions": missing_prompt_audio,
            "decode_q": decode_q,
            "scaling": float(observer.scaling),
            "head_dim": int(observer.head_dim),
        }
        payload["debug"]["layer_stats"][str(int(layer_idx))] = {
            "selected_head_count": len(observer.selected_heads),
            "prompt_audio_capture_count": int(prompt_written_cpu.sum().item()),
            "decode_q_count": int(decode_q_count),
            "missing_prompt_audio_count": len(missing_prompt_audio),
        }

    return payload


def _apply_install_audio_qk_observer_on_model(
    model,
    selected_heads: Sequence[dict[str, int]],
    audio_prompt_positions: Sequence[int],
    prompt_length: int,
    patch_mode: str,
) -> dict[str, Any]:
    return _install_audio_qk_observer_on_model(
        model,
        selected_heads=selected_heads,
        audio_prompt_positions=audio_prompt_positions,
        prompt_length=prompt_length,
        patch_mode=patch_mode,
    )


def _rpc_install_audio_qk_observer(
    worker,
    selected_heads: Sequence[dict[str, int]],
    audio_prompt_positions: Sequence[int],
    prompt_length: int,
    patch_mode: str = "postload_instance",
) -> dict[str, Any]:
    return _install_audio_qk_observer_on_model(
        worker.get_model(),
        selected_heads=selected_heads,
        audio_prompt_positions=audio_prompt_positions,
        prompt_length=prompt_length,
        patch_mode=patch_mode,
    )


def _rpc_fetch_audio_qk_observer(worker) -> dict[str, Any] | None:
    return _fetch_audio_qk_observer_from_model(worker.get_model())


def reconstruct_vllm_audio_attention_rows(
    capture_payload: dict[str, Any] | None,
    *,
    alignatt_heads: Sequence[AlignAttHead],
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    if not capture_payload or not alignatt_heads:
        return [], {
            "captured_layer_count": 0,
            "effective_head_count": 0,
            "generated_token_count": 0,
            "audio_span_length": 0,
            "missing_heads": [],
        }

    layer_captures = capture_payload.get("layer_captures", {}) or {}
    decode_counts = [
        int(layer_payload["decode_q"].shape[0])
        for layer_payload in layer_captures.values()
        if layer_payload.get("decode_q") is not None
    ]
    generated_token_count = min(decode_counts) if decode_counts else 0

    audio_span_length = 0
    for layer_payload in layer_captures.values():
        prompt_audio_k = layer_payload.get("prompt_audio_k")
        if prompt_audio_k is not None:
            audio_span_length = int(prompt_audio_k.shape[1])
            break

    missing_heads: list[dict[str, int]] = []
    effective_heads: list[tuple[AlignAttHead, dict[str, Any], int]] = []
    for head in alignatt_heads:
        layer_payload = layer_captures.get(int(head.layer))
        if layer_payload is None:
            missing_heads.append({"layer": int(head.layer), "head": int(head.head)})
            continue
        prompt_audio_k = layer_payload.get("prompt_audio_k")
        decode_q = layer_payload.get("decode_q")
        selected_heads = list(layer_payload.get("selected_heads", []))
        if (
            prompt_audio_k is None
            or decode_q is None
            or int(head.head) not in selected_heads
        ):
            missing_heads.append({"layer": int(head.layer), "head": int(head.head)})
            continue
        effective_heads.append(
            (head, layer_payload, selected_heads.index(int(head.head)))
        )

    if not effective_heads or generated_token_count <= 0 or audio_span_length <= 0:
        return [], {
            "captured_layer_count": len(layer_captures),
            "effective_head_count": len(effective_heads),
            "generated_token_count": generated_token_count,
            "audio_span_length": audio_span_length,
            "missing_heads": missing_heads,
        }

    rows_per_token: list[torch.Tensor] = []
    for token_index in range(generated_token_count):
        per_head_rows: list[torch.Tensor] = []
        for _head, layer_payload, local_head_index in effective_heads:
            decode_q = layer_payload["decode_q"]
            prompt_audio_k = layer_payload["prompt_audio_k"]
            query = torch.from_numpy(
                np.asarray(decode_q[token_index, local_head_index], dtype=np.float32)
            )
            keys = torch.from_numpy(
                np.asarray(prompt_audio_k[local_head_index], dtype=np.float32)
            )
            scores = torch.matmul(query, keys.transpose(0, 1))
            scaling = float(layer_payload.get("scaling", 1.0))
            if scaling != 1.0:
                scores = scores * scaling
            per_head_rows.append(scores)
        rows_per_token.append(torch.stack(per_head_rows, dim=0))

    return rows_per_token, {
        "captured_layer_count": len(layer_captures),
        "effective_head_count": len(effective_heads),
        "generated_token_count": generated_token_count,
        "audio_span_length": audio_span_length,
        "missing_heads": missing_heads,
    }


def _compute_prompt_observer_cache_key(
    *,
    prompt_token_ids: Sequence[int],
    audio_prompt_positions: Sequence[int],
    selected_heads: Sequence[dict[str, int]],
) -> str:
    hasher = hashlib.sha256()
    hasher.update(np.asarray(prompt_token_ids, dtype=np.int32).tobytes())
    hasher.update(np.asarray(audio_prompt_positions, dtype=np.int32).tobytes())
    for head in selected_heads:
        hasher.update(int(head["layer"]).to_bytes(4, byteorder="little", signed=True))
        hasher.update(int(head["head"]).to_bytes(4, byteorder="little", signed=True))
    return hasher.hexdigest()


def _build_prompt_observer_cache_entry(
    capture_payload: dict[str, Any] | None,
) -> _PromptObserverCacheEntry | None:
    if not capture_payload:
        return None
    audio_prompt_positions = tuple(
        int(pos) for pos in (capture_payload.get("audio_prompt_positions") or [])
    )
    if not audio_prompt_positions:
        return None

    layer_prompt_audio_k: dict[int, np.ndarray] = {}
    for layer_idx, layer_payload in (capture_payload.get("layer_captures") or {}).items():
        prompt_audio_k = layer_payload.get("prompt_audio_k")
        missing = layer_payload.get("prompt_audio_missing_positions") or []
        if prompt_audio_k is None or missing:
            return None
        layer_prompt_audio_k[int(layer_idx)] = np.asarray(prompt_audio_k, dtype=np.float32).copy()

    if not layer_prompt_audio_k:
        return None

    return _PromptObserverCacheEntry(
        prompt_length=int(capture_payload["prompt_length"]),
        audio_prompt_positions=audio_prompt_positions,
        layer_prompt_audio_k=layer_prompt_audio_k,
    )


def _hydrate_capture_payload_from_prompt_observer_cache(
    capture_payload: dict[str, Any] | None,
    *,
    cache_entry: _PromptObserverCacheEntry | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not capture_payload:
        return capture_payload, {
            "hit": False,
            "restored_layer_count": 0,
            "complete_after_restore": False,
        }
    if cache_entry is None:
        return capture_payload, {
            "hit": False,
            "restored_layer_count": 0,
            "complete_after_restore": False,
        }

    payload_positions = tuple(
        int(pos) for pos in (capture_payload.get("audio_prompt_positions") or [])
    )
    if (
        int(capture_payload.get("prompt_length", -1)) != int(cache_entry.prompt_length)
        or payload_positions != cache_entry.audio_prompt_positions
    ):
        return capture_payload, {
            "hit": False,
            "restored_layer_count": 0,
            "complete_after_restore": False,
        }

    restored_layer_count = 0
    hydrated_payload = dict(capture_payload)
    hydrated_layers: dict[int, dict[str, Any]] = {}
    layer_captures = capture_payload.get("layer_captures") or {}
    for layer_idx_raw, layer_payload in layer_captures.items():
        layer_idx = int(layer_idx_raw)
        layer_copy = dict(layer_payload)
        missing = layer_copy.get("prompt_audio_missing_positions") or []
        if layer_copy.get("prompt_audio_k") is None and missing:
            cached_prompt_audio_k = cache_entry.layer_prompt_audio_k.get(layer_idx)
            if cached_prompt_audio_k is not None:
                layer_copy["prompt_audio_k"] = np.asarray(
                    cached_prompt_audio_k,
                    dtype=np.float32,
                ).copy()
                layer_copy["prompt_audio_missing_positions"] = []
                restored_layer_count += 1
        hydrated_layers[layer_idx] = layer_copy
    hydrated_payload["layer_captures"] = hydrated_layers

    complete_after_restore = True
    for layer_payload in hydrated_layers.values():
        if layer_payload.get("prompt_audio_k") is None or (
            layer_payload.get("prompt_audio_missing_positions") or []
        ):
            complete_after_restore = False
            break

    debug_payload = dict(hydrated_payload.get("debug") or {})
    layer_stats = {
        str(layer_idx): dict(stats)
        for layer_idx, stats in (debug_payload.get("layer_stats") or {}).items()
    }
    for layer_idx, layer_payload in hydrated_layers.items():
        stats = dict(layer_stats.get(str(layer_idx), {}))
        prompt_audio_k = layer_payload.get("prompt_audio_k")
        if prompt_audio_k is not None:
            stats["prompt_audio_capture_count"] = int(prompt_audio_k.shape[1])
        stats["missing_prompt_audio_count"] = len(
            layer_payload.get("prompt_audio_missing_positions") or []
        )
        layer_stats[str(layer_idx)] = stats
    debug_payload["layer_stats"] = layer_stats
    hydrated_payload["debug"] = debug_payload

    return hydrated_payload, {
        "hit": restored_layer_count > 0,
        "restored_layer_count": int(restored_layer_count),
        "complete_after_restore": bool(complete_after_restore),
    }


class GemmaVLLMASRBackend(AlignmentBackend):
    """Experimental Gemma ASR + AlignAtt backend through vLLM."""

    name = "gemma_vllm_qk_fast"

    def __init__(
        self,
        *,
        model_name: str,
        runtime_config: SimpleNamespace,
        audio_heads_path: str | None = None,
        audio_heads_top_k: int = 8,
        filter_width: int = 7,
        max_new_tokens: int = 256,
        audio_token_id: int = GEMMA_AUDIO_TOKEN_ID_DEFAULT,
        audio_ms_per_token: float = GEMMA_AUDIO_MS_PER_TOKEN_DEFAULT,
        max_audio_seconds: float = GEMMA_AUDIO_MAX_SECONDS_DEFAULT,
    ) -> None:
        self.model_name = model_name
        self.runtime_config = runtime_config
        self.audio_heads_path = audio_heads_path
        self.audio_heads_top_k = int(audio_heads_top_k)
        self.filter_width = int(filter_width)
        self.max_new_tokens = int(max_new_tokens)
        self.audio_token_id = int(audio_token_id)
        self.audio_ms_per_token = float(audio_ms_per_token)
        self.max_audio_seconds = float(max_audio_seconds)
        # Gemma's audio encoder is trained at 16 kHz. This attribute
        # lets the stream and helpers avoid hardcoding the sample rate
        # in multiple places.
        self.sample_rate = 16000
        self.max_model_len = int(getattr(runtime_config, "gemma_max_model_len", 1024))
        # Keep a small reserve below the nominal decoder limit, analogous to
        # SimulStreaming's `max_text_len - margin` trimming: vLLM multimodal
        # execution adds a small amount of backend-managed prompt overhead that
        # is not perfectly reflected by the local tokenizer count alone.
        self.prompt_budget_reserve_tokens = 32
        self.llm = None
        self.processor = None
        self.tokenizer = None
        self.alignatt_heads: list[AlignAttHead] = []
        self.word_end_offset_s = 0.0
        self.allowed_local_media_path = str(Path.cwd().resolve())
        self.max_audio_tokens = 0
        self.worker_mode = str(
            getattr(runtime_config, "gemma_vllm_worker_mode", "custom_tensor")
        )
        self.executor_backend = str(
            getattr(runtime_config, "gemma_vllm_executor_backend", "mp")
        )
        self.patch_mode = str(
            getattr(runtime_config, "gemma_vllm_patch_mode", "postload_instance")
        )
        self.enforce_eager = bool(
            getattr(runtime_config, "gemma_vllm_enforce_eager", True)
        )
        self.disable_engine_multiprocessing = bool(
            getattr(
                runtime_config,
                "gemma_vllm_disable_engine_multiprocessing",
                True,
            )
        )
        self.compilation_mode = getattr(
            runtime_config,
            "gemma_vllm_compilation_mode",
            None,
        )
        self.cudagraph_mode = getattr(
            runtime_config,
            "gemma_vllm_cudagraph_mode",
            None,
        )
        self.compile_cache_dir = getattr(
            runtime_config,
            "gemma_vllm_compile_cache_dir",
            None,
        )
        self.disable_compile_cache = bool(
            getattr(runtime_config, "gemma_vllm_disable_compile_cache", False)
        )
        self.enable_prefix_caching = bool(
            getattr(runtime_config, "gemma_vllm_enable_prefix_caching", False)
        )
        self._prompt_observer_cache: dict[str, _PromptObserverCacheEntry] = {}
        self._last_generated_token_ids: list[int] | None = None

    def reset_caches(self) -> None:
        self._prompt_observer_cache.clear()
        self._last_generated_token_ids = None

    def decode_single_token(self, token_id: int) -> str:
        """Return the surface string for a single decoder token ID.

        Token-level commit records keep the exact subword piece so that
        concatenating ``text`` over committed tokens reproduces the
        model's decoded output verbatim (including spacing). Skipping
        special tokens here avoids meta-markers bleeding into the
        transcript; AlignAtt already handles ordering.
        """
        if self.tokenizer is None:
            raise RuntimeError("Gemma vLLM backend is not loaded.")
        return self.tokenizer.decode([int(token_id)], skip_special_tokens=True)

    def decode_token_ids(self, token_ids: Sequence[int]) -> str:
        if self.tokenizer is None:
            raise RuntimeError("Gemma vLLM backend is not loaded.")
        return self.tokenizer.decode(
            [int(t) for t in token_ids],
            skip_special_tokens=True,
        )

    def _build_compilation_config(self) -> dict[str, Any] | None:
        compilation_config: dict[str, Any] = {}
        if self.compilation_mode is not None:
            compilation_config["mode"] = str(self.compilation_mode)
        if self.cudagraph_mode is not None:
            compilation_config["cudagraph_mode"] = str(self.cudagraph_mode)
        if self.compile_cache_dir:
            compilation_config["cache_dir"] = str(
                Path(self.compile_cache_dir).expanduser().resolve()
            )

        inductor_compile_config: dict[str, Any] = {}
        if self.disable_compile_cache:
            inductor_compile_config["force_disable_caches"] = True
        if inductor_compile_config:
            compilation_config["inductor_compile_config"] = inductor_compile_config

        return compilation_config or None

    def load(self) -> None:
        from transformers import AutoProcessor

        if self.executor_backend == "uni" and self.disable_engine_multiprocessing:
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

        from vllm import LLM

        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                local_files_only=True,
            )
            self.tokenizer = self.processor.tokenizer
            ms_per_token = getattr(self.processor, "audio_ms_per_token", None)
            if ms_per_token is not None:
                self.audio_ms_per_token = float(ms_per_token)
            audio_seq_length = getattr(self.processor, "audio_seq_length", None)
            if audio_seq_length is not None and ms_per_token is not None:
                self.max_audio_seconds = (
                    float(audio_seq_length) * float(ms_per_token) / 1000.0
                )
                self.max_audio_tokens = int(audio_seq_length)
            elif ms_per_token is not None:
                self.max_audio_tokens = max(
                    1,
                    int(round(self.max_audio_seconds * 1000.0 / float(ms_per_token))),
                )

        if not self.alignatt_heads and self.audio_heads_path and Path(self.audio_heads_path).exists():
            self.alignatt_heads, self.word_end_offset_s = load_audio_alignment_heads(
                self.audio_heads_path,
                top_k=self.audio_heads_top_k,
            )

        if self.llm is None:
            compilation_config = self._build_compilation_config()
            llm_kwargs: dict[str, Any] = {}
            bootstrap_prev = None
            bootstrap_active = False
            if self.worker_mode == "custom_tensor":
                llm_kwargs["worker_cls"] = (
                    "cascade.alignment.gemma_vllm_asr_worker.GemmaVLLMASRWorker"
                )
                if not self.alignatt_heads:
                    raise RuntimeError(
                        "Cannot configure tensor observer without AlignAtt heads."
                    )
                bootstrap_prev = os.environ.get(_ALIGNATT_TENSOR_OBSERVER_BOOTSTRAP_ENV)
                os.environ[_ALIGNATT_TENSOR_OBSERVER_BOOTSTRAP_ENV] = (
                    _encode_tensor_observer_bootstrap(
                        selected_heads=[
                            {"layer": int(head.layer), "head": int(head.head)}
                            for head in self.alignatt_heads
                        ],
                        max_audio_tokens=int(self.max_audio_tokens),
                        max_decode_tokens=int(self.max_new_tokens),
                    )
                )
                bootstrap_active = True
            elif self.patch_mode == "preload_class":
                if self.executor_backend != "uni":
                    raise ValueError(
                        "preload_class patch mode currently requires "
                        "gemma_vllm_executor_backend='uni'."
                    )
                install_global_gemma4_attention_patch()
            try:
                self.llm = LLM(
                    model=self.model_name,
                    trust_remote_code=True,
                    dtype="bfloat16",
                    max_model_len=int(self.max_model_len),
                    gpu_memory_utilization=float(
                        getattr(
                            self.runtime_config,
                            "gemma_vllm_gpu_memory_utilization",
                            0.5,
                        )
                    ),
                    allowed_local_media_path=self.allowed_local_media_path,
                    enforce_eager=self.enforce_eager,
                    enable_prefix_caching=self.enable_prefix_caching,
                    distributed_executor_backend=self.executor_backend,
                    compilation_config=compilation_config,
                    **llm_kwargs,
                )
            finally:
                if bootstrap_active:
                    if bootstrap_prev is None:
                        os.environ.pop(_ALIGNATT_TENSOR_OBSERVER_BOOTSTRAP_ENV, None)
                    else:
                        os.environ[_ALIGNATT_TENSOR_OBSERVER_BOOTSTRAP_ENV] = bootstrap_prev
        if self.worker_mode == "custom_tensor":
            self._configure_tensor_observer()

    def warmup(self, duration_seconds: float = 18.0, *, sample_rate: int = 16000) -> None:
        """Trigger cudagraph capture on synthetic noise.

        The cold-run vs hot-run text gap observed under prefix caching + cudagraph=full
        appears to originate from the first decode's cudagraph capture pass producing
        slightly different numerics than subsequent replays. Calling this once after
        load() pays the capture cost on a throwaway request (synthetic noise, distinct
        token IDs from real audio so it does not pollute the prefix cache).
        """
        if self.llm is None:
            raise RuntimeError("warmup() requires the backend to be loaded.")
        samples = max(1, int(float(duration_seconds) * float(sample_rate)))
        rng = np.random.default_rng(seed=0)
        noise = rng.standard_normal(samples).astype(np.float32) * 0.01
        try:
            self.transcribe_and_align(
                noise,
                sample_rate=sample_rate,
                language="English",
            )
        except RuntimeError as exc:
            # Silence/noise may not produce any generated tokens; that's fine
            # for a warmup whose only purpose is to capture cudagraphs. Other
            # runtime errors should still surface.
            if "no valid" in str(exc) or "did not recover any generated-token" in str(exc):
                return
            raise

    def _enforce_audio_cap(self, audio: np.ndarray, *, sample_rate: int) -> float:
        duration_s = float(len(audio)) / float(sample_rate)
        if duration_s > self.max_audio_seconds + 1e-3:
            raise GemmaAudioTooLongError(
                f"Audio is {duration_s:.3f}s but Gemma encoder cap is "
                f"{self.max_audio_seconds:.3f}s. Chunk the input or raise "
                "the cap explicitly via max_audio_seconds."
            )
        return duration_s

    def _build_prompt_layout(
        self,
        audio_sample_count: int,
        *,
        forced_prefix_token_ids: Sequence[int] = (),
    ) -> _GemmaASRPromptLayout:
        """Build the one-and-only Gemma ASR prompt layout.

        The user turn is rendered once via the chat template with
        ``add_generation_prompt=True`` (so the ``<start_of_turn>model``
        preamble is included). Optionally ``forced_prefix_token_ids``
        are appended verbatim: these are real decoder token IDs for
        already-committed AlignAtt words. We never re-tokenize text at
        step time — the stream owns the IDs and hands them in directly.
        """
        if self.processor is None or self.tokenizer is None:
            raise RuntimeError("Gemma vLLM backend is not loaded.")

        processor_messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": np.asarray([], dtype=np.float32)},
                    {"type": "text", "text": GEMMA_ASR_INSTRUCTION},
                ],
            }
        ]
        prompt_text = self.processor.apply_chat_template(
            processor_messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        user_only_token_ids = tuple(
            int(token_id)
            for token_id in self.tokenizer.encode(
                prompt_text,
                add_special_tokens=False,
            )
        )
        audio_placeholder_span = detect_audio_span(
            user_only_token_ids,
            audio_token_id=self.audio_token_id,
            audio_ms_per_token=self.audio_ms_per_token,
        )
        if audio_placeholder_span is None:
            raise RuntimeError(
                "Could not detect the audio placeholder span in the Gemma prompt."
            )
        audio_placeholder_token_count = int(audio_placeholder_span.length)
        if audio_placeholder_token_count <= 0:
            raise RuntimeError("Gemma audio placeholder span must contain at least one token.")
        duration_ms = float(audio_sample_count) * 1000.0 / float(self.sample_rate)
        dynamic_audio_tokens = max(
            1,
            int(np.ceil(duration_ms / float(self.audio_ms_per_token))),
        )
        max_audio_seq_length = int(
            getattr(self.processor, "audio_seq_length", 0) or dynamic_audio_tokens
        )
        audio_token_count = min(dynamic_audio_tokens, max_audio_seq_length)
        audio_span = AudioSpan(
            prompt_start=int(audio_placeholder_span.prompt_start),
            prompt_end=int(audio_placeholder_span.prompt_start) + int(audio_token_count),
            ms_per_token=float(self.audio_ms_per_token),
        )
        prefix_token_ids = tuple(int(t) for t in forced_prefix_token_ids)
        return _GemmaASRPromptLayout(
            user_only_token_ids=user_only_token_ids,
            prefix_token_ids=prefix_token_ids,
            prompt_token_ids=user_only_token_ids + prefix_token_ids,
            audio_span=audio_span,
            audio_placeholder_token_count=audio_placeholder_token_count,
        )

    def can_fit_step(
        self,
        *,
        audio_window_samples: int,
        forced_prefix_token_count: int,
        language: str,
    ) -> bool:
        """Return True iff the AlignAtt streaming step fits under the prompt budget.

        The stream-side contract is: over-budget is a hard fail for the
        step (we skip and return an empty delta). We never silently trim
        audio out from under the forced prefix, since that is exactly
        the desync class the redesign is trying to remove.
        """
        layout = self._build_prompt_layout(int(audio_window_samples))
        non_audio_prompt_tokens = int(layout.non_audio_prompt_tokens)
        audio_token_count = int(layout.audio_token_count)
        total_prompt_tokens = (
            non_audio_prompt_tokens
            + int(forced_prefix_token_count)
            + audio_token_count
        )
        budget = int(self.max_model_len) - int(self.prompt_budget_reserve_tokens)
        return total_prompt_tokens <= budget

    def _build_compact_observer_tokens(
        self,
        *,
        token_ids: Sequence[int],
        aligned_source_positions: Sequence[int | None],
    ) -> tuple[AlignAttObserverToken, ...]:
        if self.tokenizer is None:
            raise RuntimeError("Gemma vLLM alignment backend is not loaded.")
        if len(token_ids) != len(aligned_source_positions):
            raise ValueError("token_ids and aligned_source_positions length mismatch")

        observer_tokens: list[AlignAttObserverToken] = []
        for token_id, aligned_position in zip(token_ids, aligned_source_positions):
            observer_tokens.append(
                AlignAttObserverToken(
                    token_id=int(token_id),
                    token_str=self.tokenizer.decode(
                        [int(token_id)],
                        skip_special_tokens=False,
                    ),
                    aligned_source_position=(
                        None if aligned_position is None else int(aligned_position)
                    ),
                )
            )
        return tuple(observer_tokens)

    def _trim_trailing_non_text_token_ids(
        self,
        token_ids: Sequence[int],
        *,
        text: str,
    ) -> list[int]:
        if self.tokenizer is None:
            raise RuntimeError("Gemma vLLM alignment backend is not loaded.")

        trimmed = [int(token_id) for token_id in token_ids]
        target_text = str(text).strip()
        special_ids = {
            int(token_id)
            for token_id in (getattr(self.tokenizer, "all_special_ids", None) or [])
        }
        while trimmed:
            last_token_id = int(trimmed[-1])
            without_last = self.tokenizer.decode(
                trimmed[:-1],
                skip_special_tokens=True,
            ).strip()
            if without_last != target_text:
                break
            last_visible = self.tokenizer.decode(
                [last_token_id],
                skip_special_tokens=True,
            ).strip()
            if last_token_id in special_ids or not last_visible:
                trimmed.pop()
                continue
            break
        return trimmed

    def _compute_decode_drift(
        self,
        current_ids: Sequence[int],
    ) -> dict[str, Any] | None:
        """Compare current generation against the previous run's token IDs.

        Returns None on the first run. On subsequent runs, reports whether
        the decoded surface is identical and, if not, the first divergence
        point. This is the concrete diagnostic the PLAN calls for to
        investigate prefix-cached decode drift.
        """
        prev = self._last_generated_token_ids
        if prev is None:
            return None
        current = [int(t) for t in current_ids]
        identical = current == prev
        first_divergence: int | None = None
        if not identical:
            for i, (a, b) in enumerate(zip(prev, current)):
                if a != b:
                    first_divergence = i
                    break
            else:
                first_divergence = min(len(prev), len(current))
        return {
            "identical": identical,
            "prev_token_count": len(prev),
            "current_token_count": len(current),
            "first_divergence_index": first_divergence,
        }

    def _configure_tensor_observer(self) -> dict[str, Any]:
        if self.llm is None:
            raise RuntimeError("Gemma vLLM backend is not loaded.")
        if not self.alignatt_heads:
            raise RuntimeError("Cannot configure tensor observer without AlignAtt heads.")
        max_audio_tokens = self.max_audio_tokens
        if max_audio_tokens <= 0:
            max_audio_tokens = max(
                1,
                int(round(self.max_audio_seconds * 1000.0 / self.audio_ms_per_token)),
            )
        results = self.llm.collective_rpc(
            "configure_audio_observer",
            args=(
                [
                    {"layer": int(head.layer), "head": int(head.head)}
                    for head in self.alignatt_heads
                ],
                int(max_audio_tokens),
                int(self.max_new_tokens),
            ),
        )
        if len(results) != 1:
            raise RuntimeError(
                f"Expected a single vLLM worker, got {len(results)} observer configurations."
            )
        return results[0]

    def _prepare_tensor_observer(
        self,
        *,
        prompt_length: int,
        audio_prompt_start: int,
        audio_prompt_length: int,
    ) -> dict[str, Any]:
        if self.llm is None:
            raise RuntimeError("Gemma vLLM backend is not loaded.")
        results = self.llm.collective_rpc(
            "prepare_audio_observer",
            args=(int(prompt_length), int(audio_prompt_start), int(audio_prompt_length)),
        )
        if len(results) != 1:
            raise RuntimeError(
                f"Expected a single vLLM worker, got {len(results)} observer preparations."
            )
        return results[0]

    def _install_observer(
        self,
        *,
        selected_heads: Sequence[dict[str, int]],
        audio_prompt_positions: Sequence[int],
        prompt_length: int,
    ) -> dict[str, Any]:
        if self.llm is None:
            raise RuntimeError("Gemma vLLM backend is not loaded.")

        if self.worker_mode == "custom_tensor":
            if not audio_prompt_positions:
                raise ValueError("audio_prompt_positions must not be empty")
            return self._prepare_tensor_observer(
                prompt_length=int(prompt_length),
                audio_prompt_start=int(audio_prompt_positions[0]),
                audio_prompt_length=len(audio_prompt_positions),
            )

        if self.executor_backend == "uni":
            install_results = self.llm.apply_model(
                partial(
                    _apply_install_audio_qk_observer_on_model,
                    selected_heads=selected_heads,
                    audio_prompt_positions=audio_prompt_positions,
                    prompt_length=prompt_length,
                    patch_mode=self.patch_mode,
                )
            )
        else:
            install_results = self.llm.collective_rpc(
                _rpc_install_audio_qk_observer,
                args=(
                    list(selected_heads),
                    list(audio_prompt_positions),
                    int(prompt_length),
                    self.patch_mode,
                ),
            )
        if len(install_results) != 1:
            raise RuntimeError(
                f"Expected a single vLLM worker, got {len(install_results)} observer installs."
            )
        return install_results[0]

    def _fetch_observer_payload(self) -> dict[str, Any] | None:
        if self.llm is None:
            raise RuntimeError("Gemma vLLM backend is not loaded.")

        if self.worker_mode == "custom_tensor":
            payloads = self.llm.collective_rpc("fetch_audio_observer_payload")
            if len(payloads) != 1:
                raise RuntimeError(
                    f"Expected a single vLLM worker, got {len(payloads)} observer payloads."
                )
            return payloads[0]

        if self.executor_backend == "uni":
            payloads = self.llm.apply_model(_fetch_audio_qk_observer_from_model)
        else:
            payloads = self.llm.collective_rpc(_rpc_fetch_audio_qk_observer)
        if len(payloads) != 1:
            raise RuntimeError(
                f"Expected a single vLLM worker, got {len(payloads)} observer payloads."
            )
        return payloads[0]

    def _run_alignatt_inference(
        self,
        audio: np.ndarray,
        *,
        forced_prefix_token_ids: Sequence[int],
        max_new_tokens: int,
    ) -> dict[str, Any] | None:
        """Shared ``generate + observer + reconstruct`` primitive.

        Returns a dict with:
          - ``prompt_layout``: the :class:`_GemmaASRPromptLayout`,
          - ``generated_ids``: trimmed generated token IDs,
          - ``aligned_audio_positions``: one argmax per generated token
            (relative to the audio window),
          - ``content_frame_len``: number of real audio frames in the
            window (before any encoder-side padding),
          - ``raw_completion_text``: untrimmed decoded string,
          - ``finish_reason``: vLLM finish reason,
          - ``diagnostics``: timing + capture diagnostics.

        Returns ``None`` when the model produced an empty completion
        (this is an ordinary empty step, not a failure).
        """
        if self.llm is None or self.processor is None or self.tokenizer is None:
            raise RuntimeError("Gemma vLLM backend is not loaded. Call load() first.")
        if not self.alignatt_heads:
            raise RuntimeError(
                "No calibrated AlignAtt heads are loaded for the vLLM backend."
            )

        audio = np.ascontiguousarray(np.asarray(audio, dtype=np.float32))
        self._enforce_audio_cap(audio, sample_rate=self.sample_rate)

        prompt_build_start = perf_counter()
        prompt_layout = self._build_prompt_layout(
            int(len(audio)),
            forced_prefix_token_ids=forced_prefix_token_ids,
        )
        audio_span = prompt_layout.audio_span
        prompt_token_ids = [int(t) for t in prompt_layout.prompt_token_ids]
        prompt_build_ms = (perf_counter() - prompt_build_start) * 1000.0

        install_result = self._install_observer(
            selected_heads=[
                {"layer": int(head.layer), "head": int(head.head)}
                for head in self.alignatt_heads
            ],
            audio_prompt_positions=list(range(audio_span.prompt_start, audio_span.prompt_end)),
            prompt_length=int(prompt_layout.prompt_token_count),
        )
        prompt_observer_cache_key = _compute_prompt_observer_cache_key(
            prompt_token_ids=prompt_token_ids,
            audio_prompt_positions=list(
                range(audio_span.prompt_start, audio_span.prompt_end)
            ),
            selected_heads=[
                {"layer": int(head.layer), "head": int(head.head)}
                for head in self.alignatt_heads
            ],
        )

        sampling_params, sampling_diagnostics = build_gemma_vllm_sampling_params(
            runtime_config=self.runtime_config,
            max_new_tokens=int(max_new_tokens),
        )

        generate_start = perf_counter()
        inp = {
            "prompt_token_ids": prompt_token_ids,
            "multi_modal_data": {"audio": [audio]},
        }
        outputs = self.llm.generate([inp], sampling_params=sampling_params, use_tqdm=False)
        generate_ms = (perf_counter() - generate_start) * 1000.0
        if not outputs or not outputs[0].outputs:
            raise RuntimeError("vLLM Gemma ASR produced no completion output.")

        output = outputs[0]
        completion = output.outputs[0]
        raw_completion_text = str(completion.text)
        text = raw_completion_text.strip()
        generated_ids = self._trim_trailing_non_text_token_ids(
            [int(t) for t in completion.token_ids],
            text=text,
        )

        fetch_start = perf_counter()
        capture_payload = self._fetch_observer_payload()
        capture_fetch_ms = (perf_counter() - fetch_start) * 1000.0
        capture_payload, prompt_observer_cache_diagnostics = (
            _hydrate_capture_payload_from_prompt_observer_cache(
                capture_payload,
                cache_entry=self._prompt_observer_cache.get(prompt_observer_cache_key),
            )
        )
        prompt_cache_entry = _build_prompt_observer_cache_entry(capture_payload)
        if prompt_cache_entry is not None:
            self._prompt_observer_cache[prompt_observer_cache_key] = prompt_cache_entry

        reconstruction_start = perf_counter()
        source_attention_rows_per_token, reconstruction_diagnostics = (
            reconstruct_vllm_audio_attention_rows(
                capture_payload,
                alignatt_heads=self.alignatt_heads,
            )
        )
        aligned_audio_positions = compute_alignatt_source_argmaxes(
            source_attention_rows_per_token,
            filter_width=self.filter_width,
        )
        reconstruction_ms = (perf_counter() - reconstruction_start) * 1000.0

        if not text or not generated_ids:
            return None

        effective_token_count = min(len(generated_ids), len(aligned_audio_positions))
        if effective_token_count <= 0:
            capture_debug = None if capture_payload is None else capture_payload.get("debug")
            raise RuntimeError(
                "The vLLM observer did not recover any generated-token alignment "
                f"rows. capture={reconstruction_diagnostics!r} "
                f"capture_debug={capture_debug!r} "
                f"finish_reason={completion.finish_reason!r} "
                f"generated_token_ids={len(generated_ids)}"
            )
        generated_ids = generated_ids[:effective_token_count]
        aligned_audio_positions = aligned_audio_positions[:effective_token_count]

        self._last_generated_token_ids = list(generated_ids)

        diagnostics = {
            "backend": self.name,
            "observer_backend": "vllm_qk_fast_experimental",
            "prompt_budget": {
                "max_model_len": int(self.max_model_len),
                "prompt_reserve_tokens": int(self.prompt_budget_reserve_tokens),
                "prompt_token_count": int(prompt_layout.prompt_token_count),
                "non_audio_prompt_tokens": int(prompt_layout.non_audio_prompt_tokens),
                "audio_prompt_tokens": int(prompt_layout.audio_token_count),
                "prefix_prompt_tokens": int(len(prompt_layout.prefix_token_ids)),
            },
            "sampling": sampling_diagnostics,
            "selected_head_count": len(self.alignatt_heads),
            "audio_span_length": int(audio_span.length),
            "generated_token_count": len(generated_ids),
            "monotonicity": monotonicity_score(aligned_audio_positions),
            "finish_reason": completion.finish_reason,
            "capture": reconstruction_diagnostics,
            "capture_debug": (
                None if capture_payload is None else capture_payload.get("debug")
            ),
            "prompt_observer_cache": {
                **prompt_observer_cache_diagnostics,
                "cache_size": len(self._prompt_observer_cache),
                "stored": prompt_cache_entry is not None,
            },
            "worker_install": install_result,
            "timings_ms": {
                "prompt_build": round(prompt_build_ms, 3),
                "generate": round(generate_ms, 3),
                "capture_fetch": round(capture_fetch_ms, 3),
                "qk_reconstruction": round(reconstruction_ms, 3),
            },
        }

        return {
            "prompt_layout": prompt_layout,
            "generated_ids": generated_ids,
            "aligned_audio_positions": aligned_audio_positions,
            "content_frame_len": int(audio_span.length),
            "raw_completion_text": raw_completion_text,
            "finish_reason": completion.finish_reason,
            "diagnostics": diagnostics,
        }

    def alignatt_step(
        self,
        *,
        audio_window: np.ndarray,
        forced_prefix_token_ids: Sequence[int],
        language: str,
        max_new_tokens: int,
    ) -> AlignAttStepRaw | None:
        """One AlignAtt streaming step: generate + capture attention.

        The stream hands in ``audio_window`` (the trailing slice of the
        live utterance) and ``forced_prefix_token_ids`` (the committed
        tokens whose alignments still fall inside the window). We never
        re-tokenize text here; IDs are already known at commit time.
        """
        result = self._run_alignatt_inference(
            np.asarray(audio_window, dtype=np.float32),
            forced_prefix_token_ids=forced_prefix_token_ids,
            max_new_tokens=int(max_new_tokens),
        )
        if result is None:
            return None
        return AlignAttStepRaw(
            generated_token_ids=tuple(int(t) for t in result["generated_ids"]),
            per_token_audio_frame_argmax=tuple(
                int(a) for a in result["aligned_audio_positions"]
            ),
            content_frame_len=int(result["content_frame_len"]),
            diagnostics=dict(result["diagnostics"]),
        )

    def transcribe_and_align(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int,
        language: str,
        streaming_prefix_text: str = "",
        streaming_prefix_words: tuple[WordAlignment, ...] = (),
    ) -> AlignmentResult | None:
        """Non-streaming full-utterance ASR + alignment.

        Streaming is driven by :class:`GemmaAlignAttStream` and does not
        go through this method. We keep ``transcribe_and_align`` for
        offline/full-audio harnesses. Non-empty ``streaming_prefix_text``
        is rejected to keep the two paths strictly separate.
        """
        if streaming_prefix_text:
            raise NotImplementedError(
                "GemmaVLLMASRBackend.transcribe_and_align does not accept a "
                "streaming prefix. Use GemmaAlignAttStream for the streaming "
                "AlignAtt path."
            )

        audio = np.asarray(audio, dtype=np.float32)
        audio_duration_s = self._enforce_audio_cap(audio, sample_rate=sample_rate)

        result = self._run_alignatt_inference(
            audio,
            forced_prefix_token_ids=(),
            max_new_tokens=self.max_new_tokens,
        )
        if result is None:
            return None

        generated_ids = result["generated_ids"]
        aligned_audio_positions = result["aligned_audio_positions"]
        raw_completion_text = result["raw_completion_text"]
        text = raw_completion_text.strip()

        token_end_times_s = [
            audio_position_to_end_seconds(
                aligned_position,
                ms_per_token=self.audio_ms_per_token,
                audio_duration_s=audio_duration_s,
            )
            for aligned_position in aligned_audio_positions
        ]
        token_end_times_s = _enforce_monotone(token_end_times_s)
        token_end_times_s = _apply_word_end_offset(
            token_end_times_s,
            offset_s=self.word_end_offset_s,
            audio_duration_s=audio_duration_s,
        )

        word_aggregation_start = perf_counter()
        words = tuple(
            aggregate_token_timings_to_words(
                text,
                generated_ids=generated_ids,
                tokenizer=self.tokenizer,
                token_end_times_s=token_end_times_s,
                audio_duration_s=audio_duration_s,
            )
        )
        word_aggregation_ms = (perf_counter() - word_aggregation_start) * 1000.0

        observer_tokens = self._build_compact_observer_tokens(
            token_ids=generated_ids,
            aligned_source_positions=aligned_audio_positions,
        )

        diagnostics = dict(result["diagnostics"])
        diagnostics["timings_ms"] = {
            **diagnostics.get("timings_ms", {}),
            "word_aggregation": round(word_aggregation_ms, 3),
        }
        diagnostics["invocation_path"] = "generate"

        return AlignmentResult(
            text=text,
            words=tuple(words),
            audio_duration_s=audio_duration_s,
            observer_tokens=observer_tokens,
            diagnostics=diagnostics,
        )
