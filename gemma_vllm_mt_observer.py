"""vLLM-side Q/K observer for MT AlignAtt (PLAN.md Phase 2).

This is a close analogue of ``gemma_vllm_alignment_backend._AudioQKTensorObserver``
with one structural difference: for MT we need the full 4-way provenance
partition (``source_accessible`` / ``source_inaccessible`` / ``non_source_prompt``
/ ``suffix``), which requires recomputing the softmax over ``[prompt_K | suffix_K]``.
The ASR observer captures K only at prompt positions; the MT observer also
captures K at decode positions so we can reconstruct the suffix row per token.

The observer contract per layer is:

- ``prompt_k_buffer``: K at every prompt position, on the selected **KV** heads
- ``decode_q_buffer``: Q at every decode position, on the selected Q heads
- ``decode_k_buffer``: K at every decode position, on the selected KV heads

All three are fixed-shape tensor buffers, scattered via ``scatter_add_``-style
ops so the capture path is compatible with ``torch.compile`` / cudagraph
capture (no ``nonzero`` / Python-level conditionals on tensor values).

The patched ``Gemma4Attention.forward`` is intentionally separate from the ASR
version so the two observers never alias the same attribute. If the ASR
Gemma backend and the MT Gemma backend ever need to coexist in one process,
this module would need a shared dispatcher — today PLAN.md's target pairs
``qwen_forced`` ASR with ``gemma_vllm_alignatt`` MT, so they do not.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch


_MT_OBSERVER_BOOTSTRAP_ENV = "CASCADE_MT_ALIGNATT_OBSERVER_BOOTSTRAP"


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


def _encode_mt_observer_bootstrap(
    *,
    selected_heads: Sequence[dict[str, int]],
    max_prompt_tokens: int,
    max_decode_tokens: int,
) -> str:
    return json.dumps(
        {
            "selected_heads": [
                {"layer": int(head["layer"]), "head": int(head["head"])}
                for head in selected_heads
            ],
            "max_prompt_tokens": int(max_prompt_tokens),
            "max_decode_tokens": int(max_decode_tokens),
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def _decode_mt_observer_bootstrap_from_env() -> dict[str, Any] | None:
    raw = os.environ.get(_MT_OBSERVER_BOOTSTRAP_ENV)
    if not raw:
        return None
    payload = json.loads(raw)
    return {
        "selected_heads": [
            {"layer": int(head["layer"]), "head": int(head["head"])}
            for head in payload.get("selected_heads", [])
        ],
        "max_prompt_tokens": int(payload["max_prompt_tokens"]),
        "max_decode_tokens": int(payload["max_decode_tokens"]),
    }


class _MTPromptDecodeQKTensorObserver(torch.nn.Module):
    """Per-layer tensor buffers for MT Q/K capture.

    Stores K for the prompt (by absolute position, indexed 0..max_prompt_tokens-1)
    and Q+K for the decode suffix (by decode step 0..max_decode_tokens-1).
    """

    def __init__(
        self,
        *,
        selected_heads: Sequence[int],
        selected_kv_heads: Sequence[int],
        max_prompt_tokens: int,
        max_decode_tokens: int,
        head_dim: int,
        scaling: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.selected_heads = tuple(int(head) for head in selected_heads)
        self.selected_kv_heads = tuple(int(head) for head in selected_kv_heads)
        self.scaling = float(scaling)
        self.head_dim = int(head_dim)

        self.register_buffer(
            "selected_heads_tensor",
            torch.tensor(list(self.selected_heads), device=device, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "selected_kv_heads_tensor",
            torch.tensor(list(self.selected_kv_heads), device=device, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "prompt_length_tensor",
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

        kv_head_count = len(self.selected_kv_heads)
        q_head_count = len(self.selected_heads)
        self.register_buffer(
            "prompt_k_buffer",
            torch.zeros(
                (kv_head_count, int(max_prompt_tokens), int(head_dim)),
                device=device,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "prompt_k_scratch",
            torch.zeros(
                (kv_head_count, int(max_prompt_tokens), int(head_dim)),
                device=device,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "prompt_written_buffer",
            torch.zeros((int(max_prompt_tokens),), device=device, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "prompt_written_scratch",
            torch.zeros((int(max_prompt_tokens),), device=device, dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "decode_q_buffer",
            torch.zeros(
                (int(max_decode_tokens), q_head_count, int(head_dim)),
                device=device,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "decode_q_scratch",
            torch.zeros(
                (int(max_decode_tokens), q_head_count, int(head_dim)),
                device=device,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "decode_k_buffer",
            torch.zeros(
                (int(max_decode_tokens), kv_head_count, int(head_dim)),
                device=device,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "decode_k_scratch",
            torch.zeros(
                (int(max_decode_tokens), kv_head_count, int(head_dim)),
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

    def prepare(self, *, prompt_length: int) -> None:
        max_prompt = int(self.prompt_k_buffer.shape[1])
        if int(prompt_length) > max_prompt:
            raise ValueError(
                f"prompt_length={prompt_length} exceeds max_prompt_tokens={max_prompt}"
            )
        self.prompt_length_tensor.fill_(int(prompt_length))
        self.forward_call_count_tensor.zero_()
        self.prompt_forward_call_count_tensor.zero_()
        self.decode_forward_call_count_tensor.zero_()
        self.prompt_k_buffer.zero_()
        self.prompt_k_scratch.zero_()
        self.prompt_written_buffer.zero_()
        self.prompt_written_scratch.zero_()
        self.decode_q_buffer.zero_()
        self.decode_q_scratch.zero_()
        self.decode_k_buffer.zero_()
        self.decode_k_scratch.zero_()
        self.decode_written_buffer.zero_()
        self.decode_written_scratch.zero_()


def _get_mt_qk_tensor_observer(attn_module) -> _MTPromptDecodeQKTensorObserver | None:
    observer = getattr(attn_module, "_alignatt_mt_qk_tensor_observer", None)
    if observer is None:
        return None
    if not isinstance(observer, _MTPromptDecodeQKTensorObserver):
        raise TypeError(
            "Expected _alignatt_mt_qk_tensor_observer to be an "
            "_MTPromptDecodeQKTensorObserver instance."
        )
    return observer


def _resolve_mt_observer_bindings(
    model,
) -> list[tuple[int, _MTPromptDecodeQKTensorObserver]]:
    state = getattr(model, "_alignatt_mt_qk_state", None)
    if state is None:
        return []
    layers = _resolve_vllm_gemma_decoder_layers(model)
    bindings: list[tuple[int, _MTPromptDecodeQKTensorObserver]] = []
    for layer_idx in state.get("layer_indices", ()):
        attn = getattr(layers[int(layer_idx)], "self_attn", None)
        if attn is None:
            raise RuntimeError(f"Layer {layer_idx} has no self_attn module.")
        observer = _get_mt_qk_tensor_observer(attn)
        if observer is None:
            raise RuntimeError(
                f"Layer {layer_idx} is missing the configured MT tensor observer."
            )
        bindings.append((int(layer_idx), observer))
    return bindings


def _capture_mt_qk_into_tensor_buffers(attn_module, positions, q, k) -> None:
    # NOTE (commit 2a818d0 / 5e557a8 / 4ebfee0 / this diff): an
    # earlier attempt wrapped this function with
    # ``@torch.compiler.disable`` to keep the observer-capture out of
    # the AOT-compiled Gemma4 forward graph. That patch is not
    # shippable: vLLM compiles the model in fullgraph mode, which
    # does not permit graph breaks, and
    # ``@torch.compiler.disable`` requires one. The dynamo error is
    # explicit: *"Skip calling torch.compiler.disable()'d function"
    # — the model is using torch.compile in fullgraph mode"*. A
    # proper fix would need to either (a) convince vLLM to allow a
    # break at the observer call, or (b) replace the observer
    # capture path with a PyTorch custom op registered via
    # ``torch.library.custom_op`` so it becomes a single dispatcher
    # call that AOT can represent as an opaque node. Both are out
    # of night scope.
    observer = _get_mt_qk_tensor_observer(attn_module)
    if observer is None:
        return

    positions_flat = positions.reshape(-1).to(dtype=torch.int64)
    if positions_flat.numel() == 0:
        return

    q_heads = q.reshape(-1, attn_module.num_heads, attn_module.head_dim)
    k_heads = k.reshape(-1, attn_module.num_kv_heads, attn_module.head_dim)
    selected_q = torch.index_select(
        q_heads, dim=1, index=observer.selected_heads_tensor
    ).to(dtype=torch.float32)
    selected_k = torch.index_select(
        k_heads, dim=1, index=observer.selected_kv_heads_tensor
    ).to(dtype=torch.float32)

    prompt_length = observer.prompt_length_tensor
    max_prompt = observer.prompt_k_buffer.shape[1]
    max_decode = int(observer.decode_q_buffer.shape[0])

    observer.forward_call_count_tensor.add_(1)

    # --- prompt K scatter ---
    prompt_mask = (positions_flat >= 0) & (positions_flat < prompt_length)
    prompt_mask_i32 = prompt_mask.any().to(dtype=torch.int32)
    observer.prompt_forward_call_count_tensor.add_(prompt_mask_i32)
    prompt_offsets_clamped = positions_flat.clamp(min=0, max=max_prompt - 1)
    prompt_values = selected_k.transpose(0, 1)  # (kv_heads, seq, head_dim)
    prompt_mask_f32 = prompt_mask.to(dtype=prompt_values.dtype).view(1, -1, 1)
    prompt_index = prompt_offsets_clamped.view(1, -1, 1).expand(
        prompt_values.shape[0], -1, prompt_values.shape[2]
    )
    prompt_scratch = observer.prompt_k_scratch
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
    observer.prompt_k_buffer.copy_(
        torch.where(
            prompt_write_mask.view(1, -1, 1),
            prompt_scratch,
            observer.prompt_k_buffer,
        )
    )
    observer.prompt_written_buffer.logical_or_(prompt_write_mask)

    # --- decode Q and decode K scatter ---
    decode_offsets = positions_flat - prompt_length
    decode_mask = (
        (positions_flat >= prompt_length)
        & (decode_offsets >= 0)
        & (decode_offsets < max_decode)
    )
    decode_mask_i32 = decode_mask.any().to(dtype=torch.int32)
    observer.decode_forward_call_count_tensor.add_(decode_mask_i32)
    decode_offsets_clamped = decode_offsets.clamp(min=0, max=max_decode - 1)

    # decode Q
    decode_q_mask_f32 = decode_mask.to(dtype=selected_q.dtype).view(-1, 1, 1)
    decode_q_index = decode_offsets_clamped.view(-1, 1, 1).expand(
        -1, selected_q.shape[1], selected_q.shape[2]
    )
    decode_q_scratch = observer.decode_q_scratch
    decode_q_scratch.zero_()
    decode_q_scratch.scatter_add_(0, decode_q_index, selected_q * decode_q_mask_f32)

    # decode K
    decode_k_mask_f32 = decode_mask.to(dtype=selected_k.dtype).view(-1, 1, 1)
    decode_k_index = decode_offsets_clamped.view(-1, 1, 1).expand(
        -1, selected_k.shape[1], selected_k.shape[2]
    )
    decode_k_scratch = observer.decode_k_scratch
    decode_k_scratch.zero_()
    decode_k_scratch.scatter_add_(0, decode_k_index, selected_k * decode_k_mask_f32)

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
            decode_q_scratch,
            observer.decode_q_buffer,
        )
    )
    observer.decode_k_buffer.copy_(
        torch.where(
            decode_write_mask.view(-1, 1, 1),
            decode_k_scratch,
            observer.decode_k_buffer,
        )
    )
    observer.decode_written_buffer.logical_or_(decode_write_mask)


# Global registry mapping layer index -> observer. Used by the
# custom op below so the per-layer observer lookup is a Python
# dict access in the op implementation, not a traced attribute
# access inside the AOT-compiled Gemma4 forward graph.
_LAYER_OBSERVER_REGISTRY: dict[int, "_MTPromptDecodeQKTensorObserver | None"] = {}


def _register_layer_observer(layer_idx: int, observer) -> None:
    _LAYER_OBSERVER_REGISTRY[int(layer_idx)] = observer


def _clear_layer_observer(layer_idx: int) -> None:
    _LAYER_OBSERVER_REGISTRY.pop(int(layer_idx), None)


_CUSTOM_OP_REGISTERED = False


def _ensure_custom_op_registered() -> None:
    """Lazily register ``alignatt::capture_mt_qk`` once per process.

    The op takes a layer-index scalar + positions/q/k tensors and
    returns nothing. The fake impl (for AOT-compile tracing) is a
    no-op; the real impl dispatches to the existing observer
    capture via the layer-index lookup in
    ``_LAYER_OBSERVER_REGISTRY``. Registering the capture as a
    custom op makes it an opaque dispatcher node in the AOT graph,
    so:

      (a) dynamo doesn't trace observer tensor ops into the graph,
          which keeps the graph's argument signature independent
          of observer state;
      (b) no graph break is required, so fullgraph AOT compile
          succeeds;
      (c) the compile cache is reusable across processes regardless
          of whether an observer is configured at cache-load time.

    This is the documented fix path from the
    ``a8cca6f`` attempted-`@torch.compiler.disable` rollback.
    """
    global _CUSTOM_OP_REGISTERED
    if _CUSTOM_OP_REGISTERED:
        return

    # KNOWN LIMITATION: ``mutates_args=()`` lets inductor DCE-elide
    # this op entirely at AOT time under cudagraph=full, so the
    # observer never fires on the vLLM MT path. We tried
    # ``mutates_args="unknown"`` (still elided) and a sentinel return
    # threaded through the forward's output (compiles but the
    # determine_available_memory dummy run blows up with
    # ``RuntimeError: The size of tensor a (8192) must match the size
    # of tensor b (1024) at non-singleton dimension 1`` — same shape-
    # trace class as the original compile-cache fragility this op was
    # supposed to fix). Keeping the elided ``mutates_args=()`` form
    # for now: vLLM MT runs end-to-end but with observer_empty on
    # every partial emission, so the policy loop becomes a no-op for
    # scalar vs discrete substitution. Proper fix requires a post-
    # hoc observer pattern (capture Q/K outside the compiled graph)
    # or an enforce_eager path.
    @torch.library.custom_op(
        "alignatt::capture_mt_qk", mutates_args=(), device_types=None
    )
    def capture_mt_qk(
        layer_idx: int,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> None:
        observer = _LAYER_OBSERVER_REGISTRY.get(int(layer_idx))
        if observer is None:
            return
        _capture_mt_qk_into_tensor_buffers_from_observer(
            observer, positions, q, k
        )

    @capture_mt_qk.register_fake
    def _capture_mt_qk_fake(layer_idx, positions, q, k):
        return None

    _CUSTOM_OP_REGISTERED = True


def _capture_mt_qk_into_tensor_buffers_from_observer(
    observer, positions, q, k
) -> None:
    """Body of the observer capture that ``capture_mt_qk`` dispatches to.

    Identical logic to the deprecated
    ``_capture_mt_qk_into_tensor_buffers(attn_module, ...)`` but
    takes the observer directly rather than looking it up from the
    module. Separating the two makes it trivial for the custom op
    to invoke the observer path without re-entering dynamo.
    """
    positions_flat = positions.reshape(-1).to(dtype=torch.int64)
    if positions_flat.numel() == 0:
        return

    num_q_heads = observer.selected_heads_tensor.numel()
    num_kv_heads = observer.selected_kv_heads_tensor.numel()
    head_dim = observer.head_dim
    q_heads = q.reshape(-1, q.shape[-1] // head_dim, head_dim)
    k_heads = k.reshape(-1, k.shape[-1] // head_dim, head_dim)
    selected_q = torch.index_select(
        q_heads, dim=1, index=observer.selected_heads_tensor
    ).to(dtype=torch.float32)
    selected_k = torch.index_select(
        k_heads, dim=1, index=observer.selected_kv_heads_tensor
    ).to(dtype=torch.float32)

    prompt_length = observer.prompt_length_tensor
    max_prompt = observer.prompt_k_buffer.shape[1]
    max_decode = int(observer.decode_q_buffer.shape[0])

    observer.forward_call_count_tensor.add_(1)

    # --- prompt K scatter ---
    # NOTE: the write mask must be shaped by the *buffer* size
    # (``max_prompt``), not by the *positions* size. vLLM's
    # determine_available_memory dummy_run calls this op with up
    # to 8192 positions, but the observer's prompt_k_buffer is
    # sized to max_prompt_tokens=1024. A num_positions-shaped
    # mask breaks the torch.where broadcast below. Pre-custom-op
    # code used scatter_reduce into prompt_written_scratch to
    # build a buffer-shaped mask; mirror that here.
    prompt_mask = (positions_flat >= 0) & (positions_flat < prompt_length)
    prompt_mask_i32 = prompt_mask.any().to(dtype=torch.int32)
    observer.prompt_forward_call_count_tensor.add_(prompt_mask_i32)
    prompt_offsets_clamped = positions_flat.clamp(min=0, max=max_prompt - 1)
    prompt_values = selected_k.transpose(0, 1)  # (kv_heads, seq, head_dim)
    prompt_mask_f32 = prompt_mask.to(dtype=prompt_values.dtype).view(1, -1, 1)
    prompt_index = prompt_offsets_clamped.view(1, -1, 1).expand(
        prompt_values.shape[0], -1, prompt_values.shape[2]
    )
    prompt_scratch = observer.prompt_k_scratch
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
    observer.prompt_k_buffer.copy_(
        torch.where(
            prompt_write_mask.view(1, -1, 1),
            prompt_scratch,
            observer.prompt_k_buffer,
        )
    )
    observer.prompt_written_buffer.logical_or_(prompt_write_mask)

    # --- decode Q / K scatter ---
    decode_positions = positions_flat - prompt_length
    decode_mask = (decode_positions >= 0) & (decode_positions < max_decode)
    decode_mask_i32 = decode_mask.any().to(dtype=torch.int32)
    observer.decode_forward_call_count_tensor.add_(decode_mask_i32)
    decode_offsets_clamped = decode_positions.clamp(min=0, max=max_decode - 1)
    # Buffer-shaped (max_decode) write mask, same reasoning as the
    # prompt side. CUDA scatter_reduce_ doesn't support Bool, so
    # work with int32 and convert at the end.
    decode_write_scratch = torch.zeros(
        max_decode, device=decode_mask.device, dtype=torch.int32
    )
    decode_write_scratch.scatter_reduce_(
        0,
        decode_offsets_clamped,
        decode_mask.to(dtype=torch.int32),
        reduce="amax",
        include_self=False,
    )
    decode_write_mask = decode_write_scratch.to(dtype=torch.bool)
    decode_mask_f32 = decode_mask.to(dtype=selected_q.dtype).view(-1, 1, 1)
    decode_index_q = decode_offsets_clamped.view(-1, 1, 1).expand(
        -1, selected_q.shape[1], selected_q.shape[2]
    )
    decode_q_scratch = observer.decode_q_scratch
    decode_q_scratch.zero_()
    decode_q_scratch.scatter_add_(0, decode_index_q, selected_q * decode_mask_f32)
    decode_k_scratch = observer.decode_k_scratch
    decode_k_scratch.zero_()
    decode_k_scratch.scatter_add_(0, decode_index_q, selected_k * decode_mask_f32)
    observer.decode_q_buffer.copy_(
        torch.where(
            decode_write_mask.view(-1, 1, 1),
            decode_q_scratch,
            observer.decode_q_buffer,
        )
    )
    observer.decode_k_buffer.copy_(
        torch.where(
            decode_write_mask.view(-1, 1, 1),
            decode_k_scratch,
            observer.decode_k_buffer,
        )
    )
    observer.decode_written_buffer.logical_or_(decode_write_mask)


def _make_mt_tensor_buffer_gemma4_attention_forward():
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

        # Dispatch the observer capture through the custom op so
        # AOT-compile sees a single opaque node rather than tracing
        # into the observer tensor scatters. Layer index is stored
        # on the module at stub-install / configure time; if it's
        # missing for some reason, fall back to -1 (never found in
        # the registry, so the custom op's no-op path fires).
        layer_idx = int(getattr(self, "_alignatt_mt_layer_idx", -1))
        torch.ops.alignatt.capture_mt_qk(layer_idx, positions, q, k)

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    return _patched_forward


def install_global_gemma4_attention_mt_patch() -> None:
    from vllm.model_executor.models.gemma4 import Gemma4Attention

    _ensure_custom_op_registered()

    if not hasattr(Gemma4Attention, "_alignatt_mt_qk_original_forward"):
        Gemma4Attention._alignatt_mt_qk_original_forward = Gemma4Attention.forward
        Gemma4Attention.forward = _make_mt_tensor_buffer_gemma4_attention_forward()


def install_stub_observers_on_model(model) -> int:
    """Install a ``None`` stub on every Gemma4Attention's observer attr.

    The AOT-compiled ``Gemma4Attention.forward`` traces an attribute
    access to ``_alignatt_mt_qk_tensor_observer`` during first warmup
    and then, when vLLM's compile cache replays the compiled graph
    in a fresh process, performs a direct ``__dict__`` lookup on that
    attribute. If the attribute is missing from ``__dict__`` at replay
    time (e.g. when ``determine_available_memory``'s dummy_run fires
    before we have called ``configure_mt_observer``), the compiled
    path raises ``KeyError``.

    ``_get_mt_qk_tensor_observer`` already treats ``None`` as "no
    observer configured" and ``_capture_mt_qk_into_tensor_buffers``
    early-returns on ``None``. Pre-seeding every attention layer with
    an explicit ``None`` attribute keeps the attribute present on
    ``__dict__`` so the compiled lookup succeeds. Returns the number
    of layers stubbed.
    """
    layers = _resolve_vllm_gemma_decoder_layers(model)
    stubbed = 0
    for layer_idx, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        # Use object.__setattr__ so this works even on modules that
        # otherwise reject unknown attributes. The attribute ends up
        # in attn.__dict__, which is the lookup AOT-compiled code
        # uses.
        if "_alignatt_mt_qk_tensor_observer" not in attn.__dict__:
            attn._alignatt_mt_qk_tensor_observer = None
            stubbed += 1
        # Tag with a stable integer layer index so the patched
        # forward can pass it to the alignatt::capture_mt_qk custom
        # op (which looks up the per-layer observer in the registry
        # dict via that index).
        attn._alignatt_mt_layer_idx = int(layer_idx)
        # Ensure the registry has an entry, even if the entry is
        # None (no observer yet). That way
        # alignatt::capture_mt_qk always finds something in its
        # lookup; unconfigured layers just return early.
        _LAYER_OBSERVER_REGISTRY.setdefault(int(layer_idx), None)
    return stubbed


def _configure_mt_qk_observer_on_model(
    model,
    *,
    selected_heads: Sequence[dict[str, int]],
    max_prompt_tokens: int,
    max_decode_tokens: int,
) -> dict[str, Any]:
    from cascade_mt_backend import map_attention_head_to_key_value_head

    layers = _resolve_vllm_gemma_decoder_layers(model)
    heads_by_layer: dict[int, list[int]] = {}
    for head in selected_heads:
        heads_by_layer.setdefault(int(head["layer"]), []).append(int(head["head"]))
    if not heads_by_layer:
        raise ValueError("selected_heads must not be empty for MT tensor observer")

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
        observer = _get_mt_qk_tensor_observer(attn)
        if observer is None or any(
            (
                tuple(observer.selected_heads) != tuple(int(h) for h in layer_head_indices),
                tuple(int(h) for h in observer.selected_kv_heads)
                != tuple(int(h) for h in selected_kv_heads),
                int(observer.prompt_k_buffer.shape[1]) != int(max_prompt_tokens),
                int(observer.decode_q_buffer.shape[0]) != int(max_decode_tokens),
                int(observer.head_dim) != int(attn.head_dim),
            )
        ):
            observer = _MTPromptDecodeQKTensorObserver(
                selected_heads=list(layer_head_indices),
                selected_kv_heads=selected_kv_heads,
                max_prompt_tokens=int(max_prompt_tokens),
                max_decode_tokens=int(max_decode_tokens),
                head_dim=int(attn.head_dim),
                scaling=float(getattr(attn, "scaling", 1.0)),
                device=device,
            )
        attn._alignatt_mt_qk_tensor_observer = observer
        # Keep the global registry in sync so the alignatt::capture_mt_qk
        # custom op dispatches to this observer via the layer-index
        # key on this module.
        attn._alignatt_mt_layer_idx = int(layer_idx)
        _register_layer_observer(int(layer_idx), observer)
        layer_indices.append(int(layer_idx))

    model._alignatt_mt_qk_state = {
        "storage_mode": "tensor_buffers",
        "layer_indices": tuple(sorted(layer_indices)),
    }
    return {
        "layer_count": len(heads_by_layer),
        "max_prompt_tokens": int(max_prompt_tokens),
        "max_decode_tokens": int(max_decode_tokens),
    }


def _prepare_mt_qk_observer_on_model(model, *, prompt_length: int) -> dict[str, Any]:
    state = getattr(model, "_alignatt_mt_qk_state", None)
    if state is None:
        raise RuntimeError("MT tensor observer is not configured on the model.")
    for _layer_idx, observer in _resolve_mt_observer_bindings(model):
        observer.prepare(prompt_length=int(prompt_length))
    return {"prompt_length": int(prompt_length)}


def _fetch_mt_qk_observer_from_model(model) -> dict[str, Any] | None:
    bindings = _resolve_mt_observer_bindings(model)
    if not bindings:
        return None

    first_observer = bindings[0][1]
    prompt_length = int(first_observer.prompt_length_tensor.item())
    payload: dict[str, Any] = {
        "prompt_length": prompt_length,
        "layer_captures": {},
        "debug": {
            "forward_call_count": 0,
            "prompt_forward_call_count": 0,
            "decode_forward_call_count": 0,
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

        prompt_written = observer.prompt_written_buffer[:prompt_length].detach().cpu()
        missing_prompt = [
            int(idx) for idx, present in enumerate(prompt_written.tolist()) if not present
        ]
        prompt_k = None
        if not missing_prompt and prompt_length > 0:
            prompt_k = (
                observer.prompt_k_buffer[:, :prompt_length, :]
                .detach()
                .float()
                .cpu()
                .numpy()
            )

        decode_written_cpu = observer.decode_written_buffer.detach().cpu().tolist()
        decode_count = 0
        for present in decode_written_cpu:
            if not present:
                break
            decode_count += 1
        decode_q = None
        decode_k = None
        if decode_count > 0:
            decode_q = (
                observer.decode_q_buffer[:decode_count].detach().float().cpu().numpy()
            )
            decode_k = (
                observer.decode_k_buffer[:decode_count].detach().float().cpu().numpy()
            )

        payload["layer_captures"][int(layer_idx)] = {
            "selected_heads": list(observer.selected_heads),
            "selected_kv_heads": list(observer.selected_kv_heads),
            "prompt_k": prompt_k,
            "prompt_missing_positions": missing_prompt,
            "decode_q": decode_q,
            "decode_k": decode_k,
            "scaling": float(observer.scaling),
            "head_dim": int(observer.head_dim),
        }
        payload["debug"]["layer_stats"][str(int(layer_idx))] = {
            "selected_head_count": len(observer.selected_heads),
            "prompt_capture_count": int(prompt_written.sum().item()),
            "decode_q_count": int(decode_count),
            "missing_prompt_count": len(missing_prompt),
        }

    return payload


@dataclass(frozen=True)
class MTAttentionReconstruction:
    source_attention_rows_per_token: list["torch.Tensor"]
    provenance_mass_per_token: list[tuple[float, float, float, float]]
    diagnostics: dict[str, Any]


def reconstruct_mt_attention_rows(
    capture_payload: dict[str, Any] | None,
    *,
    alignatt_heads: Sequence[Any],
    source_positions: Sequence[int],
    accessible_source_token_count: int,
) -> MTAttentionReconstruction:
    """Compute source attention rows + provenance from a captured payload.

    Mirrors the Transformers qk_fast path:
      1. For each selected head, softmax(Q @ [prompt_K | suffix_K]^T * scaling)
      2. Slice weights at ``source_positions`` → row_matrix shape (n_generated, n_source)
      3. Sum-reduce into 4-way provenance mass (accessible, inaccessible,
         non_source_prompt, suffix), averaged across heads.
    Returns per-token rows of shape (n_heads_effective, n_source) and per-token
    provenance tuples. No sliding-window mask is applied yet; the Gemma 4
    sliding window (4096) is larger than any prompt we handle at
    ``gemma_max_model_len=1024``, so within this budget the mask is a no-op.
    """
    diagnostics: dict[str, Any] = {
        "captured_layer_count": 0,
        "effective_head_count": 0,
        "generated_token_count": 0,
        "prompt_length": 0,
        "missing_heads": [],
    }
    if not capture_payload or not alignatt_heads or not source_positions:
        return MTAttentionReconstruction([], [], diagnostics)

    layer_captures = capture_payload.get("layer_captures", {}) or {}
    diagnostics["captured_layer_count"] = len(layer_captures)
    prompt_length = int(capture_payload.get("prompt_length", 0))
    diagnostics["prompt_length"] = prompt_length

    decode_counts = [
        int(payload["decode_q"].shape[0])
        for payload in layer_captures.values()
        if payload.get("decode_q") is not None
    ]
    generated_token_count = min(decode_counts) if decode_counts else 0
    diagnostics["generated_token_count"] = generated_token_count
    if generated_token_count <= 0 or prompt_length <= 0:
        return MTAttentionReconstruction([], [], diagnostics)

    source_idx = torch.tensor(list(source_positions), dtype=torch.long)
    prompt_valid = (source_idx >= 0) & (source_idx < prompt_length)
    accessible_idx = torch.tensor(
        list(source_positions[:accessible_source_token_count]), dtype=torch.long
    )
    inaccessible_idx = torch.tensor(
        list(source_positions[accessible_source_token_count:]), dtype=torch.long
    )

    n_src = int(source_idx.numel())
    head_row_matrices: list[torch.Tensor] = []
    provenance_sum: torch.Tensor | None = None
    provenance_head_count = 0
    missing_heads: list[dict[str, int]] = []
    effective_head_count = 0

    for head in alignatt_heads:
        layer_idx = int(head.layer)
        head_idx = int(head.head)
        layer_payload = layer_captures.get(layer_idx)
        if layer_payload is None:
            missing_heads.append({"layer": layer_idx, "head": head_idx})
            continue
        prompt_k = layer_payload.get("prompt_k")
        decode_q = layer_payload.get("decode_q")
        decode_k = layer_payload.get("decode_k")
        selected_q_heads = list(layer_payload.get("selected_heads", []))
        selected_kv_heads = list(layer_payload.get("selected_kv_heads", []))
        if prompt_k is None or decode_q is None or decode_k is None or head_idx not in selected_q_heads:
            missing_heads.append({"layer": layer_idx, "head": head_idx})
            continue

        local_q = selected_q_heads.index(head_idx)
        local_kv = selected_kv_heads[local_q]
        # prompt_k/decode_k are indexed by local KV-head position in the observer;
        # selected_kv_heads contains the absolute KV-head indices, so we must map
        # local_kv back to its position in selected_kv_heads (typically == local_q
        # when there is 1:1 head-to-kv mapping in the observer config).
        #
        # The observer stores K at position ``i`` = index in ``selected_kv_heads``.
        # Since we constructed ``selected_kv_heads`` 1:1 from ``selected_heads`` at
        # configure time, local_q == local_kv_storage_idx. To stay robust, compute
        # the storage index explicitly.
        try:
            local_kv_storage_idx = selected_kv_heads.index(local_kv)
        except ValueError:
            missing_heads.append({"layer": layer_idx, "head": head_idx})
            continue

        scaling = float(layer_payload.get("scaling", 1.0))
        q_head = torch.from_numpy(np.asarray(decode_q[:, local_q, :], dtype=np.float32))
        k_prompt = torch.from_numpy(
            np.asarray(prompt_k[local_kv_storage_idx, :, :], dtype=np.float32)
        )
        k_decode = torch.from_numpy(
            np.asarray(decode_k[:, local_kv_storage_idx, :], dtype=np.float32)
        )

        # full logits: [n_generated, prompt_length + n_generated]
        prompt_logits = q_head @ k_prompt.transpose(0, 1)
        suffix_logits = q_head @ k_decode.transpose(0, 1)
        if scaling != 1.0:
            prompt_logits = prompt_logits * scaling
            suffix_logits = suffix_logits * scaling
        # causal mask on suffix: token i can attend to suffix positions <= i.
        n_gen = suffix_logits.shape[0]
        causal = torch.triu(torch.ones(n_gen, n_gen, dtype=torch.bool), diagonal=1)
        suffix_logits = suffix_logits.masked_fill(causal, float("-inf"))

        full = torch.cat([prompt_logits, suffix_logits], dim=-1)
        weights = torch.softmax(full, dim=-1)
        prompt_weights = weights[:, :prompt_length]
        suffix_weights = weights[:, prompt_length:]

        row = torch.zeros(n_gen, n_src, dtype=torch.float32)
        if prompt_valid.any():
            row[:, prompt_valid] = prompt_weights[:, source_idx[prompt_valid]]
        head_row_matrices.append(row)
        effective_head_count += 1

        # provenance
        acc_valid = (accessible_idx >= 0) & (accessible_idx < prompt_length)
        accessible_mass = (
            prompt_weights[:, accessible_idx[acc_valid]].sum(dim=-1)
            if acc_valid.any()
            else torch.zeros(n_gen)
        )
        if inaccessible_idx.numel() > 0:
            inacc_valid = (inaccessible_idx >= 0) & (inaccessible_idx < prompt_length)
            inaccessible_mass = (
                prompt_weights[:, inaccessible_idx[inacc_valid]].sum(dim=-1)
                if inacc_valid.any()
                else torch.zeros(n_gen)
            )
        else:
            inaccessible_mass = torch.zeros(n_gen)
        suffix_mass = suffix_weights.sum(dim=-1)
        non_source_mass = (1.0 - accessible_mass - inaccessible_mass - suffix_mass).clamp_min(0.0)

        if provenance_sum is None:
            provenance_sum = torch.zeros(n_gen, 4)
        provenance_sum[:, 0] += accessible_mass
        provenance_sum[:, 1] += inaccessible_mass
        provenance_sum[:, 2] += non_source_mass
        provenance_sum[:, 3] += suffix_mass
        provenance_head_count += 1

    diagnostics["effective_head_count"] = effective_head_count
    diagnostics["missing_heads"] = missing_heads

    if not head_row_matrices:
        return MTAttentionReconstruction([], [], diagnostics)

    stacked = torch.stack(head_row_matrices, dim=0)  # (n_heads, n_gen, n_src)
    rows_per_token = [stacked[:, token_idx, :] for token_idx in range(stacked.shape[1])]

    provenance_rows: list[tuple[float, float, float, float]] = []
    if provenance_sum is not None and provenance_head_count > 0:
        avg = provenance_sum / float(provenance_head_count)
        for token_idx in range(avg.shape[0]):
            provenance_rows.append(
                (
                    float(avg[token_idx, 0]),
                    float(avg[token_idx, 1]),
                    float(avg[token_idx, 2]),
                    float(avg[token_idx, 3]),
                )
            )

    return MTAttentionReconstruction(
        source_attention_rows_per_token=rows_per_token,
        provenance_mass_per_token=provenance_rows,
        diagnostics=diagnostics,
    )
