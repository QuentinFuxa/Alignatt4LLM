#!/usr/bin/env python3
"""Validate qk_fast vs eager agreement for Gemma audio forced alignment.

Strategy: run one forward pass with EAGER + capture both attention weights
AND layer inputs, then manually recompute Q*K from layer inputs and compare
against the captured eager weights. This isolates whether Q*K recomputation
is correct independently of sdpa vs eager differences.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np
import torch

from cascade.mt.base import (
    SelectedLayerInputRecorder,
    compute_alignatt_source_argmaxes,
    compute_key_states_from_layer_input_capture,
    compute_query_states_from_layer_input_capture,
    extract_source_attention_rows_per_token,
    map_attention_head_to_key_value_head,
)
from cascade.alignment.gemma_transformers_asr_backend import (
    GemmaTransformersASRBackend,
    detect_audio_span,
)
from cascade.runtime import CascadeRuntimeConfig, gemma_model_name


def load_wav(path: str) -> tuple[np.ndarray, int]:
    import wave
    with wave.open(path, "rb") as wav:
        sr = wav.getframerate()
        raw = wav.readframes(wav.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if sr != 16000:
        duration = len(audio) / sr
        new_length = int(duration * 16000)
        old_times = np.linspace(0.0, duration, num=len(audio), endpoint=False)
        new_times = np.linspace(0.0, duration, num=new_length, endpoint=False)
        audio = np.interp(new_times, old_times, audio).astype(np.float32)
        sr = 16000
    return audio, sr


def main():
    wav_path = "tmp/alignatt_smoke18.wav"
    config = CascadeRuntimeConfig()
    heads_path = config.gemma_audio_alignment_heads_path
    transcript = (
        "Hi, I'm Siyu Yuan from Fudan University. "
        "Today I'd like to share with you our paper, "
        "Plan, Verify and Switch, Integrated Reasoning with Diverse X of Thoughts."
    )

    audio, sr = load_wav(wav_path)
    print(f"Audio: {len(audio) / sr:.1f}s")

    backend = GemmaTransformersASRBackend(
        model_name=gemma_model_name,
        runtime_config=config,
        audio_heads_path=heads_path,
        audio_heads_top_k=8,
    )
    backend.load()
    heads = backend.alignatt_heads
    print(f"{len(heads)} heads: {[(h.layer, h.head) for h in heads]}")

    audio_arr = np.asarray(audio, dtype=np.float32)
    backend._enforce_audio_cap(audio_arr, sample_rate=sr)
    inputs, input_ids, transcript_span = backend._prepare_forced_alignment_inputs(
        audio_arr, language="en", transcript=transcript,
    )
    audio_span = detect_audio_span(
        input_ids,
        audio_token_id=backend.audio_token_id,
        audio_ms_per_token=backend.audio_ms_per_token,
    )
    t_start, t_end = transcript_span
    print(f"Seq={len(input_ids)}, audio=[{audio_span.prompt_start},{audio_span.prompt_end}), "
          f"transcript=[{t_start},{t_end})")

    # === Run EAGER with BOTH recorders simultaneously ===
    print("\n--- Single eager forward with both recorders ---")
    all_layer_attn_recorder = backend._build_all_layer_recorder()
    layer_input_recorder = SelectedLayerInputRecorder(
        model=backend.model,
        alignatt_heads=heads,
    )

    with backend._eager_attention_implementation(), torch.no_grad():
        with all_layer_attn_recorder.capture() as captured_attn:
            with layer_input_recorder.capture() as captured_inputs:
                _ = backend.model(**inputs, use_cache=False, return_dict=True)

    # Remove extra hooks to avoid accumulation
    for h in layer_input_recorder._hooks:
        h.remove()
    for h in all_layer_attn_recorder._hooks:
        h.remove()

    print(f"Captured attention layers: {sorted(captured_attn.keys())}")
    print(f"Captured input layers: {sorted(captured_inputs.keys())}")

    # === For one head, compare eager attention row vs manual Q*K ===
    head = heads[0]  # (23, 3)
    layer_idx = head.layer
    head_idx = head.head
    print(f"\n--- Detailed comparison for head L{layer_idx} H{head_idx} ---")

    # Eager attention: shape (batch=1, num_heads, seq_len, seq_len)
    eager_attn = captured_attn[layer_idx]
    print(f"Eager attn shape: {tuple(eager_attn.shape)}")

    # Check if eager attention is post-softmax (rows should sum to ~1)
    row0_sum = eager_attn[0, head_idx, t_start, :].sum().item()
    print(f"Eager row sum (transcript token 0, head {head_idx}, over all keys): {row0_sum:.6f}")

    # Extract eager attention from transcript token 0 to audio positions
    audio_positions = list(range(audio_span.prompt_start, audio_span.prompt_end))
    eager_audio_attn = eager_attn[0, head_idx, t_start, audio_span.prompt_start:audio_span.prompt_end]
    print(f"Eager audio attn shape: {tuple(eager_audio_attn.shape)}")
    print(f"Eager audio attn sum: {eager_audio_attn.sum().item():.6f}")
    print(f"Eager audio attn argmax: {eager_audio_attn.argmax().item()}")

    # Now recompute from layer inputs
    capture = captured_inputs[layer_idx]
    qs = compute_query_states_from_layer_input_capture(capture)
    ks = compute_key_states_from_layer_input_capture(capture)
    print(f"\nManual Q shape: {tuple(qs.shape)}, K shape: {tuple(ks.shape)}")

    num_attn_heads = qs.shape[1]
    num_kv_heads = ks.shape[1]
    kv_head_idx = map_attention_head_to_key_value_head(
        head_idx, num_attention_heads=num_attn_heads, num_key_value_heads=num_kv_heads,
    )
    print(f"num_attn_heads={num_attn_heads}, num_kv_heads={num_kv_heads}, "
          f"kv_head for head {head_idx} = {kv_head_idx}")

    q_head = qs[0, head_idx, t_start, :].float()  # (head_dim,)
    k_all = ks[0, kv_head_idx, :, :].float()       # (seq_len, head_dim)
    full_seq_len = k_all.shape[0]

    logits = torch.matmul(q_head.unsqueeze(0), k_all.transpose(0, 1)).squeeze(0)  # (seq_len,)
    scaling = float(getattr(capture.module, "scaling", 1.0))
    if scaling != 1.0:
        logits = logits * scaling

    # Apply causal mask (position t_start can attend to 0..t_start)
    causal_mask = torch.arange(full_seq_len, device=logits.device) > t_start
    logits_masked = logits.clone()
    logits_masked[causal_mask] = float("-inf")

    manual_weights = torch.softmax(logits_masked, dim=-1)
    manual_audio_attn = manual_weights[audio_span.prompt_start:audio_span.prompt_end]

    print(f"\nManual row sum (full seq): {manual_weights.sum().item():.6f}")
    print(f"Manual audio attn sum: {manual_audio_attn.sum().item():.6f}")
    print(f"Manual audio attn argmax: {manual_audio_attn.argmax().item()}")
    print(f"Eager  audio attn argmax: {eager_audio_attn.argmax().item()}")

    # Compare
    eager_cpu = eager_audio_attn.cpu().float()
    manual_cpu = manual_audio_attn.cpu().float()
    max_diff = (eager_cpu - manual_cpu).abs().max().item()
    mean_diff = (eager_cpu - manual_cpu).abs().mean().item()
    print(f"\nmax_diff={max_diff:.8f}, mean_diff={mean_diff:.8f}")

    # Check where the attention mass is for both
    print(f"\nEager top-5 audio positions: {eager_cpu.topk(5).indices.tolist()}")
    print(f"Manual top-5 audio positions: {manual_cpu.topk(5).indices.tolist()}")

    # === Check: where does the eager attention mass go? ===
    full_row = eager_attn[0, head_idx, t_start, :].cpu().float()
    print(f"\n--- Where does eager attention mass go? ---")
    print(f"Total: {full_row.sum().item():.4f}")
    print(f"Audio region [{audio_span.prompt_start}:{audio_span.prompt_end}]: "
          f"{full_row[audio_span.prompt_start:audio_span.prompt_end].sum().item():.4f}")
    print(f"Before audio [0:{audio_span.prompt_start}]: "
          f"{full_row[:audio_span.prompt_start].sum().item():.4f}")
    print(f"After audio [{audio_span.prompt_end}:{t_start}]: "
          f"{full_row[audio_span.prompt_end:t_start].sum().item():.4f}")
    print(f"Transcript [{t_start}:{t_end}]: "
          f"{full_row[t_start:t_end].sum().item():.4f}")
    print(f"After transcript [{t_end}:]: {full_row[t_end:].sum().item():.4f}")

    # === Check: where does manual attention mass go? ===
    print(f"\n--- Where does manual Q*K attention mass go? ---")
    mw = manual_weights.cpu().float()
    print(f"Total: {mw.sum().item():.4f}")
    print(f"Audio region: {mw[audio_span.prompt_start:audio_span.prompt_end].sum().item():.4f}")
    print(f"Before audio: {mw[:audio_span.prompt_start].sum().item():.4f}")
    print(f"After audio to transcript: {mw[audio_span.prompt_end:t_start].sum().item():.4f}")
    print(f"Transcript: {mw[t_start:t_end].sum().item():.4f}")

    # === Raw logits comparison ===
    print(f"\n--- Raw logits at a few positions ---")
    # The eager attention pre-softmax logits aren't directly available,
    # but we can check if the distributions are fundamentally different
    print(f"Manual logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
    print(f"Manual logits at audio argmax ({manual_audio_attn.argmax().item()}): "
          f"{logits[audio_span.prompt_start + manual_audio_attn.argmax().item()].item():.4f}")

    # === Now run the full qk_fast path (with sdpa) ===
    print("\n" + "=" * 60)
    print("Full pipeline comparison: qk_fast (sdpa) vs eager")
    print("=" * 60)

    # qk_fast
    t0 = perf_counter()
    result_qk = backend.align_transcript(
        audio_arr, sample_rate=sr, language="en", transcript=transcript,
    )
    qk_ms = (perf_counter() - t0) * 1000
    qk_positions = result_qk.diagnostics.get("aligned_audio_positions", [])
    print(f"qk_fast ({result_qk.diagnostics.get('probe_backend')}): "
          f"{len(result_qk.words)} words, {qk_ms:.0f}ms, "
          f"mono={result_qk.diagnostics.get('monotonicity', 0):.3f}")

    # Disable layer_input_recorder to force eager
    saved_recorder = backend.alignatt_layer_input_recorder
    backend.alignatt_layer_input_recorder = None
    t0 = perf_counter()
    result_eager = backend.align_transcript(
        audio_arr, sample_rate=sr, language="en", transcript=transcript,
    )
    eager_ms = (perf_counter() - t0) * 1000
    eager_positions = result_eager.diagnostics.get("aligned_audio_positions", [])
    backend.alignatt_layer_input_recorder = saved_recorder
    print(f"eager ({result_eager.diagnostics.get('probe_backend')}): "
          f"{len(result_eager.words)} words, {eager_ms:.0f}ms, "
          f"mono={result_eager.diagnostics.get('monotonicity', 0):.3f}")

    n = min(len(qk_positions), len(eager_positions))
    match = sum(1 for i in range(n) if qk_positions[i] == eager_positions[i])
    print(f"\nArgmax match: {match}/{n}")
    if n > 0:
        diffs = [abs((qk_positions[i] or 0) - (eager_positions[i] or 0)) for i in range(n)]
        print(f"Max diff: {max(diffs)} tokens ({max(diffs) * 40}ms)")
        print(f"Mean diff: {sum(diffs)/len(diffs):.1f} tokens")


if __name__ == "__main__":
    main()
