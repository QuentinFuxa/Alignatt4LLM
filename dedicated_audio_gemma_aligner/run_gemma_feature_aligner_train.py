"""Train the Gemma feature aligner on the full ACL split.

Architecture inspired by Qwen3-ForcedAligner:
  - Discrete classification over audio positions (not regression)
  - Label smoothing for regularization
  - LIS-based monotonicity enforcement

Supervision: Qwen teacher timestamps (training only).
Features: frozen Gemma audio tower output + frozen Gemma text embeddings.

Usage:
    /home/fuxa/iwslt-2026-baselines/.venv-inference/bin/python dedicated_audio_gemma_aligner/run_gemma_feature_aligner_train.py
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.optim import Adam

from cascade.alignment.base import AlignmentResult, WordAlignment
from dedicated_audio_gemma_aligner.gemma_audio_features import GemmaAudioFeatures, extract_audio_features
from dedicated_audio_gemma_aligner.gemma_feature_aligner import TranscriptAudioAligner, enforce_monotone_lis
from cascade.alignment.gemma_transformers_asr_backend import (
    aggregate_token_timings_to_words,
    split_text_into_word_spans,
)

GEMMA_MODEL_PATH = "/home/fuxa/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/83df0a889143b1dbfc61b591bbc639540fd9ce4c"
DEVICE = "cuda:0"
DTYPE = torch.bfloat16
from dedicated_audio_gemma_aligner.paths import (
    ARTIFACT_DIR,
    FULL_SPLIT_MANIFEST,
    TEACHER_DIR as FEATURE_ALIGNER_TEACHER_DIR,
)

CHECKPOINT_DIR = ARTIFACT_DIR
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = FULL_SPLIT_MANIFEST
TEACHER_DIR = FEATURE_ALIGNER_TEACHER_DIR

HIDDEN_DIM = 256
NUM_LAYERS = 4
NUM_HEADS = 8
EPOCHS = 300
LR = 3e-4
LABEL_SMOOTHING = 0.1


def load_gemma():
    from transformers import AutoModelForMultimodalLM, AutoProcessor, modeling_utils

    print("Loading Gemma processor...")
    processor = AutoProcessor.from_pretrained(
        GEMMA_MODEL_PATH, trust_remote_code=True, local_files_only=True,
    )

    print("Loading Gemma model (sdpa attention, frozen)...")
    original_warmup = getattr(modeling_utils, "caching_allocator_warmup", None)
    if original_warmup is not None:
        modeling_utils.caching_allocator_warmup = lambda *a, **kw: None
    try:
        model = AutoModelForMultimodalLM.from_pretrained(
            GEMMA_MODEL_PATH,
            dtype=DTYPE,
            device_map=DEVICE,
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
    finally:
        if original_warmup is not None:
            modeling_utils.caching_allocator_warmup = original_warmup
    model.eval()
    print(f"Gemma loaded on {DEVICE}")
    return model, processor


def get_text_embeddings(model, token_ids: torch.Tensor) -> torch.Tensor:
    base = getattr(model, "model", model)
    lm = getattr(base, "language_model", base)
    embed_layer = lm.embed_tokens
    with torch.no_grad():
        return embed_layer(token_ids).detach()


def load_teacher(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def build_token_targets(
    teacher: dict,
    tokenizer,
    audio_features: GemmaAudioFeatures,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert word-level Qwen timestamps to per-token discrete audio position targets."""
    text = teacher["text"]
    words = teacher["words"]

    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    token_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]

    word_char_spans = split_text_into_word_spans(text)

    char_to_word = {}
    for w_idx, (ws, we, _) in enumerate(word_char_spans):
        for c in range(ws, we):
            char_to_word[c] = w_idx

    target_positions = []
    for start_char, end_char in offsets:
        word_indices = set()
        for c in range(start_char, end_char):
            if c in char_to_word:
                word_indices.add(char_to_word[c])

        if word_indices:
            w_idx = min(word_indices)
            if w_idx < len(words):
                w = words[w_idx]
                mid_time = (w["start_time"] + w["end_time"]) / 2.0
            else:
                mid_time = 0.0
        else:
            mid_time = 0.0

        target_pos = mid_time / audio_features.seconds_per_token
        target_pos = max(0.0, min(target_pos, audio_features.num_tokens - 1))
        target_positions.append(target_pos)

    return (
        torch.tensor(token_ids, dtype=torch.long),
        torch.tensor(target_positions, dtype=torch.float32),
    )


def _gaussian_target(positions: torch.Tensor, num_audio: int, sigma: float) -> torch.Tensor:
    audio_idx = torch.arange(num_audio, device=positions.device, dtype=torch.float32)
    diff = positions.unsqueeze(-1) - audio_idx.unsqueeze(0).unsqueeze(0)
    logits = -(diff ** 2) / (2 * sigma ** 2)
    return F.softmax(logits, dim=-1)


def train_aligner(
    aligner: TranscriptAudioAligner,
    train_data: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]],
    *,
    epochs: int = 300,
    lr: float = 3e-4,
    label_smoothing: float = 0.1,
) -> list[float]:
    """Train with Gaussian soft targets over audio positions."""
    optimizer = Adam(aligner.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for text_embeds, audio_feats, target_positions, num_audio in train_data:
            te = text_embeds.unsqueeze(0)
            af = audio_feats.unsqueeze(0)
            tp = target_positions.unsqueeze(0)

            logits = aligner(te, af)  # (1, T, A)

            target_dist = _gaussian_target(tp, num_audio, sigma=5.0)
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                target_dist.view(-1, target_dist.shape[-1]),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        losses.append(epoch_loss / len(train_data))
        if (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch+1:4d}  loss={losses[-1]:.4f}")

    return losses


def predict_alignment(
    aligner: TranscriptAudioAligner,
    audio_features: GemmaAudioFeatures,
    text_embeds: torch.Tensor,
    transcript_ids: torch.Tensor,
    tokenizer,
    text: str,
    audio_duration_s: float,
) -> AlignmentResult:
    aligner.eval()
    af = audio_features.features.unsqueeze(0).float()
    te = text_embeds.unsqueeze(0)

    with torch.no_grad():
        predicted = aligner.predict_positions(te, af)  # (1, T)

    positions = predicted[0].cpu().tolist()
    positions = enforce_monotone_lis(positions)

    token_end_times = [
        min((p + 1) * audio_features.seconds_per_token, audio_duration_s)
        for p in positions
    ]

    word_alignments = aggregate_token_timings_to_words(
        text,
        generated_ids=transcript_ids.tolist(),
        tokenizer=tokenizer,
        token_end_times_s=token_end_times,
        audio_duration_s=audio_duration_s,
    )

    return AlignmentResult(
        text=text,
        words=tuple(word_alignments),
        audio_duration_s=audio_duration_s,
        diagnostics={"backend": "gemma_feature_aligner_v3"},
    )


def evaluate_alignment(predicted: AlignmentResult, teacher: dict | None) -> dict:
    result = {
        "num_words": len(predicted.words),
        "monotone": all(
            predicted.words[i].end_time <= predicted.words[i + 1].end_time
            for i in range(len(predicted.words) - 1)
        ),
    }
    if teacher is None:
        return result

    teacher_words = teacher["words"]
    min_len = min(len(predicted.words), len(teacher_words))
    errors = [abs(predicted.words[i].end_time - teacher_words[i]["end_time"]) for i in range(min_len)]

    if errors:
        ea = np.array(errors)
        result["mae_s"] = float(np.mean(ea))
        result["median_error_s"] = float(np.median(ea))
        result["p90_error_s"] = float(np.percentile(ea, 90))
        result["max_error_s"] = float(np.max(ea))
        result["word_count_match"] = len(predicted.words) == len(teacher_words)

    return result


def load_manifest() -> dict:
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def load_clip_data(clip: dict, model, processor, tokenizer):
    tag = clip["tag"]
    audio, sr = sf.read(clip["audio"])

    teacher_path = TEACHER_DIR / f"{tag}_qwen_teacher.json"
    if not teacher_path.exists():
        return None
    teacher = load_teacher(str(teacher_path))

    t0 = time.time()
    feat = extract_audio_features(model, processor, audio, sample_rate=sr, device=DEVICE, dtype=DTYPE)
    feat_time = time.time() - t0

    tids, tpos = build_token_targets(teacher, tokenizer, feat)
    tids_dev = tids.to(DEVICE)
    tpos_dev = tpos.to(DEVICE)
    text_emb = get_text_embeddings(model, tids_dev).float()
    audio_feats = feat.features.float()

    return {
        "tag": tag,
        "teacher": teacher,
        "features": feat,
        "token_ids": tids_dev,
        "target_positions": tpos_dev,
        "text_embeds": text_emb,
        "audio_feats": audio_feats,
        "feat_extraction_time_s": feat_time,
    }


def main():
    manifest = load_manifest()
    train_clips = manifest["splits"]["train"]
    test_clips = manifest["splits"]["test"]

    model, processor = load_gemma()
    tokenizer = processor.tokenizer

    test_ids = torch.tensor([1], dtype=torch.long, device=DEVICE)
    test_emb = get_text_embeddings(model, test_ids)
    text_embed_dim = test_emb.shape[-1]
    print(f"Text embedding dim: {text_embed_dim}")

    # Load train data
    print(f"\n=== Loading {len(train_clips)} train clips ===")
    train_data = []
    train_loaded = []
    skipped = 0
    for i, clip in enumerate(train_clips):
        loaded = load_clip_data(clip, model, processor, tokenizer)
        if loaded is None:
            skipped += 1
            continue
        train_data.append((loaded["text_embeds"], loaded["audio_feats"],
                          loaded["target_positions"], loaded["features"].num_tokens))
        train_loaded.append(loaded)
        if (i + 1) % 50 == 0:
            print(f"  loaded {i+1}/{len(train_clips)} ({len(train_loaded)} usable)")
    print(f"  Total: {len(train_loaded)} train clips loaded, {skipped} skipped")

    # Load a small sample of test clips for quick validation during training
    print(f"\n=== Loading test clips for held-out check ===")
    test_loaded = []
    for clip in test_clips[:10]:
        loaded = load_clip_data(clip, model, processor, tokenizer)
        if loaded is not None:
            test_loaded.append(loaded)
    print(f"  {len(test_loaded)} test clips loaded for quick eval")

    # Build aligner
    audio_dim = train_loaded[0]["features"].feature_dim
    aligner = TranscriptAudioAligner(
        text_embed_dim=text_embed_dim,
        audio_dim=audio_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout=0.1,
    ).to(DEVICE)

    param_count = sum(p.numel() for p in aligner.parameters())
    print(f"\n=== Training ({param_count:,} params, {HIDDEN_DIM}d, {NUM_LAYERS}L, {NUM_HEADS}H) ===")
    print(f"  {len(train_data)} clips, {EPOCHS} epochs, lr={LR}, label_smoothing={LABEL_SMOOTHING}")

    t0 = time.time()
    losses = train_aligner(
        aligner, train_data,
        epochs=EPOCHS, lr=LR, label_smoothing=LABEL_SMOOTHING,
    )
    train_time = time.time() - t0
    print(f"Training: {train_time:.1f}s ({train_time/60:.1f} min), final loss: {losses[-1]:.4f}")

    # Quick train-fit check (sample of 10)
    print("\n=== Train-fit check (first 10) ===")
    train_maes = []
    for loaded in train_loaded[:10]:
        result = predict_alignment(
            aligner, loaded["features"], loaded["text_embeds"],
            loaded["token_ids"], tokenizer, loaded["teacher"]["text"],
            loaded["features"].audio_duration_s,
        )
        metrics = evaluate_alignment(result, loaded["teacher"])
        mae = metrics.get("mae_s", float("nan"))
        train_maes.append(mae)
        print(f"  {loaded['tag']}: MAE={mae:.3f}s, mono={metrics['monotone']}")
    print(f"  Train sample mean MAE: {np.mean(train_maes):.3f}s")

    # Held-out test check
    print("\n=== Held-out test check ===")
    test_maes = []
    for loaded in test_loaded:
        t0_inf = time.time()
        result = predict_alignment(
            aligner, loaded["features"], loaded["text_embeds"],
            loaded["token_ids"], tokenizer, loaded["teacher"]["text"],
            loaded["features"].audio_duration_s,
        )
        head_time = time.time() - t0_inf

        metrics = evaluate_alignment(result, loaded["teacher"])
        metrics["head_inference_time_s"] = head_time
        metrics["feat_extraction_time_s"] = loaded["feat_extraction_time_s"]
        metrics["total_alignment_time_s"] = head_time + loaded["feat_extraction_time_s"]
        mae = metrics.get("mae_s", float("nan"))
        test_maes.append(mae)
        print(f"  {loaded['tag']}: MAE={mae:.3f}s, mono={metrics['monotone']}, "
              f"total={metrics['total_alignment_time_s']:.3f}s")
    print(f"  Test mean MAE: {np.mean(test_maes):.3f}s")

    # Save checkpoint
    ckpt_path = CHECKPOINT_DIR / "aligner_v3.pt"
    torch.save({
        "model_state": aligner.state_dict(),
        "config": {
            "text_embed_dim": text_embed_dim,
            "audio_dim": audio_dim,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "num_heads": NUM_HEADS,
        },
        "final_loss": losses[-1],
        "train_time_s": train_time,
        "param_count": param_count,
        "train_tags": [l["tag"] for l in train_loaded],
        "test_tags": [l["tag"] for l in test_loaded],
        "epochs": EPOCHS,
        "lr": LR,
        "label_smoothing": LABEL_SMOOTHING,
    }, ckpt_path)
    print(f"\nSaved checkpoint -> {ckpt_path}")

    # Summary
    summary = {
        "version": "v3",
        "architecture": {
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "num_heads": NUM_HEADS,
            "params": param_count,
            "loss": "cross_entropy_discrete",
            "label_smoothing": LABEL_SMOOTHING,
            "monotonicity": "LIS-based (inspired by Qwen3-ForcedAligner)",
        },
        "training": {
            "num_train_clips": len(train_loaded),
            "epochs": EPOCHS,
            "lr": LR,
            "train_time_s": train_time,
            "final_loss": losses[-1],
        },
        "train_fit_sample_mae_s": float(np.mean(train_maes)),
        "test_sample_mae_s": float(np.mean(test_maes)),
        "train_talks": list(set(c["talk_id"] for c in train_clips)),
        "test_talks": list(set(c["talk_id"] for c in test_clips)),
    }
    with open(CHECKPOINT_DIR / "training_summary_v3.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary -> {CHECKPOINT_DIR / 'training_summary_v3.json'}")


if __name__ == "__main__":
    main()
