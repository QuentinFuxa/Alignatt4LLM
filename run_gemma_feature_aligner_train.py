"""Train + evaluate the dedicated Gemma feature aligner.

Supervision: Qwen teacher timestamps (training only).
Features: frozen Gemma audio tower output + frozen Gemma text embeddings.
Evaluation: offline forced alignment on held-out clips.

Usage:
    /home/fuxa/iwslt26-sst/.venv-qwen35-vllm/bin/python run_gemma_feature_aligner_train.py
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

from alignment_backend import AlignmentResult, WordAlignment
from gemma_audio_features import GemmaAudioFeatures, extract_audio_features
from gemma_feature_aligner import TranscriptAudioAligner
from gemma_alignment_probe import (
    aggregate_token_timings_to_words,
    split_text_into_word_spans,
)

GEMMA_MODEL_PATH = "/home/fuxa/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/83df0a889143b1dbfc61b591bbc639540fd9ce4c"
DEVICE = "cuda:0"
DTYPE = torch.bfloat16
CHECKPOINT_DIR = Path("tmp/feature_aligner")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CLIPS = [
    {
        "audio": "tmp/alignatt_smoke18.wav",
        "teacher": "tmp/alignment_research/frontier_smoke18_qwen_teacher.json",
        "tag": "smoke18",
    },
    {
        "audio": "tmp/ccpXHNfaoy_first75.wav",
        "teacher": "tmp/alignment_research/ccpXHNfaoy_30s_48s_qwen_teacher.json",
        "tag": "ccpXHNfaoy_30s_48s",
        "slice_seconds": (30, 48),
    },
]

EVAL_CLIPS = [
    {
        "audio": "tmp/alignatt_smoke18.wav",
        "teacher": "tmp/alignment_research/frontier_smoke18_qwen_teacher.json",
        "tag": "smoke18",
    },
    {
        "audio": "tmp/ccpXHNfaoy_first75.wav",
        "teacher": "tmp/alignment_research/ccpXHNfaoy_30s_48s_qwen_teacher.json",
        "tag": "ccpXHNfaoy_30s_48s",
        "slice_seconds": (30, 48),
    },
]


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
    """Extract frozen text embeddings from Gemma's language model."""
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
    """Convert word-level Qwen timestamps to per-token audio position targets.

    Maps each subword token to its overlapping word's midpoint time,
    then converts to audio-token index.
    """
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
    """Soft target: Gaussian centered at target position over audio axis."""
    audio_idx = torch.arange(num_audio, device=positions.device, dtype=torch.float32)
    diff = positions.unsqueeze(-1) - audio_idx.unsqueeze(0).unsqueeze(0)
    logits = -(diff ** 2) / (2 * sigma ** 2)
    return F.softmax(logits, dim=-1)


def train_aligner(
    aligner: TranscriptAudioAligner,
    train_data: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]],
    *,
    epochs: int = 1000,
    lr: float = 3e-4,
) -> list[float]:
    """Train on tiny dataset. Returns per-epoch losses.

    train_data items: (text_embeds, audio_features, target_positions, num_audio_tokens)
    """
    optimizer = Adam(aligner.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for text_embeds, audio_feats, target_positions, num_audio in train_data:
            te = text_embeds.unsqueeze(0)    # (1, T, E)
            af = audio_feats.unsqueeze(0)    # (1, A, D)
            tp = target_positions.unsqueeze(0)  # (1, T)

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
        if (epoch + 1) % 100 == 0:
            print(f"  epoch {epoch+1:4d}  loss={losses[-1]:.4f}")

    return losses


def enforce_monotone(positions: list[float]) -> list[float]:
    result = []
    running_max = 0.0
    for p in positions:
        running_max = max(running_max, p)
        result.append(running_max)
    return result


def predict_alignment(
    aligner: TranscriptAudioAligner,
    audio_features: GemmaAudioFeatures,
    text_embeds: torch.Tensor,
    transcript_ids: torch.Tensor,
    tokenizer,
    text: str,
    audio_duration_s: float,
) -> AlignmentResult:
    """Run inference and produce repo-compatible AlignmentResult."""
    aligner.eval()
    af = audio_features.features.unsqueeze(0).float()
    te = text_embeds.unsqueeze(0)

    with torch.no_grad():
        expected = aligner.predict_positions(te, af)  # (1, T)

    positions = expected[0].cpu().tolist()
    positions = enforce_monotone(positions)

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
        diagnostics={"backend": "gemma_feature_aligner"},
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


def main():
    model, processor = load_gemma()
    tokenizer = processor.tokenizer

    # Phase 1: Extract features
    print("\n=== Phase 1: Feature Extraction ===")
    all_clips = {c["tag"]: c for c in TRAIN_CLIPS + EVAL_CLIPS}
    clip_features = {}
    for tag, clip in all_clips.items():
        if tag in clip_features:
            continue
        audio, sr = sf.read(clip["audio"])
        if "slice_seconds" in clip:
            s, e = clip["slice_seconds"]
            audio = audio[int(s * sr):int(e * sr)]
        t0 = time.time()
        feat = extract_audio_features(model, processor, audio, sample_rate=sr, device=DEVICE, dtype=DTYPE)
        dt = time.time() - t0
        print(f"  {tag}: ({feat.num_tokens}, {feat.feature_dim}), {feat.audio_duration_s:.1f}s, {dt:.2f}s")
        clip_features[tag] = feat

    with open(CHECKPOINT_DIR / "feature_inspection.json", "w") as f:
        json.dump({t: {"num_tokens": ft.num_tokens, "feature_dim": ft.feature_dim,
                       "ms_per_token": ft.ms_per_token, "audio_duration_s": ft.audio_duration_s}
                   for t, ft in clip_features.items()}, f, indent=2)

    # Get text embedding dim
    test_ids = torch.tensor([1], dtype=torch.long, device=DEVICE)
    test_emb = get_text_embeddings(model, test_ids)
    text_embed_dim = test_emb.shape[-1]
    print(f"  Text embedding dim: {text_embed_dim}")

    # Phase 3: Build training data
    print("\n=== Phase 3: Training Data ===")
    train_data = []
    for clip in TRAIN_CLIPS:
        teacher = load_teacher(clip["teacher"])
        feat = clip_features[clip["tag"]]
        tids, tpos = build_token_targets(teacher, tokenizer, feat)
        tids_dev = tids.to(DEVICE)
        tpos_dev = tpos.to(DEVICE)
        text_emb = get_text_embeddings(model, tids_dev).float()
        audio_feats = feat.features.float()
        train_data.append((text_emb, audio_feats, tpos_dev, feat.num_tokens))
        print(f"  {clip['tag']}: {len(tids)} tokens, {len(teacher['words'])} words")

        # Debug: print target position range
        print(f"    target positions: min={tpos.min():.1f}, max={tpos.max():.1f}, "
              f"audio_tokens={feat.num_tokens}")

    # Build and train aligner
    print("\n=== Training ===")
    audio_dim = clip_features[TRAIN_CLIPS[0]["tag"]].feature_dim
    aligner = TranscriptAudioAligner(
        text_embed_dim=text_embed_dim,
        audio_dim=audio_dim,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
    ).to(DEVICE)

    param_count = sum(p.numel() for p in aligner.parameters())
    print(f"  Aligner params: {param_count:,}")

    t0 = time.time()
    losses = train_aligner(aligner, train_data, epochs=2000, lr=3e-4)
    train_time = time.time() - t0
    print(f"  Training: {train_time:.1f}s, final loss: {losses[-1]:.4f}")

    # Save checkpoint
    ckpt_path = CHECKPOINT_DIR / "aligner_v1.pt"
    torch.save({
        "model_state": aligner.state_dict(),
        "config": {
            "text_embed_dim": text_embed_dim,
            "audio_dim": audio_dim,
            "hidden_dim": 128,
            "num_layers": 2,
            "num_heads": 4,
        },
        "final_loss": losses[-1],
        "train_time_s": train_time,
        "param_count": param_count,
    }, ckpt_path)
    print(f"  Saved -> {ckpt_path}")

    # Phase 5: Evaluation
    print("\n=== Phase 5: Evaluation ===")
    eval_results = {}
    for clip in EVAL_CLIPS:
        tag = clip["tag"]
        feat = clip_features[tag]
        teacher = load_teacher(clip["teacher"]) if clip["teacher"] else None
        text = teacher["text"] if teacher else None
        if text is None:
            continue

        encoded = tokenizer(text, add_special_tokens=False)
        tids = torch.tensor(encoded["input_ids"], dtype=torch.long, device=DEVICE)
        text_emb = get_text_embeddings(model, tids).float()

        t0 = time.time()
        result = predict_alignment(aligner, feat, text_emb, tids, tokenizer, text, feat.audio_duration_s)
        inference_time = time.time() - t0

        metrics = evaluate_alignment(result, teacher)
        metrics["inference_time_s"] = inference_time
        eval_results[tag] = metrics

        print(f"\n  {tag}:")
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

        # Show first 10 words
        if teacher:
            print("    Word comparison (first 10):")
            for i in range(min(10, len(result.words), len(teacher["words"]))):
                pw, tw = result.words[i], teacher["words"][i]
                print(f"      {tw['text']:15s} teacher={tw['end_time']:.2f}  pred={pw.end_time:.2f}  "
                      f"err={abs(pw.end_time - tw['end_time']):.2f}")

        alignment_out = {
            "tag": tag, "backend": "gemma_feature_aligner",
            "text": result.text, "audio_duration_s": result.audio_duration_s,
            "words": [asdict(w) for w in result.words], "metrics": metrics,
        }
        with open(CHECKPOINT_DIR / f"alignment_{tag}.json", "w") as f:
            json.dump(alignment_out, f, indent=2)

    # Summary
    summary = {
        "aligner_params": param_count,
        "train_clips": [c["tag"] for c in TRAIN_CLIPS],
        "epochs": 2000,
        "final_loss": losses[-1],
        "train_time_s": train_time,
        "feature_source": "gemma_audio_tower_output_proj (1536-dim)",
        "text_embeddings": f"frozen gemma embed_tokens ({text_embed_dim}-dim)",
        "supervision": "qwen_teacher_timestamps",
        "eval_results": eval_results,
    }
    with open(CHECKPOINT_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved summary -> {CHECKPOINT_DIR / 'training_summary.json'}")


if __name__ == "__main__":
    main()
