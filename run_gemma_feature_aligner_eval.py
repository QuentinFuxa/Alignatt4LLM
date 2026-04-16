"""Offline multi-clip evaluation of the trained Gemma feature aligner.

Loads a trained checkpoint and evaluates on all available clips with
Qwen teacher timestamps. Reports per-clip and aggregate metrics.

Usage:
    /home/fuxa/iwslt26-sst/.venv-qwen35-vllm/bin/python run_gemma_feature_aligner_eval.py
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from gemma_audio_features import extract_audio_features
from gemma_feature_aligner import TranscriptAudioAligner
from run_gemma_feature_aligner_train import (
    CHECKPOINT_DIR,
    DEVICE,
    DTYPE,
    GEMMA_MODEL_PATH,
    evaluate_alignment,
    get_text_embeddings,
    load_gemma,
    load_teacher,
    predict_alignment,
)

EVAL_CLIPS = [
    {
        "audio": "tmp/alignatt_smoke18.wav",
        "teacher": "tmp/alignment_research/frontier_smoke18_qwen_teacher.json",
        "tag": "smoke18",
    },
    {
        "audio": "tmp/ccpXHNfaoy_first75.wav",
        "teacher": "tmp/alignment_research/ccpXHNfaoy_18s_qwen_teacher.json",
        "tag": "ccpXHNfaoy_18s",
        "slice_seconds": (0, 18),
    },
    {
        "audio": "tmp/ccpXHNfaoy_first75.wav",
        "teacher": "tmp/alignment_research/ccpXHNfaoy_30s_48s_qwen_teacher.json",
        "tag": "ccpXHNfaoy_30s_48s",
        "slice_seconds": (30, 48),
    },
]


def load_checkpoint(path: Path, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg = ckpt["config"]
    aligner = TranscriptAudioAligner(
        text_embed_dim=cfg["text_embed_dim"],
        audio_dim=cfg["audio_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
    ).to(device)
    aligner.load_state_dict(ckpt["model_state"])
    aligner.eval()
    return aligner


def main():
    ckpt_path = CHECKPOINT_DIR / "aligner_v1.pt"
    if not ckpt_path.exists():
        print(f"No checkpoint at {ckpt_path}. Run training first.")
        return

    model, processor = load_gemma()
    tokenizer = processor.tokenizer
    aligner = load_checkpoint(ckpt_path, DEVICE)
    print(f"Loaded aligner from {ckpt_path}")

    results = {}
    for clip in EVAL_CLIPS:
        tag = clip["tag"]
        audio, sr = sf.read(clip["audio"])

        if "slice_seconds" in clip:
            s, e = clip["slice_seconds"]
            audio = audio[int(s * sr):int(e * sr)]

        if len(audio) / sr > 30.0:
            print(f"  {tag}: audio too long ({len(audio)/sr:.1f}s > 30s), skipping")
            continue

        teacher = load_teacher(clip["teacher"]) if clip["teacher"] else None
        text = teacher["text"] if teacher else None
        if text is None:
            continue

        print(f"\nEvaluating {tag} ({len(audio)/sr:.1f}s)...")
        feat = extract_audio_features(model, processor, audio, sample_rate=sr, device=DEVICE, dtype=DTYPE)
        encoded = tokenizer(text, add_special_tokens=False)
        tids = torch.tensor(encoded["input_ids"], dtype=torch.long, device=DEVICE)
        text_emb = get_text_embeddings(model, tids).float()

        t0 = time.time()
        result = predict_alignment(aligner, feat, text_emb, tids, tokenizer, text, feat.audio_duration_s)
        inference_time = time.time() - t0

        metrics = evaluate_alignment(result, teacher)
        metrics["inference_time_s"] = inference_time
        metrics["tag"] = tag
        results[tag] = metrics

        print(f"  MAE: {metrics.get('mae_s', 'N/A'):.4f}s" if "mae_s" in metrics else "  MAE: N/A")
        print(f"  Median: {metrics.get('median_error_s', 'N/A'):.4f}s" if "median_error_s" in metrics else "")
        print(f"  P90: {metrics.get('p90_error_s', 'N/A'):.4f}s" if "p90_error_s" in metrics else "")
        print(f"  Monotone: {metrics['monotone']}")
        print(f"  Inference: {inference_time:.4f}s")

        if teacher:
            print("  Word comparison (first 10):")
            for i in range(min(10, len(result.words), len(teacher["words"]))):
                pw, tw = result.words[i], teacher["words"][i]
                print(f"    {tw['text']:15s} teacher={tw['end_time']:.2f}  pred={pw.end_time:.2f}  "
                      f"err={abs(pw.end_time - tw['end_time']):.2f}")

        out = {
            "tag": tag, "backend": "gemma_feature_aligner",
            "text": result.text, "audio_duration_s": result.audio_duration_s,
            "words": [asdict(w) for w in result.words], "metrics": metrics,
        }
        with open(CHECKPOINT_DIR / f"eval_{tag}.json", "w") as f:
            json.dump(out, f, indent=2)

    if results:
        all_mae = [r["mae_s"] for r in results.values() if "mae_s" in r]
        all_p90 = [r["p90_error_s"] for r in results.values() if "p90_error_s" in r]
        all_inf = [r["inference_time_s"] for r in results.values()]

        agg = {
            "num_clips": len(results),
            "mean_mae_s": float(np.mean(all_mae)) if all_mae else None,
            "mean_p90_s": float(np.mean(all_p90)) if all_p90 else None,
            "mean_inference_s": float(np.mean(all_inf)),
            "all_monotone": all(r["monotone"] for r in results.values()),
            "per_clip": results,
        }
        with open(CHECKPOINT_DIR / "multi_clip_eval.json", "w") as f:
            json.dump(agg, f, indent=2)
        print(f"\n=== Aggregate ({len(results)} clips) ===")
        print(f"  Mean MAE: {agg['mean_mae_s']:.4f}s" if agg["mean_mae_s"] else "")
        print(f"  Mean P90: {agg['mean_p90_s']:.4f}s" if agg["mean_p90_s"] else "")
        print(f"  All monotone: {agg['all_monotone']}")
        print(f"  Saved -> {CHECKPOINT_DIR / 'multi_clip_eval.json'}")


if __name__ == "__main__":
    main()
