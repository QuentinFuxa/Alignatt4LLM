"""Full held-out evaluation of the Gemma feature aligner v3.

Evaluates on ALL test clips (talk 111), reports per-clip and aggregate.

Usage:
    /home/fuxa/iwslt-2026-baselines/.venv-inference/bin/python run_gemma_feature_aligner_eval.py
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
    MANIFEST_PATH,
    TEACHER_DIR,
    evaluate_alignment,
    get_text_embeddings,
    load_gemma,
    load_teacher,
    predict_alignment,
)


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
    return aligner, ckpt


def main():
    ckpt_path = CHECKPOINT_DIR / "aligner_v3.pt"
    if not ckpt_path.exists():
        print(f"No checkpoint at {ckpt_path}. Run training first.")
        return

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    model, processor = load_gemma()
    tokenizer = processor.tokenizer
    aligner, ckpt = load_checkpoint(ckpt_path, DEVICE)
    print(f"Loaded aligner from {ckpt_path}")
    print(f"  Config: {ckpt['config']}")
    print(f"  Params: {ckpt.get('param_count', '?'):,}")

    test_clips = manifest["splits"]["test"]
    print(f"\n=== FULL TEST evaluation ({len(test_clips)} clips, talk 111) ===")

    results = {}
    for clip in test_clips:
        tag = clip["tag"]
        teacher_path = TEACHER_DIR / f"{tag}_qwen_teacher.json"
        if not teacher_path.exists():
            print(f"  {tag}: SKIP (no teacher)")
            continue

        audio, sr = sf.read(clip["audio"])
        teacher = load_teacher(str(teacher_path))

        t0 = time.time()
        feat = extract_audio_features(
            model, processor, audio, sample_rate=sr, device=DEVICE, dtype=DTYPE,
        )
        feat_time = time.time() - t0

        encoded = tokenizer(teacher["text"], add_special_tokens=False)
        tids = torch.tensor(encoded["input_ids"], dtype=torch.long, device=DEVICE)
        text_emb = get_text_embeddings(model, tids).float()

        t0 = time.time()
        result = predict_alignment(
            aligner, feat, text_emb, tids, tokenizer,
            teacher["text"], feat.audio_duration_s,
        )
        head_time = time.time() - t0

        metrics = evaluate_alignment(result, teacher)
        metrics["head_inference_time_s"] = head_time
        metrics["feat_extraction_time_s"] = feat_time
        metrics["total_alignment_time_s"] = head_time + feat_time
        metrics["audio_duration_s"] = feat.audio_duration_s
        metrics["tag"] = tag
        metrics["talk_id"] = clip.get("talk_id", "unknown")
        results[tag] = metrics

        print(f"  {tag} ({feat.audio_duration_s:.1f}s): "
              f"MAE={metrics.get('mae_s', 0):.3f}s  "
              f"P90={metrics.get('p90_error_s', 0):.3f}s  "
              f"mono={metrics['monotone']}  "
              f"total={metrics['total_alignment_time_s']:.3f}s")

        out = {
            "tag": tag, "split": "test",
            "backend": "gemma_feature_aligner_v3",
            "text": result.text,
            "audio_duration_s": result.audio_duration_s,
            "words": [asdict(w) for w in result.words],
            "metrics": metrics,
        }
        with open(CHECKPOINT_DIR / f"eval_v3_{tag}.json", "w") as f:
            json.dump(out, f, indent=2)

    if results:
        all_mae = [r["mae_s"] for r in results.values() if "mae_s" in r]
        all_median = [r["median_error_s"] for r in results.values() if "median_error_s" in r]
        all_p90 = [r["p90_error_s"] for r in results.values() if "p90_error_s" in r]
        all_head = [r["head_inference_time_s"] for r in results.values()]
        all_feat = [r["feat_extraction_time_s"] for r in results.values()]
        all_total = [r["total_alignment_time_s"] for r in results.values()]

        agg = {
            "version": "v3",
            "split": "test",
            "num_clips": len(results),
            "mean_mae_s": float(np.mean(all_mae)) if all_mae else None,
            "mean_median_error_s": float(np.mean(all_median)) if all_median else None,
            "mean_p90_s": float(np.mean(all_p90)) if all_p90 else None,
            "all_monotone": all(r["monotone"] for r in results.values()),
            "mean_head_time_s": float(np.mean(all_head)),
            "mean_feat_time_s": float(np.mean(all_feat)),
            "mean_total_time_s": float(np.mean(all_total)),
            "per_clip": results,
        }
        print(f"\n=== TEST AGGREGATE ({len(results)} clips) ===")
        print(f"  Mean MAE:    {agg['mean_mae_s']:.4f}s" if agg["mean_mae_s"] else "")
        print(f"  Mean median: {agg['mean_median_error_s']:.4f}s" if agg["mean_median_error_s"] else "")
        print(f"  Mean P90:    {agg['mean_p90_s']:.4f}s" if agg["mean_p90_s"] else "")
        print(f"  All monotone: {agg['all_monotone']}")
        print(f"  Mean total time: {agg['mean_total_time_s']:.3f}s")

        out_path = CHECKPOINT_DIR / "heldout_eval_v3.json"
        with open(out_path, "w") as f:
            json.dump(agg, f, indent=2)
        print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
