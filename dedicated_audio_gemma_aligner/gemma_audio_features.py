"""Extract frozen audio features from Gemma's audio encoder.

Given a loaded Gemma multimodal model and raw audio, this module runs
only the audio tower (conformer encoder + output projection) and returns
the resulting feature tensor together with timing metadata.

The audio tower is always frozen — no gradients flow through it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from transformers import AutoProcessor

GEMMA_AUDIO_MS_PER_TOKEN = 40.0
GEMMA_AUDIO_MAX_TOKENS = 750


@dataclass(frozen=True)
class GemmaAudioFeatures:
    features: torch.Tensor       # (num_tokens, feature_dim)
    num_tokens: int
    feature_dim: int
    ms_per_token: float
    audio_duration_s: float

    @property
    def seconds_per_token(self) -> float:
        return self.ms_per_token / 1000.0

    def token_to_seconds(self, token_idx: int) -> float:
        return min((token_idx + 1) * self.seconds_per_token, self.audio_duration_s)


def extract_audio_features(
    model,
    processor: "AutoProcessor",
    audio: np.ndarray,
    *,
    sample_rate: int = 16000,
    device: str | torch.device = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
) -> GemmaAudioFeatures:
    """Run Gemma's frozen audio tower on raw audio, return features + timing.

    Returns the output of audio_tower (conformer encoder + output_proj),
    shape (num_audio_tokens, 1536) for E4B. This is before the LM embedder
    projection — these are the richest audio-specific features.
    """
    audio_duration_s = len(audio) / sample_rate

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": np.asarray(audio, dtype=np.float32)},
                {"type": "text", "text": "Transcribe."},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    input_features = inputs.get("input_features")
    input_features_mask = inputs.get("input_features_mask")
    if input_features is None:
        raise ValueError("Processor did not produce input_features for this audio")

    input_features = input_features.to(device=device, dtype=dtype)
    input_features_mask = input_features_mask.to(device=device)

    base_model = getattr(model, "model", model)
    audio_tower = base_model.audio_tower

    with torch.no_grad():
        audio_out = audio_tower(input_features, input_features_mask, return_dict=True)

    hidden = audio_out.last_hidden_state  # (1, num_tokens, 1536)
    mask = audio_out.attention_mask        # (1, num_tokens)

    valid_mask = mask[0].bool()
    features = hidden[0][valid_mask]  # (num_valid_tokens, 1536)

    ms_per_token = getattr(processor, "audio_ms_per_token", GEMMA_AUDIO_MS_PER_TOKEN)

    return GemmaAudioFeatures(
        features=features,
        num_tokens=features.shape[0],
        feature_dim=features.shape[1],
        ms_per_token=float(ms_per_token),
        audio_duration_s=audio_duration_s,
    )
