"""Transcript-conditioned aligner on frozen Gemma audio features.

Inspired by Qwen3-ForcedAligner: timestamps as discrete classification
over audio positions, not continuous regression.

Architecture:
  1. Frozen Gemma text embeddings projected to hidden dim
  2. Frozen Gemma audio features projected to hidden dim
  3. Cross-attention transformer: transcript queries attend to audio keys
  4. Classification head: per-token logits over audio positions

Supervision: Qwen teacher timestamps (training only).
At inference, predicts audio positions independently of Qwen.
Monotonicity enforced via LIS-based post-processing (à la Qwen3).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TranscriptAudioAligner(nn.Module):
    def __init__(
        self,
        text_embed_dim: int,
        audio_dim: int = 1536,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.text_proj = nn.Linear(text_embed_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)

        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)

    def forward(
        self,
        text_embeds: torch.Tensor,       # (B, T, text_embed_dim)
        audio_features: torch.Tensor,    # (B, A, audio_dim)
    ) -> torch.Tensor:
        """Returns logits over audio positions for each transcript token.

        Returns: (B, T, A) — unnormalized scores.
        """
        tgt = self.text_proj(text_embeds)       # (B, T, H)
        memory = self.audio_proj(audio_features) # (B, A, H)

        tgt = tgt + self._sinusoidal_pe(tgt.shape[1], self.hidden_dim, tgt.device)
        memory = memory + self._sinusoidal_pe(memory.shape[1], self.hidden_dim, memory.device)

        decoded = self.decoder(tgt, memory)  # (B, T, H)
        scores = torch.bmm(decoded, memory.transpose(1, 2))  # (B, T, A)
        return scores

    def predict_positions(
        self,
        text_embeds: torch.Tensor,
        audio_features: torch.Tensor,
    ) -> torch.Tensor:
        """Predict audio position per transcript token (argmax)."""
        logits = self.forward(text_embeds, audio_features)  # (B, T, A)
        return logits.argmax(dim=-1).float()  # (B, T)

    @staticmethod
    def _sinusoidal_pe(length: int, dim: int, device: torch.device) -> torch.Tensor:
        pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe = torch.zeros(1, length, dim, device=device)
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        return pe


def enforce_monotone_lis(positions: list[float]) -> list[float]:
    """LIS-based monotonicity enforcement, inspired by Qwen3-ForcedAligner.

    Finds the longest increasing subsequence, then interpolates
    non-monotone positions between valid boundaries.
    """
    n = len(positions)
    if n <= 1:
        return positions

    # Find LIS indices
    dp = [1] * n
    parent = [-1] * n
    for i in range(1, n):
        for j in range(i):
            if positions[j] <= positions[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j

    # Reconstruct LIS
    best_end = max(range(n), key=lambda i: dp[i])
    lis_indices = set()
    idx = best_end
    while idx != -1:
        lis_indices.add(idx)
        idx = parent[idx]

    result = list(positions)

    # Fix non-LIS positions by interpolation
    sorted_lis = sorted(lis_indices)
    for i in range(n):
        if i in lis_indices:
            continue
        # Find left and right LIS boundaries
        left_val = 0.0
        right_val = positions[sorted_lis[-1]] if sorted_lis else float(n)
        left_idx = 0
        right_idx = n - 1
        for li in sorted_lis:
            if li < i:
                left_val = positions[li]
                left_idx = li
            elif li > i:
                right_val = positions[li]
                right_idx = li
                break
        # Linear interpolation
        if right_idx > left_idx:
            frac = (i - left_idx) / (right_idx - left_idx)
            result[i] = left_val + frac * (right_val - left_val)
        else:
            result[i] = left_val

    return result
