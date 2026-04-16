"""Small dedicated transcript-conditioned aligner on frozen Gemma audio features.

Architecture:
  1. Frozen Gemma text embeddings projected to aligner hidden dim
  2. Cross-attention transformer: transcript queries attend to audio keys
  3. Output: dot-product scores over audio positions per transcript token

Supervision: Qwen teacher timestamps (training only).
At inference, predicts audio positions independently of Qwen.

Input contract:
  - audio_features: (num_audio_tokens, audio_dim)  — frozen Gemma audio tower output
  - transcript_ids: (num_transcript_tokens,)        — tokenized transcript

Output contract:
  - Per transcript token: predicted audio position (index into audio_features)
  - Monotonicity is enforced in post-processing, not in the model
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
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
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
        """Predict expected audio position per transcript token (soft argmax)."""
        logits = self.forward(text_embeds, audio_features)  # (B, T, A)
        probs = F.softmax(logits, dim=-1)
        positions = torch.arange(logits.shape[2], device=logits.device, dtype=logits.dtype)
        return (probs * positions.unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # (B, T)

    @staticmethod
    def _sinusoidal_pe(length: int, dim: int, device: torch.device) -> torch.Tensor:
        pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe = torch.zeros(1, length, dim, device=device)
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        return pe
