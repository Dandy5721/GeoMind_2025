"""Transformer baseline on SPD tangent features (same protocol as GeoMind).

Mirrors :class:`geomind.GeoMind2` but replaces the Mamba selective-SSM backbone with a
standard Transformer encoder over the same tangent-space (matrix-log) FC features.
"""

import torch
from torch import Tensor, nn

from .mamba_model import vech
from .spd import spd_logm

__all__ = ["TransformerClassifier"]


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        dim: int,
        num_classes: int,
        time_points: int,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        readout: str = "mean",
        dropout: float = 0.1,
    ):
        super().__init__()
        if readout not in ("mean", "last"):
            raise ValueError(f"readout must be 'mean' or 'last', got {readout!r}")
        self.readout = readout
        feat_dim = dim * (dim + 1) // 2
        self.proj = nn.Linear(feat_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, time_points, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, num_classes)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, None]:
        v = vech(spd_logm(x))  # (B, L, feat_dim) tangent features
        h = self.proj(v) + self.pos[:, : x.shape[1]]
        h = self.encoder(h)
        pooled = h.mean(dim=1) if self.readout == "mean" else h[:, -1]
        return self.head(pooled), None
