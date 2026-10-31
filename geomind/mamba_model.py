"""GeoMind2: a geometric Mamba — SPD tangent features with a selective-SSM (Mamba) backbone.

GeoMind's original geometric SSM (Eqs. 5--6) uses *static* conjugators and mixing weights,
so its dynamics are not input-dependent. Mamba's key advance is *selectivity*: the SSM
parameters are functions of the input, letting the model selectively propagate or forget
information over time. GeoMind2 keeps GeoMind's geometric feature representation -- every
windowed FC matrix is mapped to the tangent space at the identity (matrix log + upper-
triangular vectorization), optionally modulated by the geometric-adaptive attention --
and replaces the weak barycenter recurrence with a stack of selective Mamba blocks.

This is an alternative backbone to compare against the on-manifold recurrence of
:class:`geomind.GeoMind`.
"""

import torch
from torch import Tensor, nn

from .attention import GeometricAdaptiveAttention
from .spd import spd_logm

__all__ = ["GeoMind2", "spd_tangent_vech"]


def vech(x: Tensor) -> Tensor:
    """Upper-triangular vectorization of a symmetric matrix.

    ``x`` has shape ``(..., N, N)``; returns ``(..., N(N+1)/2)`` with off-diagonal entries
    scaled by ``sqrt(2)`` so the map is an isometry of the Frobenius norm.
    """
    n = x.shape[-1]
    iu, ju = torch.triu_indices(n, n, device=x.device)
    vec = x[..., iu, ju]
    scale = torch.where(iu == ju, 1.0, 2.0 ** 0.5).to(vec.dtype)
    return vec * scale


def spd_tangent_vech(x: Tensor) -> Tensor:
    """Map SPD matrices to tangent vectors: ``vech(logm(X))`` (geometric front-end)."""
    return vech(spd_logm(x))


class GeoMind2(nn.Module):
    """Geometric Mamba classifier.

    Args:
        dim: matrix dimension ``N``.
        num_classes: number of output classes.
        time_points: sequence length ``L`` (sizes the geometric attention).
        d_model: Mamba hidden width.
        n_layers: number of stacked Mamba blocks.
        d_state: SSM state dimension of each Mamba block.
        d_conv: short convolution width of each Mamba block.
        expand: Mamba block expansion factor.
        use_attention: apply geometric-adaptive attention before vectorization.
        readout: ``"mean"`` pools over time, ``"last"`` uses the final step.
        dropout: dropout on the pooled representation.
    """

    def __init__(
        self,
        dim: int,
        num_classes: int,
        time_points: int,
        d_model: int = 128,
        n_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_attention: bool = True,
        readout: str = "mean",
        dropout: float = 0.1,
        geometric: bool = True,
    ):
        super().__init__()
        # Imported here rather than at module load: mamba_ssm pulls in transformers and
        # hundreds of files, which GeoMind (geometric SSM) runs never need.
        try:
            from mamba_ssm import Mamba
        except ImportError as err:
            raise ImportError("mamba_ssm is required for GeoMind2 (pip install mamba-ssm).") from err
        if readout not in ("mean", "last"):
            raise ValueError(f"readout must be 'mean' or 'last', got {readout!r}")
        self.use_attention = use_attention
        self.readout = readout
        # geometric=True -> tangent-space (logm) features; False -> raw FC (vanilla Mamba).
        self.geometric = geometric

        if use_attention:
            self.attention = GeometricAdaptiveAttention(time_points, 3)
        feat_dim = dim * (dim + 1) // 2
        self.proj = nn.Linear(feat_dim, d_model)
        self.in_norm = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList(
            [Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
             for _ in range(n_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, num_classes)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor | None]:
        """Args: ``x`` of shape ``(B, L, N, N)``. Returns ``(logits, attention_map)``."""
        attention = None
        if self.use_attention:
            x, attention = self.attention(x)

        v = spd_tangent_vech(x) if self.geometric else vech(x)  # (B, L, feat_dim)
        h = self.in_norm(self.proj(v))
        for block, norm in zip(self.blocks, self.norms):
            h = h + block(norm(h))  # residual selective-SSM block

        pooled = h.mean(dim=1) if self.readout == "mean" else h[:, -1]
        return self.head(pooled), attention
