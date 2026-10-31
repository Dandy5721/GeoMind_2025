"""GeoMind: a geometric deep state-space model for brain dynamics.

End-to-end model (Fig. 2c). The pipeline is:

    X (B, L, N, N) SPD
      -> Geometric-adaptive attention (GaA, Eq. 11)      -> X', Omega
      -> Geometric SSM (Eqs. 5--6)                        -> Xi (B, L, N, N)
      -> matrix-log readout + linear classifier (Eq. 8)  -> logits

The model returns ``(logits, Omega)``; ``Omega`` is the interpretable geometric
attention map averaged for saliency visualization. The same architecture runs unchanged
for fMRI (N in the hundreds of regions) and EEG (N in the tens of channels), since every
SPD operation is defined independently of ``N`` (Sec. V-B).
"""

import torch
from torch import Tensor, nn

from .attention import GeometricAdaptiveAttention
from .conv import SPDConvFrontend
from .spd import spd_logm
from .ssm import GeometricSSM

__all__ = ["GeoMind"]


class GeoMind(nn.Module):
    """Full GeoMind classifier.

    Args:
        dim: matrix dimension ``N`` (brain regions or EEG channels).
        num_classes: number of output classes.
        time_points: sequence length ``L`` (used to size the attention convolution).
        kernel_size: kernel size of the geometric-adaptive attention.
        dt_init: initial ZOH step of the geometric SSM.
        use_attention: whether to apply GaA before the SSM (ablation: ``Geo-Mamba``).
        readout: ``"mean"`` averages the readout over time, ``"last"`` uses ``Xi_T``.
    """

    def __init__(
        self,
        dim: int,
        num_classes: int,
        time_points: int,
        kernel_size: int = 3,
        dt_init: float = 0.1,
        use_attention: bool = True,
        readout: str = "last",
        head: str = "linear",
        stein_iters: int = 8,
        dropout: float = 0.0,
        conv_channels: int = 0,
        conv_layers: int = 2,
    ):
        super().__init__()
        if readout not in ("mean", "last"):
            raise ValueError(f"readout must be 'mean' or 'last', got {readout!r}")
        self.dim = dim
        self.use_attention = use_attention
        self.readout = readout

        # Optional multi-channel SPD convolution feature front-end (Eqs. 4, 9-10).
        self.conv_frontend = (
            SPDConvFrontend(conv_channels, kernel_size, conv_layers) if conv_channels > 0 else None
        )
        if use_attention:
            self.attention = GeometricAdaptiveAttention(time_points, kernel_size)
        self.ssm = GeometricSSM(dim, dt_init=dt_init, stein_iters=stein_iters)

        # Eq. (8): logits from the matrix-log of the SPD readout. "linear" is the
        # paper-faithful single linear classifier; "mlp" adds one hidden layer for capacity.
        if head == "mlp":
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(dim * dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim, num_classes),
            )
        else:
            self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim * dim, num_classes))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor | None]:
        """Args: ``x`` of shape ``(B, L, N, N)``. Returns ``(logits, attention_map)``."""
        attention = None
        if self.conv_frontend is not None:
            x = self.conv_frontend(x)  # multi-channel SPD convolution features
        if self.use_attention:
            x, attention = self.attention(x)

        xi = self.ssm(x)  # (B, L, N, N)

        if self.readout == "mean":
            pooled = xi.mean(dim=1)
        else:
            pooled = xi[:, -1]

        # Map the SPD readout to the tangent space at the identity before the linear head.
        feat = spd_logm(pooled).reshape(pooled.shape[0], -1)
        logits = self.classifier(feat)
        return logits, attention
