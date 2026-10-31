"""SPD-preserving convolution on the manifold (Sec. IV-A, Eq. 4).

The convolution kernel is constrained to be SPD by construction. Following Eq. (4),
the kernel is built as ``H = Z^T Z + eps * I`` so it is dominantly diagonal and SPD;
we therefore only learn the free parameter ``Z`` instead of constraining ``H`` directly.
A standard 2D convolution with such a kernel keeps the spatial structure of the SPD
feature maps intact.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .spd import geodesic_barycenter, spd_project, sym

__all__ = ["SPDConv", "SPDConvFrontend"]


class SPDConv(nn.Module):
    """Convolution whose kernel is guaranteed to be SPD (Eq. 4).

    Args:
        in_channels: number of input channels (e.g. the sequence length / time points).
        out_channels: number of output channels.
        kernel_size: size of the (square) convolution kernel.
        stride: convolution stride.
        epsilon: ridge added to the kernel Gram matrix to guarantee positive definiteness.
        padding: optional SPD-aware padding width; pads with a small value on the diagonal.
        padding_num: diagonal value used when padding so the padded matrix stays SPD.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        epsilon: float = 1e-3,
        padding: int | None = None,
        padding_num: float = 1e-4,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.register_buffer("epsilon", torch.tensor(epsilon))
        self.v = nn.Parameter(torch.empty((out_channels, in_channels) + kernel_size))
        nn.init.xavier_uniform_(self.v)
        self.padding = padding
        self.padding_num = padding_num
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() < 4:
            x = x.unsqueeze(1)

        if self.padding is not None:
            new_shape = list(x.shape)
            new_shape[-2] += 2 * self.padding
            new_shape[-1] += 2 * self.padding
            padded = torch.zeros(new_shape, device=x.device, dtype=x.dtype)
            row, col = np.diag_indices(new_shape[-1])
            padded[..., row, col] = self.padding_num
            padded[
                ...,
                self.padding : padded.shape[-2] - self.padding,
                self.padding : padded.shape[-1] - self.padding,
            ] = x
            x = padded

        # H = Z^T Z + eps * I  ->  SPD kernel (Eq. 4).
        weight = self.v.transpose(-2, -1) @ self.v
        weight = weight + self.epsilon * torch.eye(self.v.shape[-1], device=x.device)
        return F.conv2d(x.float(), weight=weight, stride=self.stride)


class SPDConvFrontend(nn.Module):
    """Multi-channel SPD convolution feature extractor (Eqs. 4, 9--10).

    Each windowed FC matrix is processed by a stack of multi-channel SPD-preserving
    convolutions (kept at dimension ``N`` via SPD-aware padding), recombined to a single
    channel, projected back onto ``Sym_N^+``, and geodesically blended with the original
    matrix through a learnable weight. The residual blend means the front-end can only add
    discriminative convolutional structure on top of the raw connectivity, never discard
    it -- it reduces to the identity when the blend weight is zero.

    Args:
        channels: number of SPD convolution feature channels.
        kernel_size: spatial kernel size ``theta`` of the SPD convolution.
        n_layers: number of stacked SPD convolution layers.
    """

    def __init__(self, channels: int = 4, kernel_size: int = 3, n_layers: int = 2):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.convs = nn.ModuleList(
            [SPDConv(1 if i == 0 else channels, channels, kernel_size, padding=pad)
             for i in range(n_layers)]
        )
        self.combine = SPDConv(channels, 1, kernel_size, padding=pad)
        self.blend_logit = nn.Parameter(torch.tensor(-2.0))  # start near identity

    def forward(self, x: Tensor) -> Tensor:  # (B, L, N, N) -> (B, L, N, N)
        b, l, n, _ = x.shape
        h = x.reshape(b * l, 1, n, n)
        for conv in self.convs:
            h = sym(conv(h))
        # Project to SPD and bound the condition number (<= 1/floor) so the convolution
        # features are well-conditioned enough to fuse into the geometric SSM.
        feat = spd_project(sym(self.combine(h)).squeeze(1), floor=1e-2)  # (B*L, N, N)
        feat = feat.reshape(b, l, n, n)
        # Residual geodesic blend with the raw FC (learnable weight in [0, 1]).
        blend = torch.sigmoid(self.blend_logit)
        return geodesic_barycenter(x, feat, blend)

