"""Geometric-adaptive attention (GaA) on the SPD manifold (Sec. IV-B, Eq. 11).

GaA weights connectivity entries by their importance while keeping both the input and
the gate on the SPD manifold. Given an SPD feature map ``X``, a SPD-preserving
convolution produces a gating field which is exponentiated and normalized to ``[0, 1]``,

    Omega(X) = expm([X (*) G]) / max_ij (expm([X (*) G])_ij),                  (Eq. 11)

and the features are modulated through the Hadamard product ``Omega(X) o X``. The gate
``Omega`` is returned alongside the modulated features and serves as an interpretable
saliency map over connections (Figs. 3 and 5 in the paper).
"""

import torch
from torch import Tensor, nn

from .conv import SPDConv
from .spd import sym

__all__ = ["GeometricAdaptiveAttention"]


class GeometricAdaptiveAttention(nn.Module):
    """Geometric-adaptive attention module (Eq. 11).

    Args:
        in_channels: number of input channels (sequence length / time points).
        kernel_size: kernel size of the SPD-preserving gating convolution.
    """

    def __init__(self, in_channels: int, kernel_size: int):
        super().__init__()
        self.gate_conv = SPDConv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(Omega o X, Omega)`` where ``Omega`` is the geometric attention map (Eq. 11).

        The SPD-preserving convolution produces the gating field ``X (*) G``; its matrix
        exponential (the paper's ``expm`` acts on symmetric matrices) is normalized by its
        largest entry to give a gate ``Omega in [0, 1]^{N x N}``. Features are modulated by
        the Hadamard product ``Omega o X``, which remains SPD by the Schur product theorem.
        """
        gate = sym(self.gate_conv(x))
        if not torch.isfinite(gate).all():
            raise FloatingPointError(
                "GaA gate contains NaN/Inf; check FC input scaling and optimizer stability")
        # Stable matrix exponential: shift by the largest eigenvalue (log-sum-exp trick) so
        # the gate cannot overflow even for large/global convolution kernels, where the
        # convolution sums over many taps. expm(gate - lambda_max * I) has eigenvalues in
        # (0, 1]; the final max-entry normalization (Eq. 11) leaves Omega in [0, 1].
        # lambda_max is only a scalar shift: it cancels exactly in the subsequent max-entry
        # normalization, so no gradient through this eigendecomposition is needed. CUDA's
        # single-precision symmetric eigensolver occasionally fails on EEG covariance gates
        # with repeated/ill-conditioned eigenvalues. Compute it in float64 and fall back to
        # CPU LAPACK, matching the robust spectral operations in spd.py.
        with torch.no_grad():
            gate64 = gate.detach().double()
            try:
                lambda_max = torch.linalg.eigvalsh(gate64)[..., -1]
            except torch._C._LinAlgError:
                lambda_max = torch.linalg.eigvalsh(gate64.cpu())[..., -1].to(gate.device)
            lambda_max = lambda_max.to(gate.dtype)[..., None, None]
        eye = torch.eye(gate.shape[-1], device=gate.device, dtype=gate.dtype)
        expm_gate = torch.matrix_exp(gate - lambda_max * eye)
        omega = expm_gate / expm_gate.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-12)
        modulated = omega * x
        return modulated, omega
