"""Geometric deep state-space model on the SPD manifold (Sec. IV-A, Eqs. 5--7).

This is the core of GeoMind. A windowed functional-connectivity sequence
``X = {X_t}`` (each ``X_t`` an SPD matrix) is treated as a discrete-time dynamical
system whose latent state ``Sigma_t`` and readout ``Xi_t`` also live on ``Sym_N^+``.

Transition map (state update, Eq. 5):
    Sigma_t = Psi_{A_d}(Sigma_{t-1})  +|+_alpha  Psi_{B_d}(X_t),   Sigma_0 = I_N

Readout map (observation, Eq. 6):
    Xi_t = Psi_C(Sigma_t)  +|+_beta  Psi_D(X_t)

Here ``Psi`` is the conjugation map (Eq. 2) restricted to ``O(N)`` so every step stays
on the manifold, and ``+|+`` is the weighted geodesic barycenter (Eq. 3). The discrete
state/input gains ``A_d, B_d`` are obtained from continuous generators by a zero-order
hold (ZOH) at step ``Delta`` (Eq. 7); we keep them in ``O(N)`` so the recursion is
isometry-preserving.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .spd import conjugation, geodesic_barycenter, orthogonal

__all__ = ["GeometricSSM"]


class GeometricSSM(nn.Module):
    """On-manifold state-space recurrence (Eqs. 5--6).

    Args:
        dim: matrix dimension ``N`` of the SPD inputs (brain regions / EEG channels).
        dt_init: initial value of the (learnable, positive) ZOH step ``Delta``.
    """

    def __init__(self, dim: int, dt_init: float = 0.1, stein_iters: int = 8):
        super().__init__()
        self.dim = dim
        self.stein_iters = stein_iters

        # Unconstrained generators; mapped to O(N) via the skew-symmetric exponential.
        self.A_gen = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.B_gen = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.C_gen = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.D_gen = nn.Parameter(torch.randn(dim, dim) * 0.01)

        # Learnable ZOH step Delta (kept positive through softplus).
        self.log_dt = nn.Parameter(torch.log(torch.expm1(torch.tensor(float(dt_init)))))

        # Learnable geodesic mixing weights alpha (transition) and beta (readout) in [0, 1].
        self.alpha_logit = nn.Parameter(torch.zeros(()))
        self.beta_logit = nn.Parameter(torch.zeros(()))

    def _discretize(self) -> tuple[Tensor, Tensor]:
        """Zero-order-hold discretization of the state/input gains (Eq. 7).

        ``A_d = expm(Delta * A)`` with ``A`` skew-symmetric is exactly the ZOH state gain
        of Eq. (7) and is orthogonal. For the input gain we keep ``B_d`` in ``O(N)`` as
        well (a manifold-preserving adaptation of the Eq. (7) ZOH term), so that the
        conjugation map ``Psi_{B_d}`` remains Stein-isometric and the recurrence never
        leaves ``Sym_N^+``.
        """
        dt = F.softplus(self.log_dt)
        a_d = orthogonal(self.A_gen, scale=dt)
        b_d = orthogonal(self.B_gen, scale=dt)
        return a_d, b_d

    def forward(self, x: Tensor) -> Tensor:
        """Run the recurrence over time.

        Args:
            x: SPD sequence of shape ``(B, L, N, N)``.

        Returns:
            Readout sequence ``Xi`` of shape ``(B, L, N, N)``, all SPD.
        """
        batch, length, n, _ = x.shape
        a_d, b_d = self._discretize()
        c_m = orthogonal(self.C_gen)
        d_m = orthogonal(self.D_gen)
        alpha = torch.sigmoid(self.alpha_logit)
        beta = torch.sigmoid(self.beta_logit)

        # Sigma_0 = I_N (Eq. 5).
        sigma = torch.eye(n, device=x.device, dtype=x.dtype).expand(batch, n, n).contiguous()

        # State recurrence must be sequential (Eq. 5); collect the states.
        states = []
        for t in range(length):
            # Transition (Eq. 5): alpha weights the state term Psi_{A_d}(Sigma_{t-1}).
            sigma = geodesic_barycenter(conjugation(a_d, sigma), conjugation(b_d, x[:, t]),
                                        alpha, iters=self.stein_iters)
            states.append(sigma)
        sigma_seq = torch.stack(states, dim=1)  # (B, L, N, N)

        # Readout (Eq. 6) does not depend on other time steps, so compute all L steps in one
        # batched barycenter call instead of L sequential calls (identical result, much faster).
        return geodesic_barycenter(conjugation(c_m, sigma_seq), conjugation(d_m, x),
                                   beta, iters=self.stein_iters)
