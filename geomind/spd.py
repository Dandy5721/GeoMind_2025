"""Riemannian operations on the manifold of Symmetric Positive Definite (SPD) matrices.

This module implements the geometric algebra used throughout GeoMind (Sec. III-B of
the paper): the Stein (Jensen--Bregman LogDet) distance, the conjugation map
``Psi_g(X) = g X g^T`` with ``g`` in the orthogonal group ``O(N)`` (Eq. 2), and the
geodesic barycenter operator ``barycenter`` (the infix ``+|+_alpha`` in Eq. 3).

All operations are batched over leading dimensions and preserve the SPD property, so
every intermediate state of the model stays on ``Sym_N^+``.
"""

import torch
from torch import Tensor

__all__ = [
    "sym",
    "spd_logm",
    "spd_expm",
    "stein_distance",
    "orthogonal",
    "conjugation",
    "geodesic_barycenter",
]

_EPS = 1e-5


def sym(x: Tensor) -> Tensor:
    """Symmetrize the last two dimensions: ``(X + X^T) / 2``."""
    return 0.5 * (x + x.transpose(-1, -2))


def _loewner(s: Tensor, fval: Tensor, fprime: Tensor, tol: float = 1e-6) -> Tensor:
    """Loewner / Daleckii--Krein matrix of divided differences for a spectral function.

    ``L_ij = (f(s_i) - f(s_j)) / (s_i - s_j)`` with the limit ``f'(s_i)`` when eigenvalues
    coincide. Using this in the backward pass keeps the gradient of the matrix log/exp
    finite even when the input has degenerate eigenvalues (e.g. the identity state
    ``Sigma_0 = I``), which a naive ``eigh`` backward cannot handle.
    """
    ds = s.unsqueeze(-1) - s.unsqueeze(-2)
    df = fval.unsqueeze(-1) - fval.unsqueeze(-2)
    near = ds.abs() < tol
    safe = torch.where(near, torch.ones_like(ds), ds)
    loewner = df / safe
    fprime_avg = 0.5 * (fprime.unsqueeze(-1) + fprime.unsqueeze(-2))
    return torch.where(near, fprime_avg, loewner)


class _SymSpectral(torch.autograd.Function):
    """Apply a scalar function to the eigenvalues of a symmetric matrix, with stable grad."""

    @staticmethod
    def forward(ctx, x, kind, eps):
        x = sym(x)
        # A small ridge plus a float64 eigendecomposition keeps eigh stable on rank-deficient
        # SPD inputs (e.g. windowed FC where the window length is smaller than N). CUDA's
        # cusolver can still fail to converge on such matrices, so fall back to CPU LAPACK.
        dtype, device = x.dtype, x.device
        xr = x.double() + 1e-5 * torch.eye(x.shape[-1], device=device, dtype=torch.float64)
        try:
            s, u = torch.linalg.eigh(xr)
        except torch._C._LinAlgError:
            s, u = torch.linalg.eigh(xr.cpu())
            s, u = s.to(device), u.to(device)
        s, u = s.to(dtype), u.to(dtype)
        if kind == "log":
            s_clamped = s.clamp(min=eps)
            fval = s_clamped.log()
            fprime = 1.0 / s_clamped
        else:  # exp
            fval = s.exp()
            fprime = fval
        out = u @ torch.diag_embed(fval) @ u.transpose(-1, -2)
        ctx.save_for_backward(s, u, fval, fprime)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        s, u, fval, fprime = ctx.saved_tensors
        loewner = _loewner(s, fval, fprime)
        g = sym(grad_out)
        g_eig = u.transpose(-1, -2) @ g @ u
        grad_x = u @ (loewner * g_eig) @ u.transpose(-1, -2)
        return sym(grad_x), None, None


def spd_logm(x: Tensor, eps: float = _EPS) -> Tensor:
    """Matrix logarithm of an SPD matrix with a numerically stable backward.

    Eigenvalues are clamped to ``eps`` so the operation is well defined even when
    numerical error makes an input slightly indefinite.
    """
    return _SymSpectral.apply(x, "log", eps)


def spd_expm(x: Tensor) -> Tensor:
    """Matrix exponential of a symmetric matrix; maps the tangent space back to ``Sym_N^+``."""
    return _SymSpectral.apply(x, "exp", _EPS)


def spd_project(x: Tensor, epsilon: float = 1e-4, floor: float | None = None) -> Tensor:
    """Project a symmetric matrix back onto ``Sym_N^+`` and (optionally) bound its conditioning.

    Used to keep the output of the multi-channel SPD convolution positive definite (a
    summed multi-channel convolution is not SPD in general). With ``floor`` set, the
    eigenvalues are rescaled to ``(0, 1]`` and clamped to ``[floor, 1]``, bounding the
    condition number to ``1 / floor`` -- this conditions the convolution features so they
    can be fused into the geometric SSM without destabilizing it. A numerically stable
    eigendecomposition with a CPU fallback handles ill-conditioned inputs.
    """
    x = sym(x)
    try:
        s, u = torch.linalg.eigh(x.double())
    except torch._C._LinAlgError:
        s, u = torch.linalg.eigh(x.double().cpu())
        s, u = s.to(x.device), u.to(x.device)
    s = s.clamp(min=epsilon)
    if floor is not None:
        s = (s / s.max(dim=-1, keepdim=True).values).clamp(min=floor)
    out = u @ torch.diag_embed(s) @ u.transpose(-2, -1)
    return out.to(x.dtype)


def stein_distance(x: Tensor, y: Tensor, eps: float = 1e-6) -> Tensor:
    """Squared Stein / Jensen--Bregman LogDet distance between SPD matrices.

    ``d^2(X, Y) = logdet((X + Y) / 2) - 1/2 * (logdet(X) + logdet(Y))``.

    Uses :func:`torch.slogdet` for numerical stability. Returns a tensor with the
    leading batch shape of the inputs.
    """
    mid = sym(0.5 * (x + y))
    jitter_mid = torch.eye(mid.shape[-1], device=mid.device, dtype=mid.dtype) * eps
    _, ld_mid = torch.linalg.slogdet(mid + jitter_mid)
    _, ld_x = torch.linalg.slogdet(sym(x) + jitter_mid)
    _, ld_y = torch.linalg.slogdet(sym(y) + jitter_mid)
    return (ld_mid - 0.5 * (ld_x + ld_y)).clamp(min=0.0)


def orthogonal(generator: Tensor, scale: float = 1.0) -> Tensor:
    """Map an unconstrained square parameter to an orthogonal matrix in ``O(N)``.

    A skew-symmetric matrix ``W - W^T`` is formed from the generator and exponentiated;
    ``expm`` of a skew-symmetric matrix is orthogonal. Restricting the conjugation map to
    ``O(N)`` keeps it Stein-isometric and SPD-preserving (Eq. 2).
    """
    skew = generator - generator.transpose(-1, -2)
    return torch.matrix_exp(scale * skew)


def conjugation(g: Tensor, x: Tensor) -> Tensor:
    """Conjugation map ``Psi_g(X) = g X g^T`` (Eq. 2).

    ``g`` has shape ``(N, N)`` and ``x`` has shape ``(..., N, N)``. With ``g`` orthogonal
    this is an isometry of the Stein metric and preserves positive definiteness.
    """
    return g @ x @ g.transpose(-1, -2)


def geodesic_barycenter(
    x: Tensor, y: Tensor, alpha: Tensor, iters: int = 8, eps: float = 1e-6
) -> Tensor:
    """Weighted Stein barycenter of two SPD matrices (the ``+|+_alpha`` operator, Eq. 3).

    Implements the intrinsic weighted geodesic combination of Eq. (3) under the Stein /
    Jensen--Bregman LogDet metric, i.e. the minimizer

        B* = argmin_{B in Sym_N^+}  alpha * d_S^2(X, B) + (1 - alpha) * d_S^2(Y, B).

    Setting the gradient ``alpha (X+B)^{-1} + (1-alpha) (Y+B)^{-1} - 1/2 B^{-1} = 0`` to
    zero yields the fixed-point iteration

        B <- [ 2 ( alpha (X+B)^{-1} + (1-alpha) (Y+B)^{-1} ) ]^{-1},

    which stays on ``Sym_N^+``, converges from the midpoint initialization, and recovers
    ``X`` at ``alpha = 1`` and ``Y`` at ``alpha = 0``. ``alpha`` weights the first argument,
    matching the paper's abbreviation ``X +|+_alpha Y := (alpha . X) +|+ ((1-alpha) . Y)``.
    Every operation is a batched inverse, so the barycenter is GPU-friendly and fully
    differentiable through the unrolled iteration.
    """
    x = sym(x)
    y = sym(y)
    eye = eps * torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)
    b = 0.5 * (x + y)  # midpoint initialization (already SPD)
    for _ in range(iters):
        inv_x = torch.linalg.inv(x + b + eye)
        inv_y = torch.linalg.inv(y + b + eye)
        m = 2.0 * (alpha * inv_x + (1.0 - alpha) * inv_y)
        b = sym(torch.linalg.inv(m + eye))
    return b
