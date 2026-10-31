"""GeoMind: Dancing with Geometric Brain Rhythm on Symmetric Positive Definite Manifolds.

A geometric deep state-space model that tracks the trajectory of functional-connectivity
dynamics on the Riemannian manifold of SPD matrices.
"""

from .attention import GeometricAdaptiveAttention
from .conv import SPDConv
from .model import GeoMind

try:
    from .mamba_model import GeoMind2
except ImportError:  # mamba_ssm not installed
    GeoMind2 = None
from .transformer_model import TransformerClassifier
from .spd import (
    conjugation,
    geodesic_barycenter,
    orthogonal,
    spd_expm,
    spd_logm,
    stein_distance,
    sym,
)
from .ssm import GeometricSSM

__all__ = [
    "GeoMind",
    "GeoMind2",
    "GeometricSSM",
    "GeometricAdaptiveAttention",
    "SPDConv",
    "sym",
    "spd_logm",
    "spd_expm",
    "stein_distance",
    "orthogonal",
    "conjugation",
    "geodesic_barycenter",
]

__version__ = "1.0.0"
