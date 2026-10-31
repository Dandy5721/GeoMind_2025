"""Minimal smoke test: build GeoMind, run a forward/backward pass, check the SPD math.

Run with ``python demo.py``. This requires no external data and verifies that every
manifold operation, the geometric SSM recurrence, and the attention map are wired up
correctly.
"""

import torch

from geomind import GeoMind, geodesic_barycenter, spd_expm, spd_logm, stein_distance


def check_spd_operations():
    torch.manual_seed(0)
    n = 8
    a = torch.randn(4, n, n)
    x = a @ a.transpose(-1, -2) + n * torch.eye(n)  # batch of SPD matrices

    # log/exp are inverse on the SPD cone.
    recon = spd_expm(spd_logm(x))
    assert torch.allclose(recon, x, atol=1e-3), "expm(logm(X)) != X"

    # Stein distance is zero between identical matrices and positive otherwise.
    assert torch.allclose(stein_distance(x, x), torch.zeros(4), atol=1e-4)

    # Stein barycenter endpoints: alpha=1 -> X (first arg), alpha=0 -> Y (second arg).
    b = torch.randn(4, n, n)
    y = b @ b.transpose(-1, -2) + n * torch.eye(n)
    at1 = geodesic_barycenter(x, y, torch.tensor(1.0), iters=40)
    at0 = geodesic_barycenter(x, y, torch.tensor(0.0), iters=40)
    assert torch.allclose(at1, x, atol=1e-2), (at1 - x).abs().max()
    assert torch.allclose(at0, y, atol=1e-2), (at0 - y).abs().max()
    # The barycenter is symmetric positive definite.
    assert torch.linalg.eigvalsh(geodesic_barycenter(x, y, torch.tensor(0.5))).min() > 0
    print("[ok] SPD operations (log/exp inverse, Stein distance, Stein barycenter)")


def check_model_forward_backward():
    torch.manual_seed(0)
    batch, length, dim, num_classes = 3, 6, 10, 2
    a = torch.randn(batch, length, dim, dim)
    x = a @ a.transpose(-1, -2) + dim * torch.eye(dim)  # SPD sequence

    model = GeoMind(dim=dim, num_classes=num_classes, time_points=length, kernel_size=3)
    logits, attention = model(x)
    assert logits.shape == (batch, num_classes), logits.shape
    assert attention.shape == (batch, length, dim, dim), attention.shape

    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0]))
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0, "no gradients flowed"
    print(f"[ok] GeoMind forward/backward (loss={loss.item():.4f}, grad_norm={grad_norm:.4f})")


if __name__ == "__main__":
    check_spd_operations()
    check_model_forward_backward()
    print("\nAll checks passed.")
