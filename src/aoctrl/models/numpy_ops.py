"""NumPy inference for learned latent models.

Training happens in PyTorch (`aoctrl.models.torch_models`) but the closed loop runs in NumPy on the
CPU, one sample at a time, where it is faster and dependency-free. Parameters travel between the
two as plain `dict[str, np.ndarray]` with the same names as the notebook.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------------- Exp 3
def mlp_residual(P: dict, x: np.ndarray, tag: str) -> np.ndarray:
    """x + tanh(x W1 + b1) W2 + b2   (tag 'e' = encoder, 'd' = decoder)."""
    h = np.tanh(x @ P["W1" + tag] + P["b1" + tag])
    return x + h @ P["W2" + tag] + P["b2" + tag]


def latent_arx_step(P: dict, z1, z2, c1, c2, c0):
    return z1 @ P["A1"].T + z2 @ P["A2"].T + c1 @ P["Ac1"].T + c2 @ P["Ac2"].T + c0 @ P["B0"].T


def latent_arx_rollout(P: dict, p: np.ndarray, c: np.ndarray, identity: bool = False) -> np.ndarray:
    """Multi-step prediction (t = 2..T) of the decoder from (E(p0), E(p1)) with the real commands."""
    f = (lambda x: x) if identity else (lambda x: mlp_residual(P, x, "e"))
    g = (lambda x: x) if identity else (lambda x: mlp_residual(P, x, "d"))
    zh = [f(p[:, 0]), f(p[:, 1])]
    for t in range(2, p.shape[1]):
        zh.append(latent_arx_step(P, zh[t - 1], zh[t - 2], c[:, t - 1], c[:, t - 2], c[:, t]))
    return g(np.stack(zh[2:], axis=1))


def latent_arx_decoder_jacobian(P: dict) -> np.ndarray:
    """Jacobian of the decoder at z = 0 (row convention)."""
    n = P["A1"].shape[0]
    return np.eye(n) + P["W1d"] @ np.diag(1.0 - np.tanh(P["b1d"]) ** 2) @ P["W2d"]


def stabilize_pair(A1, A2, rmax: float = 0.9995):
    n = A1.shape[0]
    C = np.block([[A1, A2], [np.eye(n), np.zeros((n, n))]])
    rho = np.max(np.abs(np.linalg.eigvals(C)))
    if rho > rmax:
        s = rmax / rho
        return A1 * s, A2 * s * s
    return A1, A2


# ---------------------------------------------------------------------------------- Exp 4
def koopman_obs(p: np.ndarray, c: np.ndarray) -> np.ndarray:
    """O_t = [p_t, p_{t-1}, c_t, c_{t-1}], t = 1..T:   (B, T+1, n) -> (B, T, 4n)."""
    return np.concatenate([p[:, 1:], p[:, :-1], c[:, 1:], c[:, :-1]], axis=-1)


def koopman_encode(P: dict, O: np.ndarray, linear: bool = False) -> np.ndarray:
    return O @ P["Pe"] if linear else O @ P["Pe"] + np.tanh(O @ P["W1e"]) @ P["W2e"]


def koopman_decode(P: dict, Z: np.ndarray, linear: bool = False) -> np.ndarray:
    return Z @ P["Pd"] if linear else Z @ P["Pd"] + np.tanh(Z @ P["W1d"]) @ P["W2d"]


def koopman_rollout(P: dict, p: np.ndarray, c: np.ndarray, linear: bool = False) -> np.ndarray:
    z = koopman_encode(P, koopman_obs(p, c)[:, 0], linear)
    out = []
    for t in range(2, p.shape[1]):
        z = z @ P["A"].T + c[:, t] @ P["B"].T
        out.append(koopman_decode(P, z, linear))
    return np.stack(out, axis=1)


def koopman_decoder_jacobian(P: dict, linear: bool = False) -> np.ndarray:
    """Jacobian of the decoder at z = 0 (no biases, tanh'(0) = 1)."""
    return P["Pd"] if linear else P["Pd"] + P["W1d"] @ P["W2d"]


def stabilize_A(A: np.ndarray, rmax: float = 0.9995):
    """Clip only eigenvalues with |λ| > rmax (keeping their phase); fall back to scaling A."""
    lam, V = np.linalg.eig(A)
    rho = float(np.max(np.abs(lam)))
    if rho <= rmax:
        return A, rho
    lam_c = np.where(np.abs(lam) > rmax, rmax * lam / np.abs(lam), lam)
    try:
        As = (V * lam_c) @ np.linalg.inv(V)
        ok = np.max(np.abs(As.imag)) < 1e-6 * max(np.max(np.abs(As.real)), 1e-12) and np.linalg.norm(
            As.real - A
        ) < 0.3 * np.linalg.norm(A)
        if ok and np.max(np.abs(np.linalg.eigvals(As.real))) <= rmax * (1 + 1e-6):
            return As.real, rho
    except np.linalg.LinAlgError:
        pass
    return A * (rmax / rho), rho
