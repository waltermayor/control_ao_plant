"""Per-mode autoregressive models of modal turbulence (notebook §6.3).

    d_k = Σ_{j=1..p} a_j ⊙ d_{k-j} + w_k

Coefficient arrays have shape (p, n_modes): row j holds the lag-(j+1) coefficient of every mode.
A "persistence" model is AR(1) with a = 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ARModel:
    coefs: np.ndarray  # (p, n)
    innov_var: np.ndarray  # (n,) variance of the one-step innovation (on training data)
    state_var: np.ndarray  # (n,) variance of the process (on training data)

    @property
    def order(self) -> int:
        return self.coefs.shape[0]

    @property
    def n(self) -> int:
        return self.coefs.shape[1]


def fit_ar(X: np.ndarray, p: int, cut: int) -> np.ndarray:
    """Least-squares AR(p) per mode on X[:cut] (X: (T, n))."""
    n = X.shape[1]
    coefs = np.zeros((p, n))
    for i in range(n):
        x = X[:cut, i]
        Phi = np.column_stack([x[p - j - 1 : len(x) - j - 1] for j in range(p)])  # lag j+1
        coefs[:, i] = np.linalg.lstsq(Phi, x[p:], rcond=None)[0]
    return coefs


def stabilize_ar(coefs: np.ndarray, rmax: float = 0.9995) -> np.ndarray:
    """Rescale characteristic roots so that all lie inside |z| <= rmax."""
    p, n = coefs.shape
    out = coefs.copy()
    for i in range(n):
        r = np.max(np.abs(np.roots(np.r_[1.0, -coefs[:, i]])))
        if r > rmax:
            out[:, i] = coefs[:, i] * (rmax / r) ** np.arange(1, p + 1)
    return out


def ar_innov_var(X: np.ndarray, coefs: np.ndarray, cut: int) -> np.ndarray:
    p = coefs.shape[0]
    V = X[:cut]
    pred = sum(coefs[j] * V[p - 1 - j : len(V) - 1 - j] for j in range(p))
    return np.var(V[p:] - pred, axis=0)


def identify_ar(X: np.ndarray, p: int, train_frac: float = 0.8, rmax: float = 0.9995) -> ARModel:
    cut = int(train_frac * len(X))
    coefs = stabilize_ar(fit_ar(X, p, cut), rmax)
    return ARModel(coefs=coefs, innov_var=ar_innov_var(X, coefs, cut), state_var=X[:cut].var(axis=0))


def persistence(n: int) -> np.ndarray:
    return np.ones((1, n))


def companion(coefs: np.ndarray) -> np.ndarray:
    """Block companion matrix F (p·n x p·n) of the stacked state [d_k, ..., d_{k-p+1}]."""
    p, n = coefs.shape
    F = np.zeros((p * n, p * n))
    for j in range(p):
        F[:n, j * n : (j + 1) * n] = np.diag(coefs[j])
    for j in range(1, p):
        F[j * n : (j + 1) * n, (j - 1) * n : j * n] = np.eye(n)
    return F


def predictor_weights(coefs: np.ndarray, theta: int) -> np.ndarray:
    """Weights (p, n) predicting d_{k-1+theta} from [d_{k-1}, ..., d_{k-p}] (first row of F^theta)."""
    p, n = coefs.shape
    G = np.zeros((p, n))
    for i in range(n):
        F = np.zeros((p, p))
        F[0, :] = coefs[:, i]
        if p > 1:
            F[1:, :-1] = np.eye(p - 1)
        G[:, i] = np.linalg.matrix_power(F, theta)[0, :]
    return G


def prediction_error_nm(X: np.ndarray, G: np.ndarray, theta: int, W: np.ndarray) -> float:
    """RMS wavefront error [nm] of the theta-step prediction over sequence X (T, n)."""
    p = G.shape[0]
    T = len(X)
    ks = np.arange(p, T - theta + 1)
    pred = sum(G[j] * X[ks - 1 - j] for j in range(p))
    err = X[ks - 1 + theta] - pred
    return float(np.sqrt(np.mean(np.einsum("ti,ij,tj->t", err, W, err))) * 1e9)


def skill_onestep(X: np.ndarray, coefs: np.ndarray, cut: int) -> np.ndarray:
    """Per-mode one-step gain over persistence on X[cut:]: 1 - MSE_AR / MSE_persistence."""
    p = coefs.shape[0]
    V = X[cut:]
    pred = sum(coefs[j] * V[p - 1 - j : len(V) - 1 - j] for j in range(p))
    pers = V[p - 1 : len(V) - 1]
    return 1 - np.sum((V[p:] - pred) ** 2, axis=0) / np.sum((V[p:] - pers) ** 2, axis=0)
