"""Datasets, normalisation and truth-free initialisations for the latent models (notebook §8.3, §9.3).

Training uses only (y, u): pv = Mrec · y and the *applied* modal command c. The true residual `s`
and turbulence `d` are kept in the runs only to *evaluate* the latent afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WindowDataset:
    train_p: np.ndarray  # (N, T+1, n)
    train_c: np.ndarray
    val_p: np.ndarray
    val_c: np.ndarray
    sigma: np.ndarray  # (n,) per-mode scale

    def describe(self) -> str:
        return f"train {self.train_p.shape} | val {self.val_p.shape}"


def normalization_sigma(runs: list[dict], sign: float) -> np.ndarray:
    """Per-mode scale from pv − c (pseudo open loop), without truth."""
    return np.concatenate([q["pv"] - sign * q["c"] for q in runs]).std(axis=0) + 1e-30


def latent_metrics(sigma: np.ndarray, W_phys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Wavefront metric in normalised units (WN) and a well-conditioned version for the DARE (WC)."""
    n = len(sigma)
    WN = sigma[:, None] * W_phys * sigma[None, :]
    WN = WN / np.trace(WN) * n
    WC = 0.9 * WN + 0.1 * np.eye(n)
    return WN, WC


def make_windows(runs: list[dict], sigma: np.ndarray, horizon: int, stride: int, val_frac: float) -> WindowDataset:
    T = horizon + 1  # windows of T + 1 frames
    tr_p, tr_c, va_p, va_c = [], [], [], []
    for q in runs:
        p = q["pv"] / sigma
        c = q["c"] / sigma
        N = len(p)
        cut = int(N * (1 - val_frac))
        for lo, hi, lp, lc in ((0, cut, tr_p, tr_c), (cut, N, va_p, va_c)):
            for i in range(lo, hi - T, stride):
                lp.append(p[i : i + T + 1])
                lc.append(c[i : i + T + 1])
    return WindowDataset(np.array(tr_p), np.array(tr_c), np.array(va_p), np.array(va_c), sigma)


def split_runs(runs: list[dict], sigma: np.ndarray, frac: float):
    """Per-run (train, val) contiguous splits of normalised p and c (same cut as the windows)."""
    p_tr = [q["pv"][: int(len(q["pv"]) * frac)] / sigma for q in runs]
    c_tr = [q["c"][: int(len(q["c"]) * frac)] / sigma for q in runs]
    return p_tr, c_tr


# ---------------------------------------------------------------------------------- Exp 3 init
def arx_init_coefs(runs: list[dict], sigma: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """Per-mode ARX on measurements: p_{t+1} ~ [p_t, p_{t-1}, c_t, c_{t-1}, c_{t+1}] -> (5, n)."""
    n = len(sigma)
    coef = np.zeros((5, n))
    for i in range(n):
        Xs, ys = [], []
        for q in runs:
            p = q["pv"][:, i] / sigma[i]
            c = q["c"][:, i] / sigma[i]
            Xs.append(np.column_stack([p[1:-1], p[:-2], c[1:-1], c[:-2], c[2:]]))
            ys.append(p[2:])
        Xi, yi = np.concatenate(Xs), np.concatenate(ys)
        coef[:, i] = np.linalg.solve(Xi.T @ Xi + ridge * len(yi) * np.eye(5), Xi.T @ yi)
    return coef


def structured_init(coef: np.ndarray, hidden: int, seed: int) -> dict[str, np.ndarray]:
    """Linear ARX + near-zero network (= the 'linear latent' model)."""
    rng = np.random.default_rng(seed)
    n = coef.shape[1]
    P = dict(A1=np.diag(coef[0]), A2=np.diag(coef[1]), Ac1=np.diag(coef[2]), Ac2=np.diag(coef[3]), B0=np.diag(coef[4]))
    for tag in ("e", "d"):
        P["W1" + tag] = rng.standard_normal((n, hidden)) / np.sqrt(n)
        P["b1" + tag] = np.zeros(hidden)
        P["W2" + tag] = 1e-2 * rng.standard_normal((hidden, n))
        P["b2" + tag] = np.zeros(n)
    return P


# ---------------------------------------------------------------------------------- Exp 4 init
def dmdc_init(p_runs, c_runs, nz: int, he: int, hd: int, seed: int, ridge: float = 1e-3) -> dict[str, np.ndarray]:
    """Linear Koopman with control (DMDc) from measurements: z = O V (PCA), linear decoder by least
    squares, (A, B) by regression z_{t+1} ~ [z_t, c_{t+1}]."""
    rng = np.random.default_rng(seed)
    n = p_runs[0].shape[1]
    Os = [np.concatenate([p[1:], p[:-1], c[1:], c[:-1]], axis=1) for p, c in zip(p_runs, c_runs)]
    Oall = np.concatenate(Os)
    V = np.linalg.svd(Oall, full_matrices=False)[2][:nz].T  # (4n, nz)
    Zall = Oall @ V
    Pall = np.concatenate([p[1:] for p in p_runs])
    Pd = np.linalg.solve(Zall.T @ Zall + ridge * len(Zall) * np.eye(nz), Zall.T @ Pall)
    X = np.concatenate([np.concatenate([O[:-1] @ V, c[2:]], axis=1) for O, c in zip(Os, c_runs)])
    Y = np.concatenate([O[1:] @ V for O in Os])
    Th = np.linalg.solve(X.T @ X + ridge * len(X) * np.eye(nz + n), X.T @ Y).T  # (nz, nz + n)
    return dict(
        A=Th[:, :nz].copy(),
        B=Th[:, nz:].copy(),
        Pe=V,
        Pd=Pd,
        W1e=rng.standard_normal((4 * n, he)) / np.sqrt(4 * n),
        W2e=1e-2 * rng.standard_normal((he, nz)),
        W1d=rng.standard_normal((nz, hd)) / np.sqrt(nz),
        W2d=1e-2 * rng.standard_normal((hd, n)),
    )
