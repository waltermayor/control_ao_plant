"""Pyramid optical-gain model (notebook §7.2–7.3).

    h_i(s) = K0_i · s_i · f(r),   f(r) = (1 + (r/r_m)^2)^(-q/2),   r = sqrt(sᵀ W s)

The optical gain *decreases* as the residual grows (the signal is compressed). q = 0 gives the
linear model h(s) = K0 · s. The Jacobian is analytic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class OpticalGainModel:
    def __init__(self, K0: np.ndarray, q: float, rm: float, W: np.ndarray):
        self.K0 = np.asarray(K0, float)
        self.q = float(q)
        self.rm = float(rm)
        self.W = W

    @classmethod
    def linear(cls, K0: np.ndarray, W: np.ndarray) -> "OpticalGainModel":
        return cls(K0, 0.0, 1.0, W)

    def _r(self, s: np.ndarray) -> float:
        return float(np.sqrt(max(s @ self.W @ s, 0.0)))

    def f(self, r):
        return (1.0 + (r / self.rm) ** 2) ** (-self.q / 2.0)

    def fp(self, r):
        return -self.q * r / self.rm**2 * (1.0 + (r / self.rm) ** 2) ** (-self.q / 2.0 - 1.0)

    def h(self, s: np.ndarray) -> np.ndarray:
        return self.K0 * s * self.f(self._r(s))

    def jac(self, s: np.ndarray) -> np.ndarray:
        r = self._r(s)
        J = np.diag(self.K0 * self.f(r))
        if self.q != 0.0 and r > 1e-15:
            J = J + np.outer(self.K0 * s, self.fp(r) * (self.W @ s) / r)
        return J

    def jacobian_check(self, s0: np.ndarray) -> float:
        """Relative error of the analytic Jacobian vs central finite differences."""
        n = len(s0)
        J = self.jac(s0)
        Jn = np.zeros_like(J)
        eps = 1e-4 * (np.abs(s0).max() + 1e-30)
        for j in range(n):
            dp = np.zeros(n)
            dp[j] = eps
            Jn[:, j] = (self.h(s0 + dp) - self.h(s0 - dp)) / (2 * eps)
        return float(np.linalg.norm(J - Jn) / np.linalg.norm(Jn))


@dataclass
class OpticalGainFit:
    K0: np.ndarray
    q: float
    rm: float
    loss: float  # relative pv error of the compressed model
    K0_linear: np.ndarray
    loss_linear: float
    rm_at_grid_edge: bool


def fit_optical_gain(PV: np.ndarray, S: np.ndarray, R: np.ndarray, qs=(0.5, 1.0, 2.0), rms_grid=None) -> OpticalGainFit:
    """Grid search over (q, r_m); K0 per mode by least squares. PV, S: (T, n); R: (T,) residual RMS [m]."""
    rms_grid = np.logspace(-7.7, -4.7, 40) if rms_grid is None else np.asarray(rms_grid)
    var_tot = np.sum(PV**2)
    best = None
    for q in qs:
        for rm in rms_grid:
            f = (1.0 + (R / rm) ** 2) ** (-q / 2.0)
            Z = S * f[:, None]
            K0 = np.sum(PV * Z, axis=0) / np.maximum(np.sum(Z * Z, axis=0), 1e-300)
            loss = np.sum((PV - K0 * Z) ** 2) / var_tot
            if best is None or loss < best[3]:
                best = (K0, q, rm, loss)
    K0l = np.sum(PV * S, axis=0) / np.maximum(np.sum(S * S, axis=0), 1e-300)
    loss_l = np.sum((PV - K0l * S) ** 2) / var_tot
    K0, q, rm, loss = best
    return OpticalGainFit(
        K0=K0,
        q=float(q),
        rm=float(rm),
        loss=float(loss),
        K0_linear=K0l,
        loss_linear=float(loss_l),
        rm_at_grid_edge=bool(rm in (rms_grid[0], rms_grid[-1])),
    )
