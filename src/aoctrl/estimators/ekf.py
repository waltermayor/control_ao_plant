"""Extended Kalman filter on the AR state of the modal turbulence (notebook §7.3).

    state        x_k = [d_k, …, d_{k-p+1}],   x_k = F x_{k-1} + w_k,  w ~ N(0, Q)
    measurement  pv_k = h(s_k) + v_k,  s_k = d_k + c_k   (c_k: effective modal command, sign included)
    update       Joseph form;  H = ∂h/∂s (analytic)

The estimator is independent of the controller: `step(pv, c_eff)` processes one measurement and
returns the filtered state. It is composed with an LQR in `control.estimator_lqr`.
"""

from __future__ import annotations

import numpy as np

from aoctrl.sensors.optical_gain import OpticalGainModel


class ModalEKF:
    def __init__(self, F: np.ndarray, Q: np.ndarray, P0: np.ndarray, R: np.ndarray, sensor: OpticalGainModel, n: int):
        self.F, self.Q, self.P0, self.R, self.sensor, self.n = F, Q, P0, R, sensor, int(n)
        self.dim = F.shape[0]
        self.reset()

    @classmethod
    def from_ar(cls, F, qvar, dvar, R_diag, sensor, n, p, q_scale: float = 1.0) -> "ModalEKF":
        pn = p * n
        Q = np.zeros((pn, pn))
        Q[:n, :n] = np.diag(qvar * q_scale)
        P0 = np.diag(np.tile(dvar * q_scale, p))
        return cls(F[:pn, :pn], Q, P0, np.diag(R_diag), sensor, n)

    def reset(self):
        self.x = np.zeros(self.dim)
        self.P = self.P0.copy()
        self.k = 0
        self.est: list[np.ndarray] = []
        self.nis: list[float] = []

    def step(self, pv: np.ndarray, c_eff: np.ndarray) -> np.ndarray:
        n = self.n
        self.k += 1
        if self.k > 1:  # prediction
            self.x = self.F @ self.x
            self.P = self.F @ self.P @ self.F.T + self.Q
        s = self.x[:n] + c_eff
        H = self.sensor.jac(s)
        e = pv - self.sensor.h(s)
        S = H @ self.P[:n, :n] @ H.T + self.R
        PHt = self.P[:, :n] @ H.T
        Kg = np.linalg.solve(S, PHt.T).T  # P Hᵀ S^-1
        self.x = self.x + Kg @ e
        IKH = np.eye(self.dim)
        IKH[:, :n] -= Kg @ H
        self.P = IKH @ self.P @ IKH.T + Kg @ self.R @ Kg.T  # Joseph form
        self.est.append(self.x[:n].copy())
        self.nis.append(float(e @ np.linalg.solve(S, e)) / n)
        return self.x
