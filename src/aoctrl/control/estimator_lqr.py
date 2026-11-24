"""Experiment 2: Pyramid measurements -> EKF -> LQR (certainty equivalence).

Same K as the Experiment 1 oracle, applied to the estimated state plus the pending-command buffer.
The measurement pv = Mrec · y of frame k-1 is processed with the command that was *effectively
applied* at that frame (accounting for the DM delay).
"""

from __future__ import annotations

import numpy as np

from aoctrl.control.base import Controller, Timer
from aoctrl.control.lqr import OracleLQRModel, build_oracle_lqr
from aoctrl.dynamics.ar import ARModel
from aoctrl.estimators.ekf import ModalEKF
from aoctrl.sensors.optical_gain import OpticalGainModel


class EstimatorLQR(Controller):
    def __init__(self, estimator: ModalEKF, lqr: OracleLQRModel, M2C, Mrec, sign: float, name: str = "EKF + LQR"):
        self.est_, self.m, self.M2C, self.Mrec = estimator, lqr, M2C, Mrec
        self.sign, self.name = float(sign), name
        self.n, self.D = lqr.n, lqr.D
        self.reset()

    def reset(self):
        super().reset()
        self.est_.reset()
        self.cmd = [np.zeros(self.n)]  # cmd[j] = modal command decided at step j
        self.raw: list[np.ndarray] = []

    def __call__(self, u, y, k):
        with Timer(self._t_ms):
            n, D = self.n, self.D
            k = len(self.cmd)  # this is step k; y is the measurement of step k-1
            pv = self.Mrec @ y
            j = (k - 1) - D  # command effectively applied at step k-1
            c_app = self.cmd[j] if j >= 0 else np.zeros(n)
            x = self.est_.step(pv, self.sign * c_app)
            self.raw.append(pv - self.sign * c_app)
            bufs = [self.cmd[k - 1 - i] if k - 1 - i >= 0 else np.zeros(n) for i in range(D)]
            c = -self.sign * (self.m.K @ np.concatenate([x] + bufs))
            self.cmd.append(c)
            return self.M2C @ c

    def diagnostics(self):
        return dict(
            t_ms=np.array(self._t_ms),
            est=np.array(self.est_.est),
            raw=np.array(self.raw),
            nis=np.array(self.est_.nis),
        )

    def __repr__(self):
        return f"{self.name}(p={self.m.p}, D={self.D})"


def build_ekf_lqr(
    ar: ARModel,
    delay: int,
    sensor: OpticalGainModel,
    noise_pv: np.ndarray,
    model_err: float,
    q_scale: float,
    rho: float,
    Wn: np.ndarray,
    M2C: np.ndarray,
    Mrec: np.ndarray,
    sign: float,
    name: str,
) -> EstimatorLQR:
    """R = measurement noise + sensor-model error; Q from AR innovations; Q, P0 scaled by q_scale
    (∝ r0^(-5/3) when the regime is assumed known)."""
    lqr = build_oracle_lqr(ar.coefs, delay, rho, Wn)
    R_diag = noise_pv**2 + (model_err**2) * ar.state_var * q_scale + 1e-30
    ekf = ModalEKF.from_ar(lqr.A, ar.innov_var, ar.state_var, R_diag, sensor, ar.n, ar.order, q_scale=q_scale)
    return EstimatorLQR(ekf, lqr, M2C, Mrec, sign, name)
