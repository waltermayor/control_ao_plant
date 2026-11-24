"""Experiment 1: LQR with the true state (KL coefficients of the turbulence).

This is an *oracle*: it uses information a real WFS does not deliver (exact state, no noise, no
sensor non-linearity). It bounds what experiments 2–4 can achieve; it is not a realisable controller.
"""

from __future__ import annotations

import numpy as np

from aoctrl.control.base import Controller, Timer
from aoctrl.control.lqr import OracleLQRModel, build_oracle_lqr


class OracleLQR(Controller):
    needs_truth = True

    def __init__(self, model: OracleLQRModel, M2C: np.ndarray, sign: float, name: str = "LQR oracle"):
        self.m, self.M2C, self.sign, self.name = model, M2C, float(sign), name
        self.reset()

    @classmethod
    def from_ar(cls, coefs, M2C, delay, rho, Wn, sign, name="LQR oracle", use_dare=True) -> "OracleLQR":
        return cls(build_oracle_lqr(coefs, delay, rho, Wn, use_dare=use_dare), M2C, sign, name)

    def reset(self):
        super().reset()
        self.hist: list[np.ndarray] = []
        self.ubuf: list[np.ndarray] = []

    def observe(self, d):
        self.hist.append(np.asarray(d, dtype=float))

    def __call__(self, u, y, k):
        with Timer(self._t_ms):
            m = self.m
            z0 = np.zeros(m.n)
            lags = [self.hist[-1 - j] if len(self.hist) > j else z0 for j in range(m.p)]
            bufs = [self.ubuf[-1 - j] if len(self.ubuf) > j else z0 for j in range(m.D)]
            c = -self.sign * (m.K @ np.concatenate(lags + bufs))  # cancels the prediction of d_{k+D}
            self.ubuf.append(c)
            return self.M2C @ c

    def __repr__(self):
        return f"{self.name}(p={self.m.p}, D={self.m.D}, K by {self.m.method})"
