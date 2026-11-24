"""Classical controllers: integrator (reference) and the cyclic-gain excitation integrator used to
collect identification data (notebook §4 and §7.1)."""

from __future__ import annotations

import numpy as np

from aoctrl.control.base import Controller, Timer


class Integrator(Controller):
    """u_{k+1} = leak · u_k − gain · R y_k."""

    def __init__(self, R: np.ndarray, gain: float = 0.4, leak: float = 1.0, name: str | None = None):
        self.R, self.gain, self.leak = R, float(gain), float(leak)
        self.name = name or f"I (g={self.gain:g})"
        self.reset()

    def __call__(self, u, y, k):
        with Timer(self._t_ms):
            return self.leak * u - self.gain * (self.R @ y)


class ExcitationIntegrator(Controller):
    """Integrator with a cyclic gain (closed / open / soft closed) plus modal dither, so that the
    data cover both small and large residuals."""

    def __init__(self, Mrec, M2C, sigma, seg: int = 50, gains=(0.5, None, 0.2), dither: float = 0.1, seed: int = 0):
        self.Mrec, self.M2C, self.sigma = Mrec, M2C, np.asarray(sigma)
        self.seg, self.gains, self.dither, self.seed = int(seg), tuple(gains), float(dither), int(seed)
        self.name = "excitation"
        self.reset()

    def reset(self):
        super().reset()
        self.base = np.zeros(self.M2C.shape[1])
        self.rng = np.random.default_rng(self.seed)

    def __call__(self, u, y, k):
        g = self.gains[(k // self.seg) % len(self.gains)]
        if g is None:
            self.base[:] = 0.0
        else:
            self.base = self.base - g * (self.Mrec @ y)
        return self.M2C @ (self.base + self.dither * self.sigma * self.rng.standard_normal(len(self.base)))
