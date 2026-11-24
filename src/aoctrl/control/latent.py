"""Controllers acting in a learned latent space (Experiments 3 and 4).

The latent state comes straight from the encoder applied to the last measurement (no filter); the
LQR uses the learned linear dynamics and the decoder's linear read-out J at z = 0 (certainty
equivalence). Both controllers share `ModalIO`, the normalisation between the plant and the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aoctrl.control.base import Controller, Timer
from aoctrl.control.lqr import koopman_lqr, latent_arx_lqr
from aoctrl.models import numpy_ops as ops


@dataclass
class ModalIO:
    Mrec: np.ndarray  # signals -> modal residual
    M2C: np.ndarray  # modal command -> actuators
    sigma: np.ndarray  # per-mode normalisation
    sign: float  # DM sign (+1: DM adds to the wavefront)
    cmax: float = 8.0  # saturation of the latent command [sigmas]
    Wc: np.ndarray | None = None  # well-conditioned wavefront metric (normalised units)
    rho: float = 1e-3

    def measure(self, y):
        return (self.Mrec @ y) / self.sigma

    def actuate(self, c):
        return self.M2C @ (c * self.sigma)


class _LatentBase(Controller):
    def __init__(self, io: ModalIO, delay: int, name: str):
        self.io, self.D, self.name = io, int(delay), name

    def reset(self):
        super().reset()
        self.cmd = [np.zeros(self.n)]
        self.est: list[np.ndarray] = []
        self.shat: list[np.ndarray] = []

    def _applied(self, j):
        return self.cmd[j - self.D] if j - self.D >= 0 else np.zeros(self.n)

    def _pending(self, k):
        return [self.cmd[k - 1 - i] if k - 1 - i >= 0 else np.zeros(self.n) for i in range(self.D)]

    def diagnostics(self):
        return dict(t_ms=np.array(self._t_ms), est=np.array(self.est), shat=np.array(self.shat))

    def __repr__(self):
        return f"{self.name}(D={self.D})"


class LatentARXLQR(_LatentBase):
    """Exp 3: X_k = [z_{k-1}, z_{k-2}, a_{k-1}, a_{k-2}, pending] with z = E(p)."""

    def __init__(self, P: dict, io: ModalIO, delay: int, name: str, identity: bool = False, rmax: float = 0.9995):
        super().__init__(io, delay, name)
        self.P = {k: np.asarray(v, float).copy() for k, v in P.items()}
        self.P["A1"], self.P["A2"] = ops.stabilize_pair(self.P["A1"], self.P["A2"], rmax)
        self.n = self.P["A1"].shape[0]
        self.identity = identity
        self.J = np.eye(self.n) if identity else ops.latent_arx_decoder_jacobian(self.P)
        self.K, self.lqr_info = latent_arx_lqr(self.P, self.D, self.J, io.Wc, io.rho)
        self.reset()

    def reset(self):
        super().reset()
        self.z_prev = np.zeros(self.n)

    def _enc(self, p):
        return p if self.identity else ops.mlp_residual(self.P, p, "e")

    def __call__(self, u, y, k):
        with Timer(self._t_ms):
            k = len(self.cmd)
            z = self._enc(self.io.measure(y))  # latent of step k-1, straight from the measurement
            X = np.concatenate([z, self.z_prev, self._applied(k - 1), self._applied(k - 2)] + self._pending(k))
            c = np.clip(-self.K @ X, -self.io.cmax, self.io.cmax)
            self.cmd.append(c)
            self.z_prev = z
            shat = self.io.sigma * z  # estimates of s and d, for metrics only
            self.shat.append(shat)
            self.est.append(shat - self.io.sign * self.io.sigma * self._applied(k - 1))
            return self.io.actuate(c)


class KoopmanLQR(_LatentBase):
    """Exp 4: z = E(O), O = [p_{k-1}, p_{k-2}, a_{k-1}, a_{k-2}];  X_k = [z_{k-1}, pending]."""

    def __init__(
        self,
        P: dict,
        io: ModalIO,
        delay: int,
        name: str,
        linear: bool = False,
        probe: np.ndarray | None = None,
        rmax: float = 0.9995,
    ):
        super().__init__(io, delay, name)
        self.P = {k: np.asarray(v, float).copy() for k, v in P.items()}
        self.linear = linear
        self.P["A"], self.rho_A = ops.stabilize_A(self.P["A"], rmax)
        self.n = self.P["B"].shape[1]
        self.J = ops.koopman_decoder_jacobian(self.P, linear)
        self.K, self.lqr_info = koopman_lqr(self.P["A"], self.P["B"], self.J, self.D, io.Wc, io.rho)
        self.probe = probe  # (nz, n) linear probe z -> s fitted with truth, diagnostics only
        self.reset()

    def reset(self):
        super().reset()
        self.p_prev = np.zeros(self.n)

    def __call__(self, u, y, k):
        with Timer(self._t_ms):
            k = len(self.cmd)
            p = self.io.measure(y)
            O = np.concatenate([p, self.p_prev, self._applied(k - 1), self._applied(k - 2)])
            z = ops.koopman_encode(self.P, O, self.linear)
            c = np.clip(-self.K @ np.concatenate([z] + self._pending(k)), -self.io.cmax, self.io.cmax)
            self.cmd.append(c)
            self.p_prev = p
            if self.probe is not None:
                shat = self.io.sigma * (z @ self.probe)
                self.shat.append(shat)
                self.est.append(shat - self.io.sign * self.io.sigma * self._applied(k - 1))
            return self.io.actuate(c)

    def __repr__(self):
        return f"{self.name}(D={self.D}, nz={self.P['A'].shape[0]})"
