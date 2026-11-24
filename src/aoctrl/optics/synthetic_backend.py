"""Fast modal surrogate of an SCAO Pyramid system (no OOPAO required).

Purpose: unit tests, CI, and fast iteration on estimators / learning code. It reproduces the
*structure* the experiments rely on, not OOPAO's numbers:

* turbulence: per-mode AR(2) process (wind-like oscillation + decay) with a Kolmogorov-like
  variance spectrum, amplitude scaling (r0_ref / r0)^(5/6), plus an out-of-basis high-order
  component (fitting floor);
* DM: one actuator per mode (M2C = identity), adds to the wavefront (DM sign +1);
* WFS: y = G · Mix · [K0 ⊙ s · f(r)] + noise, with the optical-gain compression
  f(r) = (1 + (r/r_m)^2)^(-q/2) used in Experiment 2 — the Pyramid-like non-linearity;
* photon noise scaling with guide-star magnitude.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aoctrl.optics.base import Conditions, InteractionData, OpticalBackend


def _cosine_modes(grid: int, pupil: np.ndarray, n: int) -> np.ndarray:
    """Orthonormal (on the pupil, unit RMS) low->high spatial-frequency modes, excluding piston."""
    x = np.linspace(-1, 1, grid)
    X, Y = np.meshgrid(x, x)
    cands = []
    kmax = int(np.ceil(np.sqrt(4 * n))) + 2
    for kx in range(kmax):
        for ky in range(kmax):
            if kx == 0 and ky == 0:
                continue
            for fx in (np.cos, np.sin):
                for fy in (np.cos, np.sin):
                    if (kx == 0 and fx is np.sin) or (ky == 0 and fy is np.sin):
                        continue
                    cands.append((kx * kx + ky * ky, fx(np.pi * kx * X / 2) * fy(np.pi * ky * Y / 2)))
    cands.sort(key=lambda t: t[0])
    M = np.stack([c[1][pupil] for c in cands], axis=1)
    M = M - M.mean(axis=0, keepdims=True)
    Q, R = np.linalg.qr(M)
    keep = np.abs(np.diag(R)) > 1e-8 * np.abs(np.diag(R)).max()
    Q = Q[:, keep][:, :n]
    return Q * np.sqrt(pupil.sum())  # unit RMS over the pupil


class SyntheticBackend(OpticalBackend):
    def __init__(self, cfg: Any):
        sc = cfg.synthetic
        self.cfg = cfg
        self._n = int(sc.n_modes)
        self._n_hi = int(sc.get("n_high_order", 20))
        self._grid = int(sc.grid)
        x = np.linspace(-1, 1, self._grid)
        X, Y = np.meshgrid(x, x)
        self._pupil = (X**2 + Y**2) <= 1.0
        allm = _cosine_modes(self._grid, self._pupil, self._n + self._n_hi)
        self.Bm, self.Bh = allm[:, : self._n], allm[:, self._n :]
        self._Bm_pinv = np.linalg.pinv(self.Bm)
        self._wavelength = float(sc.get("wavelength", 790e-9))
        self._ts = float(cfg.telescope.sampling_time)
        rng = np.random.default_rng(int(sc.get("structure_seed", 1234)))

        # temporal model of each mode: AR(2) with poles rho·exp(±iω); faster for higher modes
        i = np.arange(self._n)
        rho = np.linspace(sc.ar_rho[0], sc.ar_rho[1], self._n)
        om = np.linspace(sc.ar_omega[0], sc.ar_omega[1], self._n)
        self.a1, self.a2 = 2 * rho * np.cos(om), -(rho**2)
        std = float(sc.rms_ref_nm) * 1e-9 * (i + 1.0) ** (-11.0 / 12.0)
        std *= float(sc.rms_ref_nm) * 1e-9 / np.sqrt(np.sum(std**2))
        a1, a2 = self.a1, self.a2
        stat_gain = (1 - a2) / ((1 + a2) * ((1 - a2) ** 2 - a1**2))
        self._w_std = std / np.sqrt(stat_gain)
        self._hi_std = float(sc.fitting_rms_nm) * 1e-9 / np.sqrt(self._n_hi)
        self._hi_rho = float(sc.get("hi_rho", 0.9))

        # WFS model
        self._K0 = 1.0 + float(sc.og_k0_spread) * rng.uniform(-1, 1, self._n)
        self._q, self._rm = float(sc.og_q), float(sc.og_rm_nm) * 1e-9
        n_sig = int(sc.signals_per_mode) * self._n
        Mix = rng.standard_normal((n_sig, self._n))
        self._mix = Mix / np.linalg.norm(Mix, axis=0, keepdims=True)
        self._wfs_gain = 1.0 / (self._wavelength / (2 * np.pi))  # metres -> radians-ish, y = O(1)
        self._noise_ref = float(sc.photon_noise_ref)
        self._r0_ref = float(sc.r0_ref)

        self._cond = Conditions(
            r0=float(cfg.atmosphere.r0),
            magnitude=float(cfg.guide_stars.stars[0].magnitude),
            photon_noise=bool(cfg.wfs.get("photon_noise", False)),
            readout_noise=float(cfg.wfs.get("readout_noise", 0.0)),
        )
        self._dm = np.zeros(self._n)
        self.new_turbulence(0)
        self.description = f"synthetic SCAO: {self._n} modes / {n_sig} signals (AR(2) turbulence, optical-gain WFS)"

    # ------------------------------------------------------------------ sizes
    @property
    def n_act(self) -> int:
        return self._n

    @property
    def n_signal(self) -> int:
        return self._mix.shape[0]

    @property
    def n_pupil_pixels(self) -> int:
        return int(self._pupil.sum())

    @property
    def pupil(self) -> np.ndarray:
        return self._pupil

    @property
    def wavelength(self) -> float:
        return self._wavelength

    @property
    def sampling_time(self) -> float:
        return self._ts

    # ------------------------------------------------------------------ turbulence
    def _scale(self) -> float:
        return (self._r0_ref / float(self._cond.r0)) ** (5.0 / 6.0)

    def new_turbulence(self, seed: int) -> None:
        self._rng = np.random.default_rng(int(seed))
        self._noise_rng = np.random.default_rng(int(seed) + 10_000_019)
        self._d = np.zeros((2, self._n))  # unit-amplitude AR state [d_k, d_{k-1}]
        self._e = np.zeros(self._n_hi)
        for _ in range(200):  # burn-in to the stationary distribution
            self._ar_step()
        self._turb = self._compose()
        self._res = self._turb.copy()

    def _ar_step(self):
        d_new = self.a1 * self._d[0] + self.a2 * self._d[1] + self._w_std * self._rng.standard_normal(self._n)
        self._d = np.stack([d_new, self._d[0]])
        self._e = self._hi_rho * self._e + np.sqrt(1 - self._hi_rho**2) * self._hi_std * self._rng.standard_normal(
            self._n_hi
        )

    def _compose(self) -> np.ndarray:
        return self._scale() * (self.Bm @ self._d[0] + self.Bh @ self._e)

    def advance_turbulence(self, disturbance=None, mode: str = "add") -> None:
        self._ar_step()
        base = self._compose()
        if disturbance is None:
            self._turb = base
            return
        dmap = np.asarray(disturbance, float)
        dmap = dmap[self._pupil] if dmap.ndim == 2 else dmap
        if mode == "add":
            self._turb = base + dmap
        elif mode == "replace":
            self._turb = dmap.copy()
        else:
            raise ValueError(f"unknown disturbance mode {mode!r}")

    # ------------------------------------------------------------------ actuation / sensing
    def set_dm(self, coefs) -> None:
        self._dm = np.full(self._n, float(coefs)) if np.isscalar(coefs) else np.asarray(coefs, float).copy()

    def _wfs(self, res_opd: np.ndarray, noise: bool) -> np.ndarray:
        s = self._Bm_pinv @ res_opd
        r = float(np.std(res_opd))
        f = (1.0 + (r / self._rm) ** 2) ** (-self._q / 2.0)
        y = self._wfs_gain * (self._mix @ (self._K0 * s * f))
        if noise:
            sigma = 0.0
            if self._cond.photon_noise:
                sigma += self._noise_ref * 10 ** (0.2 * (float(self._cond.magnitude) - 8.0))
            sigma += float(self._cond.readout_noise or 0.0) * self._noise_ref
            if sigma > 0:
                y = y + sigma * self._noise_rng.standard_normal(y.shape)
        return y

    def propagate(self) -> np.ndarray:
        self._res = self._turb + self.Bm @ self._dm
        self._y = self._wfs(self._res, noise=True)
        return self._y

    def turbulence_opd(self, direction: int = 0) -> np.ndarray:
        return self._turb

    def residual_opd(self, direction: int = 0) -> np.ndarray:
        return self._res

    def dm_opd(self, coefs, direction: int = 0) -> np.ndarray:
        return self.Bm @ np.asarray(coefs, float)

    # ------------------------------------------------------------------ calibration
    def modal_basis(self, n_modes) -> np.ndarray:
        n = min(int(n_modes if np.isscalar(n_modes) else n_modes[0]), self._n)
        return np.eye(self._n)[:, :n]

    def interaction_matrix(self, M2C: np.ndarray, stroke: float) -> InteractionData:
        D = np.zeros((self.n_signal, M2C.shape[1]))
        for j in range(M2C.shape[1]):
            opd = self.Bm @ M2C[:, j] * stroke
            D[:, j] = (self._wfs(opd, False) - self._wfs(-opd, False)) / (2 * stroke)
        U, s, Vt = np.linalg.svd(D, full_matrices=False)
        M = (Vt.T / s) @ U.T
        return InteractionData(D=D, M=M, singular_values=s, cond=float(s[0] / s[-1]))

    # ------------------------------------------------------------------ conditions
    def get_conditions(self) -> Conditions:
        return Conditions(**self._cond.__dict__)

    def set_conditions(self, cond: Conditions) -> None:
        self._cond = self._cond.updated(**cond.__dict__)

    # ------------------------------------------------------------------ visualisation
    def _to_map(self, v):
        out = np.zeros(self._pupil.shape)
        out[self._pupil] = v
        return out

    def maps(self) -> dict[str, np.ndarray]:
        y = getattr(self, "_y", np.zeros(self.n_signal))
        side = int(np.ceil(np.sqrt(len(y))))
        sig = np.zeros(side * side)
        sig[: len(y)] = y
        return dict(turb=self._to_map(self._turb), res=self._to_map(self._res), sig=sig.reshape(side, side))

    def wfs_frame(self):
        return self.maps()["sig"]

    def psf(self, zero_padding: int = 2):
        field = self._pupil * np.exp(1j * 2 * np.pi * self._to_map(self._res) / self._wavelength)
        n = self._grid * zero_padding
        p = np.abs(np.fft.fftshift(np.fft.fft2(field, s=(n, n)))) ** 2
        return p

    def seeing_arcsec(self):
        return float(0.98 * 500e-9 / float(self._cond.r0) * 206265)
