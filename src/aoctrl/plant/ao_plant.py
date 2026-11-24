"""Control-facing AO plant: `step(u) -> (y, info)`.

Wraps an `OpticalBackend` with what a control engineer cares about (from notebook §2):

| control concept        | here                                   | units       |
|------------------------|----------------------------------------|-------------|
| input  u_k             | absolute DM command (all DMs)          | m (surface) |
| measured output y_k    | concatenated WFS signals               | normalised  |
| disturbance d_k        | turbulence (+ optional injected OPD)   | m           |
| performance output z_k | residual OPD RMS, Maréchal Strehl      | m, -        |

Timing convention (unchanged from the notebook): `step(u_k)` applies u_k (after `delay` frames and
optional saturation), advances the turbulence one frame and measures y_k *already corrected*.
With delay=0 the command acts on the same frame it is measured on; the loop itself adds 1 frame
because the controller only sees y_{k-1} when deciding u_k.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from aoctrl.optics.base import Conditions, OpticalBackend


@dataclass
class StepInfo:
    rms: float  # residual OPD RMS [m] (mean over science directions)
    strehl: float  # Maréchal approximation


class AOPlant:
    def __init__(self, backend: OpticalBackend, delay: int = 0, u_limit: float | None = None):
        self.backend = backend
        self.n_act = backend.n_act
        self.n_meas = backend.n_signal
        self.delay = int(delay)
        self.u_limit = u_limit
        self.record_maps = False
        self.last: dict[str, np.ndarray] = {}
        self.reset()

    # ------------------------------------------------------------------ lifecycle
    def reset(self) -> None:
        self.backend.set_dm(0)
        self._buf = [np.zeros(self.n_act) for _ in range(self.delay)]
        self.u_applied = np.zeros(self.n_act)

    def new_turbulence(self, seed: int) -> None:
        self.backend.new_turbulence(seed)

    # ------------------------------------------------------------------ dynamics
    def step(self, u: np.ndarray, d: np.ndarray | None = None, mode: str = "add") -> tuple[np.ndarray, StepInfo]:
        u = np.asarray(u, dtype=float).copy()
        if self.u_limit is not None:
            u = np.clip(u, -self.u_limit, self.u_limit)
        self._buf.append(u)
        self.u_applied = self._buf.pop(0)
        self.backend.set_dm(self.u_applied)
        self.backend.advance_turbulence(d, mode)
        y = self.backend.propagate()
        rms_dirs = [float(self.backend.residual_opd(j).std()) for j in range(self.backend.n_directions)]
        rms = float(np.mean(rms_dirs))
        strehl = float(np.mean([np.exp(-((2 * np.pi * r / self.backend.wavelength) ** 2)) for r in rms_dirs]))
        if self.record_maps:
            self.last = self.backend.maps()
        return y, StepInfo(rms=rms, strehl=strehl)

    # ------------------------------------------------------------------ truth (simulation only)
    def turbulence_opd(self, direction: int = 0) -> np.ndarray:
        return self.backend.turbulence_opd(direction)

    def residual_opd(self, direction: int = 0) -> np.ndarray:
        return self.backend.residual_opd(direction)

    # ------------------------------------------------------------------ operating conditions
    def get_conditions(self) -> dict:
        c = self.backend.get_conditions()
        return dict(conditions=c, delay=self.delay, u_limit=self.u_limit)

    def set_conditions(self, *, delay: int | None = None, u_limit: float | None = None, **physical) -> None:
        """Change conditions. `physical` keys: r0, magnitude, photon_noise, readout_noise, wind_speed."""
        if physical:
            self.backend.set_conditions(Conditions().updated(**physical))
        if delay is not None:
            self.delay = int(delay)
            self._buf = [np.zeros(self.n_act) for _ in range(self.delay)]
        self.u_limit = u_limit

    @contextmanager
    def conditions(self, *, delay: int | None = None, u_limit: float | None = None, **physical) -> Iterator["AOPlant"]:
        """Temporarily change operating conditions, restoring them on exit (replaces the notebook's
        `apply_conditions` / `restore_conditions` pair)."""
        saved = self.get_conditions()
        try:
            self.set_conditions(delay=delay, u_limit=u_limit, **physical)
            yield self
        finally:
            self.backend.set_conditions(saved["conditions"])
            self.set_conditions(delay=saved["delay"], u_limit=saved["u_limit"])
