"""Open-loop modal turbulence series (notebook §5): input for AR identification and studies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aoctrl.calibration import Calibration
from aoctrl.optics.base import OpticalBackend


@dataclass
class TurbulenceSeries:
    coefs: np.ndarray  # (T, n_modes) modal turbulence coefficients
    explained_variance: float  # fraction of OPD variance captured by the modal basis
    opd_rms: np.ndarray  # (T,) total OPD RMS [m]


def collect_turbulence(
    backend: OpticalBackend, calib: Calibration, n_steps: int, seed: int, check_every: int = 100
) -> TurbulenceSeries:
    """Advance the atmosphere without DM/WFS (fast) and project the OPD onto the modal basis."""
    backend.new_turbulence(seed)
    X = np.zeros((n_steps, calib.n_modes))
    rms = np.zeros(n_steps)
    ratios = []
    for k in range(n_steps):
        backend.advance_turbulence()
        o = backend.propagate_turbulence_only()
        X[k] = calib.Bpinv @ o
        rms[k] = o.std()
        if k % check_every == 0:
            ratios.append(np.var(calib.B @ X[k]) / np.var(o))
    return TurbulenceSeries(coefs=X, explained_variance=float(np.mean(ratios)), opd_rms=rms)
