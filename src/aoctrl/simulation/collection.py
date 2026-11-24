"""Data collection for identification (notebook §7.1 and §8.3).

All functions take an explicit plant/calibration and return plain dicts of arrays so that results
are cacheable and backend-agnostic.
"""

from __future__ import annotations

import numpy as np

from aoctrl.calibration import Calibration
from aoctrl.control.classic import ExcitationIntegrator
from aoctrl.plant import AOPlant
from aoctrl.simulation.runner import simulate


def measure_noise_floor(
    plant: AOPlant, calib: Calibration, n_steps: int = 150, burn: int = 10, **conditions
) -> np.ndarray:
    """Per-mode std of pv = Mrec·y with zero turbulence and a flat DM (measurement noise)."""
    zero = np.zeros(plant.backend.pupil.shape)
    with plant.conditions(**conditions):
        tel = simulate(plant, n_steps, None, seed=1, disturbance=lambda k: zero, dist_mode="replace")
    return (calib.Mrec @ tel.y.T).T[burn:].std(axis=0)


def collect_excitation(
    plant: AOPlant,
    calib: Calibration,
    n_steps: int,
    seed: int,
    sigma_modes: np.ndarray,
    seg: int = 50,
    dither: float = 0.1,
    gains=(0.5, None, 0.2),
    **conditions,
) -> dict:
    """Closed/open-loop excitation run. Returns pv (measured modal residual), c (applied modal
    command), and the truth s (modal residual) and d (modal turbulence) for evaluation."""
    ctrl = ExcitationIntegrator(calib.Mrec, calib.M2C, sigma_modes, seg=seg, gains=gains, dither=dither, seed=seed)
    with plant.conditions(**conditions):
        tel = simulate(plant, n_steps, ctrl, seed=seed, projector=calib.Bpinv)
    return dict(pv=(calib.Mrec @ tel.y.T).T, c=(calib.Minv @ tel.u.T).T, s=tel.s, d=tel.d)
