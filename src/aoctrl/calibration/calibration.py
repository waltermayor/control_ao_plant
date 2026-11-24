"""System calibration (notebook §3 and §5).

1. Modal basis M2C (modes -> actuator commands), KL ordered low -> high frequency.
2. Interaction matrix D (signals x modes) and modal reconstructor M = D^+ (modes <- signals).
   Zonal reconstructor R = M2C · M (commands <- signals): integrator u_{k+1} = u_k - g R y_k.
3. Modal projection of OPD maps: B (pixels x modes) is the OPD of each mode *measured from the DM
   response* (no assumption about index ordering); Bpinv = B^+ maps an OPD to modal coefficients.
4. Wavefront metric W = BᵀB / n_pix, so eᵀ W e is the OPD variance of a modal error e [m^2].
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aoctrl.optics.base import OpticalBackend


@dataclass
class Calibration:
    M2C: np.ndarray  # (n_act, n_modes)
    D: np.ndarray  # (n_signal, n_modes)
    Mrec: np.ndarray  # (n_modes, n_signal)   pv = Mrec @ y (measured modal residual)
    singular_values: np.ndarray
    cond: float
    B: np.ndarray  # (n_pix, n_modes)
    Bpinv: np.ndarray  # (n_modes, n_pix)
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ derived quantities
    @property
    def n_modes(self) -> int:
        return self.M2C.shape[1]

    @property
    def R(self) -> np.ndarray:
        """Zonal reconstructor (n_act x n_signal)."""
        return self.M2C @ self.Mrec

    @property
    def Minv(self) -> np.ndarray:
        """Actuator commands -> modal coefficients."""
        return np.linalg.pinv(self.M2C)

    @property
    def W_phys(self) -> np.ndarray:
        """eᵀ W e = OPD variance of a modal error e [m^2]."""
        return self.B.T @ self.B / self.B.shape[0]

    @property
    def W_norm(self) -> np.ndarray:
        """Trace-normalised metric used in the DAREs (better conditioned)."""
        W = self.W_phys
        return W / np.trace(W) * W.shape[0]

    def project(self, opd_pupil: np.ndarray) -> np.ndarray:
        return self.Bpinv @ opd_pupil


def calibrate(backend: OpticalBackend, n_modes, stroke: float) -> Calibration:
    M2C = backend.modal_basis(n_modes)
    inter = backend.interaction_matrix(M2C, stroke)
    B = np.stack([backend.dm_opd(M2C[:, i]) for i in range(M2C.shape[1])], axis=1)
    backend.set_dm(0)
    return Calibration(
        M2C=M2C,
        D=inter.D,
        Mrec=inter.M,
        singular_values=inter.singular_values,
        cond=inter.cond,
        B=B,
        Bpinv=np.linalg.pinv(B),
        meta=dict(stroke=stroke, n_modes_requested=n_modes),
    )


def check_dm_sign(d: np.ndarray, s: np.ndarray, u: np.ndarray, Minv: np.ndarray) -> dict:
    """Fit `residual - turbulence ≈ slope · command` in modal space (notebook §6.2).

    d, s: (T, n_modes) modal turbulence / residual; u: (T, n_act) applied commands.
    Returns slope, relative fit error, sign (+1: the DM adds to the wavefront) and a validity flag.
    """
    c_app = (Minv @ u.T).T
    dm_part = s - d
    slope = float(np.sum(dm_part * c_app) / np.sum(c_app * c_app))
    rel = float(np.linalg.norm(dm_part - slope * c_app) / np.linalg.norm(dm_part))
    return dict(slope=slope, rel_error=rel, sign=1.0 if slope > 0 else -1.0, ok=rel <= 0.2 and 0.5 < abs(slope) < 2)
