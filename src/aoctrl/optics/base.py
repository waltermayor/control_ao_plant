"""Physical-simulation backend interface.

A backend owns the optical system (sources, atmosphere, telescope, DMs, WFSs) and exposes the
*physics* operations. It knows nothing about delays, controllers or experiments: those live in
`aoctrl.plant`. Two implementations exist:

* `OOPAOBackend`    — the real simulator (SCAO and MCAO).
* `SyntheticBackend` — a fast modal surrogate with a Pyramid-like optical-gain non-linearity, used
                       for unit tests, CI and fast iteration on estimators/learning code.

Conventions
-----------
* DM commands are a single concatenated vector over all DMs (`n_act` entries, metres).
* WFS measurements are a single concatenated vector over all WFSs (`n_signal` entries).
* OPDs returned by `turbulence_opd` / `residual_opd` are 1-D arrays of the valid pupil pixels
  (metres), for science direction `direction`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Conditions:
    """Operating conditions that experiments may change at run time."""

    r0: float | None = None
    magnitude: float | None = None
    photon_noise: bool | None = None
    readout_noise: float | None = None
    wind_speed: list[float] | None = None

    def updated(self, **kw: Any) -> "Conditions":
        data = {k: v for k, v in self.__dict__.items()}
        data.update({k: v for k, v in kw.items() if k in data and v is not None})
        return Conditions(**data)


@dataclass
class InteractionData:
    D: np.ndarray  # (n_signal, n_modes) interaction matrix
    M: np.ndarray  # (n_modes, n_signal) modal reconstructor (pseudo-inverse of D)
    singular_values: np.ndarray
    cond: float
    extra: dict = field(default_factory=dict)


class OpticalBackend(ABC):
    """Abstract optical system. All arrays are NumPy, SI units."""

    #: human-readable description of the topology, e.g. "SCAO 1 GS / 1 WFS / 1 DM"
    description: str = ""

    # ------------------------------------------------------------------ geometry / sizes
    @property
    @abstractmethod
    def n_act(self) -> int: ...

    @property
    @abstractmethod
    def n_signal(self) -> int: ...

    @property
    @abstractmethod
    def n_pupil_pixels(self) -> int: ...

    @property
    @abstractmethod
    def pupil(self) -> np.ndarray:
        """2-D boolean pupil mask (for maps / videos)."""

    @property
    @abstractmethod
    def wavelength(self) -> float:
        """Science wavelength [m] (Maréchal Strehl)."""

    @property
    @abstractmethod
    def sampling_time(self) -> float: ...

    @property
    def n_directions(self) -> int:
        """Number of science directions on which performance is evaluated."""
        return 1

    @property
    def dm_slices(self) -> list[slice]:
        """Slices of the concatenated command vector belonging to each DM."""
        return [slice(0, self.n_act)]

    @property
    def wfs_slices(self) -> list[slice]:
        """Slices of the concatenated measurement vector belonging to each WFS."""
        return [slice(0, self.n_signal)]

    # ------------------------------------------------------------------ turbulence
    @abstractmethod
    def new_turbulence(self, seed: int) -> None:
        """Draw new phase screens (deterministic given `seed`)."""

    @abstractmethod
    def advance_turbulence(self, disturbance: np.ndarray | None = None, mode: str = "add") -> None:
        """Advance the atmosphere one sampling period. `disturbance` is an OPD map [m]."""

    # ------------------------------------------------------------------ actuation / sensing
    @abstractmethod
    def set_dm(self, coefs: np.ndarray | float) -> None: ...

    @abstractmethod
    def propagate(self) -> np.ndarray:
        """Propagate all guide stars through atmosphere, telescope, DMs and WFSs.

        Returns the concatenated WFS measurement. After this call `residual_opd` reflects the
        current DM shape.
        """

    @abstractmethod
    def turbulence_opd(self, direction: int = 0) -> np.ndarray:
        """Current turbulent OPD (no DM) on the valid pupil pixels [m]."""

    @abstractmethod
    def residual_opd(self, direction: int = 0) -> np.ndarray:
        """Residual OPD (turbulence + DM) after the last `propagate` [m]."""

    @abstractmethod
    def dm_opd(self, coefs: np.ndarray, direction: int = 0) -> np.ndarray:
        """OPD produced by the DMs alone for command `coefs`, seen from `direction` [m]."""

    def propagate_turbulence_only(self, direction: int = 0) -> np.ndarray:
        """Fast path for open-loop identification: turbulence OPD without DM/WFS."""
        return self.turbulence_opd(direction)

    # ------------------------------------------------------------------ calibration
    @abstractmethod
    def modal_basis(self, n_modes: int) -> np.ndarray:
        """Modes-to-commands matrix M2C (n_act x n_modes), ordered low -> high spatial frequency."""

    @abstractmethod
    def interaction_matrix(self, M2C: np.ndarray, stroke: float) -> InteractionData: ...

    # ------------------------------------------------------------------ conditions
    @abstractmethod
    def get_conditions(self) -> Conditions: ...

    @abstractmethod
    def set_conditions(self, cond: Conditions) -> None: ...

    # ------------------------------------------------------------------ visualisation (optional)
    def maps(self) -> dict[str, np.ndarray]:
        """2-D maps of the last step: 'turb', 'res' (OPD [m]) and 'sig' (WFS signal image)."""
        return {}

    def wfs_frame(self) -> np.ndarray | None:
        return None

    def psf(self, zero_padding: int = 2) -> np.ndarray | None:
        return None

    def seeing_arcsec(self) -> float | None:
        return None

    def summary(self) -> dict[str, Any]:
        return dict(
            description=self.description,
            n_act=self.n_act,
            n_signal=self.n_signal,
            n_pupil_pixels=self.n_pupil_pixels,
            wavelength_nm=self.wavelength * 1e9,
            seeing_arcsec=self.seeing_arcsec(),
        )
