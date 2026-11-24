"""OOPAO implementation of `OpticalBackend` (SCAO and MCAO).

SCAO (1 guide star, 1 WFS, 1 ground DM, no extra science directions) reproduces the reference
notebook `OOPAO_init.ipynb` call-for-call:

    ngs ** atm * tel * dm * wfs        (propagation)
    atm.OPD                            (turbulence truth)
    dm.OPD                             (DM response, used for the modal projection B)
    compute_KL_basis / InteractionMatrix (calibration)

MCAO generalises this to N guide stars (one WFS each), M DMs (optionally conjugated at altitude)
and optional science directions. Each guide star is propagated through the shared atmosphere and
the full DM stack. The interaction matrix is assembled block-wise (WFS i x DM j).

NOTE: OOPAO's support for altitude-conjugated DMs and off-axis sources depends on its version.
The MCAO path is written against the current public API but is NOT validated yet (see TODO.md,
stage 4). Constructor keyword arguments unknown to the installed OOPAO version are dropped with a
warning (see `_construct`) so that version drift shows up as a log line, not a crash.
"""

from __future__ import annotations

import inspect
import logging
import os
from typing import Any

import numpy as np

from aoctrl.optics.base import Conditions, InteractionData, OpticalBackend

log = logging.getLogger(__name__)


def _require_oopao():
    try:
        import OOPAO  # noqa: F401
    except ImportError as e:  # pragma: no cover - depends on environment
        raise ImportError(
            "OOPAO is not installed. Install it with `pip install -e '.[oopao]'` "
            "(or `pip install git+https://github.com/cheritier/OOPAO.git`), "
            "or run with `system=synthetic` for the fast surrogate backend."
        ) from e


def _construct(cls, **kwargs):
    """Instantiate an OOPAO class, dropping kwargs its signature does not accept."""
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):  # pragma: no cover
        return cls(**kwargs)
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return cls(**kwargs)
    accepted = {k: v for k, v in kwargs.items() if k in params}
    dropped = sorted(set(kwargs) - set(accepted))
    if dropped:
        log.warning("%s does not accept %s in this OOPAO version; ignored.", cls.__name__, dropped)
    return cls(**accepted)


class OOPAOBackend(OpticalBackend):
    def __init__(self, cfg: Any):
        _require_oopao()
        from OOPAO.Atmosphere import Atmosphere
        from OOPAO.DeformableMirror import DeformableMirror
        from OOPAO.Source import Source
        from OOPAO.Telescope import Telescope

        self.cfg = cfg
        n_cpu = max(1, min(os.cpu_count() or 1, int(cfg.get("n_threads", 8))))
        tc, ac = cfg.telescope, cfg.atmosphere
        stars = list(cfg.guide_stars.stars)
        sci_dirs = list(cfg.get("science", {}).get("directions", []) or [])

        # Off-axis sources need a field of view large enough for the atmosphere screens.
        max_off = max([float(s.coordinates[0]) for s in stars] + [float(d[0]) for d in sci_dirs] + [0.0])
        fov = float(tc.get("fov_arcsec", 0.0) or 0.0) or (2.2 * max_off if max_off > 0 else 0.0)
        tel_kw = dict(
            resolution=tc.resolution,
            diameter=tc.diameter,
            samplingTime=tc.sampling_time,
            centralObstruction=tc.central_obstruction,
        )
        if fov > 0:
            tel_kw["fov"] = fov
        self.tel = _construct(Telescope, **tel_kw)

        def make_source(band, mag, coords):
            kw = dict(optBand=band, magnitude=mag)
            if coords is not None and float(coords[0]) != 0.0:
                kw["coordinates"] = [float(coords[0]), float(coords[1])]
            return _construct(Source, **kw)

        self.gs = [make_source(s.band, s.magnitude, s.get("coordinates")) for s in stars]
        sci_band = cfg.get("science", {}).get("band", stars[0].band)
        self.sci = [make_source(sci_band, stars[0].magnitude, d) for d in sci_dirs]
        self.gs[0] ** self.tel

        layers = list(ac.layers)
        atm_kw = dict(
            telescope=self.tel,
            r0=ac.r0,
            L0=ac.L0,
            windSpeed=[float(layer.wind_speed) for layer in layers],
            fractionalR0=[float(layer.fractional_r0) for layer in layers],
            windDirection=[float(layer.wind_direction) for layer in layers],
            altitude=[float(layer.altitude) for layer in layers],
        )
        if len(self.gs) == 1:
            atm_kw["src"] = self.gs[0]
        self.atm = _construct(Atmosphere, **atm_kw)
        if not hasattr(self.atm, "layer_1"):  # constructor does not create the layers
            try:
                self.atm.initializeAtmosphere(self.tel)
            except TypeError:
                self.atm.initializeAtmosphere()

        self.dms = []
        for m in cfg.dm.mirrors:
            kw = dict(telescope=self.tel, nSubap=m.n_subap, mechCoupling=m.mech_coupling, nJobs=n_cpu, nThreads=n_cpu)
            if float(m.get("altitude", 0.0) or 0.0) != 0.0:
                kw["altitude"] = float(m.altitude)
            self.dms.append(_construct(DeformableMirror, **kw))

        self.wfss = [self._make_wfs(cfg.wfs) for _ in self.gs]
        for gs, wfs in zip(self.gs, self.wfss):
            gs**self.tel * wfs  # first propagation initialises the WFS reference

        self._pupil = self.tel.pupil.astype(bool)
        self._dm_sl = _slices([dm.nValidAct for dm in self.dms])
        self._wfs_sl = _slices([int(w.nSignal) for w in self.wfss])
        # SCAO fast path identical to the notebook: truth = atm.OPD, evaluate on the guide star
        self._scao_like = len(self.gs) == 1 and len(self.sci) == 0 and len(self.dms) == 1
        self._turb: list[np.ndarray | None] = [None] * self.n_directions
        self._res: list[np.ndarray | None] = [None] * self.n_directions
        self.description = (
            f"{'SCAO' if self._scao_like else 'MCAO'}: {len(self.gs)} GS / {len(self.wfss)} "
            f"{cfg.wfs.type} / {len(self.dms)} DM / {len(layers)} layer(s)"
        )

    # ------------------------------------------------------------------ construction helpers
    def _make_wfs(self, wc):
        if wc.type == "pyramid":
            from OOPAO.Pyramid import Pyramid

            wfs = _construct(
                Pyramid, nSubap=wc.n_subap, telescope=self.tel, modulation=wc.modulation, lightRatio=wc.light_ratio
            )
        elif wc.type in ("shack_hartmann", "sh"):
            from OOPAO.ShackHartmann import ShackHartmann

            wfs = _construct(ShackHartmann, nSubap=wc.n_subap, telescope=self.tel, lightRatio=wc.light_ratio)
        else:
            raise ValueError(f"Unknown WFS type {wc.type!r}")
        if hasattr(wfs, "cam"):
            wfs.cam.photonNoise = bool(wc.get("photon_noise", False))
            wfs.cam.readoutNoise = float(wc.get("readout_noise", 0.0))
        return wfs

    # ------------------------------------------------------------------ sizes
    @property
    def n_act(self) -> int:
        return int(sum(dm.nValidAct for dm in self.dms))

    @property
    def n_signal(self) -> int:
        return int(sum(w.nSignal for w in self.wfss))

    @property
    def n_pupil_pixels(self) -> int:
        return int(self._pupil.sum())

    @property
    def pupil(self) -> np.ndarray:
        return self._pupil

    @property
    def wavelength(self) -> float:
        return float((self.sci[0] if self.sci else self.gs[0]).wavelength)

    @property
    def sampling_time(self) -> float:
        return float(self.tel.samplingTime)

    @property
    def n_directions(self) -> int:
        return max(1, len(self.sci))

    @property
    def dm_slices(self) -> list[slice]:
        return self._dm_sl

    @property
    def wfs_slices(self) -> list[slice]:
        return self._wfs_sl

    # ------------------------------------------------------------------ turbulence
    def new_turbulence(self, seed: int) -> None:
        self.atm.generateNewPhaseScreen(seed=int(seed))

    def advance_turbulence(self, disturbance=None, mode: str = "add") -> None:
        if disturbance is None:
            self.atm.update()
        elif mode == "add":
            self.atm.update()
            self.atm.update(OPD=np.array(self.atm.OPD) + disturbance)
        elif mode == "replace":
            self.atm.update(OPD=disturbance)
        else:
            raise ValueError(f"unknown disturbance mode {mode!r}")

    # ------------------------------------------------------------------ actuation / sensing
    def set_dm(self, coefs) -> None:
        if np.isscalar(coefs):
            for dm in self.dms:
                dm.coefs = coefs
            return
        coefs = np.asarray(coefs, dtype=float)
        for dm, sl in zip(self.dms, self._dm_sl):
            dm.coefs = coefs[sl]

    def _through_dms(self, src):
        for dm in self.dms:
            src * dm
        return src

    def propagate(self) -> np.ndarray:
        if self._scao_like:
            gs, wfs = self.gs[0], self.wfss[0]
            gs**self.atm * self.tel * self.dms[0] * wfs
            self._res[0] = None  # read lazily from gs.OPD
            return np.array(wfs.signal)
        sigs = []
        for i, (gs, wfs) in enumerate(zip(self.gs, self.wfss)):
            gs**self.atm * self.tel
            if not self.sci and i == 0:
                self._turb[0] = np.array(gs.OPD)[self._pupil]
            self._through_dms(gs)
            gs * wfs
            sigs.append(np.array(wfs.signal))
            if not self.sci and i == 0:
                self._res[0] = np.array(gs.OPD)[self._pupil]
        for j, sci in enumerate(self.sci):
            sci**self.atm * self.tel
            self._turb[j] = np.array(sci.OPD)[self._pupil]
            self._through_dms(sci)
            self._res[j] = np.array(sci.OPD)[self._pupil]
        return np.concatenate(sigs)

    def turbulence_opd(self, direction: int = 0) -> np.ndarray:
        if self._scao_like:
            return np.array(self.atm.OPD)[self._pupil]
        if self._turb[direction] is None:
            self.propagate_turbulence_only(direction)
        return self._turb[direction]

    def residual_opd(self, direction: int = 0) -> np.ndarray:
        if self._scao_like:
            return np.array(self.gs[0].OPD)[self._pupil]
        return self._res[direction]

    def propagate_turbulence_only(self, direction: int = 0) -> np.ndarray:
        src = self.sci[direction] if self.sci else self.gs[0]
        src**self.atm * self.tel
        opd = np.array(src.OPD)[self._pupil]
        if not self._scao_like:
            self._turb[direction] = opd
        return opd

    def dm_opd(self, coefs, direction: int = 0) -> np.ndarray:
        prev = [np.array(dm.coefs, dtype=float).copy() for dm in self.dms]
        self.set_dm(coefs)
        if len(self.dms) == 1 and (not self.sci or direction == 0) and self._scao_like:
            opd = np.array(self.dms[0].OPD)[self._pupil]
        else:
            src = self.sci[direction] if self.sci else self.gs[0]
            src**self.tel
            self._through_dms(src)
            opd = np.array(src.OPD)[self._pupil]
        for dm, c in zip(self.dms, prev):
            dm.coefs = c
        return opd

    # ------------------------------------------------------------------ calibration
    def modal_basis(self, n_modes) -> np.ndarray:
        from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis

        per_dm = list(n_modes) if isinstance(n_modes, (list, tuple)) or hasattr(n_modes, "__len__") else [n_modes]
        if len(per_dm) == 1 and len(self.dms) > 1:
            per_dm = per_dm * len(self.dms)
        basis = str(self.cfg.calibration.get("basis", "kl"))
        # OOPAO's KL basis assumes the DM footprint equals the pupil: not true for altitude DMs.
        basis_alt = str(self.cfg.calibration.get("basis_altitude_dm", "zonal"))
        blocks = []
        for dm, nm, mc in zip(self.dms, per_dm, self.cfg.dm.mirrors):
            b = basis_alt if float(mc.get("altitude", 0.0) or 0.0) != 0.0 else basis
            if b == "kl":
                full = compute_KL_basis(self.tel, self.atm, dm)
            elif b == "zonal":
                full = np.eye(dm.nValidAct)
            else:
                raise ValueError(f"unknown modal basis {b!r}")
            blocks.append(full[:, : min(int(nm), full.shape[1])])
        return _block_diag(blocks)

    def interaction_matrix(self, M2C: np.ndarray, stroke: float) -> InteractionData:
        from OOPAO.calibration.InteractionMatrix import InteractionMatrix

        self.set_dm(0)
        if self._scao_like:
            self.gs[0] ** self.tel * self.wfss[0]
            calib = InteractionMatrix(
                ngs=self.gs[0], tel=self.tel, dm=self.dms[0], wfs=self.wfss[0], M2C=M2C, stroke=stroke, noise="off"
            )
            self.set_dm(0)
            return InteractionData(
                D=np.array(calib.D), M=np.array(calib.M), singular_values=np.array(calib.s), cond=float(calib.cond)
            )
        # MCAO: block (WFS i, DM j). Columns of M2C belonging to DM j are those non-zero in its rows.
        D = np.zeros((self.n_signal, M2C.shape[1]))
        for j, (dm, dsl) in enumerate(zip(self.dms, self._dm_sl)):
            cols = np.where(np.abs(M2C[dsl]).sum(axis=0) > 0)[0]
            for i, (gs, wfs, wsl) in enumerate(zip(self.gs, self.wfss, self._wfs_sl)):
                gs**self.tel * wfs
                calib = InteractionMatrix(
                    ngs=gs, tel=self.tel, dm=dm, wfs=wfs, M2C=M2C[dsl][:, cols], stroke=stroke, noise="off"
                )
                D[wsl, cols] = np.array(calib.D)
        self.set_dm(0)
        U, s, Vt = np.linalg.svd(D, full_matrices=False)
        rcond = float(self.cfg.calibration.get("rcond", 1e-3))
        keep = s > rcond * s[0]
        M = (Vt[keep].T / s[keep]) @ U[:, keep].T
        return InteractionData(D=D, M=M, singular_values=s, cond=float(s[0] / s[keep][-1]))

    # ------------------------------------------------------------------ conditions
    def get_conditions(self) -> Conditions:
        cam = getattr(self.wfss[0], "cam", None)
        return Conditions(
            r0=float(self.atm.r0),
            magnitude=float(self.gs[0].magnitude),
            photon_noise=bool(getattr(cam, "photonNoise", False)),
            readout_noise=float(getattr(cam, "readoutNoise", 0.0)),
            wind_speed=None,
        )

    def set_conditions(self, cond: Conditions) -> None:
        if cond.r0 is not None:
            self.atm.r0 = cond.r0
        if cond.magnitude is not None:
            for gs in self.gs:
                gs.magnitude = cond.magnitude
        for wfs in self.wfss:
            if hasattr(wfs, "cam"):
                if cond.photon_noise is not None:
                    wfs.cam.photonNoise = cond.photon_noise
                if cond.readout_noise is not None:
                    wfs.cam.readoutNoise = cond.readout_noise
        if cond.wind_speed is not None:
            try:
                self.atm.windSpeed = list(cond.wind_speed)
            except Exception as e:  # pragma: no cover
                log.warning("could not change wind speed at run time: %r", e)

    # ------------------------------------------------------------------ visualisation
    def maps(self) -> dict[str, np.ndarray]:
        try:
            turb = np.array(self.atm.OPD) if self._scao_like else _unflatten(self._turb[0], self._pupil)
        except Exception:
            turb = np.zeros(self._pupil.shape)
        res = np.array(self.gs[0].OPD) if self._scao_like else _unflatten(self._res[0], self._pupil)
        return dict(turb=turb, res=res, sig=np.array(self.wfss[0].signal_2D))

    def wfs_frame(self):
        return np.array(self.wfss[0].cam.frame)

    def psf(self, zero_padding: int = 2):
        self.tel.computePSF(zeroPaddingFactor=zero_padding)
        return np.array(self.tel.PSF, dtype=float)

    def seeing_arcsec(self):
        return float(self.atm.seeingArcsec)


def _slices(sizes: list[int]) -> list[slice]:
    out, start = [], 0
    for n in sizes:
        out.append(slice(start, start + int(n)))
        start += int(n)
    return out


def _block_diag(blocks: list[np.ndarray]) -> np.ndarray:
    rows = sum(b.shape[0] for b in blocks)
    cols = sum(b.shape[1] for b in blocks)
    out = np.zeros((rows, cols))
    r = c = 0
    for b in blocks:
        out[r : r + b.shape[0], c : c + b.shape[1]] = b
        r += b.shape[0]
        c += b.shape[1]
    return out


def _unflatten(v, pupil):
    out = np.zeros(pupil.shape)
    if v is not None:
        out[pupil] = v
    return out
