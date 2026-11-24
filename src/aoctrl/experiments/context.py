"""Shared, lazily-built and disk-cached artifacts for experiments.

In the notebook, Experiment 4 asserts that Experiments 1–3 already ran in the same kernel because it
reuses their globals. Here every experiment asks the context for what it needs; the context builds
it on first use and caches it on disk keyed by the relevant config, so:

* any experiment runs standalone (`python -m aoctrl.cli experiment=exp4_deep_koopman`), and
* nothing expensive is recomputed across experiments or sessions (calibration, turbulence series,
  reference runs, identification data). Trained models are handled by `training.CheckpointManager`.
"""

from __future__ import annotations

import logging
from functools import cached_property
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from aoctrl.calibration import Calibration, calibrate, check_dm_sign
from aoctrl.control import Integrator, OracleLQR
from aoctrl.control.latent import ModalIO
from aoctrl.dynamics import ARModel, TurbulenceSeries, collect_turbulence, identify_ar, persistence
from aoctrl.evaluation import OPEN_LOOP, ResultSet, evaluate_controllers
from aoctrl.models.data import WindowDataset, latent_metrics, make_windows, normalization_sigma
from aoctrl.optics import OpticalBackend, build_backend
from aoctrl.plant import AOPlant
from aoctrl.simulation import collect_excitation, measure_noise_floor, simulate
from aoctrl.utils import ArtifactCache, config_hash, resolve_device, resolve_dtype, resolve_path

log = logging.getLogger(__name__)

# Bump when the code that produces an artifact changes (invalidates its cache entries).
CACHE_VERSIONS = dict(
    calibration="1", turbulence="1", dm_sign="1", references="1", noise_floor="1", og_data="1", latent_runs="1"
)


class ExperimentContext:
    def __init__(self, cfg: Any, logger=None):
        self.cfg = cfg
        self.logger = logger
        self.cache = ArtifactCache(resolve_path(cfg.paths.cache_dir), enabled=bool(cfg.cache.enabled))

    # ------------------------------------------------------------------ config keys
    @cached_property
    def system_key(self) -> dict:
        sysc = OmegaConf.to_container(self.cfg.system, resolve=True)
        sysc.pop("n_threads", None)
        return sysc

    @property
    def system_hash(self) -> str:
        return config_hash(self.system_key)

    def _key(self, *parts) -> list:
        return [
            self.system_key,
            *[OmegaConf.to_container(p, resolve=True) if OmegaConf.is_config(p) else p for p in parts],
        ]

    # ------------------------------------------------------------------ physical system
    @cached_property
    def backend(self) -> OpticalBackend:
        b = build_backend(self.cfg.system)
        log.info("backend: %s", b.description)
        return b

    @cached_property
    def plant(self) -> AOPlant:
        return AOPlant(self.backend)

    @cached_property
    def calibration(self) -> Calibration:
        cc = self.cfg.system.calibration
        n_modes = OmegaConf.to_container(cc.n_modes) if OmegaConf.is_config(cc.n_modes) else cc.n_modes
        return self.cache.get_or_compute(
            "calibration",
            self._key(),
            lambda: calibrate(self.backend, n_modes, float(cc.stroke)),
            CACHE_VERSIONS["calibration"],
        )

    @property
    def n_modes(self) -> int:
        return self.calibration.n_modes

    @property
    def ts_ms(self) -> float:
        return float(self.cfg.system.telescope.sampling_time) * 1e3

    # ------------------------------------------------------------------ operating regimes
    @cached_property
    def regimes(self) -> dict[str, dict]:
        """Physical conditions per regime (notebook `apply_conditions`)."""
        ev = self.cfg.eval
        mag = ev.get("magnitude")
        mag = float(self.cfg.system.guide_stars.stars[0].magnitude) if mag is None else float(mag)
        out = {}
        for name, rc in ev.regimes.items():
            d = dict(OmegaConf.to_container(rc, resolve=True))
            d.setdefault("magnitude", mag)
            d.setdefault("photon_noise", bool(ev.photon_noise))
            out[str(name)] = d
        return out

    @property
    def r0_ref(self) -> float:
        return float(self.cfg.system.atmosphere.r0)

    def q_scale(self, regime: str) -> float:
        """Turbulence variance scaling (r0_ref / r0)^(5/3) of a regime."""
        return (self.r0_ref / float(self.regimes[regime]["r0"])) ** (5.0 / 3.0)

    # ------------------------------------------------------------------ dynamics
    @cached_property
    def turbulence(self) -> TurbulenceSeries:
        tc = self.cfg.identification.turbulence
        return self.cache.get_or_compute(
            "turbulence",
            self._key(tc),
            lambda: collect_turbulence(self.backend, self.calibration, int(tc.n_steps), int(tc.seed)),
            CACHE_VERSIONS["turbulence"],
        )

    @cached_property
    def ar_models(self) -> dict[int, ARModel]:
        ic = self.cfg.identification
        X = self.turbulence.coefs
        return {
            int(p): identify_ar(X, int(p), float(ic.turbulence.train_frac), float(ic.ar.rmax)) for p in ic.ar.orders
        }

    @property
    def ar_max(self) -> int:
        return max(self.ar_models)

    @cached_property
    def dm_sign(self) -> dict:
        """Short integrator run checking residual − turbulence ≈ slope · command (notebook §6.2)."""
        dc = self.cfg.identification.dm_sign

        def compute():
            tel = simulate(
                self.plant, int(dc.n_steps), self.integrator(), seed=int(dc.seed), projector=self.calibration.Bpinv
            )
            return check_dm_sign(tel.d, tel.s, tel.u, self.calibration.Minv)

        res = self.cache.get_or_compute(
            "dm_sign", self._key(dc, self.cfg.control.integrator), compute, CACHE_VERSIONS["dm_sign"]
        )
        if not res["ok"]:
            log.warning(
                "DM sign check failed (slope %.3f, rel. error %.1e): the turbulence truth may not be the "
                "current OPD in this OOPAO version.",
                res["slope"],
                res["rel_error"],
            )
        return res

    @property
    def sign(self) -> float:
        return float(self.dm_sign["sign"])

    # ------------------------------------------------------------------ controllers
    def integrator(self) -> Integrator:
        ic = self.cfg.control.integrator
        return Integrator(self.calibration.R, gain=float(ic.gain), leak=float(ic.leak))

    @property
    def integrator_name(self) -> str:
        return self.integrator().name

    def oracle_name(self, p: int | None = None) -> str:
        return f"LQR oracle AR({p or self.ar_max})"

    def reference_factories(self) -> dict:
        calib, rho, Wn = self.calibration, float(self.cfg.control.lqr.rho), self.calibration.W_norm
        f = {OPEN_LOOP: lambda dl, rg: None, self.integrator_name: lambda dl, rg: self.integrator()}
        for p, ar in self.ar_models.items():
            f[self.oracle_name(p)] = lambda dl, rg, ar=ar, p=p: OracleLQR.from_ar(
                ar.coefs, calib.M2C, dl, rho, Wn, self.sign, name=self.oracle_name(p)
            )
        f["LQR oracle persist."] = lambda dl, rg: OracleLQR.from_ar(
            persistence(self.n_modes), calib.M2C, dl, rho, Wn, self.sign, name="LQR oracle persist.", use_dare=False
        )
        return f

    def evaluate(self, factories: dict, keep_y: bool = False) -> ResultSet:
        ev = self.cfg.eval
        return evaluate_controllers(
            self.plant,
            factories,
            self.regimes,
            list(ev.delays),
            list(ev.seeds),
            int(ev.n_steps),
            self.calibration.Bpinv,
            keep_y=keep_y,
        )

    @cached_property
    def reference_runs(self) -> ResultSet:
        """Open loop, integrator and oracles on the evaluation protocol (shared by experiments 1–4)."""
        key = self._key(self.cfg.eval, self.cfg.identification, self.cfg.control)
        return self.cache.get_or_compute(
            "references", key, lambda: self.evaluate(self.reference_factories()).light(), CACHE_VERSIONS["references"]
        )

    # ------------------------------------------------------------------ identification data (exp 2–4)
    @cached_property
    def noise_floor(self) -> np.ndarray:
        oc = self.cfg.identification.optical_gain
        first = next(iter(self.regimes.values()))
        cond = dict(r0=first["r0"], magnitude=first["magnitude"], photon_noise=first["photon_noise"], delay=0)
        return self.cache.get_or_compute(
            "noise_floor",
            self._key(oc.noise_floor_steps, cond),
            lambda: measure_noise_floor(self.plant, self.calibration, int(oc.noise_floor_steps), **cond),
            CACHE_VERSIONS["noise_floor"],
        )

    def _excitation_runs(self, name: str, n_steps: int, seed_offset: int, seg: int, dither: float) -> list[dict]:
        sigma = self.turbulence.coefs.std(axis=0)

        def compute():
            return [
                collect_excitation(
                    self.plant,
                    self.calibration,
                    n_steps,
                    seed_offset + i,
                    sigma,
                    seg=seg,
                    dither=dither,
                    delay=0,
                    **cond,
                )
                for i, cond in enumerate(self.regimes.values())
            ]

        key = self._key(self.regimes, n_steps, seed_offset, seg, dither, self.cfg.identification.turbulence)
        return self.cache.get_or_compute(name, key, compute, CACHE_VERSIONS.get(name, "1"))

    @cached_property
    def og_data(self) -> list[dict]:
        oc = self.cfg.identification.optical_gain
        return self._excitation_runs("og_data", int(oc.n_steps), int(oc.seed_offset), int(oc.seg), float(oc.dither))

    @cached_property
    def latent_runs(self) -> list[dict]:
        lc = self.cfg.identification.latent_data
        return self._excitation_runs("latent_runs", int(lc.n_steps), int(lc.seed_offset), int(lc.seg), float(lc.dither))

    @cached_property
    def latent_sigma(self) -> np.ndarray:
        return normalization_sigma(self.latent_runs, self.sign)

    @cached_property
    def latent_dataset(self) -> WindowDataset:
        lc = self.cfg.identification.latent_data
        return make_windows(self.latent_runs, self.latent_sigma, int(lc.horizon), int(lc.stride), float(lc.val_frac))

    def modal_io(self) -> ModalIO:
        _, WC = latent_metrics(self.latent_sigma, self.calibration.W_phys)
        lc = self.cfg.control.latent
        return ModalIO(
            self.calibration.Mrec,
            self.calibration.M2C,
            self.latent_sigma,
            self.sign,
            cmax=float(lc.cmax),
            Wc=WC,
            rho=float(self.cfg.control.lqr.rho),
        )

    def data_fingerprint(self) -> list:
        """What determines the latent training data (part of the checkpoint key)."""
        return self._key(
            self.regimes,
            self.cfg.identification.latent_data,
            self.cfg.identification.turbulence,
            self.cfg.identification.dm_sign,
        )

    # ------------------------------------------------------------------ compute
    @cached_property
    def device(self):
        return resolve_device(str(self.cfg.training.device), bool(self.cfg.training.allow_mps))

    @cached_property
    def dtype(self):
        return resolve_dtype(str(self.cfg.training.dtype), self.device)
