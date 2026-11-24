"""Closed-loop simulation (notebook `run()` and `run_truth()` merged).

    for k in range(n_steps):
        events[k](plant)                     # optional: change r0, noise, ... at step k
        u = controller(u, y, k)              # once a measurement exists
        y, info = plant.step(u, disturbance(k))
        d, s = Bpinv·turbulence, Bpinv·residual   (if a projector is given: modal truth)
        controller.observe(d)                # oracles only (needs_truth)

Telemetry holds everything an analysis needs; controller diagnostics (state estimates, NIS,
compute time, ...) are attached under `diag`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np

from aoctrl.control.base import Controller
from aoctrl.plant import AOPlant


@dataclass
class Telemetry:
    rms: np.ndarray  # (T,) residual RMS [m]
    strehl: np.ndarray  # (T,)
    u: np.ndarray  # (T, n_act) applied commands
    y: np.ndarray | None  # (T, n_signal)
    d: np.ndarray | None = None  # (T, n_modes) modal turbulence (truth)
    s: np.ndarray | None = None  # (T, n_modes) modal residual (truth)
    diag: dict = field(default_factory=dict)
    psf: np.ndarray | None = None  # long-exposure PSF
    frames: dict | None = None  # recorded maps for videos
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.rms)

    def drop_heavy(self) -> "Telemetry":
        """Copy without per-step measurements / frames (for caching large sweeps)."""
        return Telemetry(self.rms, self.strehl, self.u, None, self.d, self.s, self.diag, self.psf, None, self.meta)


def simulate(
    plant: AOPlant,
    n_steps: int,
    controller: Controller | None = None,
    *,
    seed: int | None = None,
    projector: np.ndarray | None = None,
    events: Mapping[int, Callable[[AOPlant], None]] | None = None,
    disturbance: Callable[[int], np.ndarray | None] | None = None,
    dist_mode: str = "add",
    record_every: int | None = None,
    record_psf: bool = False,
    psf_from: int | None = None,
    psf_crop: int = 40,
    keep_y: bool = True,
) -> Telemetry:
    if seed is not None:
        plant.new_turbulence(seed)
    plant.reset()
    if controller is not None:
        controller.reset()
    plant.record_maps = bool(record_every)
    frames = dict(turb=[], res=[], sig=[], k=[], psf=[], psf_peak=[]) if record_every else None
    u = np.zeros(plant.n_act)
    y = None
    rms, strehl, us, ys, ds, ss = [], [], [], [], [], []
    psf_acc, n_psf = None, 0
    for k in range(n_steps):
        if events and k in events:
            events[k](plant)
        if controller is not None and y is not None:
            u = controller(u, y, k)
        dk = disturbance(k) if disturbance is not None else None
        y, info = plant.step(u, dk, dist_mode)
        rms.append(info.rms)
        strehl.append(info.strehl)
        us.append(plant.u_applied.copy())
        if keep_y:
            ys.append(y)
        if projector is not None:
            d = projector @ plant.turbulence_opd()
            ds.append(d)
            ss.append(projector @ plant.residual_opd())
            if controller is not None and controller.needs_truth:
                controller.observe(d)
        if record_every and k % record_every == 0 and plant.last:
            for key in ("turb", "res", "sig"):
                frames[key].append(np.asarray(plant.last[key], dtype=np.float32))
            frames["k"].append(k)
            if record_psf:
                p = plant.backend.psf()
                if p is not None:
                    p = p / p.sum()
                    cy, cx = p.shape[0] // 2, p.shape[1] // 2
                    hw = min(psf_crop, cy, cx)
                    frames["psf"].append(p[cy - hw : cy + hw, cx - hw : cx + hw].astype(np.float32))
                    frames["psf_peak"].append(float(p.max()))
        if psf_from is not None and k >= psf_from:
            p = plant.backend.psf()
            psf_acc = p if psf_acc is None else psf_acc + p
            n_psf += 1
    plant.record_maps = False
    return Telemetry(
        rms=np.array(rms),
        strehl=np.array(strehl),
        u=np.array(us),
        y=np.array(ys) if keep_y else None,
        d=np.array(ds) if projector is not None else None,
        s=np.array(ss) if projector is not None else None,
        diag=controller.diagnostics() if controller is not None else {},
        psf=None if psf_acc is None else psf_acc / n_psf,
        frames=frames,
        meta=dict(seed=seed, controller=getattr(controller, "name", "open loop"), delay=plant.delay),
    )
