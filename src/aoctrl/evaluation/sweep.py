"""Evaluation protocol: every controller × regime × delay × seed, with paired seeds.

Controllers are given as *factories* `f(delay, regime) -> Controller | None` because several
controllers depend on the delay (LQR buffers) or on the regime (EKF noise scaling). `None` means
open loop.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from aoctrl.control.base import Controller
from aoctrl.metrics import compute_ms, estimation_error_nm, fitting_floor_nm, gap_closed, nis_mean, run_stats
from aoctrl.plant import AOPlant
from aoctrl.simulation.runner import Telemetry, simulate

log = logging.getLogger(__name__)

Factory = Callable[[int, str], "Controller | None"]
OPEN_LOOP = "open loop"


@dataclass
class ResultSet:
    """runs[(regime, delay)][controller_name] -> list of Telemetry (one per seed)."""

    runs: dict = field(default_factory=dict)

    def add(self, regime: str, delay: int, name: str, runs: list[Telemetry]) -> None:
        self.runs.setdefault((regime, int(delay)), {})[name] = runs

    def conditions(self) -> list[tuple[str, int]]:
        return list(self.runs)

    def at(self, regime: str, delay: int) -> dict[str, list[Telemetry]]:
        return self.runs[(regime, int(delay))]

    def merged(self, *others: "ResultSet") -> "ResultSet":
        out = ResultSet({k: dict(v) for k, v in self.runs.items()})
        for o in others:
            for k, v in o.runs.items():
                out.runs.setdefault(k, {}).update(v)
        return out

    def light(self) -> "ResultSet":
        return ResultSet({k: {n: [r.drop_heavy() for r in rs] for n, rs in v.items()} for k, v in self.runs.items()})


def evaluate_controllers(
    plant: AOPlant,
    factories: Mapping[str, Factory],
    regimes: Mapping[str, Mapping],
    delays: Iterable[int],
    seeds: Iterable[int],
    n_steps: int,
    projector: np.ndarray | None,
    keep_y: bool = False,
) -> ResultSet:
    rs = ResultSet()
    t0 = time.time()
    for rg, phys in regimes.items():
        for dl in delays:
            with plant.conditions(delay=int(dl), **dict(phys)):
                for name, make in factories.items():
                    ctrl = make(int(dl), rg)
                    runs = [
                        simulate(plant, n_steps, ctrl, seed=int(sd), projector=projector, keep_y=keep_y) for sd in seeds
                    ]
                    rs.add(rg, dl, name, runs)
            log.info("[%s, delay=%d] done (%.0f s)", rg, dl, time.time() - t0)
    return rs


def summarize(
    rs: ResultSet,
    W: np.ndarray,
    ref_low: str | None = None,
    ref_high: str | None = None,
    tags: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """One row per (regime, delay, controller): RMS ± std, Strehl, ‖u‖, % of the integrator→oracle gap
    retained, estimation error, NIS, compute cost; plus the fitting floor per condition."""
    rows = []
    for (rg, dl), by_name in rs.runs.items():
        S = {nm: run_stats(runs) for nm, runs in by_name.items()}
        floor = (
            fitting_floor_nm(by_name[OPEN_LOOP], W)
            if OPEN_LOOP in by_name and by_name[OPEN_LOOP][0].s is not None
            else np.nan
        )
        lo = S[ref_low]["rms"].mean() if ref_low in S else np.nan
        hi = S[ref_high]["rms"].mean() if ref_high in S else np.nan
        for nm, runs in by_name.items():
            s = S[nm]
            rows.append(
                dict(
                    regime=rg,
                    delay=dl,
                    controller=nm,
                    tag=(tags or {}).get(nm, ""),
                    rms=s["rms"].mean(),
                    rms_std=s["rms"].std(),
                    strehl=s["strehl"].mean(),
                    u=s["u"].mean(),
                    retained=np.nan if nm == OPEN_LOOP else gap_closed(s["rms"].mean(), lo, hi),
                    est_err=estimation_error_nm(runs, "est", W) if runs[0].d is not None else np.nan,
                    nis=nis_mean(runs),
                    t_ms=compute_ms(runs),
                    fitting_floor=floor,
                    n_seeds=len(runs),
                )
            )
    return pd.DataFrame(rows)


def format_table(
    df: pd.DataFrame, cols=("controller", "rms", "rms_std", "strehl", "u", "retained", "est_err", "nis", "t_ms")
) -> str:
    out = []
    for (rg, dl), g in df.groupby(["regime", "delay"], sort=False):
        floor = g["fitting_floor"].iloc[0]
        out.append(f"\n=== regime {rg} | DM delay {dl} frame(s) | fitting floor {floor:.0f} nm ===")
        out.append(g[list(c for c in cols if c in g)].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    return "\n".join(out)
