"""Experiment base class and registry.

An experiment is a class with a `name` and a `run()` method that composes library components
(through the shared `ExperimentContext`) and reports through `self.figure / self.table / self.metrics`.
Those helpers both keep the objects in the returned `ExperimentResult` (for notebooks) and log
them (local files + W&B).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar

import matplotlib.pyplot as plt
import pandas as pd

from aoctrl.experiments.context import ExperimentContext
from aoctrl.tracking import ExperimentLogger

log = logging.getLogger(__name__)

REGISTRY: dict[str, type["Experiment"]] = {}


def register(cls: type["Experiment"]) -> type["Experiment"]:
    if cls.name in REGISTRY:
        raise ValueError(f"duplicate experiment name {cls.name!r}")
    REGISTRY[cls.name] = cls
    return cls


def get_experiment(name: str) -> type["Experiment"]:
    import aoctrl.experiments  # noqa: F401  (populates the registry)

    if name not in REGISTRY:
        raise KeyError(f"unknown experiment {name!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[name]


@dataclass
class ExperimentResult:
    name: str
    summary: pd.DataFrame | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    figures: dict[str, Any] = field(default_factory=dict)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    output_dir: str | None = None


class Experiment:
    name: ClassVar[str] = "base"
    description: ClassVar[str] = ""

    def __init__(self, cfg: Any, ctx: ExperimentContext, logger: ExperimentLogger):
        self.cfg = cfg
        self.ecfg = cfg.experiment
        self.ctx = ctx
        self.logger = logger
        self.result = ExperimentResult(name=self.name, output_dir=str(logger.dir))
        self.close_figures = bool(cfg.get("close_figures", True))

    # ------------------------------------------------------------------ reporting helpers
    def figure(self, name: str, fig) -> None:
        self.logger.log_figure(name, fig)
        self.result.figures[name] = fig
        if self.close_figures:
            plt.close(fig)

    def table(self, name: str, df: pd.DataFrame) -> None:
        self.logger.log_table(name, df)
        self.result.tables[name] = df

    def metrics(self, values: dict[str, Any], summary: bool = True) -> None:
        self.logger.log_metrics(values)
        if summary:
            self.logger.summary(values)
        self.result.metrics.update(values)

    def info(self, msg: str, *args) -> None:
        log.info(msg, *args)

    def video(self, name: str, path) -> None:
        self.logger.log_video(name, path, fps=int(self.cfg.video.fps))
        self.result.artifacts.setdefault("videos", {})[name] = str(path)

    def comparison_video(self, factories: dict, tag: str = "comparison") -> None:
        """Residual maps of several controllers on the same turbulence (first regime / delay)."""
        from aoctrl.simulation import simulate
        from aoctrl.viz import render_comparison_video

        vc = self.cfg.video
        if not bool(vc.enabled):
            return
        ctx = self.ctx
        rg = str(vc.get("regime") or next(iter(ctx.regimes)))
        dl = int(vc.get("delay") if vc.get("delay") is not None else self.cfg.eval.delays[0])
        with ctx.plant.conditions(delay=dl, **ctx.regimes[rg]):
            runs = {
                nm: simulate(
                    ctx.plant,
                    int(vc.n_steps),
                    f(dl, rg),
                    seed=int(vc.seed),
                    projector=ctx.calibration.Bpinv,
                    record_every=int(vc.record_every),
                    keep_y=False,
                )
                for nm, f in factories.items()
            }
        path = render_comparison_video(
            runs,
            self.logger.dir / "videos" / f"{tag}.{vc.format}",
            pupil=ctx.backend.pupil,
            fps=int(vc.fps),
            ts_ms=ctx.ts_ms,
            title=f"{self.name}: regime {rg}, delay {dl}",
        )
        self.video(tag, path)

    # ------------------------------------------------------------------ main
    def run(self) -> ExperimentResult:  # pragma: no cover
        raise NotImplementedError
