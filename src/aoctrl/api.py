"""Programmatic entry points shared by the CLI and notebooks.

from aoctrl.api import load_config, run_experiment, build_context

cfg = load_config(["experiment=exp1_oracle_lqr", "system=synthetic", "eval=quick"])
res = run_experiment(cfg)                # same code path as the CLI
res.summary, res.figures["rms_bars"]     # inspect in the notebook

ctx = build_context(cfg)                 # or explore components interactively
ctx.plant, ctx.calibration, ctx.ar_models
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf, open_dict

from aoctrl.utils import seed_everything
from aoctrl.utils.cache import config_hash
from aoctrl.utils.paths import config_dir, resolve_path

log = logging.getLogger(__name__)


def load_config(overrides: Sequence[str] = (), config_name: str = "config") -> Any:
    """Compose the Hydra config without changing the working directory (notebook-friendly)."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir()), version_base="1.3"):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    return cfg


def _default_output_dir(cfg) -> Path:
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    return resolve_path("outputs") / str(cfg.experiment.name) / stamp


def build_context(cfg: Any, logger=None):
    from aoctrl.experiments.context import ExperimentContext

    return ExperimentContext(cfg, logger)


def setup_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level, format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s", datefmt="%H:%M:%S"
        )
    root.setLevel(level)


def run_experiment(cfg: Any, ctx=None, output_dir: str | Path | None = None):
    """Run the experiment named by `cfg.experiment.name`; returns an `ExperimentResult`."""
    from aoctrl.experiments.base import get_experiment
    from aoctrl.experiments.context import ExperimentContext
    from aoctrl.tracking import ExperimentLogger

    setup_logging(str(cfg.logging.level))
    seed_everything(int(cfg.seed))
    out = (
        Path(output_dir)
        if output_dir
        else (Path(cfg.paths.output_dir) if cfg.paths.get("output_dir") else _default_output_dir(cfg))
    )
    with open_dict(cfg):
        cfg.paths.output_dir = str(out)
    name = str(cfg.experiment.name)
    run_name = f"{name}-{cfg.system.name}-{time.strftime('%m%d-%H%M%S')}"
    logger = ExperimentLogger(
        out, cfg, cfg.logging.wandb, run_name=run_name, save_figures=bool(cfg.logging.save_figures)
    )
    if ctx is not None:
        same = lambda k: (
            config_hash(OmegaConf.to_container(ctx.cfg[k], resolve=True))
            == config_hash(  # noqa: E731
                OmegaConf.to_container(cfg[k], resolve=True)
            )
        )
        if not all(same(k) for k in ("system", "eval", "identification", "control", "training", "paths", "cache")):
            log.warning("the given context was built from a different config; building a new one")
            ctx = None
    ctx = ctx or ExperimentContext(cfg, logger)
    ctx.logger = logger
    exp_cls = get_experiment(name)
    log.info("experiment %s | system %s | output %s", name, cfg.system.name, out)
    try:
        result = exp_cls(cfg, ctx, logger).run()
        if result.summary is not None:
            res_dir = resolve_path(cfg.paths.results_dir) / name / ctx.system_hash
            res_dir.mkdir(parents=True, exist_ok=True)
            result.summary.to_csv(res_dir / "summary.csv", index=False)
            (res_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
            logger.log_artifact(res_dir / "summary.csv", name=f"{name}-summary", type="results")
            log.info("summary saved to %s", res_dir / "summary.csv")
        result.artifacts.setdefault("context", ctx)
        result.metrics.setdefault("config_hash", config_hash(OmegaConf.to_container(cfg, resolve=True)))
        return result
    finally:
        logger.finish()
