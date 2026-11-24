"""Experiment logging: always to the local run directory, optionally to Weights & Biases.

Local layout (`<output_dir>/`):
    config.yaml        resolved config
    metrics.jsonl      every `log_metrics` call
    figures/*.png      every figure
    tables/*.csv       every table
    videos/*           videos
    summary.json       final scalar summary

W&B mirrors the same calls (config, metrics, figures as images, tables, videos, artifacts) when
`logging.wandb.enabled=true` and `wandb` is importable; otherwise it is skipped with a warning.
`mode: offline` logs locally for a later `wandb sync`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)


def _jsonable(v):
    try:
        json.dumps(v)
        return v
    except TypeError:
        if hasattr(v, "item"):
            return v.item()
        return str(v)


class ExperimentLogger:
    def __init__(
        self,
        output_dir: str | Path,
        config: Any = None,
        wandb_cfg: Any = None,
        run_name: str | None = None,
        save_figures: bool = True,
    ):
        self.dir = Path(output_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.save_figures = save_figures
        self._step = 0
        self._summary: dict[str, Any] = {}
        self.wandb = None
        plain_cfg = None
        if config is not None:
            from omegaconf import OmegaConf

            (self.dir / "config.yaml").write_text(OmegaConf.to_yaml(config, resolve=True))
            plain_cfg = OmegaConf.to_container(config, resolve=True)
        if wandb_cfg is not None and bool(wandb_cfg.get("enabled", False)):
            try:
                import wandb

                self.wandb = wandb.init(
                    project=wandb_cfg.get("project", "aoctrl"),
                    entity=wandb_cfg.get("entity") or None,
                    group=wandb_cfg.get("group") or None,
                    tags=list(wandb_cfg.get("tags") or []),
                    mode=wandb_cfg.get("mode", "online"),
                    name=run_name,
                    config=plain_cfg,
                    dir=str(self.dir),
                    reinit=True,
                )
            except Exception as e:  # noqa: BLE001 - never crash an experiment because of logging
                log.warning("W&B disabled (%s). Logging locally only.", e)
                self.wandb = None

    # ------------------------------------------------------------------ scalars
    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self._step = self._step + 1 if step is None else int(step)
        rec = {k: _jsonable(v) for k, v in metrics.items()}
        with open(self.dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps({"step": self._step, **rec}) + "\n")
        if self.wandb is not None:
            self.wandb.log(rec)

    def summary(self, values: dict[str, Any]) -> None:
        self._summary.update({k: _jsonable(v) for k, v in values.items()})
        if self.wandb is not None:
            self.wandb.summary.update(self._summary)

    # ------------------------------------------------------------------ rich media
    def log_figure(self, name: str, fig) -> Path | None:
        path = None
        if self.save_figures:
            path = self.dir / "figures" / f"{name}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, bbox_inches="tight", dpi=110)
        if self.wandb is not None:
            import wandb

            self.wandb.log({f"figures/{name}": wandb.Image(fig)})
        return path

    def log_table(self, name: str, df: pd.DataFrame) -> Path:
        path = self.dir / "tables" / f"{name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        if self.wandb is not None:
            import wandb

            self.wandb.log({f"tables/{name}": wandb.Table(dataframe=df.reset_index(drop=True))})
        return path

    def log_video(self, name: str, path: str | Path, fps: int = 15) -> None:
        if self.wandb is not None:
            import wandb

            self.wandb.log({f"videos/{name}": wandb.Video(str(path), fps=fps)})

    def log_artifact(self, path: str | Path, name: str, type: str = "result") -> None:
        if self.wandb is not None:
            import wandb

            art = wandb.Artifact(name.replace("/", "-"), type=type)
            art.add_file(str(path))
            self.wandb.log_artifact(art)

    # ------------------------------------------------------------------ lifecycle
    def finish(self) -> None:
        (self.dir / "summary.json").write_text(json.dumps(self._summary, indent=2, default=str))
        if self.wandb is not None:
            self.wandb.finish()
            self.wandb = None
