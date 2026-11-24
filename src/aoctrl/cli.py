"""Command-line entry point (Hydra).

python -m aoctrl.cli experiment=exp1_oracle_lqr
python -m aoctrl.cli experiment=exp3_latent_lqr system=synthetic eval=quick training.device=cuda
python -m aoctrl.cli -m experiment=exp1_oracle_lqr control.lqr.rho=1e-3,1e-2   # Hydra multirun
aoctrl experiment=study_atmosphere system/atmosphere=three_layer               # console script
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: figures go to files / W&B

import hydra  # noqa: E402
from omegaconf import DictConfig, open_dict  # noqa: E402

from aoctrl.utils.paths import config_dir  # noqa: E402


@hydra.main(config_path=str(config_dir()), config_name="config", version_base="1.3")
def _main(cfg: DictConfig) -> None:
    from hydra.core.hydra_config import HydraConfig

    from aoctrl.api import run_experiment

    with open_dict(cfg):
        cfg.paths.output_dir = HydraConfig.get().runtime.output_dir
    res = run_experiment(cfg)  # the summary table is already logged by the experiment
    print(f"\noutputs: {res.output_dir}")
    for k, v in res.metrics.items():
        if isinstance(v, (int, float, str)):
            print(f"{k}: {v}")


def main() -> None:
    _main()


if __name__ == "__main__":
    main()
