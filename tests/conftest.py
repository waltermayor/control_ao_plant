import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from aoctrl.api import load_config  # noqa: E402


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def tiny_overrides(tmp_path):
    """Synthetic backend + tiny protocol: an experiment runs in seconds."""
    return [
        "system=synthetic",
        "system.synthetic.n_modes=8",
        "system.calibration.n_modes=8",
        "eval.delays=[0,1]",
        "eval.seeds=[21]",
        "eval.n_steps=40",
        "identification.turbulence.n_steps=300",
        "identification.optical_gain.n_steps=150",
        "identification.optical_gain.noise_floor_steps=30",
        "identification.latent_data.n_steps=200",
        f"paths.cache_dir={tmp_path}/cache",
        f"paths.checkpoint_dir={tmp_path}/ckpt",
        f"paths.results_dir={tmp_path}/results",
        f"paths.output_dir={tmp_path}/out",
    ]


@pytest.fixture
def tiny_cfg(tiny_overrides):
    def make(*extra):
        return load_config([*tiny_overrides, *extra])

    return make


def stable_ar2(n, rng):
    rho = rng.uniform(0.85, 0.99, n)
    om = rng.uniform(0.01, 0.3, n)
    return np.stack([2 * rho * np.cos(om), -(rho**2)])
