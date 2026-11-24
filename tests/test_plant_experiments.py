import importlib.util

import numpy as np
import pytest

from aoctrl.api import build_context, run_experiment
from aoctrl.calibration import calibrate
from aoctrl.optics import build_backend
from aoctrl.plant import AOPlant
from aoctrl.simulation import simulate


def test_plant_delay_and_conditions(tiny_cfg):
    cfg = tiny_cfg()
    plant = AOPlant(build_backend(cfg.system), delay=2)
    u1, u2 = np.ones(plant.n_act), 2 * np.ones(plant.n_act)
    plant.step(u1)
    assert np.allclose(plant.u_applied, 0)
    plant.step(u2)
    assert np.allclose(plant.u_applied, 0)
    plant.step(u2)
    assert np.allclose(plant.u_applied, u1)  # applied after 2 frames
    r0 = plant.backend.get_conditions().r0
    with plant.conditions(r0=0.05, delay=0):
        assert plant.backend.get_conditions().r0 == 0.05 and plant.delay == 0
    assert plant.backend.get_conditions().r0 == r0 and plant.delay == 2


def test_synthetic_calibration_and_dm_sign(tiny_cfg):
    ctx = build_context(tiny_cfg())
    calib = ctx.calibration
    assert calib.M2C.shape[1] == 8 and calib.cond < 100
    assert ctx.dm_sign["ok"] and ctx.dm_sign["sign"] == 1.0
    tel = simulate(ctx.plant, 50, ctx.integrator(), seed=1, projector=calib.Bpinv)
    ol = simulate(ctx.plant, 50, None, seed=1)
    assert tel.rms[25:].mean() < 0.5 * ol.rms[25:].mean()


@pytest.mark.parametrize(
    "exp,extra",
    [
        ("exp1_oracle_lqr", ["experiment.rejection.n_steps=256"]),
        ("exp2_ekf_lqr", []),
        ("exp3_latent_lqr", ["model.train.steps=20", "model.hidden=8", "training.eval_every=10"]),
        (
            "exp4_deep_koopman",
            [
                "model.train.steps=20",
                "model.hidden_enc=8",
                "model.hidden_dec=8",
                "model.nz_factor=2",
                "training.eval_every=10",
            ],
        ),
        (
            "study_plant",
            [
                "experiment.n_steps=40",
                "experiment.sweep_steps=30",
                "experiment.sweep_gains=[0.3]",
                "experiment.sweep_delays=[0]",
            ],
        ),
        ("study_calibration", []),
    ],
)
def test_experiment_smoke(tiny_cfg, exp, extra):
    res = run_experiment(tiny_cfg(f"experiment={exp}", *extra))
    assert res.name == exp
    if exp.startswith("exp"):
        df = res.summary
        assert {"rms", "strehl", "retained"} <= set(df.columns)
        oracle = df[df.controller == "LQR oracle AR(2)"].rms
        integ = df[df.controller == "I (g=0.4)"].rms
        assert (oracle.values <= integ.values * 1.05).all()


@pytest.mark.oopao
@pytest.mark.skipif(importlib.util.find_spec("OOPAO") is None, reason="OOPAO not installed")
def test_oopao_scao_small():
    from aoctrl.api import load_config

    cfg = load_config(
        [
            "system.telescope.resolution=40",
            "system.dm.mirrors.0.n_subap=4",
            "system.wfs.n_subap=4",
            "system.calibration.n_modes=10",
            "system.n_threads=1",
        ]
    )
    b = build_backend(cfg.system)
    calib = calibrate(b, 10, 1e-9)
    plant = AOPlant(b)
    tel = simulate(plant, 20, None, seed=1, projector=calib.Bpinv)
    assert tel.d.shape == (20, calib.n_modes) and np.isfinite(tel.rms).all()
