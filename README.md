# aoctrl

Adaptive-optics control with learned linear latents, using OOPAO as the plant. A modular rewrite of
`OOPAO_init.ipynb` (Experiments 1–4: oracle LQR, EKF + LQR, structured neural latent + LQR,
Deep Koopman + LQR) with Hydra configs, SCAO/MCAO systems, GPU training with checkpoints, W&B logging
and videos.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,wandb,video,notebook]"
pip install -e ".[oopao]"                       # the real simulator (optional for tests)

pytest -q
python -m aoctrl.cli experiment=exp1_oracle_lqr system=synthetic eval=quick   # seconds, no OOPAO
python -m aoctrl.cli experiment=exp1_oracle_lqr                               # OOPAO SCAO
```

Experiments: `exp1_oracle_lqr`, `exp2_ekf_lqr`, `exp3_latent_lqr`, `exp4_deep_koopman`, `compare`,
`study_atmosphere`, `study_sensor`, `study_calibration`, `study_plant`.

Interactive use: `notebooks/quickstart.ipynb`. Architecture, conventions and the module map to the
original notebook: `CLAUDE.md`. Roadmap: `TODO.md`.
