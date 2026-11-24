"""Shared pieces of the learned-latent experiments (3 and 4): train-or-load and diagnostics."""

from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf

from aoctrl.metrics import linear_probe, prediction_error_by_horizon_nm, wavefront_rms_nm
from aoctrl.models.torch_models import LatentModel
from aoctrl.training import CheckpointManager, TrainConfig, TrainResult, train_latent_model
from aoctrl.utils.device import describe_device
from aoctrl.utils.paths import resolve_path


def train_config(cfg) -> TrainConfig:
    mt = OmegaConf.to_container(cfg.model.train, resolve=True)
    tc = cfg.training
    return TrainConfig(
        steps=int(mt["steps"]),
        batch=int(mt["batch"]),
        lr=float(mt["lr"]),
        lin_lr_factor=float(mt["lin_lr_factor"]),
        reg=float(mt["reg"]),
        reg_dyn=float(mt["reg_dyn"]),
        weights=dict(mt["weights"]),
        seed=int(mt["seed"]),
        eval_every=int(tc.eval_every),
        grad_clip=float(tc.grad_clip),
        checkpoint_every=int(tc.checkpoint_every),
    )


def train_or_load(exp, model: LatentModel, tag: str) -> TrainResult:
    """Train `model` on the context's latent dataset, or load the matching checkpoint."""
    ctx, cfg = exp.ctx, exp.cfg
    tcfg = train_config(cfg)
    key = dict(
        model=OmegaConf.to_container(cfg.model, resolve=True),
        train=tcfg.__dict__,
        dtype=str(cfg.training.dtype),
        data=ctx.data_fingerprint(),
    )
    ckpt = CheckpointManager(
        resolve_path(cfg.paths.checkpoint_dir),
        tag,
        key,
        enabled=bool(cfg.training.checkpoints),
        force_retrain=bool(cfg.training.force_retrain),
    )
    exp.info(
        "[%s] device %s | dataset %s | checkpoint %s",
        tag,
        describe_device(ctx.device),
        ctx.latent_dataset.describe(),
        ckpt.dir,
    )
    res = train_latent_model(
        model, ctx.latent_dataset, tcfg, ctx.device, ctx.dtype, ckpt=ckpt, logger=exp.logger, tag=tag
    )
    exp.metrics(
        {
            f"{tag}/from_checkpoint": res.from_checkpoint,
            f"{tag}/best_val": res.best_val,
            f"{tag}/best_it": res.best_it,
            f"{tag}/train_seconds": res.seconds,
        }
    )
    exp.result.artifacts[f"{tag}_checkpoint"] = str(ckpt.final_path)
    return res


def persistence_prediction(val_p: np.ndarray, val_c: np.ndarray, sign: float) -> np.ndarray:
    """Constant turbulence: p_t = p_1 − c_1 + c_t (in measurement units)."""
    return np.stack([val_p[:, 1] - sign * val_c[:, 1] + sign * val_c[:, t] for t in range(2, val_p.shape[1])], axis=1)


def multistep_errors(ctx, preds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    ds = ctx.latent_dataset
    return {
        nm: prediction_error_by_horizon_nm(p, ds.val_p, ds.sigma, ctx.calibration.W_phys) for nm, p in preds.items()
    }


def probe_errors(ctx, features: dict, frac: float, need_O: bool = False):
    """Linear recoverability of the true residual s from several representations (validation, nm).

    `features[name](O, p) -> array` with O = [p_t, p_{t-1}, c_t, c_{t-1}] and p = p_t (normalised).
    Returns (errors_nm, predictions, truth, probes)."""
    sig, W = ctx.latent_sigma, ctx.calibration.W_phys

    def seq(q, lo, hi):
        p = q["pv"][lo:hi] / sig
        c = q["c"][lo:hi] / sig
        return np.concatenate([p[1:], p[:-1], c[1:], c[:-1]], axis=1), q["s"][lo + 1 : hi] / sig, p[1:]

    tr = [seq(q, 0, int(len(q["pv"]) * frac)) for q in ctx.latent_runs]
    va = [seq(q, int(len(q["pv"]) * frac), len(q["pv"])) for q in ctx.latent_runs]
    truth = np.concatenate([s for _, s, _ in va]) * sig
    errs, preds, probes = {}, {}, {}
    for nm, feat in features.items():
        T, pv = linear_probe(
            np.concatenate([feat(O, p) for O, _, p in tr]),
            np.concatenate([s for _, s, _ in tr]),
            np.concatenate([feat(O, p) for O, _, p in va]),
        )
        preds[nm] = pv * sig
        probes[nm] = T
        errs[nm] = wavefront_rms_nm(preds[nm] - truth, W)
    return errs, preds, truth, probes
