"""Generic trainer for `LatentModel`s (notebook `nn_train` / `dk_train`, ported to PyTorch).

Identical recipe: Adam (β=(0.9, 0.999), ε=1e-8), global-norm gradient clipping, cosine schedule
lr_t = lr · (0.1 + 0.45 (1 + cos(π t / steps))), a smaller LR for the linear parts, an anchor of
the dynamics to the truth-free initial model, and selection of the best *validation* model —
including the initial (linear) one, so "training did not help" is an explicit, valid outcome.

Additions over the notebook: device placement (GPU when available), W&B/local logging of the
training curves, resumable periodic checkpoints and reuse of completed checkpoints.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from aoctrl.models.data import WindowDataset
from aoctrl.models.torch_models import LatentModel
from aoctrl.training.checkpoint import CheckpointManager

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    steps: int = 4000
    batch: int = 128
    lr: float = 2e-3
    lin_lr_factor: float = 0.3
    reg: float = 1e-4
    reg_dyn: float = 3e-3
    weights: dict = field(default_factory=lambda: dict(rec=1.0, one=1.0, lat1=0.1, pred=0.3, lat=0.03))
    eval_every: int = 250
    grad_clip: float = 5.0
    seed: int = 0
    checkpoint_every: int = 1000


@dataclass
class TrainResult:
    params: dict  # best parameters (NumPy)
    init_params: dict  # initial (linear) parameters (NumPy)
    history: dict  # it, train, val (+ per-term val)
    best_val: float
    best_it: int
    from_checkpoint: bool
    device: str
    seconds: float = 0.0


def _tensor(x, device, dtype):
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)


def train_latent_model(
    model: LatentModel,
    data: WindowDataset,
    cfg: TrainConfig,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    ckpt: CheckpointManager | None = None,
    logger=None,
    tag: str = "model",
) -> TrainResult:
    init_params = model.to_numpy()
    if ckpt is not None and ckpt.has_final():
        state = ckpt.load_final()
        return TrainResult(
            params=state["params"],
            init_params=state.get("init_params", init_params),
            history=state["history"],
            best_val=state["best_val"],
            best_it=state["best_it"],
            from_checkpoint=True,
            device="checkpoint",
        )

    t_start = time.time()
    model = model.to(device=device, dtype=dtype)
    anchor = model.anchor()
    tp, tc = _tensor(data.train_p, device, dtype), _tensor(data.train_c, device, dtype)
    vp, vc = _tensor(data.val_p, device, dtype), _tensor(data.val_c, device, dtype)
    W = dict(cfg.weights)
    params = [p for p in model.parameters()]
    opt = torch.optim.Adam(model.param_groups(cfg.lr, cfg.lin_lr_factor), betas=(0.9, 0.999), eps=1e-8)
    steps = int(cfg.steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: 0.1 + 0.45 * (1 + math.cos(math.pi * min(e + 1, steps) / steps))
    )
    rng = np.random.default_rng(cfg.seed)

    def validate():
        with torch.no_grad():
            loss, terms = model.loss(vp, vc, W, 0.0)
        return float(loss), {k: float(v) for k, v in terms.items()}

    best_val, vterms = validate()
    best = model.to_numpy()
    best_it, start_it = 0, 1
    hist = dict(it=[0], val=[best_val], train=[float("nan")], **{f"val_{k}": [v] for k, v in vterms.items()})
    log.info("[%s] it 0/%d | val %.4f (initial model)", tag, steps, best_val)

    if ckpt is not None and ckpt.has_last():
        st = ckpt.load_last()
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        rng.bit_generator.state = st["rng"]
        best, best_val, best_it, hist, start_it = st["best"], st["best_val"], st["best_it"], st["history"], st["it"] + 1
        log.info("[%s] resumed from iteration %d", tag, st["it"])

    run_loss: list[torch.Tensor] = []
    for it in range(start_it, steps + 1):
        idx = torch.as_tensor(rng.integers(0, len(data.train_p), size=cfg.batch), device=device)
        loss, _ = model.loss(tp[idx], tc[idx], W, cfg.reg, anchor=anchor, reg_dyn=cfg.reg_dyn)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step()
        sched.step()
        run_loss.append(loss.detach())
        if it % cfg.eval_every == 0 or it == steps:
            vl, vterms = validate()
            tr = float(torch.stack(run_loss[-cfg.eval_every :]).mean())
            hist["it"].append(it)
            hist["val"].append(vl)
            hist["train"].append(tr)
            for k, v in vterms.items():
                hist[f"val_{k}"].append(v)
            improved = vl < best_val
            if improved:
                best_val, best, best_it = vl, model.to_numpy(), it
            log.info("[%s] it %d/%d | train %.4f | val %.4f%s", tag, it, steps, tr, vl, "  * best" if improved else "")
            if logger is not None:
                logger.log_metrics(
                    {
                        f"train/{tag}/loss": tr,
                        f"val/{tag}/loss": vl,
                        f"train/{tag}/lr": sched.get_last_lr()[0],
                        **{f"val/{tag}/{k}": v for k, v in vterms.items()},
                    },
                    step=None,
                )
        if ckpt is not None and cfg.checkpoint_every and it % cfg.checkpoint_every == 0 and it < steps:
            ckpt.save_last(
                dict(
                    model=model.state_dict(),
                    opt=opt.state_dict(),
                    sched=sched.state_dict(),
                    rng=rng.bit_generator.state,
                    best=best,
                    best_val=best_val,
                    best_it=best_it,
                    history=hist,
                    it=it,
                )
            )

    log.info("[%s] best validation %.4f (iteration %d)", tag, best_val, best_it)
    result = TrainResult(
        params=best,
        init_params=init_params,
        history=hist,
        best_val=best_val,
        best_it=best_it,
        from_checkpoint=False,
        device=str(device),
        seconds=time.time() - t_start,
    )
    if ckpt is not None:
        path = ckpt.save_final(
            dict(
                params=best,
                init_params=init_params,
                history=hist,
                best_val=best_val,
                best_it=best_it,
                kind=model.kind,
                device=str(device),
            )
        )
        if logger is not None and path is not None:
            logger.log_artifact(path, name=f"{tag}-model", type="model")
    return result


def evaluate_loss_terms(model_cls, P: dict, p: np.ndarray, c: np.ndarray, weights: dict) -> dict:
    """Loss terms of NumPy params `P` on (p, c), computed on CPU in float64 (diagnostics)."""
    m = model_cls.from_numpy(P)
    with torch.no_grad():
        loss, terms = m.loss(torch.as_tensor(p), torch.as_tensor(c), weights, 0.0)
    return dict(total=float(loss), **{k: float(v) for k, v in terms.items()})
