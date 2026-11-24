"""The PyTorch models must reproduce the notebook's NumPy losses exactly, and train/checkpoint."""

import numpy as np
import torch

from aoctrl.models import DeepKoopman, StructuredLatentARX, WindowDataset, dmdc_init, structured_init
from aoctrl.models import numpy_ops as ops
from aoctrl.training import CheckpointManager, TrainConfig, train_latent_model

W = dict(rec=1.0, one=0.7, lat1=0.2, pred=0.5, lat=0.1)


# ---------------- NumPy forward losses, transcribed from the notebook (nn_loss_grads / dk_loss_grads)
def nb_structured_terms(P, p, c):
    Bn, T1, n = p.shape
    X = p.reshape(-1, n)
    Z = ops.mlp_residual(P, X, "e")
    Zr = Z.reshape(Bn, T1, n)
    Rec = ops.mlp_residual(P, Z, "d")
    Z1 = ops.latent_arx_step(P, Zr[:, 1:-1], Zr[:, :-2], c[:, 1:-1], c[:, :-2], c[:, 2:]).reshape(-1, n)
    PD1 = ops.mlp_residual(P, Z1, "d")
    zh = [Zr[:, 0], Zr[:, 1]]
    for t in range(2, T1):
        zh.append(ops.latent_arx_step(P, zh[t - 1], zh[t - 2], c[:, t - 1], c[:, t - 2], c[:, t]))
    ZH = np.stack(zh[2:], axis=1).reshape(-1, n)
    PD = ops.mlp_residual(P, ZH, "d")
    Pt, Zt = p[:, 2:].reshape(-1, n), Zr[:, 2:].reshape(-1, n)
    return dict(
        rec=np.mean((Rec - X) ** 2),
        one=np.mean((PD1 - Pt) ** 2),
        lat1=np.mean((Z1 - Zt) ** 2),
        pred=np.mean((PD - Pt) ** 2),
        lat=np.mean((ZH - Zt) ** 2),
    )


def nb_koopman_terms(P, p, c):
    Bn, T1, n = p.shape
    nz = P["A"].shape[0]
    O = ops.koopman_obs(p, c).reshape(-1, 4 * n)
    Zf = ops.koopman_encode(P, O)
    Z = Zf.reshape(Bn, T1 - 1, nz)
    Pt1, Pt2 = p[:, 1:].reshape(-1, n), p[:, 2:].reshape(-1, n)
    Rec = ops.koopman_decode(P, Zf)
    Z1 = (Z[:, :-1] @ P["A"].T + c[:, 2:] @ P["B"].T).reshape(-1, nz)
    D1 = ops.koopman_decode(P, Z1)
    Zt = Z[:, 1:].reshape(-1, nz)
    zr = [Z[:, 0]]
    for t in range(2, T1):
        zr.append(zr[-1] @ P["A"].T + c[:, t] @ P["B"].T)
    ZR = np.stack(zr[1:], axis=1).reshape(-1, nz)
    DR = ops.koopman_decode(P, ZR)
    return dict(
        rec=np.mean((Rec - Pt1) ** 2),
        one=np.mean((D1 - Pt2) ** 2),
        lat1=np.mean((Z1 - Zt) ** 2),
        pred=np.mean((DR - Pt2) ** 2),
        lat=np.mean((ZR - Zt) ** 2),
    )


def _structured_params(rng, n=5, hid=7):
    P = structured_init(rng.uniform(-0.5, 0.9, (5, n)), hid, 0)
    for k in P:
        P[k] = P[k] + 0.2 * rng.standard_normal(P[k].shape)
    return P


def _koopman_params(rng, n=3, nz=7):
    p = rng.standard_normal((4, 30, n))
    c = rng.standard_normal((4, 30, n))
    P = dmdc_init(list(p), list(c), nz, 5, 6, 0)
    for k in P:
        P[k] = P[k] + 0.2 * rng.standard_normal(P[k].shape)
    return P


def test_structured_loss_matches_notebook(rng):
    P = _structured_params(rng)
    p, c = rng.standard_normal((4, 6, 5)), rng.standard_normal((4, 6, 5))
    ref = nb_structured_terms(P, p, c)
    with torch.no_grad():
        _, terms = StructuredLatentARX(P).loss(torch.as_tensor(p), torch.as_tensor(c), W, 0.0)
    for k in ref:
        assert np.isclose(float(terms[k]), ref[k], rtol=1e-10), k


def test_koopman_loss_matches_notebook(rng):
    P = _koopman_params(rng)
    p, c = rng.standard_normal((3, 6, 3)), rng.standard_normal((3, 6, 3))
    ref = nb_koopman_terms(P, p, c)
    with torch.no_grad():
        _, terms = DeepKoopman(P).loss(torch.as_tensor(p), torch.as_tensor(c), W, 0.0)
    for k in ref:
        assert np.isclose(float(terms[k]), ref[k], rtol=1e-10), k


def test_autograd_vs_finite_differences(rng):
    """Replaces the notebook's hand-written gradient checks (§8.2, §9.2)."""
    for model, (p, c) in (
        (StructuredLatentARX(_structured_params(rng)), (rng.standard_normal((2, 5, 5)),) * 2),
        (DeepKoopman(_koopman_params(rng)), (rng.standard_normal((2, 5, 3)),) * 2),
    ):
        anchor = {k: v.detach() + 0.1 for k, v in model.anchor().items()}
        params = tuple(model.parameters())
        pt, ct = torch.as_tensor(p), torch.as_tensor(c * 0.5)

        def f():
            return model.loss(pt, ct, W, 1e-3, anchor=anchor, reg_dyn=0.05)[0]

        loss = f()
        grads = torch.autograd.grad(loss, params)
        g = dict(zip([k for k, _ in model.named_parameters()], grads))
        for name, prm in model.named_parameters():
            idx = tuple(int(rng.integers(s)) for s in prm.shape)
            with torch.no_grad():
                old = prm[idx].item()
                prm[idx] = old + 1e-6
                lp = f().item()
                prm[idx] = old - 1e-6
                lm = f().item()
                prm[idx] = old
            num = (lp - lm) / 2e-6
            assert abs(num - g[name][idx].item()) <= 1e-5 * (abs(num) + 1e-8) + 1e-9, name


def _dataset(rng, n=4):
    def win(N):
        return rng.standard_normal((N, 8, n)), rng.standard_normal((N, 8, n))

    tp, tc = win(64)
    vp, vc = win(16)
    return WindowDataset(tp, tc, vp, vc, np.ones(n))


def test_train_checkpoint_and_reuse(rng, tmp_path):
    ds = _dataset(rng)
    P0 = structured_init(rng.uniform(-0.3, 0.8, (5, 4)), 8, 0)
    cfg = TrainConfig(steps=30, batch=16, eval_every=10, checkpoint_every=10)
    ck = CheckpointManager(tmp_path, "m", {"k": 1})
    r1 = train_latent_model(StructuredLatentARX(P0), ds, cfg, torch.device("cpu"), torch.float64, ckpt=ck)
    assert not r1.from_checkpoint and ck.final_path.exists() and r1.best_val <= r1.history["val"][0]
    r2 = train_latent_model(
        StructuredLatentARX(P0),
        ds,
        cfg,
        torch.device("cpu"),
        torch.float64,
        ckpt=CheckpointManager(tmp_path, "m", {"k": 1}),
    )
    assert r2.from_checkpoint
    for k in r1.params:
        assert np.allclose(r1.params[k], r2.params[k])
    ck3 = CheckpointManager(tmp_path, "m", {"k": 2})  # different key -> retrain
    assert not ck3.has_final()
