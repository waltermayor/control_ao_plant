"""PyTorch latent models (training side). Losses are identical to the notebook's hand-derived
NumPy versions; gradients come from autograd instead of manual BPTT.

Both models share the `LatentModel` interface used by the generic trainer:

    loss(p, c, weights, reg, anchor=None, reg_dyn=0.0) -> (loss, terms)
    param_groups(lr, lin_lr_factor)      parameter groups (linear parts can have a smaller LR)
    to_numpy() / from_numpy(P)           exchange with the NumPy inference ops / controllers
    anchor_keys                          parameters anchored to their initialisation

p, c: (batch, T+1, n) normalised modal measurements and *applied* modal commands.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import torch
from torch import nn

LOSS_TERMS = ("rec", "one", "lat1", "pred", "lat")


class LatentModel(nn.Module):
    anchor_keys: ClassVar[tuple[str, ...]] = ()
    lin_keys: ClassVar[tuple[str, ...]] = ()
    reg_keys: ClassVar[tuple[str, ...]] = ("W2e", "W2d")
    kind: ClassVar[str] = "latent"

    def __init__(self, P: dict[str, np.ndarray]):
        super().__init__()
        for k, v in P.items():
            self.register_parameter(k, nn.Parameter(torch.as_tensor(np.asarray(v, dtype=np.float64))))

    # ------------------------------------------------------------------ exchange
    def to_numpy(self) -> dict[str, np.ndarray]:
        return {k: v.detach().cpu().double().numpy().copy() for k, v in self.named_parameters()}

    @classmethod
    def from_numpy(cls, P: dict[str, np.ndarray]) -> "LatentModel":
        return cls(P)

    def anchor(self) -> dict[str, torch.Tensor]:
        return {k: getattr(self, k).detach().clone() for k in self.anchor_keys}

    def param_groups(self, lr: float, lin_lr_factor: float):
        lin = [p for k, p in self.named_parameters() if k in self.lin_keys]
        other = [p for k, p in self.named_parameters() if k not in self.lin_keys]
        groups = [dict(params=other, lr=lr)]
        if lin:
            groups.append(dict(params=lin, lr=lr * lin_lr_factor))
        return groups

    # ------------------------------------------------------------------ loss
    def loss_terms(self, p: torch.Tensor, c: torch.Tensor) -> dict[str, torch.Tensor]:  # pragma: no cover
        raise NotImplementedError

    def loss(self, p, c, weights: dict, reg: float, anchor: dict | None = None, reg_dyn: float = 0.0):
        terms = self.loss_terms(p, c)
        loss = sum(float(weights[k]) * terms[k] for k in LOSS_TERMS)
        if reg > 0:
            loss = loss + reg * sum(torch.sum(getattr(self, k) ** 2) for k in self.reg_keys)
        if anchor is not None and reg_dyn > 0:
            loss = loss + reg_dyn * sum(torch.sum((getattr(self, k) - anchor[k]) ** 2) for k in self.anchor_keys)
        return loss, terms


class StructuredLatentARX(LatentModel):
    """Experiment 3. Encoder/decoder = identity + residual MLP; latent ARX(2) dynamics with feedthrough:

    z_t = E(p_t) = p_t + MLP(p_t)
    z_{t+1} = A1 z_t + A2 z_{t-1} + Ac1 c_t + Ac2 c_{t-1} + B0 c_{t+1}
    p̂_t = D(z_t) = z_t + MLP(z_t)
    """

    anchor_keys = ("A1", "A2", "Ac1", "Ac2", "B0")
    lin_keys = anchor_keys
    kind = "structured_latent_arx"

    def _mlp(self, x, tag):
        h = torch.tanh(x @ getattr(self, "W1" + tag) + getattr(self, "b1" + tag))
        return x + h @ getattr(self, "W2" + tag) + getattr(self, "b2" + tag)

    def encode(self, x):
        return self._mlp(x, "e")

    def decode(self, z):
        return self._mlp(z, "d")

    def dyn(self, z1, z2, c1, c2, c0):
        return z1 @ self.A1.T + z2 @ self.A2.T + c1 @ self.Ac1.T + c2 @ self.Ac2.T + c0 @ self.B0.T

    def loss_terms(self, p, c):
        T1 = p.shape[1]
        Z = self.encode(p)
        Rec = self.decode(Z)
        # (a) one step forced by the encoder
        Z1 = self.dyn(Z[:, 1:-1], Z[:, :-2], c[:, 1:-1], c[:, :-2], c[:, 2:])
        PD1 = self.decode(Z1)
        # (b) free rollout from (E(p0), E(p1))
        zh = [Z[:, 0], Z[:, 1]]
        for t in range(2, T1):
            zh.append(self.dyn(zh[t - 1], zh[t - 2], c[:, t - 1], c[:, t - 2], c[:, t]))
        ZH = torch.stack(zh[2:], dim=1)
        PD = self.decode(ZH)
        Pt, Zt = p[:, 2:], Z[:, 2:]
        return dict(
            rec=torch.mean((Rec - p) ** 2),
            one=torch.mean((PD1 - Pt) ** 2),
            lat1=torch.mean((Z1 - Zt) ** 2),
            pred=torch.mean((PD - Pt) ** 2),
            lat=torch.mean((ZH - Zt) ** 2),
        )


class DeepKoopman(LatentModel):
    """Experiment 4. Koopman autoencoder with control (Lusch–Kutz–Brunton losses):

    O_t = [p_t, p_{t-1}, c_t, c_{t-1}]
    z_t = O_t Pe + tanh(O_t W1e) W2e            (no biases: E(0) = 0)
    p̂_t = z_t Pd + tanh(z_t W1d) W2d            (no biases: D(0) = 0)
    z_{t+1} = A z_t + B c_{t+1}
    """

    anchor_keys = ("A", "B")
    lin_keys = ("A", "B", "Pe", "Pd")
    kind = "deep_koopman"

    @staticmethod
    def obs(p, c):
        return torch.cat([p[:, 1:], p[:, :-1], c[:, 1:], c[:, :-1]], dim=-1)

    def encode(self, O):
        return O @ self.Pe + torch.tanh(O @ self.W1e) @ self.W2e

    def decode(self, Z):
        return Z @ self.Pd + torch.tanh(Z @ self.W1d) @ self.W2d

    def loss_terms(self, p, c):
        T1 = p.shape[1]
        Z = self.encode(self.obs(p, c))  # (B, T, nz)
        Pt1, Pt2 = p[:, 1:], p[:, 2:]
        Rec = self.decode(Z)
        Z1 = Z[:, :-1] @ self.A.T + c[:, 2:] @ self.B.T
        D1 = self.decode(Z1)
        Zt = Z[:, 1:]
        zr = [Z[:, 0]]
        for t in range(2, T1):
            zr.append(zr[-1] @ self.A.T + c[:, t] @ self.B.T)
        ZR = torch.stack(zr[1:], dim=1)
        DR = self.decode(ZR)
        return dict(
            rec=torch.mean((Rec - Pt1) ** 2),
            one=torch.mean((D1 - Pt2) ** 2),
            lat1=torch.mean((Z1 - Zt) ** 2),
            pred=torch.mean((DR - Pt2) ** 2),
            lat=torch.mean((ZR - Zt) ** 2),
        )


MODEL_REGISTRY: dict[str, type[LatentModel]] = {
    StructuredLatentARX.kind: StructuredLatentARX,
    DeepKoopman.kind: DeepKoopman,
}
