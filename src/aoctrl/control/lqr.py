"""Discrete-time LQR machinery shared by all experiments.

* `dare_sda` / `lqr_gain_robust`: robust DARE solver (structure-preserving doubling first, SciPy QZ
  as fallback), validated by Riccati residual and closed-loop stability (notebook §8.6).
* `build_oracle_lqr`: true-state LQR of Experiment 1 (AR(p) + pending-command buffer, cost with a
  cross term). Closed form K = G/(1+ρ) and DARE are both computed and compared (notebook §6.4).
* `latent_arx_lqr`: augmented system for the structured latent ARX model (Experiment 3).
* `koopman_lqr`: augmented system for a first-order Koopman latent (Experiment 4).
* `verify_*`: with the *physical* dynamics written in the latent formats, the latent LQRs must
  reproduce the oracle law u = -G d / (1+ρ). These are used by tests and by the experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_discrete_are

from aoctrl.dynamics.ar import companion, predictor_weights


# ---------------------------------------------------------------------------------- solvers
def dare_sda(A, B, Q, R, tol: float = 1e-12, maxit: int = 200):
    """Structure-preserving doubling for P = Q + Aᵀ P (I + G P)^-1 A, G = B R^-1 Bᵀ."""
    G = B @ np.linalg.solve(R, B.T)
    Ak, Gk, Hk = A.copy(), 0.5 * (G + G.T), 0.5 * (Q + Q.T)
    I = np.eye(len(A))
    for it in range(maxit):
        Wk = I + Gk @ Hk
        WA = np.linalg.solve(Wk, Ak)
        WG = np.linalg.solve(Wk, Gk)
        Hn = Hk + Ak.T @ Hk @ WA
        Gn = Gk + Ak @ WG @ Ak.T
        An = Ak @ WA
        d = np.linalg.norm(Hn - Hk, 1) / max(np.linalg.norm(Hn, 1), 1e-300)
        Ak, Gk, Hk = An, 0.5 * (Gn + Gn.T), 0.5 * (Hn + Hn.T)
        if d < tol:
            return Hk, it + 1
    raise RuntimeError(f"SDA did not converge in {maxit} iterations")


def lqr_gain_robust(A, B, Q, R) -> tuple[np.ndarray, dict]:
    errs = []
    for name in ("SDA", "scipy"):
        try:
            P = dare_sda(A, B, Q, R)[0] if name == "SDA" else solve_discrete_are(A, B, Q, R)
            K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
            G = B @ np.linalg.solve(R, B.T)
            I = np.eye(len(A))
            res = np.linalg.norm(A.T @ P @ np.linalg.solve(I + G @ P, A) + Q - P, 1) / max(np.linalg.norm(P, 1), 1e-300)
            rho_cl = float(np.max(np.abs(np.linalg.eigvals(A - B @ K))))
            if res < 1e-6 and rho_cl < 1.0:
                return K, dict(method=name, residual=float(res), rho_cl=rho_cl)
            errs.append(f"{name}: residual {res:.1e}, closed-loop spectral radius {rho_cl:.4f}")
        except Exception as e:  # noqa: BLE001 - we aggregate solver failures
            errs.append(f"{name}: {str(e)[:70]}")
    raise RuntimeError(
        "Could not solve the DARE (" + " | ".join(errs) + "). The model is probably unstable: "
        "increase regularisation (reg / reg_dyn) or lower the spectral-radius cap."
    )


# ---------------------------------------------------------------------------------- Exp 1: oracle
@dataclass
class OracleLQRModel:
    K: np.ndarray  # (n, p·n + D·n)
    A: np.ndarray
    B: np.ndarray
    G: np.ndarray  # E[d_{k+D} | ζ_k]
    p: int
    D: int
    n: int
    method: str
    dare_vs_closed_form: float
    info: dict = field(default_factory=dict)


def build_oracle_lqr(
    coefs: np.ndarray, delay: int, rho: float, Wn: np.ndarray, use_dare: bool = True
) -> OracleLQRModel:
    """Decision state ζ_k = [d_{k-1} … d_{k-p} | u_{k-1} … u_{k-D}];  cost (Gζ+u)ᵀW(Gζ+u) + ρ uᵀWu."""
    p, n = coefs.shape
    D = int(delay)
    theta = D + 1
    Fd = companion(coefs)
    nz = p * n + D * n
    A = np.zeros((nz, nz))
    A[: p * n, : p * n] = Fd
    Bm = np.zeros((nz, n))
    if D >= 1:
        Bm[p * n : p * n + n] = np.eye(n)
        for j in range(1, D):
            A[p * n + j * n : p * n + (j + 1) * n, p * n + (j - 1) * n : p * n + j * n] = np.eye(n)
    G = np.zeros((n, nz))
    G[:, : p * n] = np.linalg.matrix_power(Fd, theta)[:n, :]
    Q = G.T @ Wn @ G
    N = G.T @ Wn
    R = (1.0 + rho) * Wn
    K_cf = np.linalg.solve(R, N.T)  # = G / (1 + rho): exact, u_k does not affect future costs
    K, method, diff = K_cf, "closed form (B=0)", 0.0
    if use_dare and D >= 1:
        try:
            P = solve_discrete_are(A, Bm, Q, R, s=N)
            K_d = np.linalg.solve(R + Bm.T @ P @ Bm, Bm.T @ P @ A + N.T)
            diff = float(np.linalg.norm(K_d - K_cf) / np.linalg.norm(K_cf))
            K, method = (K_d, "DARE") if diff < 1e-3 else (K_cf, "closed form (DARE differs)")
        except Exception as e:  # noqa: BLE001
            method = f"closed form (DARE failed: {type(e).__name__})"
    elif not use_dare:
        method = "closed form"
    return OracleLQRModel(K=K, A=A, B=Bm, G=G, p=p, D=D, n=n, method=method, dare_vs_closed_form=diff)


# ---------------------------------------------------------------------------------- Exp 3: latent ARX
def latent_arx_lqr(L: dict, D: int, J: np.ndarray, Wc: np.ndarray, rho: float) -> tuple[np.ndarray, dict]:
    """X_k = [z_{k-1}, z_{k-2}, a_{k-1}, a_{k-2}, u_{k-1}, ..., u_{k-D}]  (a_j: command applied at step j)."""
    n = L["A1"].shape[0]
    nx = n * (4 + D)
    I = np.eye(n)
    A = np.zeros((nx, nx))
    Bm = np.zeros((nx, n))

    def blk(i):
        return slice(i * n, (i + 1) * n)

    A[blk(0), blk(0)] = L["A1"]
    A[blk(0), blk(1)] = L["A2"]
    A[blk(0), blk(2)] = L["Ac1"]
    A[blk(0), blk(3)] = L["Ac2"]
    A[blk(1), blk(0)] = I
    A[blk(3), blk(2)] = I
    if D == 0:
        Bm[blk(0)] = L["B0"]
        Bm[blk(2)] = I  # a_k = u_k
    else:
        last = 4 + D - 1
        A[blk(0), blk(last)] = L["B0"]
        A[blk(2), blk(last)] = I  # a_k = u_{k-D} (oldest pending)
        Bm[blk(4)] = I
        for j in range(1, D):
            A[blk(4 + j), blk(4 + j - 1)] = I
    Q = np.zeros((nx, nx))
    Q[blk(0), blk(0)] = J @ Wc @ J.T
    return lqr_gain_robust(A, Bm, Q, rho * Wc)


# ---------------------------------------------------------------------------------- Exp 4: Koopman
def koopman_lqr(A: np.ndarray, Bm: np.ndarray, J: np.ndarray, D: int, Wc: np.ndarray, rho: float):
    """X_k = [z_{k-1}, u_{k-1}, ..., u_{k-D}];  z_k = A z_{k-1} + B a_k,  a_k = u_{k-D}."""
    nz, n = Bm.shape
    nx = nz + D * n
    I = np.eye(n)
    Aa = np.zeros((nx, nx))
    Ba = np.zeros((nx, n))
    Aa[:nz, :nz] = A
    if D == 0:
        Ba[:nz] = Bm
    else:
        Aa[:nz, nz + (D - 1) * n : nz + D * n] = Bm
        Ba[nz : nz + n] = I
        for j in range(1, D):
            Aa[nz + j * n : nz + (j + 1) * n, nz + (j - 1) * n : nz + j * n] = I
    Q = np.zeros((nx, nx))
    Q[:nz, :nz] = J @ Wc @ J.T
    return lqr_gain_robust(Aa, Ba, Q, rho * Wc)


def physical_koopman_params(ar2: np.ndarray) -> dict:
    """Exact AR(2) dynamics in Koopman format (z = O = [p_t, p_{t-1}, c_t, c_{t-1}], decoder reads p)."""
    n = ar2.shape[1]
    I = np.eye(n)
    Z0 = np.zeros((n, n))
    a1, a2 = np.diag(ar2[0]), np.diag(ar2[1])
    return dict(
        A=np.block([[a1, a2, -a1, -a2], [I, Z0, Z0, Z0], [Z0, Z0, Z0, Z0], [Z0, Z0, I, Z0]]),
        B=np.vstack([I, Z0, I, Z0]),
        Pe=np.eye(4 * n),
        Pd=np.vstack([I, Z0, Z0, Z0]),
        W1e=np.zeros((4 * n, 2)),
        W2e=np.zeros((2, 4 * n)),
        W1d=np.zeros((4 * n, 2)),
        W2d=np.zeros((2, n)),
    )


# ---------------------------------------------------------------------------------- verification
def _oracle_law(ar2, X, n, D, rho):
    zk1, zk2, ak1, ak2 = X[:n], X[n : 2 * n], X[2 * n : 3 * n], X[3 * n : 4 * n]
    Gw = predictor_weights(ar2, D + 1)
    return -(Gw[0] * (zk1 - ak1) + Gw[1] * (zk2 - ak2)) / (1 + rho)


def verify_latent_arx_lqr(ar2: np.ndarray, delays, rho: float, Wc: np.ndarray, seed: int = 1) -> dict[int, dict]:
    """Physical dynamics in ARX-latent format (E = D = I) must reproduce the oracle law."""
    n = ar2.shape[1]
    a1, a2 = ar2[0], ar2[1]
    Lph = dict(A1=np.diag(a1), A2=np.diag(a2), Ac1=-np.diag(a1), Ac2=-np.diag(a2), B0=np.eye(n))
    rng = np.random.default_rng(seed)
    out = {}
    for Dv in sorted(set(int(d) for d in delays)):
        K, info = latent_arx_lqr(Lph, Dv, np.eye(n), Wc, rho)
        X = rng.standard_normal(n * (4 + Dv))
        u_or = _oracle_law(ar2, X, n, Dv, rho)
        out[Dv] = dict(rel_error=float(np.linalg.norm(-K @ X - u_or) / np.linalg.norm(u_or)), **info)
    return out


def verify_koopman_lqr(ar2: np.ndarray, delays, rho: float, Wc: np.ndarray, seed: int = 1) -> dict[int, dict]:
    n = ar2.shape[1]
    P = physical_koopman_params(ar2)
    rng = np.random.default_rng(seed)
    out = {}
    for Dv in sorted(set(int(d) for d in delays)):
        K, info = koopman_lqr(P["A"], P["B"], P["Pd"], Dv, Wc, rho)
        X = rng.standard_normal(4 * n + Dv * n)
        u_or = _oracle_law(ar2, X, n, Dv, rho)
        out[Dv] = dict(rel_error=float(np.linalg.norm(-K @ X - u_or) / np.linalg.norm(u_or)), **info)
    return out
