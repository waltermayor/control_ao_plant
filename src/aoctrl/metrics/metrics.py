"""Performance, modal, temporal and estimation metrics (notebook §6.1 "Métricas", §7, §8, §9).

All functions are pure; they take telemetry arrays (or lists of `Telemetry`) and return numbers.
Wavefront-weighted norms use W (eᵀ W e = OPD variance of a modal error e).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.signal import welch


def tail_mean(x: np.ndarray, frac: float = 0.5) -> float:
    """Mean over the last `frac` of the run (steady state)."""
    return float(np.mean(x[int(len(x) * (1 - frac)) :]))


def wavefront_rms_nm(E: np.ndarray, W: np.ndarray) -> float:
    """RMS over time of the wavefront norm of modal errors E (T, n) [nm]."""
    return float(np.sqrt(np.mean(np.einsum("ti,ij,tj->t", E, W, E))) * 1e9)


def run_stats(runs: Sequence) -> dict[str, np.ndarray]:
    """Per-seed steady-state RMS [nm], Strehl and mean command norm [nm]."""
    return dict(
        rms=np.array([tail_mean(r.rms) * 1e9 for r in runs]),
        strehl=np.array([tail_mean(r.strehl) for r in runs]),
        u=np.array([np.mean(np.linalg.norm(r.u, axis=1)) * 1e9 for r in runs]),
    )


def fitting_floor_nm(open_loop_runs: Sequence, W: np.ndarray) -> float:
    """Part of the turbulence outside the modal basis: sqrt(RMS² − sᵀWs) in open loop [nm]."""
    v_tot = np.mean([np.mean(r.rms**2) for r in open_loop_runs])
    v_mod = np.mean([np.mean(np.einsum("ti,ij,tj->t", r.s, W, r.s)) for r in open_loop_runs])
    return float(np.sqrt(max(v_tot - v_mod, 0.0)) * 1e9)


def gap_closed(rms: float, ref_low: float, ref_high: float) -> float:
    """Percent of the distance from `ref_low` (integrator) to `ref_high` (oracle) recovered (0..100)."""
    return float((ref_low - rms) / (ref_low - ref_high) * 100) if abs(ref_low - ref_high) > 1e-9 else float("nan")


def modal_rms(runs: Sequence, key: str, frac: float = 0.5) -> np.ndarray:
    """Per-mode RMS over the tail of the runs of field `key` ('d' or 's')."""
    vals = [getattr(r, key) for r in runs]
    return np.sqrt(np.mean([np.mean(v[int(len(v) * (1 - frac)) :] ** 2, axis=0) for v in vals], axis=0))


def modal_rejection(runs: Sequence, frac: float = 0.5) -> np.ndarray:
    """Residual / turbulence RMS per mode (< 1 = corrected)."""
    return modal_rms(runs, "s", frac) / modal_rms(runs, "d", frac)


def temporal_rejection(run, mode: int, fs: float, nperseg: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """PSD(residual)/PSD(turbulence) for one mode, Welch estimate."""
    f, Pd = welch(run.d[:, mode], fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    _, Ps = welch(run.s[:, mode], fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    return f[1:], (Ps / Pd)[1:]


def estimation_error_nm(runs: Sequence, key: str, W: np.ndarray, target: str = "d") -> float:
    """RMS wavefront error of a controller's state estimate `diag[key]` vs the truth (est[j] estimates d[j])."""
    v = []
    for r in runs:
        est = r.diag.get(key)
        if est is None or len(est) == 0:
            return float("nan")
        truth = getattr(r, target)[: len(est)]
        v.append(wavefront_rms_nm(est - truth, W))
    return float(np.mean(v))


def nis_mean(runs: Sequence, burn: int = 10) -> float:
    vals = [np.mean(r.diag["nis"][burn:]) for r in runs if "nis" in r.diag and len(r.diag["nis"]) > burn]
    return float(np.mean(vals)) if vals else float("nan")


def compute_ms(runs: Sequence) -> float:
    vals = [np.mean(r.diag["t_ms"]) for r in runs if len(r.diag.get("t_ms", [])) > 0]
    return float(np.mean(vals)) if vals else float("nan")


def prediction_error_by_horizon_nm(pred: np.ndarray, p: np.ndarray, sigma: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Multi-step prediction error of normalised measurements, in nm of wavefront (per horizon)."""
    e = (pred - p[:, 2:]) * sigma
    return np.array([np.sqrt(np.mean(np.einsum("bi,ij,bj->b", e[:, h], W, e[:, h]))) * 1e9 for h in range(e.shape[1])])


def linear_probe(feat_train: np.ndarray, target_train: np.ndarray, feat_val: np.ndarray, ridge: float = 1e-4):
    """Ridge-regression probe target ≈ Tᵀ feat; returns (T, prediction on validation).

    Makes the "how linear is this representation w.r.t. the true state" comparison invariant to the
    basis and scale of a latent."""
    G = feat_train.T @ feat_train
    T = np.linalg.solve(G + ridge * np.trace(G) / len(G) * np.eye(len(G)), feat_train.T @ target_train)
    return T, feat_val @ T
