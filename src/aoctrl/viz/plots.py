"""Plot functions. Each returns a matplotlib Figure and never calls `plt.show()`, so the same code
serves the CLI (saved / logged to W&B) and notebooks (displayed inline)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})


def _masked(m, pupil):
    return np.ma.array(m, mask=~pupil) if pupil is not None and m.shape == pupil.shape else m


def plot_system_overview(pupil, turb_map, frame=None, signal_2d=None):
    n = 2 + (frame is not None) + (signal_2d is not None)
    fig, ax = plt.subplots(1, n, figsize=(4.6 * n, 4))
    ax[0].imshow(pupil, cmap="gray")
    ax[0].set_title("Pupil")
    im = ax[1].imshow(_masked(turb_map * 1e9, pupil), cmap="RdBu_r")
    ax[1].set_title("Turbulent OPD [nm]")
    plt.colorbar(im, ax=ax[1], fraction=0.046)
    i = 2
    if frame is not None:
        ax[i].imshow(frame, cmap="gray")
        ax[i].set_title("WFS frame")
        i += 1
    if signal_2d is not None:
        im = ax[i].imshow(signal_2d, cmap="RdBu_r")
        ax[i].set_title("WFS signal 2D")
        plt.colorbar(im, ax=ax[i], fraction=0.046)
    for a in ax:
        a.grid(False)
    fig.tight_layout()
    return fig


def plot_calibration(singular_values, D, mode_maps: Sequence[np.ndarray], pupil, mode_ids: Sequence[int]):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].semilogy(singular_values, "o-", ms=3)
    ax[0].set_title("Singular values of D")
    ax[0].set_xlabel("mode")
    im = ax[1].imshow(D, aspect="auto", cmap="RdBu_r")
    ax[1].set_title("Interaction matrix D (signal x mode)")
    ax[1].grid(False)
    plt.colorbar(im, ax=ax[1], fraction=0.046)
    if mode_maps:
        side = int(np.ceil(np.sqrt(len(mode_maps))))
        H, Wd = mode_maps[0].shape
        canvas = np.ma.masked_all((side * H, side * Wd))
        for i, m in enumerate(mode_maps):
            r, c = divmod(i, side)
            canvas[r * H : (r + 1) * H, c * Wd : (c + 1) * Wd] = _masked(m, pupil)
        ax[2].imshow(canvas, cmap="RdBu_r")
    ax[2].set_title("Modes " + ",".join(map(str, mode_ids)))
    ax[2].grid(False)
    fig.tight_layout()
    return fig


def plot_timeseries(runs: Mapping[str, object], ts_ms: float, title: str = "", key: str = "rms"):
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    for nm, r in runs.items():
        t = np.arange(len(r.rms)) * ts_ms
        ax[0].plot(t, r.rms * 1e9, label=nm, lw=1.2)
        ax[1].plot(t, r.strehl, label=nm, lw=1.2)
    ax[0].set_ylabel("residual RMS [nm]")
    ax[1].set_ylabel("Strehl (Maréchal)")
    for a in ax:
        a.set_xlabel("t [ms]")
        a.legend(fontsize=8)
    ax[0].set_title(title)
    fig.tight_layout()
    return fig


def plot_condition_bars(df: pd.DataFrame, controllers: Sequence[str], labels: Sequence[str] | None = None):
    """Grid regimes x delays of RMS bars (± std over seeds), fitting floor as a dotted line."""
    regimes = list(dict.fromkeys(df["regime"]))
    delays = sorted(set(df["delay"]))
    labels = labels or controllers
    colors = plt.cm.tab10(np.arange(len(controllers)) % 10)
    fig, axs = plt.subplots(len(regimes), len(delays), figsize=(6.4 * len(delays), 4.6 * len(regimes)), squeeze=False)
    for i, rg in enumerate(regimes):
        for j, dl in enumerate(delays):
            ax = axs[i, j]
            g = df[(df.regime == rg) & (df.delay == dl)].set_index("controller")
            present = [c for c in controllers if c in g.index]
            vals = [g.loc[c, "rms"] for c in present]
            errs = [g.loc[c, "rms_std"] for c in present]
            ax.bar(range(len(present)), vals, yerr=errs, color=colors[: len(present)], capsize=3)
            floor = g["fitting_floor"].iloc[0] if len(g) else np.nan
            if np.isfinite(floor):
                ax.axhline(floor, color="r", ls=":", lw=1.5, label="fitting floor")
                ax.legend(fontsize=8)
            if vals:
                ax.set_ylim(0, 1.25 * max(v + e for v, e in zip(vals, errs)))
            ax.set_xticks(range(len(present)))
            ax.set_xticklabels([labels[controllers.index(c)] for c in present], rotation=35, ha="right", fontsize=8)
            ax.set_ylabel("residual RMS [nm]")
            ax.set_title(f"regime {rg}, delay {dl}")
    fig.tight_layout()
    return fig


def plot_curves(
    curves: Mapping[str, np.ndarray],
    xlabel: str,
    ylabel: str,
    title: str = "",
    logy: bool = False,
    x: np.ndarray | None = None,
    hline: float | None = None,
):
    fig, ax = plt.subplots(figsize=(7, 4))
    for nm, v in curves.items():
        xx = np.arange(1, len(v) + 1) if x is None else x
        (ax.semilogy if logy else ax.plot)(xx, v, "o-", ms=3, label=nm)
    if hline is not None:
        ax.axhline(hline, color="k", ls=":")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_temporal_rejection(curves: Mapping[str, Sequence[tuple[np.ndarray, np.ndarray]]], modes: Sequence[int]):
    fig, axs = plt.subplots(1, len(modes), figsize=(5.3 * len(modes), 4), squeeze=False)
    for mi, ax in zip(range(len(modes)), axs[0]):
        for nm, per_mode in curves.items():
            f, r = per_mode[mi]
            ax.loglog(f, r, label=nm)
        ax.axhline(1, color="k", ls=":")
        ax.set_xlabel("frequency [Hz]")
        ax.set_ylabel("PSD residual / PSD turbulence")
        ax.set_title(f"temporal rejection, mode {modes[mi]}")
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_prediction_table(pred: Mapping[int, Mapping[str, float]]):
    horizons = sorted(pred)
    labels = list(next(iter(pred.values())).keys())
    w = 0.8 / len(labels)
    fig, ax = plt.subplots(figsize=(7, 4))
    for j, lab in enumerate(labels):
        ax.bar(np.array(horizons) + (j - (len(labels) - 1) / 2) * w, [pred[h][lab] for h in horizons], w, label=lab)
    ax.set_xticks(horizons)
    ax.set_xlabel("prediction horizon θ [frames]")
    ax.set_ylabel("error [nm]")
    ax.set_title("Modal prediction error (validation)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_optical_gain(S, PV, R, fit):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.4))
    sc = ax[0].scatter(S[:, 0], PV[:, 0], c=R * 1e9, s=6, cmap="viridis")
    plt.colorbar(sc, ax=ax[0], label="residual RMS [nm]")
    xs = np.linspace(S[:, 0].min(), S[:, 0].max(), 100)
    f_all = (1 + (R / fit.rm) ** 2) ** (-fit.q / 2)
    pred_og = S * f_all[:, None] * fit.K0
    ax[0].plot(xs, fit.K0_linear[0] * xs, "r--", label="linear model")
    ax[0].plot(S[:, 0], pred_og[:, 0], "k.", ms=2, label="compression model")
    ax[0].set_xlabel("true s, mode 0")
    ax[0].set_ylabel("measured pv, mode 0")
    ax[0].legend()
    rr = np.logspace(np.log10(max(R.min(), 1e-9)), np.log10(R.max()), 100)
    ax[1].semilogx(rr * 1e9, (1 + (rr / fit.rm) ** 2) ** (-fit.q / 2), "b-")
    ax[1].axhline(1, color="k", ls=":")
    ax[1].set_xlabel("residual RMS r [nm]")
    ax[1].set_ylabel("f(r)")
    ax[1].set_title("optical gain vs residual")
    sd = PV.std(axis=0)
    ax[2].semilogy(np.sqrt(np.mean((PV - S * fit.K0_linear) ** 2, axis=0)) / sd, "o-", ms=3, label="linear")
    ax[2].semilogy(np.sqrt(np.mean((PV - pred_og) ** 2, axis=0)) / sd, "s-", ms=3, label="compression")
    ax[2].set_xlabel("mode")
    ax[2].set_ylabel("pv error / pv RMS")
    ax[2].legend()
    fig.tight_layout()
    return fig


def plot_training(history: Mapping[str, Mapping[str, list]]):
    fig, ax = plt.subplots(figsize=(7, 4))
    for nm, h in history.items():
        ax.semilogy(h["it"][1:], h["train"][1:], "o-", ms=3, label=f"{nm} train")
        ax.semilogy(h["it"], h["val"], "s-", ms=3, label=f"{nm} val")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.set_title("Training (it = 0: initial linear model)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_eigenvalues(eigs: Mapping[str, np.ndarray]):
    fig, ax = plt.subplots(figsize=(5, 5))
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), "k-", lw=0.8)
    for (nm, ev), mk in zip(eigs.items(), "oxs^"):
        ax.plot(ev.real, ev.imag, mk, ms=4, label=f"{nm} (|λ|max={np.max(np.abs(ev)):.3f})")
    ax.set_aspect("equal")
    ax.set_xlabel("Re λ")
    ax.set_ylabel("Im λ")
    ax.set_title("Koopman spectrum")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_probe_scatter(truth: np.ndarray, preds: Mapping[str, np.ndarray], mode: int = 0):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for nm, p in preds.items():
        ax.plot(truth[:, mode], p[:, mode], ".", ms=3, label=nm)
    lim = np.abs(truth[:, mode]).max()
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="ideal")
    ax.set_xlabel(f"true s, mode {mode}")
    ax.set_ylabel("linear-probe estimate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_bars(values: Mapping[str, float], ylabel: str, title: str = "", hline: float | None = None):
    fig, ax = plt.subplots(figsize=(max(5, 1.1 * len(values)), 4))
    ax.bar(range(len(values)), list(values.values()), color="tab:blue")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(list(values.keys()), fontsize=8, rotation=30, ha="right")
    if hline is not None:
        ax.axhline(hline, color="k", ls="--")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig
