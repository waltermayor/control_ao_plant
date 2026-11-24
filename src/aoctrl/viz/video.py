"""Video generation for simulations and component behaviour.

* `render_loop_video`  — closed-loop telemetry frames recorded by `simulate(record_every=...)`:
                          turbulence | residual | WFS signal | (PSF) + residual-RMS trace.
* `render_maps_video`  — any sequence of 2-D maps (phase screens, DM shapes, WFS frames, ...).
* `render_comparison_video` — several controllers side by side on the same turbulence.

Writes MP4 through ffmpeg when available, otherwise an animated GIF (Pillow).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

log = logging.getLogger(__name__)


def _writer(path: Path, fps: int):
    if path.suffix == ".mp4" and animation.writers.is_available("ffmpeg"):
        return path, animation.FFMpegWriter(fps=fps, bitrate=2400)
    if path.suffix == ".mp4":
        log.warning("ffmpeg not available; writing a GIF instead of %s", path.name)
        path = path.with_suffix(".gif")
    return path, animation.PillowWriter(fps=fps)


def _mask(m, pupil):
    return np.ma.array(m, mask=~pupil) if pupil is not None and m.shape == pupil.shape else m


def render_maps_video(
    maps: Sequence[np.ndarray],
    path: str | Path,
    fps: int = 15,
    title: str = "",
    pupil=None,
    cmap: str = "RdBu_r",
    scale: float = 1.0,
    unit: str = "",
    labels: Sequence[str] | None = None,
) -> Path:
    path, writer = _writer(Path(path), fps)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(maps) * scale
    vmax = float(np.percentile(np.abs(arr), 99.5)) or 1.0
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(_mask(arr[0], pupil), cmap=cmap, vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, fraction=0.046, label=unit)
    ax.grid(False)

    def update(i):
        im.set_data(_mask(arr[i], pupil))
        ax.set_title(f"{title} {labels[i] if labels else i}")
        return (im,)

    anim = animation.FuncAnimation(fig, update, frames=len(arr), blit=False)
    anim.save(str(path), writer=writer)
    plt.close(fig)
    return path


def render_loop_video(
    frames: Mapping[str, list],
    rms_nm: np.ndarray,
    path: str | Path,
    pupil=None,
    fps: int = 15,
    ts_ms: float = 1.0,
    title: str = "",
) -> Path:
    path, writer = _writer(Path(path), fps)
    path.parent.mkdir(parents=True, exist_ok=True)
    has_psf = bool(frames.get("psf"))
    ncols = 4 + has_psf
    turb = np.asarray(frames["turb"]) * 1e9
    res = np.asarray(frames["res"]) * 1e9
    sig = np.asarray(frames["sig"])
    ks = list(frames["k"])
    vt = float(np.percentile(np.abs(turb), 99.5)) or 1.0
    vr = float(np.percentile(np.abs(res), 99.5)) or 1.0
    vs = float(np.percentile(np.abs(sig), 99.5)) or 1.0
    fig, ax = plt.subplots(1, ncols, figsize=(4.2 * ncols, 4))
    im_t = ax[0].imshow(_mask(turb[0], pupil), cmap="RdBu_r", vmin=-vt, vmax=vt)
    ax[0].set_title("turbulence [nm]")
    im_r = ax[1].imshow(_mask(res[0], pupil), cmap="RdBu_r", vmin=-vr, vmax=vr)
    ax[1].set_title("residual [nm]")
    im_s = ax[2].imshow(sig[0], cmap="RdBu_r", vmin=-vs, vmax=vs)
    ax[2].set_title("WFS signal")
    col = 3
    if has_psf:
        psf = np.log10(np.asarray(frames["psf"]) + 1e-8)
        im_p = ax[3].imshow(psf[0], cmap="inferno", vmin=psf.min(), vmax=psf.max())
        ax[3].set_title("PSF (log)")
        col = 4
    for a in ax[:col]:
        a.grid(False)
        a.set_xticks([])
        a.set_yticks([])
    t = np.arange(len(rms_nm)) * ts_ms
    ax[col].plot(t, rms_nm, color="0.6", lw=1)
    (marker,) = ax[col].plot([t[ks[0]]], [rms_nm[ks[0]]], "ro")
    ax[col].set_xlabel("t [ms]")
    ax[col].set_ylabel("residual RMS [nm]")
    fig.suptitle(title)
    fig.tight_layout()

    def update(i):
        im_t.set_data(_mask(turb[i], pupil))
        im_r.set_data(_mask(res[i], pupil))
        im_s.set_data(sig[i])
        if has_psf:
            im_p.set_data(psf[i])
        marker.set_data([t[ks[i]]], [rms_nm[ks[i]]])
        return ()

    anim = animation.FuncAnimation(fig, update, frames=len(ks), blit=False)
    anim.save(str(path), writer=writer)
    plt.close(fig)
    return path


def render_comparison_video(
    runs: Mapping[str, object], path: str | Path, pupil=None, fps: int = 15, ts_ms: float = 1.0, title: str = ""
) -> Path:
    """Residual maps of several controllers side by side (runs must have recorded frames)."""
    path, writer = _writer(Path(path), fps)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(runs)
    res = {n: np.asarray(runs[n].frames["res"]) * 1e9 for n in names}
    ks = runs[names[0]].frames["k"]
    v = max(float(np.percentile(np.abs(r), 99.5)) for r in res.values()) or 1.0
    fig, ax = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4.2), squeeze=False)
    ims = []
    for a, n in zip(ax[0], names):
        ims.append(a.imshow(_mask(res[n][0], pupil), cmap="RdBu_r", vmin=-v, vmax=v))
        a.grid(False)
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(title)

    def update(i):
        for a, im, n in zip(ax[0], ims, names):
            im.set_data(_mask(res[n][i], pupil))
            a.set_title(f"{n}\nRMS {runs[n].rms[ks[i]] * 1e9:.0f} nm | t={ks[i] * ts_ms:.0f} ms", fontsize=9)
        return ims

    anim = animation.FuncAnimation(fig, update, frames=len(ks), blit=False)
    anim.save(str(path), writer=writer)
    plt.close(fig)
    return path
