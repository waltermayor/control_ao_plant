"""Component studies: examine one part of the system without running the full control pipeline.

study_atmosphere   turbulence statistics, modal spectrum, temporal PSDs, AR predictability, video
study_sensor       WFS images, linearity sweeps, optical gain vs turbulence strength, noise vs magnitude
study_calibration  modal basis, interaction matrix, singular values, conditioning
study_plant        open vs closed loop, DM-sign check, integrator gain x delay sweep, loop videos
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import welch

from aoctrl.dynamics import skill_onestep
from aoctrl.evaluation import OPEN_LOOP
from aoctrl.experiments.base import Experiment, register
from aoctrl.metrics import tail_mean
from aoctrl.simulation import measure_noise_floor, simulate
from aoctrl.viz import plots, render_loop_video, render_maps_video


def _unflatten(v, pupil):
    out = np.zeros(pupil.shape)
    out[pupil] = v
    return out


@register
class StudyAtmosphere(Experiment):
    name = "study_atmosphere"
    description = "Turbulence statistics, modal variance and temporal spectra, AR predictability"

    def run(self):
        ctx, ec = self.ctx, self.ecfg
        b = ctx.backend
        b.new_turbulence(int(ec.seed))
        b.advance_turbulence()
        b.propagate()
        opd = b.turbulence_opd()
        self.metrics(
            dict(
                opd_rms_nm=float(opd.std() * 1e9),
                seeing_arcsec=b.seeing_arcsec() or float("nan"),
                n_layers=len(self.cfg.system.atmosphere.layers),
            )
        )
        self.figure("overview", plots.plot_system_overview(b.pupil, _unflatten(opd, b.pupil)))

        ts = ctx.turbulence
        X = ts.coefs
        self.metrics(dict(explained_variance=ts.explained_variance))
        self.figure(
            "modal_variance",
            plots.plot_curves(
                {"modal std [nm]": X.std(axis=0) * 1e9}, "mode", "std [nm]", "Modal turbulence spectrum", logy=True
            ),
        )
        fs = 1.0 / b.sampling_time
        curves = {}
        for m in ec.psd_modes:
            if m < X.shape[1]:
                f, P = welch(X[:, m], fs=fs, nperseg=int(ec.nperseg))
                curves[f"mode {m}"] = (f[1:], P[1:])
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        for nm, (f, P) in curves.items():
            ax.loglog(f, P, label=nm)
        ax.set_xlabel("frequency [Hz]")
        ax.set_ylabel("PSD [m²/Hz]")
        ax.set_title("Temporal PSD of modal turbulence")
        ax.legend(fontsize=8)
        self.figure("temporal_psd", fig)
        cut = int(float(self.cfg.identification.turbulence.train_frac) * len(X))
        self.figure(
            "ar_skill",
            plots.plot_curves(
                {f"AR({p})": skill_onestep(X, ar.coefs, cut) for p, ar in ctx.ar_models.items()},
                "mode",
                "1 - MSE/MSE_persistence",
                "AR one-step gain over persistence",
                hline=0.0,
            ),
        )
        self.table(
            "ar_coefficients",
            pd.DataFrame(
                {f"AR({p}) a{j + 1}": ar.coefs[j] for p, ar in ctx.ar_models.items() for j in range(ar.order)}
            ),
        )
        if bool(self.cfg.video.enabled):
            b.new_turbulence(int(ec.seed))
            maps = []
            for k in range(int(self.cfg.video.n_steps)):
                b.advance_turbulence()
                if k % int(self.cfg.video.record_every) == 0:
                    maps.append(_unflatten(b.propagate_turbulence_only(), b.pupil))
            path = render_maps_video(
                maps,
                self.logger.dir / "videos" / f"turbulence.{self.cfg.video.format}",
                fps=int(self.cfg.video.fps),
                title="turbulent OPD",
                pupil=b.pupil,
                scale=1e9,
                unit="nm",
            )
            self.video("turbulence", path)
        return self.result


@register
class StudySensor(Experiment):
    name = "study_sensor"
    description = "WFS response: images, linearity sweeps, optical gain vs r0, noise vs magnitude"

    def run(self):
        ctx, ec = self.ctx, self.ecfg
        b, calib, plant = ctx.backend, ctx.calibration, ctx.plant
        zero = np.zeros(b.pupil.shape)
        b.new_turbulence(int(ec.seed))
        b.set_dm(0)
        b.advance_turbulence()
        b.propagate()
        m = b.maps()
        self.figure("wfs_images", plots.plot_system_overview(b.pupil, m.get("turb", zero), b.wfs_frame(), m.get("sig")))

        # linearity sweep of selected modes, no turbulence
        amps = np.linspace(-float(ec.sweep_amp_nm), float(ec.sweep_amp_nm), int(ec.sweep_points)) * 1e-9
        modes = [i for i in ec.sweep_modes if i < calib.n_modes]
        curves = {}
        for i in modes:
            resp = []
            for a in amps:
                b.advance_turbulence(zero, "replace")
                b.set_dm(calib.M2C[:, i] * a)
                resp.append((calib.Mrec @ b.propagate())[i])
            curves[f"mode {i}"] = np.array(resp) * 1e9
        b.set_dm(0)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for nm, r in curves.items():
            ax.plot(amps * 1e9, r, "o-", ms=3, label=nm)
        ax.plot(amps * 1e9, amps * 1e9, "k--", lw=1, label="ideal")
        ax.set_xlabel("applied modal amplitude [nm]")
        ax.set_ylabel("reconstructed amplitude [nm]")
        ax.set_title("WFS linearity (no turbulence)")
        ax.legend(fontsize=8)
        self.figure("linearity", fig)

        # optical gain under open-loop turbulence for each regime: slope of a small push-pull
        a = float(ec.og_probe_nm) * 1e-9
        rows = []
        for rg, cond in ctx.regimes.items():
            with plant.conditions(**cond):
                gains = np.zeros((int(ec.og_realisations), len(modes)))
                for r in range(int(ec.og_realisations)):
                    b.new_turbulence(int(ec.seed) + 100 + r)
                    b.advance_turbulence()
                    for j, i in enumerate(modes):
                        b.set_dm(calib.M2C[:, i] * a)
                        yp = calib.Mrec @ b.propagate()
                        b.set_dm(-calib.M2C[:, i] * a)
                        ym = calib.Mrec @ b.propagate()
                        gains[r, j] = (yp[i] - ym[i]) / (2 * a)
                b.set_dm(0)
            for j, i in enumerate(modes):
                rows.append(
                    dict(regime=rg, r0=cond["r0"], mode=i, optical_gain=gains[:, j].mean(), std=gains[:, j].std())
                )
        og = pd.DataFrame(rows)
        self.table("optical_gain_open_loop", og)
        self.figure(
            "optical_gain_vs_regime",
            plots.plot_bars(
                {f"{r.regime} m{r.mode}": r.optical_gain for r in og.itertuples()},
                "optical gain",
                "Open-loop optical gain",
                1.0,
            ),
        )

        # measurement noise vs guide-star magnitude
        noise = {}
        for mag in ec.noise_magnitudes:
            nf = measure_noise_floor(
                plant, calib, int(ec.noise_steps), magnitude=float(mag), photon_noise=True, delay=0
            )
            noise[f"mag {mag}"] = float(np.sqrt(np.mean(nf**2)) * 1e9)
        self.metrics({f"noise_pv_rms_nm/{k}": v for k, v in noise.items()})
        self.figure(
            "noise_vs_magnitude", plots.plot_bars(noise, "pv noise RMS per mode [nm]", "Measurement noise (photon)")
        )
        return self.result


@register
class StudyCalibration(Experiment):
    name = "study_calibration"
    description = "Modal basis, interaction matrix, singular values"

    def run(self):
        ctx, calib = self.ctx, self.ctx.calibration
        b = ctx.backend
        ids = [i for i in self.ecfg.show_modes if i < calib.n_modes]
        maps = []
        for i in ids:
            mmap = _unflatten(calib.B[:, i], b.pupil)
            maps.append(mmap / (calib.B[:, i].std() + 1e-30))
        self.figure("calibration", plots.plot_calibration(calib.singular_values, calib.D, maps, b.pupil, ids))
        self.metrics(
            dict(
                n_modes=calib.n_modes,
                n_act=b.n_act,
                n_signal=b.n_signal,
                cond=calib.cond,
                explained_variance=ctx.turbulence.explained_variance,
            )
        )
        self.result.artifacts.update(calibration=calib, backend_summary=b.summary())
        return self.result


@register
class StudyPlant(Experiment):
    name = "study_plant"
    description = "Open vs closed loop, DM-sign check, integrator gain x delay sweep, loop videos"

    def run(self):
        ctx, ec = self.ctx, self.ecfg
        calib, plant = ctx.calibration, ctx.plant
        n = int(ec.n_steps)
        ol = simulate(plant, n, None, seed=int(ec.seed), projector=calib.Bpinv)
        cl = simulate(plant, n, ctx.integrator(), seed=int(ec.seed), projector=calib.Bpinv)
        self.metrics(
            {
                "open_loop_rms_nm": tail_mean(ol.rms) * 1e9,
                "closed_loop_rms_nm": tail_mean(cl.rms) * 1e9,
                "open_loop_strehl": tail_mean(ol.strehl),
                "closed_loop_strehl": tail_mean(cl.strehl),
            }
        )
        self.figure(
            "open_vs_closed",
            plots.plot_timeseries({OPEN_LOOP: ol, ctx.integrator_name: cl}, ctx.ts_ms, "Open vs closed loop"),
        )
        sg = ctx.dm_sign
        self.metrics(
            dict(dm_sign=sg["sign"], dm_slope=sg["slope"], dm_fit_rel_error=sg["rel_error"], dm_sign_ok=sg["ok"])
        )

        rows = []
        for dl in ec.sweep_delays:
            for g in ec.sweep_gains:
                with plant.conditions(delay=int(dl)):
                    from aoctrl.control import Integrator

                    tel = simulate(
                        plant, int(ec.sweep_steps), Integrator(calib.R, float(g)), seed=int(ec.seed), keep_y=False
                    )
                rows.append(
                    dict(delay=int(dl), gain=float(g), rms_nm=tail_mean(tel.rms) * 1e9, strehl=tail_mean(tel.strehl))
                )
        sweep = pd.DataFrame(rows)
        self.table("integrator_gain_sweep", sweep)
        self.figure(
            "integrator_gain_sweep",
            plots.plot_curves(
                {f"delay {d}": g.set_index("gain")["rms_nm"].values for d, g in sweep.groupby("delay")},
                "gain",
                "residual RMS [nm]",
                "Integrator gain x delay",
                x=np.array(sorted(set(sweep.gain))),
            ),
        )

        vc = self.cfg.video
        if bool(vc.enabled):
            tel = simulate(
                plant,
                int(vc.n_steps),
                ctx.integrator(),
                seed=int(vc.seed),
                record_every=int(vc.record_every),
                record_psf=bool(vc.psf),
                keep_y=False,
            )
            path = render_loop_video(
                tel.frames,
                tel.rms * 1e9,
                self.logger.dir / "videos" / f"closed_loop.{vc.format}",
                pupil=ctx.backend.pupil,
                fps=int(vc.fps),
                ts_ms=ctx.ts_ms,
                title="integrator closed loop",
            )
            self.video("closed_loop", path)
            self.comparison_video(
                {OPEN_LOOP: lambda dl, rg: None, ctx.integrator_name: lambda dl, rg: ctx.integrator()}, "open_vs_closed"
            )
        self.result.artifacts.update(open_loop=ol, closed_loop=cl)
        return self.result
