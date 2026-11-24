"""Experiment 2 — Pyramid measurements + EKF -> LQR. Notebook §7.

Same LQR gain as the Experiment 1 oracle, but the turbulence state is *estimated* from the
Pyramid with an EKF whose measurement model is an identified optical-gain law. Variants isolate:
filtering + prediction (KF, K=1), linear optical gain (KF, K0 fitted), compression (EKF).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aoctrl.control import build_ekf_lqr
from aoctrl.evaluation import OPEN_LOOP, format_table, summarize
from aoctrl.experiments.base import Experiment, register
from aoctrl.metrics import estimation_error_nm
from aoctrl.sensors import OpticalGainModel, fit_optical_gain
from aoctrl.viz import plots


@register
class Exp2EKFLQR(Experiment):
    name = "exp2_ekf_lqr"
    description = "Pyramid + EKF (optical-gain model) -> LQR; gap decomposition vs the oracle"

    def identify_sensor(self):
        ctx, calib = self.ctx, self.ctx.calibration
        oc = self.cfg.identification.optical_gain
        burn = int(oc.burn)
        PV = np.concatenate([d["pv"][burn:] for d in ctx.og_data])
        S = np.concatenate([d["s"][burn:] for d in ctx.og_data])
        R = np.sqrt(np.maximum(np.einsum("ti,ij,tj->t", S, calib.W_phys, S), 0.0))
        grid = np.logspace(float(oc.rm_grid.log10_min), float(oc.rm_grid.log10_max), int(oc.rm_grid.n))
        fit = fit_optical_gain(PV, S, R, qs=tuple(oc.qs), rms_grid=grid)
        self.metrics(
            dict(
                og_q=fit.q,
                og_rm_nm=fit.rm * 1e9,
                og_loss=fit.loss,
                og_loss_linear=fit.loss_linear,
                og_K0_mean=float(fit.K0.mean()),
            )
        )
        if fit.rm_at_grid_edge:
            self.info("!! r_m at the edge of the grid: widen the excitation or the grid")
        self.figure("optical_gain_fit", plots.plot_optical_gain(S, PV, R, fit))
        og = OpticalGainModel(fit.K0, fit.q, fit.rm, calib.W_phys)
        jac = {lab: og.jacobian_check(s0) for lab, s0 in [("typical", S[len(S) // 2]), ("large", S[np.argmax(R)])]}
        self.metrics({f"jacobian_rel_error/{k}": v for k, v in jac.items()})
        return fit, og

    def run(self):
        ctx, calib = self.ctx, self.ctx.calibration
        W, rho, ec = calib.W_phys, float(self.cfg.control.lqr.rho), self.ecfg.ekf
        fit, og = self.identify_sensor()
        noise = ctx.noise_floor
        n, pm = ctx.n_modes, ctx.ar_max
        sensors = {
            f"KF linear (K=1) + LQR AR({pm})": (OpticalGainModel.linear(np.ones(n), W), pm),
            f"KF linear (K0 fitted) + LQR AR({pm})": (OpticalGainModel.linear(fit.K0, W), pm),
            **{f"EKF + LQR AR({p})": (og, p) for p in ctx.ar_models},
        }

        def factory(sensor, p, name):
            def make(dl, rg):
                qs = ctx.q_scale(rg) if bool(ec.knows_r0) else 1.0
                return build_ekf_lqr(
                    ctx.ar_models[p],
                    dl,
                    sensor,
                    noise,
                    float(ec.model_err),
                    qs,
                    rho,
                    calib.W_norm,
                    calib.M2C,
                    calib.Mrec,
                    ctx.sign,
                    name,
                )

            return make

        rs2 = ctx.evaluate({nm: factory(s, p, nm) for nm, (s, p) in sensors.items()})
        rs = ctx.reference_runs.merged(rs2)
        orc, ekf = ctx.oracle_name(), f"EKF + LQR AR({pm})"
        tags = {OPEN_LOOP: "-", ctx.integrator_name: "ref", orc: "Exp 1", **{k: "Exp 2" for k in sensors}}
        keep = [OPEN_LOOP, ctx.integrator_name, *sensors, orc]
        df = summarize(rs, W, ref_low=ctx.integrator_name, ref_high=orc, tags=tags)
        df = df[df.controller.isin(keep)].reset_index(drop=True)
        self.table("summary", df)
        self.result.summary = df
        self.info("%s", format_table(df))

        # gap decomposition integrator -> oracle
        rows = []
        k1, k0 = list(sensors)[:2]
        for (rg, dl), g in df.groupby(["regime", "delay"]):
            r = g.set_index("controller")["rms"]
            tot = r[ctx.integrator_name] - r[orc]
            steps = [
                ("filtering + prediction (I -> KF K=1)", r[ctx.integrator_name] - r[k1]),
                ("linear optical gain (K=1 -> K0)", r[k1] - r[k0]),
                ("non-linearity / compression (K0 -> EKF)", r[k0] - r[ekf]),
                ("remaining sensing cost (EKF -> oracle)", r[ekf] - r[orc]),
            ]
            for lab, v in steps:
                rows.append(
                    dict(regime=rg, delay=dl, step=lab, nm=v, percent=v / tot * 100 if abs(tot) > 1e-9 else np.nan)
                )
            rr = rs2.at(rg, dl)[ekf]
            rows.append(
                dict(
                    regime=rg,
                    delay=dl,
                    step="state error: EKF [nm]",
                    nm=estimation_error_nm(rr, "est", W),
                    percent=np.nan,
                )
            )
            rows.append(
                dict(
                    regime=rg,
                    delay=dl,
                    step="state error: direct pv - c [nm]",
                    nm=estimation_error_nm(rr, "raw", W),
                    percent=np.nan,
                )
            )
        self.table("gap_decomposition", pd.DataFrame(rows))

        # plots
        show = [ctx.integrator_name, *sensors, orc]
        self.figure("rms_bars", plots.plot_condition_bars(df, show, [s.replace(" + LQR", "\n+ LQR") for s in show]))
        d0 = int(self.cfg.eval.delays[0])
        sw = np.sqrt(np.diag(W)) * 1e9
        for rg in ctx.regimes:
            runs = rs.at(rg, d0)
            self.figure(
                f"timeseries_{rg}_d{d0}",
                plots.plot_timeseries(
                    {nm: runs[nm][0] for nm in [OPEN_LOOP, ctx.integrator_name, ekf, orc]},
                    ctx.ts_ms,
                    f"regime {rg}, delay {d0}",
                ),
            )
            er = rs2.at(rg, d0)[ekf]

            def per_mode(key):
                return (
                    np.sqrt(
                        np.mean([np.mean((r.diag[key] - r.d[: len(r.diag[key])]) ** 2, axis=0) for r in er], axis=0)
                    )
                    * sw
                )

            self.figure(
                f"state_error_per_mode_{rg}_d{d0}",
                plots.plot_curves(
                    {
                        "direct pv - c": per_mode("raw"),
                        "EKF": per_mode("est"),
                        "turbulence RMS": np.sqrt(np.mean([np.mean(r.d**2, axis=0) for r in er], axis=0)) * sw,
                    },
                    "KL mode",
                    "estimation error [nm]",
                    f"State estimate, {rg}, delay {d0}",
                    logy=True,
                ),
            )
        nis = df[df.controller == ekf].set_index(["regime", "delay"])["nis"]
        self.figure(
            "nis",
            plots.plot_bars(
                {f"{rg} D={dl}": v for (rg, dl), v in nis.items()}, "mean NIS", "EKF consistency (ideal = 1)", hline=1.0
            ),
        )
        art = self.logger.dir / "exp2_ekf.npz"
        np.savez(art, K0=fit.K0, q=fit.q, rm=fit.rm, noise_pv=noise)
        self.logger.log_artifact(art, name="exp2-sensor-model", type="model")
        self.result.artifacts.update(results=rs, optical_gain=fit)
        return self.result
