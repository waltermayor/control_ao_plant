"""Experiment 1 — LQR with the true state (KL coefficients of the turbulence). Notebook §6.

Isolates the error due to the temporal model (AR), the DM delay and modal fitting, *without a
sensor*. Its oracle bounds experiments 2–4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aoctrl.control import build_oracle_lqr
from aoctrl.dynamics import persistence, prediction_error_nm, predictor_weights, skill_onestep
from aoctrl.evaluation import OPEN_LOOP, format_table, summarize
from aoctrl.experiments.base import Experiment, register
from aoctrl.metrics import modal_rejection, temporal_rejection
from aoctrl.simulation import simulate
from aoctrl.viz import plots


@register
class Exp1OracleLQR(Experiment):
    name = "exp1_oracle_lqr"
    description = "True-state LQR (oracle) vs integrator, normal/strong turbulence, several DM delays"

    def run(self):
        ctx, calib = self.ctx, self.ctx.calibration
        W = calib.W_phys
        rho = float(self.cfg.control.lqr.rho)

        # --- 1. truth and DM-sign sanity checks
        ts, sg = ctx.turbulence, ctx.dm_sign
        self.metrics(
            dict(
                explained_variance=ts.explained_variance,
                dm_sign=sg["sign"],
                dm_slope=sg["slope"],
                dm_fit_rel_error=sg["rel_error"],
            )
        )
        self.info(
            "modal basis explains %.1f %% of OPD variance; DM sign %+d (slope %.3f, rel. err %.1e)",
            100 * ts.explained_variance,
            sg["sign"],
            sg["slope"],
            sg["rel_error"],
        )

        # --- 2. AR model quality (validation split)
        X = ts.coefs
        cut = int(float(self.cfg.identification.turbulence.train_frac) * len(X))
        Xv = X[cut:]
        horizons = list(self.ecfg.horizons)
        pred = {}
        for th in horizons:
            row = {
                "no control": float(np.sqrt(np.mean(np.einsum("ti,ij,tj->t", Xv, W, Xv))) * 1e9),
                "persistence": prediction_error_nm(Xv, persistence(ctx.n_modes), th, W),
            }
            for p, ar in ctx.ar_models.items():
                row[f"AR({p})"] = prediction_error_nm(Xv, predictor_weights(ar.coefs, th), th, W)
            pred[th] = row
        pred_df = pd.DataFrame(pred).T.rename_axis("horizon").reset_index()
        self.table("ar_prediction_error_nm", pred_df)
        self.figure(
            "ar_prediction_error",
            plots.plot_prediction_table(
                {th: {k: v for k, v in r.items() if k != "no control"} for th, r in pred.items()}
            ),
        )
        self.figure(
            "ar_skill_per_mode",
            plots.plot_curves(
                {f"AR({p})": skill_onestep(X, ar.coefs, cut) for p, ar in ctx.ar_models.items()},
                "KL mode",
                "1 - MSE/MSE_persistence",
                "One-step prediction gain over persistence",
                hline=0.0,
            ),
        )

        # --- 3. LQR construction check: DARE vs closed form K = G/(1+rho)
        Wn = calib.W_norm
        rows = []
        for p, ar in ctx.ar_models.items():
            for D in sorted(set(int(d) for d in self.cfg.eval.delays)):
                mo = build_oracle_lqr(ar.coefs, D, rho, Wn)
                rows.append(
                    dict(
                        ar=f"AR({p})",
                        delay=D,
                        state_dim=mo.A.shape[0],
                        method=mo.method,
                        dare_vs_closed_form=mo.dare_vs_closed_form,
                    )
                )
        self.table("oracle_lqr_construction", pd.DataFrame(rows))

        # --- 4. closed-loop evaluation (cached reference runs: shared with experiments 2–4)
        rs = ctx.reference_runs
        orc = ctx.oracle_name()
        df = summarize(
            rs,
            W,
            ref_low=ctx.integrator_name,
            ref_high=orc,
            tags={OPEN_LOOP: "-", ctx.integrator_name: "ref", orc: "Exp 1"},
        )
        # expected oracle RMS from the AR model (offline validation) vs measured
        expected = []
        for (rg, dl), g in df.groupby(["regime", "delay"]):
            scale = (ctx.r0_ref / ctx.regimes[rg]["r0"]) ** (5 / 6)
            th = dl + 1
            if th in pred:
                pm = pred[th][f"AR({ctx.ar_max})"] * scale
                expected.append(
                    dict(
                        regime=rg,
                        delay=dl,
                        expected_oracle_rms=np.sqrt(pm**2 + g.fitting_floor.iloc[0] ** 2),
                        measured_oracle_rms=float(g.set_index("controller").loc[orc, "rms"]),
                    )
                )
        self.table("oracle_expected_vs_measured", pd.DataFrame(expected))
        self.table("summary", df)
        self.result.summary = df
        self.info("%s", format_table(df))

        ctrls = [c for c in rs.at(*rs.conditions()[0]) if c != OPEN_LOOP]
        self.figure("rms_bars", plots.plot_condition_bars(df, ctrls))
        d0 = int(self.cfg.eval.delays[0])
        for rg in ctx.regimes:
            runs = rs.at(rg, d0)
            show = [OPEN_LOOP, ctx.integrator_name, "LQR oracle persist.", orc]
            self.figure(
                f"timeseries_{rg}_d{d0}",
                plots.plot_timeseries(
                    {n: runs[n][0] for n in show if n in runs}, ctx.ts_ms, f"regime {rg}, delay {d0}"
                ),
            )
            self.figure(
                f"modal_rejection_{rg}_d{d0}",
                plots.plot_curves(
                    {n: modal_rejection(runs[n]) for n in [ctx.integrator_name, "LQR oracle persist.", orc]},
                    "KL mode",
                    "RMS residual / RMS turbulence",
                    f"Modal rejection, {rg}, delay {d0} (<1 = corrects)",
                    logy=True,
                    hline=1.0,
                ),
            )

        # --- 5. temporal rejection (longer run)
        rc = self.ecfg.rejection
        fac = ctx.reference_factories()
        fs = 1.0 / float(self.cfg.system.telescope.sampling_time)
        curves = {}
        with ctx.plant.conditions(delay=int(rc.delay), **ctx.regimes[str(rc.regime)]):
            for nm in [ctx.integrator_name, "LQR oracle persist.", orc]:
                tel = simulate(
                    ctx.plant,
                    int(rc.n_steps),
                    fac[nm](int(rc.delay), str(rc.regime)),
                    seed=int(rc.seed),
                    projector=calib.Bpinv,
                    keep_y=False,
                )
                curves[nm] = [temporal_rejection(tel, m, fs) for m in rc.modes]
        self.figure("temporal_rejection", plots.plot_temporal_rejection(curves, list(rc.modes)))

        # --- 6. headline metrics + artifacts
        for (rg, dl), g in df.groupby(["regime", "delay"]):
            gi = g.set_index("controller")
            self.metrics({f"rms/{rg}/d{dl}/{n}": float(gi.loc[n, "rms"]) for n in gi.index})
        art = self.logger.dir / "exp1_oracle.npz"
        np.savez(
            art,
            **{f"ar{p}": ar.coefs for p, ar in ctx.ar_models.items()},
            K_D0=build_oracle_lqr(ctx.ar_models[ctx.ar_max].coefs, 0, rho, Wn).K,
            rho=rho,
        )
        self.logger.log_artifact(art, name="exp1-oracle", type="model")
        self.comparison_video({n: fac[n] for n in (OPEN_LOOP, ctx.integrator_name, orc)}, "controllers")
        self.result.artifacts.update(results=rs, ar_models=ctx.ar_models)
        return self.result
