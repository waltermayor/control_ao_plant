"""Experiment 3 — Neural latent representation learned from the Pyramid -> LQR. Notebook §8.

Structured latent (encoder/decoder ≈ identity + residual MLP, ARX(2) latent dynamics with command
feedthrough) trained only on (y, u). Truth is used only to *evaluate* the latent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aoctrl.control import LatentARXLQR, verify_latent_arx_lqr
from aoctrl.evaluation import OPEN_LOOP, format_table, summarize
from aoctrl.experiments.base import Experiment, register
from aoctrl.experiments.latent_common import multistep_errors, persistence_prediction, probe_errors, train_or_load
from aoctrl.models import StructuredLatentARX, arx_init_coefs, latent_metrics, structured_init
from aoctrl.models import numpy_ops as ops
from aoctrl.training import evaluate_loss_terms
from aoctrl.viz import plots

LIN_NAME, NN_NAME = "LQR latent linear (ARX)", "LQR latent neural"


@register
class Exp3LatentLQR(Experiment):
    name = "exp3_latent_lqr"
    description = "Structured neural latent (E/D + ARX(2)) learned from (y, u) -> latent LQR"

    def run(self):
        ctx, mc = self.ctx, self.cfg.model
        if mc.kind != "structured_latent_arx":
            raise ValueError("exp3 expects model=structured_latent_arx")
        ds, sign = ctx.latent_dataset, ctx.sign

        # --- initial linear model (truth-free ARX on measurements) and training
        coef = arx_init_coefs(ctx.latent_runs, ctx.latent_sigma)
        P0 = structured_init(coef, int(mc.hidden), int(mc.init_seed))
        res = train_or_load(self, StructuredLatentARX(P0), "exp3_structured_latent")
        P = res.params
        W = dict(self.cfg.model.train.weights)
        v0 = evaluate_loss_terms(StructuredLatentARX, P0, ds.val_p, ds.val_c, W)
        v1 = evaluate_loss_terms(StructuredLatentARX, P, ds.val_p, ds.val_c, W)
        self.table("validation_terms", pd.DataFrame([dict(model="linear (initial)", **v0), dict(model="neural", **v1)]))
        self.figure("training", plots.plot_training({"structured latent": res.history}))

        # --- offline diagnostics: multi-step prediction and sensor linearisation (linear probe)
        E = multistep_errors(
            ctx,
            {
                "persistence": persistence_prediction(ds.val_p, ds.val_c, sign),
                "latent linear (ARX)": ops.latent_arx_rollout(P0, ds.val_p, ds.val_c, identity=True),
                "latent neural": ops.latent_arx_rollout(P, ds.val_p, ds.val_c),
            },
        )
        self.table(
            "multistep_prediction_nm",
            pd.DataFrame(E).rename_axis("horizon").reset_index().assign(horizon=lambda d: d.horizon + 1),
        )
        self.figure(
            "multistep_prediction",
            plots.plot_curves(E, "horizon [frames]", "prediction error [nm]", "Multi-step prediction (validation)"),
        )
        frac = 1.0 - float(self.cfg.identification.latent_data.val_frac)
        errs, preds, truth, _ = probe_errors(
            ctx, {"raw p": lambda O, p: p, "latent E(p)": lambda O, p: ops.mlp_residual(P, p, "e")}, frac
        )
        self.metrics({f"probe_error_nm/{k}": v for k, v in errs.items()})
        self.figure("sensor_linearisation", plots.plot_probe_scatter(truth, preds))

        # --- LQR in the latent space; sanity check with the physical dynamics
        if sign > 0 and 2 in ctx.ar_models:
            _, WC = latent_metrics(ctx.latent_sigma, ctx.calibration.W_phys)
            chk = verify_latent_arx_lqr(
                ctx.ar_models[2].coefs, self.cfg.eval.delays, float(self.cfg.control.lqr.rho), WC
            )
            self.metrics({f"latent_lqr_vs_oracle/D{d}": v["rel_error"] for d, v in chk.items()})
        io = ctx.modal_io()
        rmax = float(self.cfg.control.latent.rmax)
        factories = {
            LIN_NAME: lambda dl, rg: LatentARXLQR(P0, io, dl, LIN_NAME, identity=True, rmax=rmax),
            NN_NAME: lambda dl, rg: LatentARXLQR(P, io, dl, NN_NAME, rmax=rmax),
        }
        rs3 = ctx.evaluate(factories)
        rs = ctx.reference_runs.merged(rs3)
        orc = ctx.oracle_name()
        df = summarize(
            rs,
            ctx.calibration.W_phys,
            ref_low=ctx.integrator_name,
            ref_high=orc,
            tags={OPEN_LOOP: "-", ctx.integrator_name: "ref", orc: "Exp 1", LIN_NAME: "Exp 3", NN_NAME: "Exp 3"},
        )
        df = df[df.controller.isin([OPEN_LOOP, ctx.integrator_name, LIN_NAME, NN_NAME, orc])].reset_index(drop=True)
        self.table("summary", df)
        self.result.summary = df
        self.info("%s", format_table(df))
        self.figure(
            "rms_bars",
            plots.plot_condition_bars(
                df, [ctx.integrator_name, LIN_NAME, NN_NAME, orc], ["I", "latent linear", "latent neural", "oracle"]
            ),
        )
        d0 = int(self.cfg.eval.delays[0])
        for rg in ctx.regimes:
            runs = rs.at(rg, d0)
            self.figure(
                f"timeseries_{rg}_d{d0}",
                plots.plot_timeseries(
                    {nm: runs[nm][0] for nm in [OPEN_LOOP, ctx.integrator_name, NN_NAME, orc]},
                    ctx.ts_ms,
                    f"regime {rg}, delay {d0}",
                ),
            )
        ref = ctx.reference_factories()
        self.comparison_video(
            {ctx.integrator_name: ref[ctx.integrator_name], NN_NAME: factories[NN_NAME], orc: ref[orc]}, "controllers"
        )
        art = self.logger.dir / "exp3_latent.npz"
        np.savez(
            art, sigma=ctx.latent_sigma, **{"P_" + k: v for k, v in P.items()}, **{"P0_" + k: v for k, v in P0.items()}
        )
        self.logger.log_artifact(art, name="exp3-latent-model", type="model")
        self.result.artifacts.update(results=rs, params=P, init_params=P0, history=res.history)
        return self.result
