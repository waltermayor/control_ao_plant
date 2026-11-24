"""Experiment 4 — Deep Koopman with control -> LQR. Notebook §9.

Generic Koopman autoencoder (free latent, first-order linear dynamics conditioned on the DM
commands, full B), delay-embedded observables O_t = [p_t, p_{t-1}, c_t, c_{t-1}], initialised by
DMDc (ridge chosen on validation, no truth).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aoctrl.control import KoopmanLQR, physical_koopman_params, verify_koopman_lqr
from aoctrl.evaluation import OPEN_LOOP, format_table, summarize
from aoctrl.experiments.base import Experiment, register
from aoctrl.experiments.latent_common import multistep_errors, persistence_prediction, probe_errors, train_or_load
from aoctrl.models import DeepKoopman, arx_init_coefs, dmdc_init, latent_metrics, structured_init
from aoctrl.models import numpy_ops as ops
from aoctrl.models.data import split_runs
from aoctrl.training import evaluate_loss_terms
from aoctrl.viz import plots

LIN_NAME, DEEP_NAME, PHYS_NAME = "LQR Koopman linear (DMDc)", "LQR Deep Koopman", "LQR Koopman physical (reference)"


@register
class Exp4DeepKoopman(Experiment):
    name = "exp4_deep_koopman"
    description = "Deep Koopman autoencoder with control (free latent) -> LQR"

    def run(self):
        ctx, mc = self.ctx, self.cfg.model
        if mc.kind != "deep_koopman":
            raise ValueError("exp4 expects model=deep_koopman")
        ds, sign, n = ctx.latent_dataset, ctx.sign, ctx.n_modes
        nz = int(mc.nz) if mc.get("nz") else int(mc.nz_factor) * n
        W = dict(mc.train.weights)
        frac = 1.0 - float(self.cfg.identification.latent_data.val_frac)

        # --- DMDc initialisation: choose the ridge on validation (one step + free rollout), no truth
        p_tr, c_tr = split_runs(ctx.latent_runs, ctx.latent_sigma, frac)
        cands = []
        for rd in mc.ridges:
            Pc = dmdc_init(p_tr, c_tr, nz, int(mc.hidden_enc), int(mc.hidden_dec), int(mc.init_seed), ridge=float(rd))
            tv = evaluate_loss_terms(DeepKoopman, Pc, ds.val_p, ds.val_c, W)
            cands.append(
                dict(
                    ridge=float(rd),
                    one=tv["one"],
                    pred=tv["pred"],
                    score=tv["one"] + tv["pred"],
                    rho_A=float(np.max(np.abs(np.linalg.eigvals(Pc["A"])))),
                    P=Pc,
                )
            )
        best = min(cands, key=lambda c: c["score"])
        P0 = best["P"]
        self.table("dmdc_ridge_selection", pd.DataFrame([{k: v for k, v in c.items() if k != "P"} for c in cands]))
        self.metrics(dict(dmdc_ridge=best["ridge"], latent_dim=nz))

        res = train_or_load(self, DeepKoopman(P0), "exp4_deep_koopman")
        P = res.params
        v0 = evaluate_loss_terms(DeepKoopman, P0, ds.val_p, ds.val_c, W)
        v1 = evaluate_loss_terms(DeepKoopman, P, ds.val_p, ds.val_c, W)
        self.table(
            "validation_terms",
            pd.DataFrame([dict(model="Koopman linear (DMDc)", **v0), dict(model="Deep Koopman", **v1)]),
        )
        self.figure("training", plots.plot_training({"deep koopman": res.history}))

        # --- offline diagnostics
        Parx = structured_init(arx_init_coefs(ctx.latent_runs, ctx.latent_sigma), 2, 0)
        E = multistep_errors(
            ctx,
            {
                "persistence": persistence_prediction(ds.val_p, ds.val_c, sign),
                "ARX (Exp 3 init)": ops.latent_arx_rollout(Parx, ds.val_p, ds.val_c, identity=True),
                "Koopman linear (DMDc)": ops.koopman_rollout(P0, ds.val_p, ds.val_c, linear=True),
                "Deep Koopman": ops.koopman_rollout(P, ds.val_p, ds.val_c),
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
        self.figure(
            "koopman_spectrum",
            plots.plot_eigenvalues({"DMDc": np.linalg.eigvals(P0["A"]), "Deep Koopman": np.linalg.eigvals(P["A"])}),
        )
        errs, preds, truth, probes = probe_errors(
            ctx,
            {
                "p only": lambda O, p: p,
                "raw O": lambda O, p: O,
                "z (DMDc)": lambda O, p: ops.koopman_encode(P0, O, True),
                "z (Deep Koopman)": lambda O, p: ops.koopman_encode(P, O),
            },
            frac,
        )
        self.metrics({f"probe_error_nm/{k}": v for k, v in errs.items()})
        self.figure(
            "sensor_linearisation",
            plots.plot_probe_scatter(truth, {k: preds[k] for k in ("p only", "z (Deep Koopman)")}),
        )

        # --- LQR in the Koopman latent
        rho = float(self.cfg.control.lqr.rho)
        has_phys = sign > 0 and 2 in ctx.ar_models
        if has_phys:
            _, WC = latent_metrics(ctx.latent_sigma, ctx.calibration.W_phys)
            chk = verify_koopman_lqr(ctx.ar_models[2].coefs, self.cfg.eval.delays, rho, WC)
            self.metrics({f"koopman_lqr_vs_oracle/D{d}": v["rel_error"] for d, v in chk.items()})
        io = ctx.modal_io()
        rmax = float(mc.rmax)
        factories = {
            LIN_NAME: lambda dl, rg: KoopmanLQR(P0, io, dl, LIN_NAME, linear=True, probe=probes["z (DMDc)"], rmax=rmax),
            DEEP_NAME: lambda dl, rg: KoopmanLQR(P, io, dl, DEEP_NAME, probe=probes["z (Deep Koopman)"], rmax=rmax),
        }
        if has_phys and bool(mc.include_physical_reference):
            Pph = physical_koopman_params(ctx.ar_models[2].coefs)
            factories[PHYS_NAME] = lambda dl, rg: KoopmanLQR(Pph, io, dl, PHYS_NAME, linear=True, rmax=rmax)
        rs4 = ctx.evaluate(factories)
        rs = ctx.reference_runs.merged(rs4)
        orc = ctx.oracle_name()
        names = [OPEN_LOOP, ctx.integrator_name, *factories, orc]
        df = summarize(
            rs,
            ctx.calibration.W_phys,
            ref_low=ctx.integrator_name,
            ref_high=orc,
            tags={
                OPEN_LOOP: "-",
                ctx.integrator_name: "ref",
                orc: "Exp 1",
                LIN_NAME: "Exp 4",
                DEEP_NAME: "Exp 4",
                PHYS_NAME: "Exp 4r",
            },
        )
        df = df[df.controller.isin(names)].reset_index(drop=True)
        self.table("summary", df)
        self.result.summary = df
        self.info("%s", format_table(df))
        self.figure("rms_bars", plots.plot_condition_bars(df, names[1:]))
        d0 = int(self.cfg.eval.delays[0])
        for rg in ctx.regimes:
            runs = rs.at(rg, d0)
            self.figure(
                f"timeseries_{rg}_d{d0}",
                plots.plot_timeseries(
                    {nm: runs[nm][0] for nm in [OPEN_LOOP, ctx.integrator_name, DEEP_NAME, orc]},
                    ctx.ts_ms,
                    f"regime {rg}, delay {d0}",
                ),
            )
        ref = ctx.reference_factories()
        self.comparison_video(
            {ctx.integrator_name: ref[ctx.integrator_name], DEEP_NAME: factories[DEEP_NAME], orc: ref[orc]},
            "controllers",
        )
        art = self.logger.dir / "exp4_koopman.npz"
        np.savez(
            art, sigma=ctx.latent_sigma, **{"P_" + k: v for k, v in P.items()}, **{"P0_" + k: v for k, v in P0.items()}
        )
        self.logger.log_artifact(art, name="exp4-koopman-model", type="model")
        self.result.artifacts.update(results=rs, params=P, init_params=P0, history=res.history)
        return self.result
