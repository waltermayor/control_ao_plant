"""Cross-experiment comparison (notebook §8.8 / §9.7 tables) from saved summaries.

Each experiment writes `results/<experiment>/<system_hash>/summary.csv`; this experiment merges
the ones available for the current system (references are deduplicated) and recomputes the
"retained" column against the common integrator and oracle.
"""

from __future__ import annotations

import pandas as pd

from aoctrl.evaluation import format_table
from aoctrl.experiments.base import Experiment, register
from aoctrl.metrics import gap_closed
from aoctrl.utils.paths import resolve_path
from aoctrl.viz import plots


@register
class Compare(Experiment):
    name = "compare"
    description = "Merge saved summaries of experiments 1-4 into one comparison"

    def run(self):
        ctx = self.ctx
        root = resolve_path(self.cfg.paths.results_dir)
        frames = []
        for exp in self.ecfg.experiments:
            p = root / exp / ctx.system_hash / "summary.csv"
            if p.exists():
                frames.append(pd.read_csv(p).assign(source=exp))
            else:
                self.info("no saved summary for %s (%s); run it first", exp, p)
        if not frames:
            raise FileNotFoundError(f"no summaries found under {root} for system hash {ctx.system_hash}")
        df = (
            pd.concat(frames)
            .drop_duplicates(subset=["regime", "delay", "controller"], keep="first")
            .reset_index(drop=True)
        )
        lo, hi = ctx.integrator_name, ctx.oracle_name()
        for (rg, dl), g in df.groupby(["regime", "delay"]):
            r = g.set_index("controller")["rms"]
            if lo in r and hi in r:
                df.loc[g.index, "retained"] = [gap_closed(v, r[lo], r[hi]) for v in g.rms]
        self.table("comparison", df)
        self.result.summary = None  # comparison is derived; do not overwrite per-experiment summaries
        self.result.tables["comparison"] = df
        self.info("%s", format_table(df))
        show = [
            c
            for c in (self.ecfg.show or list(dict.fromkeys(df.controller)))
            if c in set(df.controller) and c != "open loop"
        ]
        self.figure("comparison_bars", plots.plot_condition_bars(df, show))
        return self.result
