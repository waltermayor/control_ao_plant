"""Runnable experiments. Importing this package registers all of them.

Add a new experiment: create a module with an `Experiment` subclass decorated with `@register`,
import it here, and add `configs/experiment/<name>.yaml`.
"""

from aoctrl.experiments import (  # noqa: F401  (registration side effects)
    compare,
    exp1_oracle_lqr,
    exp2_ekf_lqr,
    exp3_latent_lqr,
    exp4_deep_koopman,
    studies,
)
from aoctrl.experiments.base import REGISTRY, Experiment, ExperimentResult, get_experiment, register
from aoctrl.experiments.context import ExperimentContext

__all__ = ["REGISTRY", "Experiment", "ExperimentResult", "ExperimentContext", "get_experiment", "register"]
