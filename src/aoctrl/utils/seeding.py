from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch global RNGs.

    Simulation code should still prefer explicit `np.random.default_rng(seed)` generators;
    this only guards against libraries that use the global state.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass
