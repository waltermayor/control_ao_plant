"""Controller interface.

A controller maps the previous command and the latest measurement to the next command:

    u_{k} = controller(u_{k-1}, y_{k-1}, k)

Stateful controllers implement `reset()`. Oracles that use the simulated truth set
`needs_truth = True`; the runner then calls `observe(d_k)` after each step with the true modal
turbulence (1 frame of latency, same as an ideal WFS). `diagnostics()` returns per-step arrays
(state estimates, NIS, compute time, ...) that the runner stores in the telemetry.
"""

from __future__ import annotations

import time

import numpy as np


class Controller:
    name: str = "controller"
    needs_truth: bool = False

    def reset(self) -> None:
        self._t_ms: list[float] = []

    def __call__(self, u: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def observe(self, d: np.ndarray) -> None:
        pass

    def diagnostics(self) -> dict[str, np.ndarray]:
        return {"t_ms": np.array(getattr(self, "_t_ms", []))}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"


class Timer:
    """Context manager appending elapsed milliseconds to a list (per-step compute cost)."""

    def __init__(self, sink: list[float]):
        self.sink = sink

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.sink.append((time.perf_counter() - self.t0) * 1e3)
        return False
