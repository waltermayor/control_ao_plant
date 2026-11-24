"""Physical simulation backends. Use `build_backend(system_cfg)` to construct one."""

from __future__ import annotations

from typing import Any

from aoctrl.optics.base import Conditions, InteractionData, OpticalBackend


def build_backend(system_cfg: Any) -> OpticalBackend:
    kind = str(system_cfg.backend)
    if kind == "oopao":
        from aoctrl.optics.oopao_backend import OOPAOBackend

        return OOPAOBackend(system_cfg)
    if kind == "synthetic":
        from aoctrl.optics.synthetic_backend import SyntheticBackend

        return SyntheticBackend(system_cfg)
    raise ValueError(f"unknown backend {kind!r} (expected 'oopao' or 'synthetic')")


__all__ = ["Conditions", "InteractionData", "OpticalBackend", "build_backend"]
