"""Compute-device selection for neural-network training.

The AO simulation and the controllers run in NumPy on the CPU (per-step latency matters more
than throughput there). Only model *training* is moved to an accelerator.
"""

from __future__ import annotations

import logging

import torch

log = logging.getLogger(__name__)


def resolve_device(spec: str = "auto", allow_mps: bool = True) -> torch.device:
    """Return a torch device.

    spec: "auto" (cuda > mps > cpu), "cpu", "cuda", "cuda:1", "mps".
    An explicit request for an unavailable accelerator falls back to CPU with a warning.
    """
    spec = (spec or "auto").lower()
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if allow_mps and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if spec.startswith("cuda") and not torch.cuda.is_available():
        log.warning("CUDA requested but not available; using CPU.")
        return torch.device("cpu")
    if spec == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        log.warning("MPS requested but not available; using CPU.")
        return torch.device("cpu")
    return torch.device(spec)


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    """float64 is not supported on MPS: downgrade to float32 there."""
    dtype = {"float32": torch.float32, "float64": torch.float64}[name]
    if device.type == "mps" and dtype == torch.float64:
        log.warning("float64 is not supported on MPS; training in float32.")
        return torch.float32
    return dtype


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        idx = device.index or 0
        return f"cuda:{idx} ({torch.cuda.get_device_name(idx)})"
    return device.type
