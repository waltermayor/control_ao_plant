"""Model checkpoints keyed by *what determines the model*.

Layout: `<root>/<name>/<hash>/` with
    final.pt     completed training (best params, history, metadata) -> reused, no retraining
    last.pt      periodic training state (model, optimiser, scheduler, RNG) -> resume after a crash
    config.json  the key material, human readable

The hash covers the model hyper-parameters, the training config and a fingerprint of the data
(system config + data-collection config). Change any of them and a new directory is used, so
stale checkpoints are never loaded silently.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

from aoctrl.utils.cache import _to_plain, config_hash

log = logging.getLogger(__name__)


class CheckpointManager:
    def __init__(self, root: str | Path, name: str, key_parts: Any, enabled: bool = True, force_retrain: bool = False):
        self.key = config_hash(key_parts)
        self.dir = Path(root) / name / self.key
        self.enabled = enabled
        self.force_retrain = force_retrain
        self._key_parts = key_parts

    # ------------------------------------------------------------------ paths
    @property
    def final_path(self) -> Path:
        return self.dir / "final.pt"

    @property
    def last_path(self) -> Path:
        return self.dir / "last.pt"

    def _write(self, path: Path, payload: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        cfg = self.dir / "config.json"
        if not cfg.exists():
            cfg.write_text(json.dumps(_to_plain(self._key_parts), indent=2, default=str))
        tmp = path.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)

    # ------------------------------------------------------------------ final
    def has_final(self) -> bool:
        return self.enabled and not self.force_retrain and self.final_path.exists()

    def load_final(self) -> dict:
        log.info("loading checkpoint %s", self.final_path)
        return torch.load(self.final_path, map_location="cpu", weights_only=False)

    def save_final(self, payload: dict) -> Path | None:
        if not self.enabled:
            return None
        self._write(self.final_path, payload)
        if self.last_path.exists():
            self.last_path.unlink()
        log.info("saved checkpoint %s", self.final_path)
        return self.final_path

    # ------------------------------------------------------------------ resumable state
    def has_last(self) -> bool:
        return self.enabled and not self.force_retrain and self.last_path.exists()

    def load_last(self) -> dict:
        return torch.load(self.last_path, map_location="cpu", weights_only=False)

    def save_last(self, payload: dict) -> None:
        if self.enabled:
            self._write(self.last_path, payload)
