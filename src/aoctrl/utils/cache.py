"""Content-addressed on-disk cache for expensive, deterministic artifacts.

Calibration (KL basis, interaction matrix), open-loop turbulence series, identification data and
reference closed-loop runs are pure functions of the configuration. They are cached under
`<cache_dir>/<name>/<hash>.pkl` where the hash covers the relevant config subtree plus a
per-artifact version string. Bump the version when the code producing an artifact changes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def _to_plain(obj: Any) -> Any:
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf

        if isinstance(obj, (DictConfig, ListConfig)):
            return OmegaConf.to_container(obj, resolve=True)
    except ImportError:  # pragma: no cover
        pass
    if isinstance(obj, dict):
        return {str(k): _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return obj


def config_hash(*parts: Any, length: int = 12) -> str:
    """Stable short hash of arbitrary (config-like) objects."""
    payload = json.dumps([_to_plain(p) for p in parts], sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:length]


class ArtifactCache:
    def __init__(self, root: str | Path, enabled: bool = True):
        self.root = Path(root)
        self.enabled = enabled

    def path(self, name: str, key: str) -> Path:
        return self.root / name / f"{key}.pkl"

    def get_or_compute(self, name: str, key_parts: Any, fn: Callable[[], Any], version: str = "1") -> Any:
        key = config_hash(key_parts, version)
        path = self.path(name, key)
        if self.enabled and path.exists():
            log.info("cache hit: %s/%s", name, key)
            with open(path, "rb") as f:
                return pickle.load(f)
        t0 = time.time()
        value = fn()
        log.info("computed %s in %.1f s", name, time.time() - t0)
        if self.enabled:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(path)
        return value
