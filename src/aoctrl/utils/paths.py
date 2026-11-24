from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Repository root: $AOCTRL_ROOT if set, else the first parent holding pyproject.toml."""
    env = os.environ.get("AOCTRL_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() and (parent / "configs").is_dir():
            return parent
    return Path.cwd().resolve()


def config_dir() -> Path:
    return project_root() / "configs"


def resolve_path(p: str | os.PathLike) -> Path:
    """Relative paths in configs are relative to the project root (not Hydra's run dir)."""
    p = Path(p).expanduser()
    return p if p.is_absolute() else (project_root() / p)
