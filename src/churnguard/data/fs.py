"""Filesystem helpers shared by every layer writer."""

from __future__ import annotations

import shutil
from pathlib import Path


def fresh_staging(final: Path) -> Path:
    """Empty sibling directory to build `final` in, e.g. bronze/.staging_user_logs."""
    staging = final.with_name(f".staging_{final.name}")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    return staging


def publish(staging: Path, final: Path) -> None:
    """Swap a fully written staging dir into place; readers never see a half-written table."""
    if final.exists():
        backup = final.with_name(f".old_{final.name}")
        shutil.rmtree(backup, ignore_errors=True)
        final.replace(backup)
        staging.replace(final)
        shutil.rmtree(backup)
    else:
        staging.replace(final)


def check_free_disk(path: Path, min_gb: float) -> float:
    path.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(path).free / 1e9
    if free_gb < min_gb:
        raise RuntimeError(f"only {free_gb:.1f} GB free at {path}; need >= {min_gb} GB")
    return free_gb
