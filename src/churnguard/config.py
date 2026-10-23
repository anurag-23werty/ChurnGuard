"""Typed configuration loaded from configs/*.yaml.

Override the file with CHURNGUARD_CONFIG=/path/to/file.yaml.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "base.yaml"


@dataclass(frozen=True)
class Paths:
    raw: Path
    bronze: Path
    silver: Path
    gold: Path
    reports: Path


@dataclass(frozen=True)
class IngestConfig:
    month_start: str | None = None
    month_end: str | None = None
    csv_block_size: int = 64 << 20
    flush_rows: int = 500_000
    validate_sample_rows: int = 20_000
    compression: str = "zstd"
    min_free_disk_gb: float = 8.0


@dataclass(frozen=True)
class DuckDBConfig:
    memory_limit: str = "6GB"
    threads: int = 6


@dataclass(frozen=True)
class StorageConfig:
    # rclone remote + base folder for cold storage, e.g. "gdrive:ChurnGuard".
    remote: str = "gdrive:ChurnGuard"
    transfers: int = 4


@dataclass(frozen=True)
class Settings:
    paths: Paths
    kaggle_competition: str
    ingest: IngestConfig = field(default_factory=IngestConfig)
    duckdb: DuckDBConfig = field(default_factory=DuckDBConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)


def _resolve(p: str, root: Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else root / path


def load_settings(config_path: str | Path | None = None, root: Path = PROJECT_ROOT) -> Settings:
    path = Path(config_path or os.environ.get("CHURNGUARD_CONFIG", DEFAULT_CONFIG))
    raw = yaml.safe_load(path.read_text())
    return Settings(
        paths=Paths(**{k: _resolve(v, root) for k, v in raw["paths"].items()}),
        kaggle_competition=raw["kaggle"]["competition"],
        ingest=IngestConfig(**raw.get("ingest", {})),
        duckdb=DuckDBConfig(**raw.get("duckdb", {})),
        storage=StorageConfig(**raw.get("storage", {})),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
