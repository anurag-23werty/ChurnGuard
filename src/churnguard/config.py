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
    models: Path


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
class ModelingConfig:
    # Time-ordered splits by label month: train < valid < test, never shuffled.
    train_months: tuple[str, ...] = ("2016-10", "2016-11", "2016-12", "2017-01")
    valid_month: str = "2017-02"
    # Test labels come from the official train_v2 file (our March labels are immature).
    test_month: str = "2017-03"
    # Rows sampled from train for in-memory models (None = all decided rows).
    train_sample: int | None = 1_000_000
    seed: int = 42
    tuning_trials: int = 25
    tuning_timeout_s: int | None = 1800


@dataclass(frozen=True)
class MLflowConfig:
    # Relative sqlite paths resolve against the project root.
    tracking_uri: str = "sqlite:///mlruns.db"
    experiment: str = "churnguard"
    registered_model: str = "churnguard"  # model-agnostic; type recorded as a version tag


@dataclass(frozen=True)
class RetentionConfig:
    """Retention economics (see decisioning/retention.py)."""

    save_rate: float = 0.20  # ASSUMPTION: no campaign data in KKBox; measure via A/B test
    horizon_months: int = 12
    baseline_monthly_churn: float = 0.06
    contact_cost: float = 2.0  # NTD per contacted member
    incentive_months: float = 1.0  # free months offered
    budget: float | None = None  # NTD per campaign; None = every positive-EV member


@dataclass(frozen=True)
class Settings:
    paths: Paths
    kaggle_competition: str
    ingest: IngestConfig = field(default_factory=IngestConfig)
    duckdb: DuckDBConfig = field(default_factory=DuckDBConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    modeling: ModelingConfig = field(default_factory=ModelingConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    retention: RetentionConfig = field(default_factory=RetentionConfig)


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
        modeling=_modeling(raw.get("modeling", {})),
        mlflow=MLflowConfig(**raw.get("mlflow", {})),
        retention=RetentionConfig(**raw.get("retention", {})),
    )


def _modeling(raw: dict) -> ModelingConfig:
    if "train_months" in raw:
        raw = {**raw, "train_months": tuple(raw["train_months"])}
    return ModelingConfig(**raw)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
