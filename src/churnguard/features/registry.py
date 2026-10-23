"""Load and enforce the feature registry (registry.yaml)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

REGISTRY_PATH = Path(__file__).with_name("registry.yaml")
DTYPES = {"int", "float", "str"}


@dataclass(frozen=True)
class Feature:
    name: str
    family: str
    source: str
    dtype: str
    description: str
    window_days: int | None = None
    categorical: bool = False
    expected_drift: bool = False


class RegistryError(ValueError):
    pass


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Feature]:
    raw = yaml.safe_load(path.read_text())
    out: dict[str, Feature] = {}
    for family, spec in raw["families"].items():
        for f in spec["features"]:
            if f["dtype"] not in DTYPES:
                raise RegistryError(f"{f['name']}: dtype must be one of {DTYPES}")
            windows = f.get("windows") or [None]
            for w in windows:
                name = f"{f['name']}_{w}d" if w else f["name"]
                if name in out:
                    raise RegistryError(f"duplicate feature {name}")
                out[name] = Feature(
                    name=name,
                    family=family,
                    source=spec["source"],
                    dtype=f["dtype"],
                    description=f["description"].replace("{w}", str(w)),
                    window_days=w,
                    categorical=bool(f.get("categorical", False)),
                    expected_drift=bool(f.get("expected_drift", False)),
                )
    return out


@lru_cache(maxsize=1)
def registry() -> dict[str, Feature]:
    return load_registry()


def categorical_features() -> list[str]:
    return [f.name for f in registry().values() if f.categorical]


def expected_drift_features() -> set[str]:
    return {f.name for f in registry().values() if f.expected_drift}


def numeric_features() -> list[str]:
    return [f.name for f in registry().values() if not f.categorical]


def by_family(family: str) -> list[Feature]:
    return [f for f in registry().values() if f.family == family]


def check_columns(produced: list[str], family: str | None = None) -> None:
    """Fail if produced feature columns differ from the registry (optionally one family)."""
    expected = {f.name for f in (by_family(family) if family else registry().values())}
    got = set(produced)
    missing, extra = sorted(expected - got), sorted(got - expected)
    if missing or extra:
        scope = f"family '{family}'" if family else "registry"
        raise RegistryError(f"{scope} mismatch: missing={missing} unregistered={extra}")
