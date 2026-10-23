"""Load time-ordered train / valid / test splits from the offline feature store.

Filtering and sampling happen in DuckDB so pandas only ever holds the rows it needs.

  train  decided rows of `train_months`, derived labels, optionally sampled
  valid  decided rows of `valid_month`, derived labels
  test   `test_month` features with the *official* labels (train_v2): our March labels are
         immature, so the holdout is scored against the published ground truth, on members
         present in both populations
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from churnguard.config import Settings
from churnguard.db import connect
from churnguard.features.registry import categorical_features, numeric_features, registry

SPLITS = ("train", "valid", "test")


@dataclass
class Split:
    name: str
    X: pd.DataFrame
    y: pd.Series
    meta: pd.DataFrame  # member_id, label_month, label_source

    def __len__(self) -> int:
        return len(self.y)


def _features_rel(settings: Settings) -> str:
    return (f"read_parquet('{settings.paths.gold}/features/**/*.parquet', "
            f"hive_partitioning = false)")


def split_sql(settings: Settings, name: str) -> str:
    m = settings.modeling
    feats = ", ".join(f"f.{c}" for c in registry())
    if name in ("train", "valid"):
        months = m.train_months if name == "train" else (m.valid_month,)
        month_list = ", ".join(f"'{x}'" for x in months)
        sql = f"""
            SELECT f.member_id, f.label_month, 'derived' AS label_source,
                   f.is_churn::INT AS y, {feats}
            FROM {_features_rel(settings)} f
            WHERE f.label_status = 'decided' AND f.label_month IN ({month_list})
        """
        if name == "train" and m.train_sample:
            # USING SAMPLE runs before WHERE, so filter in a subquery first.
            sql = (f"SELECT * FROM ({sql}) "
                   f"USING SAMPLE reservoir({m.train_sample} ROWS) REPEATABLE ({m.seed})")
        return sql
    if name == "test":
        return f"""
            SELECT f.member_id, f.label_month, 'official' AS label_source,
                   o.is_churn::INT AS y, {feats}
            FROM {_features_rel(settings)} f
            JOIN read_parquet('{settings.paths.silver}/labels/labels.parquet') o
              ON o.member_id = f.member_id AND o.label_month = f.label_month
            WHERE f.label_month = '{m.test_month}'
        """
    raise ValueError(f"unknown split {name!r}; expected one of {SPLITS}")


def category_codes(s: pd.Series) -> pd.Series:
    """Canonical string codes. Integer codes arrive as int, Int32 or float64 depending on
    who wrote the parquet (pandas keeps Int32, DuckDB-written int-with-nulls reads back as
    float64), and str(13.0) == "13.0" would be an unseen level. Numeric -> "13"."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype("Float64").round().astype("Int64").astype("string")
    return s.astype("string")


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """The one place model-input dtypes are decided; training and serving both call it."""
    X = df[list(registry())].copy()
    for c in categorical_features():
        X[c] = category_codes(X[c])  # codes, not quantities
    num = numeric_features()
    X[num] = X[num].astype("float32")  # NULL -> NaN; half the memory of float64
    return X


def load_split(settings: Settings, name: str, con=None) -> Split:
    con = con or connect(settings)
    df = con.execute(f"SELECT * FROM ({split_sql(settings, name)}) ORDER BY member_id").df()
    meta = df[["member_id", "label_month", "label_source"]]
    return Split(name=name, X=prepare_features(df), y=df["y"].astype("int8"), meta=meta)
