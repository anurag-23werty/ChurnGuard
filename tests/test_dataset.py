from __future__ import annotations

from dataclasses import replace

import duckdb
import pytest

from churnguard.features.registry import categorical_features, registry
from churnguard.modeling.dataset import load_split


@pytest.fixture
def store(settings):
    """200 rows per month 2016-10..2017-03; every 4th row of each month is pending."""
    con = duckdb.connect()
    feats = ", ".join(
        f"'1'::VARCHAR AS {f}" if f in categorical_features() else f"i::INTEGER AS {f}"
        for f in registry())
    for i, month in enumerate(["2016-10", "2016-11", "2016-12", "2017-01", "2017-02", "2017-03"]):
        out = settings.paths.gold / "features" / f"label_month={month}" / "data.parquet"
        out.parent.mkdir(parents=True)
        con.execute(f"""COPY (
            SELECT i + {i * 1000} AS member_id, '{month}' AS label_month,
                   DATE '2017-01-01' AS as_of_date,
                   CASE WHEN i % 4 = 0 THEN NULL ELSE i % 10 = 0 END AS is_churn,
                   CASE WHEN i % 4 = 0 THEN 'pending' ELSE 'decided' END AS label_status,
                   {feats}
            FROM range(200) t(i)) TO '{out}' (FORMAT parquet)""")
    labels = settings.paths.silver / "labels" / "labels.parquet"
    labels.parent.mkdir(parents=True)
    # Official March labels exist for half of the March members, all churned.
    con.execute(f"""COPY (SELECT i + 5000 AS member_id, '2017-03' AS label_month,
                         true AS is_churn FROM range(0, 200, 2) t(i)) TO '{labels}'""")
    return settings


def test_splits_are_time_ordered_and_decided_only(store):
    m = store.modeling
    train, valid = load_split(store, "train"), load_split(store, "valid")
    assert set(train.meta.label_month) == set(m.train_months)
    assert set(valid.meta.label_month) == {m.valid_month}
    assert len(valid) == 150  # pending rows dropped
    assert list(train.X.columns) == list(registry())


def test_train_sample_is_taken_after_filtering(store):
    sampled = replace(store, modeling=replace(store.modeling, train_sample=300))
    train = load_split(sampled, "train")
    # 4 months x 150 decided rows = 600 eligible; sampling before WHERE would return fewer.
    assert len(train) == 300
    assert set(train.meta.label_month) <= set(store.modeling.train_months)
    assert load_split(sampled, "train").meta.member_id.tolist() == train.meta.member_id.tolist()


def test_test_split_uses_official_labels(store):
    test = load_split(store, "test")
    assert len(test) == 100 and test.y.eq(1).all()  # official says churn; derived would not
    assert set(test.meta.label_source) == {"official"}


def test_dtypes(store):
    X = load_split(store, "valid").X
    assert all(X[c].dtype == "string" for c in categorical_features())
    assert {str(X[c].dtype) for c in X if c not in categorical_features()} == {"float32"}
