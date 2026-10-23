"""Batch scorer and API: correctness and batch/API parity on a synthetic silver layer."""

from __future__ import annotations

from datetime import date, timedelta

import duckdb
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from churnguard.features.registry import categorical_features, numeric_features, registry
from churnguard.modeling.candidates import SklearnChurnModel
from churnguard.modeling.preprocessing import make_logreg
from churnguard.serving.api import create_app
from churnguard.serving.batch import next_month, score_month
from churnguard.serving.scoring import features_for

N = 40
INFO = {"name": "test", "version": "7", "model_type": "logreg"}


def _write(con, path, sql):
    path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY ({sql}) TO '{path}' (FORMAT parquet)")


@pytest.fixture
def silver(settings):
    rng = np.random.default_rng(0)
    txn, logs = [], []
    for m in range(1, N + 1):
        auto = bool(m % 3)
        # Three renewals, the last one in March. Members 1..30 are on 30-day plans; 31..40
        # bought 61-day plans: an earlier one expires in April, the last runs into May.
        plan = 30 if m <= 30 else 61
        last_txn = date(2017, 3, 1 + m % 25) if m <= 30 else date(2017, 3, 10)
        for k in range(2, -1, -1):
            d = last_txn - timedelta(days=30 * k)
            txn.append((m, 41 if auto else 38, plan, 149, 149 if m % 7 else 0, auto,
                        d, d + timedelta(days=plan), False))
        if m % 5 == 0:  # a cancellation in March
            txn.append((m, 41, 30, 149, 0, auto, date(2017, 3, 20), date(2017, 3, 20), True))
        for day in rng.choice(59, size=rng.integers(0, 40), replace=False):
            logs.append((m, date(2017, 2, 1) + timedelta(days=int(day)),
                         *rng.integers(0, 20, 6).tolist(), float(rng.exponential(3000))))
    con = duckdb.connect()
    s = settings.paths.silver
    con.execute("CREATE TABLE t (member_id INT, payment_method_id SMALLINT, payment_plan_days "
                "SMALLINT, plan_list_price INT, actual_amount_paid INT, is_auto_renew BOOL, "
                "transaction_date DATE, membership_expire_date DATE, is_cancel BOOL)")
    con.executemany("INSERT INTO t VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", txn)
    _write(con, s / "transactions" / "transactions.parquet", "SELECT *, 'v1' AS source FROM t")
    con.execute("CREATE TABLE u (member_id INT, date DATE, num_25 INT, num_50 INT, num_75 INT, "
                "num_985 INT, num_100 INT, num_unq INT, total_secs DOUBLE)")
    con.executemany("INSERT INTO u VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", logs)
    _write(con, s / "user_logs" / "month=2017-02" / "data.parquet", "SELECT * FROM u")
    _write(con, s / "members" / "members.parquet",
           f"SELECT i AS member_id, 1 + i % 5 AS city, 20 + i AS age, "
           f"CASE WHEN i % 2 = 0 THEN 'male' ELSE 'unknown' END AS gender, 7 AS registered_via, "
           f"DATE '2015-01-01' AS registration_date FROM range(1, {N + 1}) t(i)")
    _write(con, s / "member_ids" / "member_ids.parquet",
           f"SELECT 'msno_' || i AS msno, i::INT AS member_id FROM range(1, {N + 1}) t(i)")
    return settings


@pytest.fixture
def model(silver):
    """A real logreg over all registry features, fit on the April population."""
    con = duckdb.connect()
    con.execute(f"""CREATE TEMP TABLE pop AS SELECT i::INT AS member_id,
        DATE '2017-04-01' AS as_of_date, DATE '2017-04-15' AS last_expire
        FROM range(1, {N + 1}) t(i)""")
    from churnguard.modeling.dataset import prepare_features
    X = prepare_features(features_for(con, silver, "pop"))
    y = np.array((X["last_is_auto_renew"] == 0).astype("int8"))  # writable copy
    y[:2] = 1 - y[:2]  # avoid a perfectly separable toy
    feats = list(registry())
    return SklearnChurnModel(make_logreg(numeric_features(), categorical_features()),
                             feats).fit(X, y)


@pytest.fixture
def batch(silver, model):
    con = duckdb.connect()
    return score_month(silver, "2017-04", model, INFO, con)


def test_next_month_follows_last_transaction(silver):
    con = duckdb.connect()
    src = f"read_parquet('{silver.paths.silver}/transactions/transactions.parquet')"
    assert next_month(con, src) == "2017-04"


def _due_in_april(settings) -> list[int]:
    """Official population rule, computed independently: any pre-April transaction whose
    expiry falls in April (this includes early renewers already paid into May)."""
    return [r[0] for r in duckdb.execute(f"""
        SELECT DISTINCT member_id FROM
        read_parquet('{settings.paths.silver}/transactions/transactions.parquet')
        WHERE transaction_date < DATE '2017-04-01'
          AND membership_expire_date BETWEEN DATE '2017-04-01' AND DATE '2017-04-30'
        ORDER BY 1""").fetchall()]


def test_batch_scores_exactly_the_due_population(batch, silver):
    s = batch["scores"]
    due = _due_in_april(silver)
    assert sorted(s.member_id) == due
    assert {31, 40} <= set(due) and 25 not in due  # early renewers in; Mar-31 expiry out
    assert s.ev_rank.tolist() == list(range(1, len(due) + 1))
    assert s.ev.is_monotonic_decreasing
    assert (s.loc[s.contact, "ev"] > 0).all() and not (s.loc[~s.contact, "ev"] > 0).any()
    assert s.msno.tolist()[:1][0].startswith("msno_")
    assert set(s.model_version) == {"7"} and set(s.score_month) == {"2017-04"}
    assert batch["manifest"]["rows"] == len(due)


def test_batch_output_is_published(batch, silver):
    out = silver.paths.gold / "scores" / "score_month=2017-04"
    assert (out / "data.parquet").exists() and (out / "_manifest.json").exists()
    assert len(pd.read_parquet(out / "data.parquet")) == len(_due_in_april(silver))


def test_api_matches_batch_exactly(batch, silver, model):
    client = TestClient(create_app(silver, model, INFO))
    rows = batch["scores"].set_index("member_id")
    for member_id in (1, 5, 10, 17, 30, 35):  # 35: an early renewer
        r = client.get(f"/members/{member_id}/score", params={"as_of": "2017-04-01"})
        assert r.status_code == 200, r.text
        body, row = r.json(), rows.loc[member_id]
        assert body["p_churn"] == pytest.approx(row.p_churn, rel=0, abs=1e-12)
        assert body["expected_value"] == pytest.approx(row.ev, rel=0, abs=1e-9)
        assert body["contact"] == bool(row.contact)
        assert [x["feature"] for x in body["reasons"]] == [
            row[f"reason_{i}"] for i in (1, 2, 3) if isinstance(row[f"reason_{i}"], str)]
        assert body["model"]["version"] == "7"


def test_post_score_uses_the_registry_schema(batch, silver, model):
    client = TestClient(create_app(silver, model, INFO))
    con = duckdb.connect()
    con.execute("""CREATE TEMP TABLE pop AS SELECT member_id, as_of_date, last_expire
        FROM read_parquet(?)""", [str(silver.paths.gold / "scores" / "score_month=2017-04"
                                      / "data.parquet")])
    con.execute("DELETE FROM pop WHERE member_id != 3")
    raw = features_for(con, silver, "pop")
    feats = {k: (None if pd.isna(v) else v.item() if hasattr(v, "item") else v)
             for k, v in raw[list(registry())].iloc[0].items()}
    ok = client.post("/score", json={"member_id": 3, "features": feats})
    assert ok.status_code == 200, ok.text
    assert ok.json()["p_churn"] == pytest.approx(
        batch["scores"].set_index("member_id").loc[3, "p_churn"], abs=1e-9)
    missing = {k: v for k, v in feats.items() if k != "days_to_expiry"}
    assert client.post("/score", json={"features": missing}).status_code == 422
    wrong = {**feats, "n_txn_30d": "lots"}
    assert client.post("/score", json={"features": wrong}).status_code == 422


def test_unknown_member_and_health(silver, model):
    client = TestClient(create_app(silver, model, INFO))
    assert client.get("/members/999/score", params={"as_of": "2017-04-01"}).status_code == 404
    assert client.get("/members/1/score").status_code == 422  # as_of is required
    assert client.get("/health").json() == {"status": "ok", "model": INFO}
