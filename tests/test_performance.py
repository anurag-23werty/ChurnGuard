from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from churnguard.monitoring.performance import performance


def _write_txn(settings, rows):
    con = duckdb.connect()
    con.execute("CREATE TABLE t (member_id INT, payment_method_id SMALLINT, payment_plan_days "
                "SMALLINT, plan_list_price INT, actual_amount_paid INT, is_auto_renew BOOL, "
                "transaction_date DATE, membership_expire_date DATE, is_cancel BOOL)")
    con.executemany("INSERT INTO t VALUES (?, 41, 30, 149, 149, true, ?, ?, false)", rows)
    out = settings.paths.silver / "transactions" / "transactions.parquet"
    out.parent.mkdir(parents=True)
    con.execute(f"COPY (SELECT *, 'v1' AS source FROM t) TO '{out}' (FORMAT parquet)")


def _write_scores(settings, month, p):
    d = settings.paths.gold / "scores" / f"score_month={month}"
    d.mkdir(parents=True)
    pd.DataFrame({"member_id": np.arange(1, len(p) + 1, dtype="int32"), "p_churn": p}
                 ).to_parquet(d / "data.parquet")


@pytest.fixture
def history(settings):
    """20 members due in Feb 2017; odd ids renew in early March, even ids never return.
    Data ends 2017-03-31, so Feb labels are fully decided; April has none yet."""
    rows = []
    for m in range(1, 21):
        rows.append((m, "2017-01-10", "2017-02-10"))
        if m % 2:
            rows.append((m, "2017-03-05", "2017-04-05"))
    rows.append((99, "2017-03-31", "2017-04-30"))  # data end; also due in April
    _write_txn(settings, rows)
    return settings


def test_complete_month_uses_served_scores(history):
    p = np.where(np.arange(1, 21) % 2 == 0, 0.9, 0.1)  # perfect ranking
    _write_scores(history, "2017-02", p)
    r = performance(history, "2017-02", duckdb.connect())
    assert (r["status"], r["scores"], r["members"], r["decided_share"]) == (
        "complete", "served", 20, 1.0)
    assert r["metrics"]["roc_auc"] == 1.0 and r["metrics"]["base_rate"] == 0.5


def test_future_month_is_pending(history):
    _write_scores(history, "2017-04", np.array([0.3]))
    r = performance(history, "2017-04", duckdb.connect())
    assert r["status"] == "pending" and "metrics" not in r


def test_no_scores_and_no_model(history):
    assert performance(history, "2016-06", duckdb.connect())["status"] == "no_scores"
