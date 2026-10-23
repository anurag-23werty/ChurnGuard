"""Scenario tests for the label spine, one member per rule of the official labeller."""

from __future__ import annotations

import duckdb
import pytest

from churnguard.labels.spine import build_spine, month_bounds, months_between

COLS = ("member_id, payment_method_id, payment_plan_days, plan_list_price, actual_amount_paid, "
        "is_auto_renew, transaction_date, membership_expire_date, is_cancel, source")


def _txn(mid, txn_date, expire, cancel=False, price=149, days=30, method=41):
    return (f"({mid}, {method}, {days}, {price}, {price}, true, DATE '{txn_date}', "
            f"DATE '{expire}', {str(cancel).lower()}, 'v1')")


@pytest.fixture
def spine(settings):
    """Label month 2017-02, data ends 2017-03-31."""
    rows = [
        # 1 renews on time (gap 3)                                   -> not churn
        _txn(1, "2017-01-05", "2017-02-05"), _txn(1, "2017-02-08", "2017-03-08"),
        # 2 never comes back, window closed (Feb 10 + 30 < Mar 31)    -> churn
        _txn(2, "2017-01-10", "2017-02-10"),
        # 3 renews exactly 30 days late                               -> churn
        _txn(3, "2017-01-01", "2017-02-01"), _txn(3, "2017-03-03", "2017-04-03"),
        # 4 renews 29 days late                                       -> not churn
        _txn(4, "2017-01-01", "2017-02-01"), _txn(4, "2017-03-02", "2017-04-02"),
        # 5 cancels first (expiry pulled to Feb 2), renews Mar 6: gap 32 -> churn
        _txn(5, "2017-01-20", "2017-02-20"), _txn(5, "2017-02-02", "2017-02-02", cancel=True),
        _txn(5, "2017-03-06", "2017-04-06"),
        # 6 early renewer: due Feb 3, renewed Jan 31 to Mar 3, cancels Feb 15, gone -> churn
        _txn(6, "2017-01-03", "2017-02-03"), _txn(6, "2017-01-31", "2017-03-03"),
        _txn(6, "2017-02-15", "2017-02-15", cancel=True),
        # 7 expiry in March, not February                             -> not in population
        _txn(7, "2017-01-15", "2017-03-15"),
        # 8 same day, same plan: subscribe then cancel; cancel is last -> last_expire Feb 9
        _txn(8, "2017-01-09", "2017-02-25"), _txn(8, "2017-01-09", "2017-02-09", cancel=True),
        # 9 only shows up in February itself                          -> not in population
        _txn(9, "2017-02-01", "2017-02-28"),
        # 10 due Feb 27, no renewal, window open until Mar 29... closed; due Feb 28 -> Mar 30
        _txn(10, "2017-01-28", "2017-02-28"),
        # sets data_end = 2017-03-31
        _txn(99, "2017-03-31", "2017-04-30"),
    ]
    txn_dir = settings.paths.silver / "transactions"
    txn_dir.mkdir(parents=True)
    con = duckdb.connect()
    con.execute(f"""
        COPY (SELECT * FROM (VALUES {", ".join(rows)}) t({COLS}))
        TO '{txn_dir}/transactions.parquet' (FORMAT parquet)""")
    labels_dir = settings.paths.silver / "labels"
    labels_dir.mkdir(parents=True)
    con.execute(f"""
        COPY (SELECT * FROM (VALUES (1, '2017-02', false), (2, '2017-02', true),
                                    (3, '2017-02', false), (42, '2017-02', true))
              t(member_id, label_month, is_churn))
        TO '{labels_dir}/labels.parquet' (FORMAT parquet)""")
    report = build_spine(settings, ["2017-02"])
    rows = con.execute(f"""
        SELECT member_id, last_expire::VARCHAR, renewal_gap_days, is_churn, label_status
        FROM read_parquet('{settings.paths.gold}/spine/**/*.parquet', hive_partitioning=false)
    """).fetchall()
    return {r[0]: r[1:] for r in rows}, report


def test_population_is_point_in_time(spine):
    rows, _ = spine
    assert set(rows) == {1, 2, 3, 4, 5, 6, 8, 10}  # 7: due in March; 9: first seen in Feb


@pytest.mark.parametrize("mid, gap, churn", [
    (1, 3, False), (3, 30, True), (4, 29, False), (5, 32, True),
])
def test_renewal_gap_rule(spine, mid, gap, churn):
    rows, _ = spine
    assert rows[mid][1:3] == (gap, churn)
    assert rows[mid][3] == "decided"


def test_no_renewal_is_churn_once_window_closes(spine):
    rows, _ = spine
    assert rows[2] == ("2017-02-10", None, True, "decided")
    assert rows[10] == ("2017-02-28", None, True, "decided")  # Feb 28 + 30 = Mar 30 <= Mar 31


def test_early_renewer_who_cancels_is_churn(spine):
    rows, _ = spine
    assert rows[6][0] == "2017-03-03"  # paid-through, after M
    assert rows[6][2:] == (True, "decided")


def test_same_day_subscribe_then_cancel_orders_cancel_last(spine):
    rows, _ = spine
    assert rows[8][0] == "2017-02-09"


def test_validation_report(spine):
    _, report = spine
    v = report["validation"][0]
    assert (v["official_members"], v["derived_members"], v["overlap"]) == (4, 8, 3)
    assert v["label_agreement_on_decided"] == round(2 / 3, 5)  # member 3: 30 days -> churn


def test_pending_when_window_still_open(settings):
    txn_dir = settings.paths.silver / "transactions"
    txn_dir.mkdir(parents=True)
    (settings.paths.silver / "labels").mkdir(parents=True)
    con = duckdb.connect()
    con.execute(f"""
        COPY (SELECT * FROM (VALUES {_txn(1, "2017-02-20", "2017-03-20")}) t({COLS}))
        TO '{txn_dir}/transactions.parquet' (FORMAT parquet)""")
    con.execute(f"""
        COPY (SELECT 1 AS member_id, '2016-01' AS label_month, false AS is_churn)
        TO '{settings.paths.silver}/labels/labels.parquet' (FORMAT parquet)""")
    report = build_spine(settings, ["2017-03"])
    assert report["months"]["2017-03"] == {"members": 1, "pending": 1, "churn_rate_decided": 0}


def test_month_helpers():
    assert month_bounds("2016-02") == (month_bounds("2016-02")[0], month_bounds("2016-02")[1])
    assert str(month_bounds("2016-02")[1]) == "2016-02-29"
    assert months_between("2016-11:2017-02") == ["2016-11", "2016-12", "2017-01", "2017-02"]
