"""Feature correctness on hand-computed members, plus the poisoned-future leakage test."""

from __future__ import annotations

import duckdb
import pytest

from churnguard.features.build import build_features
from churnguard.features.registry import registry

TXN_COLS = ("member_id, payment_method_id, payment_plan_days, plan_list_price, "
            "actual_amount_paid, is_auto_renew, transaction_date, membership_expire_date, "
            "is_cancel, source")
LOG_COLS = ("member_id, date, num_25, num_50, num_75, num_985, num_100, num_unq, total_secs")


def _txn(mid, d, exp, paid=149, cancel=False, auto=True, method=41):
    return (f"({mid}, {method}, 30, 149, {paid}, {str(auto).lower()}, DATE '{d}', "
            f"DATE '{exp}', {str(cancel).lower()}, 'v1')")


def _log(mid, d, secs, n25=1, n100=3, unq=4):
    return f"({mid}, DATE '{d}', {n25}, 0, 0, 0, {n100}, {unq}, {secs})"


def _write(con, path, cols, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY (SELECT * FROM (VALUES {', '.join(rows)}) t({cols})) "
                f"TO '{path}' (FORMAT parquet)")


BASE_TXN = [
    _txn(1, "2016-11-20", "2016-11-20", cancel=True),
    _txn(1, "2016-12-05", "2017-01-05"),
    _txn(1, "2017-01-05", "2017-02-05", method=38),
    _txn(2, "2017-01-10", "2017-02-10", paid=0, auto=False),
    _txn(3, "2017-01-15", "2017-02-15"),
]
BASE_LOGS = [
    _log(1, "2017-01-31", 100), _log(1, "2017-01-25", 200), _log(1, "2016-12-15", 400),
    _log(3, "2017-01-20", 50),
]
BASE_MEMBERS = ["(1, 5, 28, 'female', 9, DATE '2016-02-01')",
                "(3, 1, NULL, 'unknown', 7, DATE '2017-03-01')"]  # registers after as_of


def _silver(settings, txn, logs, members):
    con = duckdb.connect()
    s = settings.paths.silver
    _write(con, s / "transactions" / "transactions.parquet", TXN_COLS, txn)
    _write(con, s / "user_logs" / "month=2017-01" / "data.parquet", LOG_COLS, logs)
    _write(con, s / "members" / "members.parquet",
           "member_id, city, age, gender, registered_via, registration_date", members)
    _write(con, settings.paths.gold / "spine" / "label_month=2017-02" / "data.parquet",
           "member_id, label_month, as_of_date, last_expire, is_churn, label_status", [
               "(1, '2017-02', DATE '2017-02-01', DATE '2017-02-05', false, 'decided')",
               "(2, '2017-02', DATE '2017-02-01', DATE '2017-02-10', true, 'decided')",
               "(3, '2017-02', DATE '2017-02-01', DATE '2017-02-15', NULL, 'pending')",
           ])


def _features(settings) -> dict[int, dict]:
    build_features(settings, ["2017-02"])
    con = duckdb.connect()
    rel = con.execute(f"SELECT * FROM '{settings.paths.gold}/features/**/*.parquet'")
    cols = [d[0] for d in rel.description]
    return {r[0]: dict(zip(cols, r, strict=True)) for r in rel.fetchall()}


@pytest.fixture
def feats(settings):
    _silver(settings, BASE_TXN, BASE_LOGS, BASE_MEMBERS)
    return _features(settings)


def test_schema_matches_registry(feats):
    row = feats[1]
    assert list(row)[:5] == ["member_id", "label_month", "as_of_date", "is_churn",
                             "label_status"]
    assert list(row)[5:] == list(registry())
    assert "renewal_gap_days" not in row  # label-derived spine columns never leak in


def test_member_features(feats):
    assert feats[1]["tenure_days"] == 366  # 2016-02-01 -> 2017-02-01 (leap year)
    assert feats[3]["tenure_days"] is None  # registered after as_of_date
    assert (feats[2]["has_member_record"], feats[2]["gender"], feats[2]["age_missing"]) == (
        0, "unknown", 1)


def test_billing_features(feats):
    f = feats[1]
    assert f["days_to_expiry"] == 4
    assert (f["last_payment_method"], f["last_amount_paid"], f["last_is_cancel"]) == (38, 149, 0)
    assert (f["days_since_last_txn"], f["days_since_first_txn"]) == (27, 73)
    assert (f["n_txn_30d"], f["n_txn_90d"], f["n_cancel_90d"], f["n_cancel_all"]) == (1, 3, 1, 1)
    assert f["amount_paid_90d"] == 149 * 3
    assert f["n_payment_methods"] == 2
    assert feats[2]["n_free_txn_365d"] == 1
    assert feats[2]["last_discount"] == 149


def test_engagement_features(feats):
    f = feats[1]
    # as_of 2017-02-01: 7d window = [Jan 25, Feb 1), 90d = [Nov 3, Feb 1)
    assert (f["active_days_7d"], f["total_secs_7d"]) == (2, 300)
    assert (f["active_days_90d"], f["total_secs_90d"]) == (3, 700)
    assert f["days_since_last_listen"] == 1
    assert f["completion_rate_30d"] == pytest.approx(6 / 8)
    assert f["secs_trend_7_30"] == pytest.approx(300 / (300 * 7 / 30))
    g = feats[2]  # never listened
    assert (g["active_days_30d"], g["total_secs_30d"], g["days_since_last_listen"]) == (0, 0, None)
    assert g["completion_rate_30d"] is None


def test_poisoned_future_changes_nothing(settings, feats):
    """Inject extreme events on/after as_of_date into every source; features must not move."""
    poison_txn = [_txn(m, "2017-02-01", "2017-03-01", paid=9999, cancel=True, method=99)
                  for m in (1, 2, 3)] + [_txn(1, "2017-02-20", "2018-01-01", paid=0)]
    poison_logs = [_log(m, d, 1e7, n25=500, unq=999)
                   for m in (1, 2, 3) for d in ("2017-02-01", "2017-02-15")]
    poison_members = ["(2, 22, 99, 'male', 3, DATE '2017-02-01')"]  # appears after as_of
    _silver(settings, BASE_TXN + poison_txn, BASE_LOGS + poison_logs, BASE_MEMBERS)
    poisoned = _features(settings)
    assert poisoned == feats
    # Member attributes are a static snapshot: a record created after as_of must not
    # produce tenure, even though its static columns are (by assumption) usable.
    _silver(settings, BASE_TXN, BASE_LOGS, BASE_MEMBERS + poison_members)
    assert _features(settings)[2]["tenure_days"] is None


def test_mixed_as_of_dates_in_one_query(settings):
    """Same member at two as-of dates in one query: the earlier row must not see the gap
    between them. This is the case serving/backfills hit, and the only one where the
    per-row `< as_of_date` guard (not the constant pruning bound) is what protects us."""
    from churnguard.features.build import feature_sql, silver_sources

    _silver(settings, BASE_TXN, BASE_LOGS, BASE_MEMBERS)
    con = duckdb.connect()
    con.execute("""CREATE TABLE pop AS SELECT * FROM (VALUES
        (1, DATE '2017-01-01', DATE '2017-01-05'),
        (1, DATE '2017-02-01', DATE '2017-02-05')) t(member_id, as_of_date, last_expire)""")
    sql = feature_sql("pop", silver_sources(settings), "2017-01-01", "2017-02-01",
                      passthrough=("as_of_date",))
    rows = {str(r[0]): r for r in con.execute(
        f"SELECT as_of_date, n_txn_365d, total_secs_90d, last_payment_method, "
        f"days_since_last_listen FROM ({sql})").fetchall()}
    # Jan 1 sees only Nov 20 + Dec 5 transactions and the Dec 15 listen.
    assert rows["2017-01-01"][1:] == (2, 400, 41, 17)
    assert rows["2017-02-01"][1:] == (3, 700, 38, 1)


def test_feature_auc_matches_sklearn_definition():
    from churnguard.features.diagnostics import feature_auc

    con = duckdb.connect()
    con.execute("""CREATE TABLE t AS SELECT * FROM (VALUES
        (1, true), (2, true), (2, false), (0, false), (NULL, false), (3, true))
        t(x, is_churn)""")
    # Pairs (pos, neg): pos {1,2,3}, neg {2,0,NULL}. Wins: 1>{0,NULL}=2, 2>{0,NULL}+tie=2.5,
    # 3>all=3 -> 7.5 / 9.
    assert feature_auc(con, "t", "x") == pytest.approx(7.5 / 9)
