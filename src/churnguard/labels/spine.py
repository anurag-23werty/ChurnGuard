"""Label spine: one row per (member, label month) with a point-in-time as-of date.

Port of KKBox's official WSDMChurnLabeller.scala (docs/reference/) to DuckDB SQL, generalised
from February 2017 to any month M:

  population  members with any transaction before M whose membership_expire_date falls in
              M. The Scala file only looks at M-1 and at the *last* transaction, which
              matches the published labels at ~86% recall; "any expiry in M over all
              history" matches at ~99% (validation in reports/labels). It notably keeps
              auto-renewers who already renewed past M, who churn at 7-17% via cancels.
  last_expire paid-through date at the as-of date: expiry of the last transaction before M
              under the official ordering (may lie after M for early renewers)
  is_churn    no transaction from M onwards, or the first renewal comes >= 30 days after
              last_expire; cancellations before that renewal pull the expiry earlier
  as_of_date  first day of M. Population and last_expire use only data before M, so this
              is the real decision point: features must use events strictly before it.

Labels mature row by row, as they do in production. A row is `decided` once a renewal is
observed (on time or late) or once last_expire + 30 days is inside the data; otherwise it is
`pending` with is_churn NULL, never silently counted as churn. Training uses decided rows
only; pending rows resolve when newer transactions arrive.

Validated against both official label files (train.csv = Feb 2017, train_v2 = Mar 2017):
population recall/precision on all rows, label agreement on decided rows.

Usage:
  python -m churnguard.labels.spine                         # default months + validation
  python -m churnguard.labels.spine --months 2016-10:2017-02
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta

import duckdb

from churnguard.config import Settings, get_settings
from churnguard.data.fs import fresh_staging, publish
from churnguard.db import connect
from churnguard.logging_utils import get_logger, setup_logging

log = get_logger(__name__)

CHURN_GAP_DAYS = 30
DEFAULT_MONTHS = "2016-10:2017-03"  # 90+ days of silver user_logs before every as-of date

# Total order used by the official labeller for one member's transactions
# (calculateLastday / calculateRenewalGap):
#   1. transaction_date ascending
#   2. plan "signature" = string concat of list price, plan days, payment method; DESCENDING
#      (string comparison, exactly as in Scala)
#   3. same signature: renewal before cancel; renewals by expiry asc, cancels by expiry desc
_SIG = ("plan_list_price::VARCHAR || payment_plan_days::VARCHAR || payment_method_id::VARCHAR")
ORDER_ASC = f"""transaction_date ASC, {_SIG} DESC, is_cancel::INT ASC,
    CASE WHEN is_cancel THEN -epoch(membership_expire_date)
         ELSE epoch(membership_expire_date) END ASC"""
ORDER_DESC = f"""transaction_date DESC, {_SIG} ASC, is_cancel::INT DESC,
    CASE WHEN is_cancel THEN -epoch(membership_expire_date)
         ELSE epoch(membership_expire_date) END DESC"""


def month_bounds(month: str) -> tuple[date, date]:
    y, m = map(int, month.split("-"))
    start = date(y, m, 1)
    end = (date(y + (m == 12), m % 12 + 1, 1)) - timedelta(days=1)
    return start, end


def months_between(spec: str) -> list[str]:
    start, _, end = spec.partition(":")
    end = end or start
    out, (y, m) = [], map(int, start.split("-"))
    while f"{y:04d}-{m:02d}" <= end:
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def label_month_sql(month: str, txn_src: str, data_end: date) -> str:
    """SELECT producing (member_id, label_month, as_of_date, last_expire, renewal_gap_days,
    is_churn, label_status) for one label month, given the last observed transaction date."""
    m_start, m_end = month_bounds(month)
    return f"""
    WITH history AS (
        SELECT * FROM {txn_src} WHERE transaction_date < DATE '{m_start}'
    ),
    due_in_month AS (
        SELECT DISTINCT member_id FROM history
        WHERE membership_expire_date BETWEEN DATE '{m_start}' AND DATE '{m_end}'
    ),
    candidates AS (
        SELECT h.member_id, h.membership_expire_date AS last_expire
        FROM history h SEMI JOIN due_in_month USING (member_id)
        QUALIFY row_number() OVER (PARTITION BY h.member_id ORDER BY {ORDER_DESC}) = 1
    ),
    future AS (
        SELECT f.*, c.last_expire,
               row_number() OVER (PARTITION BY f.member_id ORDER BY {ORDER_ASC}) AS seq
        FROM {txn_src} f JOIN candidates c USING (member_id)
        WHERE f.transaction_date >= DATE '{m_start}'
    ),
    first_renewal AS (
        SELECT member_id, min(seq) AS seq FROM future WHERE NOT is_cancel GROUP BY 1
    ),
    outcome AS (
        -- Cancels ordered before the first renewal (all cancels, if there is none) can only
        -- move the expiry earlier. That effective expiry drives both the gap and maturity.
        SELECT f.member_id,
               least(any_value(f.last_expire),
                     coalesce(min(f.membership_expire_date)
                              FILTER (WHERE f.is_cancel AND f.seq < coalesce(r.seq, 1e9)),
                              any_value(f.last_expire))) AS effective_expire,
               any_value(f.transaction_date) FILTER (WHERE f.seq = r.seq) AS renewal_date
        FROM future f LEFT JOIN first_renewal r USING (member_id)
        GROUP BY f.member_id
    ),
    resolved AS (
        SELECT c.member_id, c.last_expire,
               coalesce(o.effective_expire, c.last_expire) AS effective_expire,
               date_diff('day', o.effective_expire, o.renewal_date) AS gap
        FROM candidates c LEFT JOIN outcome o USING (member_id)
    )
    SELECT member_id,
           '{month}' AS label_month,
           DATE '{m_start}' AS as_of_date,
           last_expire,
           effective_expire,
           gap AS renewal_gap_days,
           CASE WHEN gap IS NOT NULL THEN gap >= {CHURN_GAP_DAYS}
                WHEN effective_expire + INTERVAL {CHURN_GAP_DAYS} DAY <= DATE '{data_end}'
                THEN TRUE
           END AS is_churn,
           CASE WHEN gap IS NOT NULL
                  OR effective_expire + INTERVAL {CHURN_GAP_DAYS} DAY <= DATE '{data_end}'
                THEN 'decided' ELSE 'pending' END AS label_status
    FROM resolved
    """


def validate_against_official(
    con: duckdb.DuckDBPyConnection, spine: str, official: str, month: str
) -> dict:
    """Population match on all rows; label agreement on decided rows only."""
    row = con.execute(f"""
        WITH d AS (SELECT member_id, is_churn FROM {spine} WHERE label_month = '{month}'),
             o AS (SELECT member_id, is_churn FROM {official} WHERE label_month = '{month}')
        SELECT
            (SELECT count(*) FROM d), (SELECT count(*) FROM o),
            count(*) FILTER (WHERE d.member_id IS NOT NULL AND o.member_id IS NOT NULL),
            count(*) FILTER (WHERE d.is_churn IS NOT NULL AND o.member_id IS NOT NULL),
            count(*) FILTER (WHERE d.is_churn = o.is_churn),
            (SELECT avg(is_churn::INT) FROM d), (SELECT avg(is_churn::INT) FROM o)
        FROM d FULL OUTER JOIN o USING (member_id)
    """).fetchone()
    n_derived, n_official, n_both, n_decided, n_agree, rate_d, rate_o = row
    return {
        "month": month,
        "derived_members": n_derived,
        "official_members": n_official,
        "overlap": n_both,
        "population_recall": round(n_both / n_official, 5) if n_official else None,
        "population_precision": round(n_both / n_derived, 5) if n_derived else None,
        "decided_overlap": n_decided,
        "label_agreement_on_decided": round(n_agree / n_decided, 5) if n_decided else None,
        "churn_rate_derived_decided": round(rate_d, 5) if rate_d is not None else None,
        "churn_rate_official": round(rate_o, 5) if rate_o is not None else None,
    }


def build_spine(settings: Settings, months: list[str], con=None) -> dict:
    con = con or connect(settings)
    txn = f"read_parquet('{settings.paths.silver}/transactions/transactions.parquet')"
    official = f"read_parquet('{settings.paths.silver}/labels/labels.parquet')"
    con.execute(f"CREATE OR REPLACE TEMP VIEW txn AS SELECT * FROM {txn}")
    data_end = con.execute("SELECT max(transaction_date) FROM txn").fetchone()[0]

    final = settings.paths.gold / "spine"
    staging = fresh_staging(final)
    per_month = {}
    for month in months:
        out = staging / f"label_month={month}" / "data.parquet"
        out.parent.mkdir(parents=True)
        con.execute(f"""
            COPY ({label_month_sql(month, 'txn', data_end)} ORDER BY member_id)
            TO '{out}' (FORMAT parquet, COMPRESSION zstd)
        """)
        n, rate, pending = con.execute(f"""
            SELECT count(*), avg(is_churn::INT), count(*) FILTER (WHERE label_status = 'pending')
            FROM '{out}'""").fetchone()
        dups = con.execute(
            f"SELECT count(*) - count(DISTINCT member_id) FROM '{out}'").fetchone()[0]
        if dups:
            raise RuntimeError(f"{month}: {dups} duplicate members in spine")
        per_month[month] = {"members": n, "pending": pending,
                            "churn_rate_decided": round(rate or 0, 5)}
        log.info("[spine] %s: %s members (%s pending), churn %.2f%% of decided",
                 month, f"{n:,}", f"{pending:,}", 100 * (rate or 0))

    spine = f"read_parquet('{staging}/**/*.parquet', hive_partitioning = false)"
    validation = [
        validate_against_official(con, spine, official, m)
        for m in months
        if con.execute(f"SELECT count(*) FROM {official} WHERE label_month = '{m}'").fetchone()[0]
    ]
    report = {
        "data_end": data_end,
        "churn_gap_days": CHURN_GAP_DAYS,
        "months": per_month,
        "validation": validation,
    }
    (staging / "_manifest.json").write_text(json.dumps(report, indent=2, default=str))
    publish(staging, final)
    return report


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Build the point-in-time label spine")
    parser.add_argument("--months", default=DEFAULT_MONTHS, help="YYYY-MM:YYYY-MM inclusive")
    args = parser.parse_args(argv)

    settings = get_settings()
    report = build_spine(settings, months_between(args.months))
    for v in report["validation"]:
        log.info("[validation] %s", v)
    out = settings.paths.reports / "labels"
    out.mkdir(parents=True, exist_ok=True)
    (out / "spine_validation.json").write_text(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
