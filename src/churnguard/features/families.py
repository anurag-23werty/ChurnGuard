"""Point-in-time feature SQL, one function per registry family.

Every function takes a population relation with (member_id, as_of_date, last_expire) and
returns a SELECT keyed by (member_id, as_of_date), so one query can score the same member
at several as-of dates (backfills, serving). Two rules make these leak-free:

1. Each source is joined with `event_date < p.as_of_date` *per row* before aggregation.
   A constant bound (< max as_of) is added only so DuckDB can prune partitions; it is not
   the guard, and tests mutate the per-row guard with mixed as-of dates to prove it.
2. Windowed expressions come from the registry (`windows: [...]`), so the code cannot drift
   from what the registry documents.

The same SQL serves training (population = label spine) and scoring (population = members
due to expire), which is what keeps training/serving features identical.
"""

from __future__ import annotations

from datetime import date

from churnguard.features.registry import by_family
from churnguard.labels.spine import ORDER_DESC

PLAYS = "(num_25 + num_50 + num_75 + num_985 + num_100)"
LOG_LOOKBACK_DAYS = 90


def _win(col: str, w: int) -> str:
    return f"{col} >= p.as_of_date - INTERVAL {w} DAY"


def _windowed(family: str, templates: dict[str, str], date_col: str) -> list[str]:
    """Expand `<base>_<w>d` registry features using templates keyed by base name."""
    out = []
    for f in by_family(family):
        if f.window_days is None:
            continue
        base = f.name.removesuffix(f"_{f.window_days}d")
        out.append(f"{templates[base].format(w=_win(date_col, f.window_days))} AS {f.name}")
    return out


def member_sql(pop: str, members: str) -> str:
    return f"""
    SELECT p.member_id, p.as_of_date,
           m.city,
           m.age,
           (m.age IS NULL)::INT AS age_missing,
           coalesce(m.gender, 'unknown') AS gender,
           m.registered_via,
           CASE WHEN m.registration_date < p.as_of_date
                THEN date_diff('day', m.registration_date, p.as_of_date) END AS tenure_days,
           (m.member_id IS NOT NULL)::INT AS has_member_record
    FROM {pop} p LEFT JOIN {members} m USING (member_id)
    """


BILLING_WINDOWED = {
    "n_txn": "count(t.transaction_date) FILTER (WHERE {w})",
    "n_cancel": "count(t.transaction_date) FILTER (WHERE t.is_cancel AND {w})",
    "amount_paid": "coalesce(sum(t.actual_amount_paid) FILTER (WHERE {w}), 0)",
    "n_free_txn": "count(t.transaction_date) FILTER (WHERE t.actual_amount_paid = 0 AND {w})",
    "auto_renew_share": "avg(t.is_auto_renew::INT) FILTER (WHERE {w})",
}


def billing_sql(pop: str, txn: str, as_of_max: date) -> str:
    windowed = ",\n           ".join(_windowed("billing", BILLING_WINDOWED, "t.transaction_date"))
    return f"""
    WITH hist AS (
        SELECT t.*, p.as_of_date
        FROM {pop} p JOIN {txn} t USING (member_id)
        WHERE t.transaction_date < p.as_of_date
          AND t.transaction_date < DATE '{as_of_max}'
    ),
    last_txn AS (
        SELECT member_id, as_of_date, payment_plan_days, plan_list_price, actual_amount_paid,
               is_auto_renew, is_cancel, payment_method_id
        FROM hist
        QUALIFY row_number() OVER (
            PARTITION BY member_id, as_of_date ORDER BY {ORDER_DESC}) = 1
    ),
    agg AS (
        SELECT p.member_id, p.as_of_date,
               date_diff('day', max(t.transaction_date), p.as_of_date) AS days_since_last_txn,
               date_diff('day', min(t.transaction_date), p.as_of_date) AS days_since_first_txn,
               count(t.transaction_date) FILTER (WHERE t.is_cancel) AS n_cancel_all,
               count(DISTINCT t.payment_method_id) AS n_payment_methods,
           {windowed}
        FROM {pop} p LEFT JOIN hist t USING (member_id, as_of_date)
        GROUP BY p.member_id, p.as_of_date
    )
    SELECT p.member_id, p.as_of_date,
           date_diff('day', p.as_of_date, p.last_expire) AS days_to_expiry,
           l.payment_plan_days AS last_plan_days,
           l.plan_list_price AS last_list_price,
           l.actual_amount_paid AS last_amount_paid,
           l.plan_list_price - l.actual_amount_paid AS last_discount,
           l.is_auto_renew::INT AS last_is_auto_renew,
           l.is_cancel::INT AS last_is_cancel,
           l.payment_method_id AS last_payment_method,
           a.* EXCLUDE (member_id, as_of_date)
    FROM {pop} p
    LEFT JOIN last_txn l USING (member_id, as_of_date)
    LEFT JOIN agg a USING (member_id, as_of_date)
    """


ENGAGEMENT_WINDOWED = {
    "active_days": "count(u.date) FILTER (WHERE {w})",
    "total_secs": "coalesce(sum(u.total_secs) FILTER (WHERE {w}), 0)",
    "num_unq": "coalesce(sum(u.num_unq) FILTER (WHERE {w}), 0)",
    "num_plays": f"coalesce(sum({PLAYS}) FILTER (WHERE {{w}}), 0)",
    "completion_rate": f"sum(u.num_100) FILTER (WHERE {{w}}) "
                       f"/ nullif(sum({PLAYS}) FILTER (WHERE {{w}}), 0)",
    "skip_rate": f"sum(u.num_25) FILTER (WHERE {{w}}) "
                 f"/ nullif(sum({PLAYS}) FILTER (WHERE {{w}}), 0)",
    "secs_per_active_day": "sum(u.total_secs) FILTER (WHERE {w}) "
                           "/ nullif(count(u.date) FILTER (WHERE {w}), 0)",
}


def engagement_sql(pop: str, logs: str, as_of_min: date, as_of_max: date) -> str:
    windowed = ",\n           ".join(_windowed("engagement", ENGAGEMENT_WINDOWED, "u.date"))
    w7, w30 = _win("u.date", 7), _win("u.date", 30)
    return f"""
    WITH recent AS (
        SELECT u.*
        FROM {logs} u
        WHERE u.date >= DATE '{as_of_min}' - INTERVAL {LOG_LOOKBACK_DAYS} DAY
          AND u.date < DATE '{as_of_max}'
    )
    SELECT p.member_id, p.as_of_date,
           {windowed},
           date_diff('day', max(u.date), p.as_of_date) AS days_since_last_listen,
           coalesce(sum(u.total_secs) FILTER (WHERE {w7}), 0)
             / nullif(sum(u.total_secs) FILTER (WHERE {w30}) * 7 / 30, 0) AS secs_trend_7_30,
           count(u.date) FILTER (WHERE {w7})
             / nullif(count(u.date) FILTER (WHERE {w30}) * 7 / 30, 0) AS active_trend_7_30
    FROM {pop} p
    LEFT JOIN recent u
      ON u.member_id = p.member_id
     AND u.date < p.as_of_date
     AND u.date >= p.as_of_date - INTERVAL {LOG_LOOKBACK_DAYS} DAY
    GROUP BY p.member_id, p.as_of_date
    """
