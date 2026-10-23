"""Bronze -> silver: conformed, de-duplicated, cleaned tables keyed by an integer member_id.

Design notes
- `msno` (44-char base64) is replaced everywhere by `member_id` (int32). The mapping lives
  only in silver/member_ids and is append-only, so ids stay stable across rebuilds.
- Cleaning never silently drops information: values are nulled/clipped and every rule's
  hit count is written to the table's _manifest.json, which also reconciles row counts
  against bronze.
- Each table is built in a staging dir, checked against SQL contracts, then published
  atomically. A failed contract leaves the previous silver table untouched.

Usage:
  python -m churnguard.data.silver                    # all tables
  python -m churnguard.data.silver --tables user_logs
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from churnguard.config import Settings, get_settings
from churnguard.data.fs import check_free_disk, fresh_staging, publish
from churnguard.db import bronze_glob, connect
from churnguard.logging_utils import get_logger, setup_logging

log = get_logger(__name__)

# Plausible member age; KKBox uses bd=0 as "unknown" for ~67% of members.
AGE_MIN, AGE_MAX = 10, 80
SECONDS_PER_DAY = 86_400
PARQUET_OPTS = "FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 500000"

BRONZE_TABLES = ("members", "transactions", "user_logs", "labels", "scoring_members")


@dataclass(frozen=True)
class Contract:
    """A SQL predicate that must match zero rows of the built table."""

    name: str
    violation_sql: str  # full query returning a single count


class ContractError(RuntimeError):
    pass


def _one(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    return con.execute(sql).fetchone()[0]


def _bronze(settings: Settings, table: str) -> str:
    return f"read_parquet('{bronze_glob(settings, table)}', union_by_name = true)"


def _enforce(con: duckdb.DuckDBPyConnection, contracts: list[Contract]) -> dict[str, int]:
    results = {c.name: _one(con, c.violation_sql) for c in contracts}
    failed = {k: v for k, v in results.items() if v}
    if failed:
        raise ContractError(f"silver contract violations: {failed}")
    return results


def _copy(con: duckdb.DuckDBPyConnection, select_sql: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY ({select_sql}) TO '{path}' ({PARQUET_OPTS})")


# --------------------------------------------------------------------------- member ids


def build_member_ids(con: duckdb.DuckDBPyConnection, settings: Settings, staging: Path) -> dict:
    """Append-only msno -> member_id map covering every bronze table."""
    existing = settings.paths.silver / "member_ids" / "member_ids.parquet"
    union = " UNION ".join(
        f"SELECT msno FROM {_bronze(settings, t)}"
        for t in BRONZE_TABLES
        if (settings.paths.bronze / t).exists()
    )
    con.execute(f"CREATE OR REPLACE TEMP TABLE all_msno AS {union}")
    if existing.exists():
        con.execute(f"CREATE OR REPLACE TEMP TABLE old_ids AS SELECT * FROM '{existing}'")
    else:
        con.execute("CREATE OR REPLACE TEMP TABLE old_ids (msno VARCHAR, member_id INTEGER)")

    con.execute("""
        CREATE OR REPLACE TEMP TABLE member_ids AS
        SELECT msno, member_id FROM old_ids
        UNION ALL
        SELECT msno,
               (SELECT coalesce(max(member_id), 0) FROM old_ids)
                 + row_number() OVER (ORDER BY msno) AS member_id
        FROM (SELECT msno FROM all_msno EXCEPT SELECT msno FROM old_ids)
    """)
    _copy(con, "SELECT msno, member_id::INTEGER AS member_id FROM member_ids ORDER BY member_id",
          staging / "member_ids.parquet")
    src = f"'{staging}/member_ids.parquet'"
    checks = _enforce(con, [
        Contract("duplicate_msno", f"SELECT count(*) - count(DISTINCT msno) FROM {src}"),
        Contract("duplicate_member_id",
                 f"SELECT count(*) - count(DISTINCT member_id) FROM {src}"),
        Contract("ids_changed", f"""
            SELECT count(*) FROM old_ids o JOIN {src} n USING (msno)
            WHERE o.member_id != n.member_id"""),
    ])
    rows = _one(con, f"SELECT count(*) FROM {src}")
    return {
        "rows": rows,
        "new_ids": rows - _one(con, "SELECT count(*) FROM old_ids"),
        "contracts": checks,
    }


def _ids_ready(con: duckdb.DuckDBPyConnection, settings: Settings) -> None:
    path = settings.paths.silver / "member_ids" / "member_ids.parquet"
    if not path.exists():
        raise FileNotFoundError("silver/member_ids missing; build it first (--tables member_ids)")
    con.execute(f"CREATE OR REPLACE TEMP TABLE ids AS SELECT * FROM '{path}'")


# --------------------------------------------------------------------------- members


def build_members(con: duckdb.DuckDBPyConnection, settings: Settings, staging: Path) -> dict:
    _ids_ready(con, settings)
    src = _bronze(settings, "members")
    out = staging / "members.parquet"
    _copy(con, f"""
        SELECT i.member_id,
               m.city,
               CASE WHEN m.bd BETWEEN {AGE_MIN} AND {AGE_MAX} THEN m.bd::SMALLINT END AS age,
               coalesce(m.gender, 'unknown') AS gender,
               m.registered_via,
               m.registration_init_time AS registration_date
        FROM {src} m JOIN ids i USING (msno)
        ORDER BY i.member_id
    """, out)
    built = f"'{out}'"
    return {
        "rows_bronze": _one(con, f"SELECT count(*) FROM {src}"),
        "rows_silver": _one(con, f"SELECT count(*) FROM {built}"),
        "rules": {
            "age_nulled_out_of_range": _one(con, f"""
                SELECT count(*) FROM {src}
                WHERE bd IS NOT NULL AND bd NOT BETWEEN {AGE_MIN} AND {AGE_MAX}"""),
            "gender_unknown": _one(con, f"SELECT count(*) FROM {built} WHERE gender = 'unknown'"),
        },
        "contracts": _enforce(con, [
            Contract("rows_match_bronze", f"""
                SELECT abs((SELECT count(*) FROM {src}) - (SELECT count(*) FROM {built}))"""),
            Contract("duplicate_member_id",
                     f"SELECT count(*) - count(DISTINCT member_id) FROM {built}"),
            Contract("age_out_of_range", f"""
                SELECT count(*) FROM {built} WHERE age NOT BETWEEN {AGE_MIN} AND {AGE_MAX}"""),
        ]),
    }


# --------------------------------------------------------------------------- transactions


def build_transactions(con: duckdb.DuckDBPyConnection, settings: Settings, staging: Path) -> dict:
    """Union v1 + v2 and drop exact duplicates only.

    v2 is not just March 2017: it also carries ~360k historical rows absent from v1, and
    (member, date) is not a key (cancel + renew on the same day is common). So the only
    safe de-duplication is on full row content; v2 wins the `source` tag on ties.
    """
    _ids_ready(con, settings)
    src = _bronze(settings, "transactions")
    out = staging / "transactions.parquet"
    content = ("msno, payment_method_id, payment_plan_days, plan_list_price, "
               "actual_amount_paid, is_auto_renew, transaction_date, membership_expire_date, "
               "is_cancel")
    _copy(con, f"""
        WITH dedup AS (
            SELECT {content}, max(source) AS source  -- 'v2' > 'v1'
            FROM {src} GROUP BY ALL
        )
        SELECT i.member_id,
               d.payment_method_id, d.payment_plan_days, d.plan_list_price,
               d.actual_amount_paid,
               d.is_auto_renew::BOOLEAN AS is_auto_renew,
               d.transaction_date,
               -- 1970-01-01 is a placeholder, not a real expiry.
               CASE WHEN year(d.membership_expire_date) > 1970
                    THEN d.membership_expire_date END AS membership_expire_date,
               d.is_cancel::BOOLEAN AS is_cancel,
               d.source
        FROM dedup d JOIN ids i USING (msno)
        ORDER BY i.member_id, d.transaction_date
    """, out)
    built = f"'{out}'"
    rows_bronze = _one(con, f"SELECT count(*) FROM {src}")
    rows_distinct = _one(con, f"SELECT count(*) FROM (SELECT DISTINCT {content} FROM {src})")
    return {
        "rows_bronze": rows_bronze,
        "rows_silver": _one(con, f"SELECT count(*) FROM {built}"),
        "rules": {
            "exact_duplicates_dropped": rows_bronze - rows_distinct,
            "expire_date_placeholder_nulled": _one(con, f"""
                SELECT count(*) FROM {src} WHERE year(membership_expire_date) <= 1970"""),
        },
        "contracts": _enforce(con, [
            Contract("rows_match_distinct_bronze",
                     f"SELECT abs({rows_distinct} - (SELECT count(*) FROM {built}))"),
            Contract("null_member_id",
                     f"SELECT count(*) FROM {built} WHERE member_id IS NULL"),
            Contract("negative_amounts", f"""
                SELECT count(*) FROM {built}
                WHERE actual_amount_paid < 0 OR plan_list_price < 0 OR payment_plan_days < 0"""),
        ]),
    }


# --------------------------------------------------------------------------- user logs


def build_user_logs(con: duckdb.DuckDBPyConnection, settings: Settings, staging: Path) -> dict:
    """Month by month (each ~17M rows sorts in memory), sorted by (member_id, date).

    Sorting by member makes zstd + dictionary encoding effective and lets feature queries
    for one member skip row groups via min/max statistics.
    """
    _ids_ready(con, settings)
    bronze_dir = settings.paths.bronze / "user_logs"
    months = sorted(p.name.removeprefix("month=") for p in bronze_dir.glob("month=*")
                    if p.name != "month=invalid")
    per_month: dict[str, dict] = {}
    for m in months:
        src = f"read_parquet('{bronze_dir}/month={m}/*.parquet')"
        out = staging / f"month={m}" / "data.parquet"
        t0 = time.monotonic()
        _copy(con, f"""
            SELECT i.member_id, u.date,
                   u.num_25, u.num_50, u.num_75, u.num_985, u.num_100, u.num_unq,
                   least(greatest(u.total_secs, 0), {SECONDS_PER_DAY}) AS total_secs
            FROM {src} u JOIN ids i USING (msno)
            ORDER BY i.member_id, u.date
        """, out)
        built = f"'{out}'"
        per_month[m] = {
            "rows_bronze": _one(con, f"SELECT count(*) FROM {src}"),
            "rows_silver": _one(con, f"SELECT count(*) FROM {built}"),
            "total_secs_clipped": _one(con, f"""
                SELECT count(*) FROM {src}
                WHERE total_secs < 0 OR total_secs > {SECONDS_PER_DAY}"""),
        }
        _enforce(con, [
            Contract(f"{m}_rows_match_bronze",
                     f"SELECT abs({per_month[m]['rows_bronze']} - {per_month[m]['rows_silver']})"),
            Contract(f"{m}_duplicate_member_date",
                     f"SELECT count(*) - count(DISTINCT hash(member_id, date)) FROM {built}"),
            Contract(f"{m}_date_outside_partition",
                     f"SELECT count(*) FROM {built} WHERE strftime(date, '%Y-%m') != '{m}'"),
        ])
        log.info("[user_logs] %s: %s rows (%.1fs)", m, f"{per_month[m]['rows_silver']:,}",
                 time.monotonic() - t0)

    invalid = bronze_dir / "month=invalid"
    return {
        "rows_bronze": sum(v["rows_bronze"] for v in per_month.values()),
        "rows_silver": sum(v["rows_silver"] for v in per_month.values()),
        "rules": {
            "total_secs_clipped": sum(v["total_secs_clipped"] for v in per_month.values()),
            "invalid_date_rows_excluded": _one(
                con, f"SELECT count(*) FROM read_parquet('{invalid}/*.parquet')"
            ) if invalid.exists() else 0,
        },
        "months": per_month,
    }


# --------------------------------------------------------------------------- labels


def build_labels(con: duckdb.DuckDBPyConnection, settings: Settings, staging: Path) -> dict:
    _ids_ready(con, settings)
    src = _bronze(settings, "labels")
    out = staging / "labels.parquet"
    _copy(con, f"""
        SELECT i.member_id, l.label_month, l.is_churn::BOOLEAN AS is_churn
        FROM (SELECT DISTINCT msno, label_month, is_churn FROM {src}) l JOIN ids i USING (msno)
        ORDER BY l.label_month, i.member_id
    """, out)
    built = f"'{out}'"
    return {
        "rows_bronze": _one(con, f"SELECT count(*) FROM {src}"),
        "rows_silver": _one(con, f"SELECT count(*) FROM {built}"),
        "churn_rate": {r[0]: round(r[1], 5) for r in con.execute(
            f"SELECT label_month, avg(is_churn::INT) FROM {built} GROUP BY 1 ORDER BY 1"
        ).fetchall()},
        "contracts": _enforce(con, [
            # Conflicting labels for the same member-month would mean a broken source.
            Contract("duplicate_member_month",
                     f"SELECT count(*) - count(DISTINCT (member_id, label_month)) FROM {built}"),
        ]),
    }


def build_scoring_members(
    con: duckdb.DuckDBPyConnection, settings: Settings, staging: Path
) -> dict:
    _ids_ready(con, settings)
    src = _bronze(settings, "scoring_members")
    out = staging / "scoring_members.parquet"
    _copy(con, f"""
        SELECT DISTINCT i.member_id FROM {src} s JOIN ids i USING (msno) ORDER BY 1
    """, out)
    return {
        "rows_bronze": _one(con, f"SELECT count(*) FROM {src}"),
        "rows_silver": _one(con, f"SELECT count(*) FROM '{out}'"),
    }


# --------------------------------------------------------------------------- driver

BUILDERS: dict[str, Callable[[duckdb.DuckDBPyConnection, Settings, Path], dict]] = {
    "member_ids": build_member_ids,  # must run first
    "members": build_members,
    "transactions": build_transactions,
    "user_logs": build_user_logs,
    "labels": build_labels,
    "scoring_members": build_scoring_members,
}


def build_table(con: duckdb.DuckDBPyConnection, settings: Settings, name: str) -> dict:
    final = settings.paths.silver / name
    staging = fresh_staging(final)
    t0 = time.monotonic()
    log.info("[%s] building", name)
    report = BUILDERS[name](con, settings, staging)
    report = {"table": name, "built_at": datetime.now(UTC).isoformat(),
              "duration_s": round(time.monotonic() - t0, 1), **report}
    (staging / "_manifest.json").write_text(json.dumps(report, indent=2, default=str))
    publish(staging, final)
    log.info("[%s] published in %.1fs: %s", name, report["duration_s"],
             {k: report[k] for k in ("rows_bronze", "rows_silver", "rules") if k in report})
    return report


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Build silver tables from bronze")
    parser.add_argument("--tables", nargs="+", choices=list(BUILDERS), default=list(BUILDERS))
    args = parser.parse_args(argv)

    settings = get_settings()
    free_gb = check_free_disk(settings.paths.silver, settings.ingest.min_free_disk_gb)
    log.info("free disk: %.1f GB", free_gb)
    con = connect(settings)
    for name in [t for t in BUILDERS if t in args.tables]:  # keep dependency order
        build_table(con, settings, name)


if __name__ == "__main__":
    main()
