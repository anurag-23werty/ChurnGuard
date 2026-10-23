"""Data-quality profile of bronze tables.

Bronze contracts (schemas.py) already rejected structurally broken data. This report
measures the *known-dirty* aspects of KKBox that silver must handle, plus volume anomalies
that indicate a bad extract. Results go to reports/dq/<table>.json; the process exits
non-zero if any check is FAIL so it can gate a pipeline.

Usage:
  python -m churnguard.data.dq                 # all ingested tables
  python -m churnguard.data.dq --tables user_logs
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

import duckdb

from churnguard.config import Settings, get_settings
from churnguard.data.catalog import TABLES
from churnguard.db import bronze_glob, connect
from churnguard.logging_utils import get_logger, setup_logging

log = get_logger(__name__)

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


@dataclass(frozen=True)
class RuleCheck:
    """Share of rows matching `predicate`; WARN above `warn`, FAIL above `fail`."""

    name: str
    predicate: str
    warn: float
    fail: float = 1.01  # default: informational/warn only


# Thresholds reflect what is known about KKBox, so a WARN means "silver must handle this"
# and a FAIL means "this extract looks wrong".
RULES: dict[str, list[RuleCheck]] = {
    "members": [
        RuleCheck("bd_out_of_range", "bd IS NULL OR bd NOT BETWEEN 1 AND 100", warn=0.10),
        RuleCheck("gender_missing", "gender IS NULL", warn=0.30),
        RuleCheck("registration_date_invalid", "registration_init_time IS NULL", warn=0.001,
                  fail=0.05),
    ],
    "transactions": [
        RuleCheck("expire_before_transaction",
                  "membership_expire_date < transaction_date AND is_cancel = 0", warn=0.001),
        RuleCheck("paid_more_than_list", "actual_amount_paid > plan_list_price", warn=0.01),
        RuleCheck("zero_plan_days", "payment_plan_days = 0", warn=0.01),
        RuleCheck("expire_date_invalid", "membership_expire_date IS NULL", warn=0.001, fail=0.05),
    ],
    "user_logs": [
        RuleCheck("total_secs_negative", "total_secs < 0", warn=0.0001),
        RuleCheck("total_secs_over_24h", "total_secs > 86400", warn=0.0001),
        RuleCheck("date_invalid", "date IS NULL", warn=0.0001, fail=0.01),
    ],
    "labels": [
        RuleCheck("churn_rate_high", "is_churn = 1", warn=0.20, fail=0.50),
    ],
    "scoring_members": [],
}

# (msno, key) pairs expected to be unique within a table, after including source.
DUPLICATE_KEYS: dict[str, list[str]] = {
    "members": ["msno"],
    "user_logs": ["msno", "date", "source"],
    "labels": ["msno", "label_month"],
    "scoring_members": ["msno"],
}


@dataclass
class CheckResult:
    check: str
    status: str
    value: float | int | None
    detail: str = ""


def _q(con: duckdb.DuckDBPyConnection, sql: str) -> list[tuple]:
    return con.execute(sql).fetchall()


def profile_columns(con: duckdb.DuckDBPyConnection, src: str) -> dict:
    # SUMMARIZE gives min/max/approx_unique/null% per column in one pass.
    rows = con.execute(f"SUMMARIZE SELECT * FROM {src}").fetchdf()
    keep = ["column_name", "column_type", "min", "max", "approx_unique", "null_percentage"]
    return {
        r["column_name"]: {k: (str(r[k]) if r[k] is not None else None) for k in keep[1:]}
        for _, r in rows[keep].iterrows()
    }


def volume_checks(con: duckdb.DuckDBPyConnection, src: str) -> tuple[list[CheckResult], dict]:
    """Per-month counts; flag gaps and months far from the trailing median."""
    rows = _q(con, f"""
        SELECT month, source, count(*) AS n
        FROM read_parquet('{src}', hive_partitioning = true)
        WHERE month != 'invalid'
        GROUP BY ALL ORDER BY month, source
    """)
    by_month: dict[str, int] = {}
    for month, _source, n in rows:
        by_month[month] = by_month.get(month, 0) + n

    results: list[CheckResult] = []
    months = sorted(by_month)
    if months:
        expected = _month_range(months[0], months[-1])
        missing = sorted(set(expected) - set(months))
        results.append(CheckResult("month_gaps", FAIL if missing else PASS, len(missing),
                                   ",".join(missing)))

    anomalies = []
    for i, m in enumerate(months):
        trailing = sorted(by_month[p] for p in months[max(0, i - 3):i])
        if len(trailing) < 2:
            continue
        median = trailing[len(trailing) // 2]
        ratio = by_month[m] / median
        if ratio < 0.5 or ratio > 2.0:
            anomalies.append(f"{m}: {by_month[m]:,} vs median {median:,} ({ratio:.2f}x)")
    results.append(CheckResult("monthly_volume", WARN if anomalies else PASS, len(anomalies),
                               "; ".join(anomalies)))
    return results, {"rows_by_month_source": [list(r) for r in rows]}


def _month_range(start: str, end: str) -> list[str]:
    y, m = map(int, start.split("-"))
    out = []
    while f"{y:04d}-{m:02d}" <= end:
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def run_table(con: duckdb.DuckDBPyConnection, settings: Settings, table: str) -> dict:
    spec = TABLES[table]
    glob = bronze_glob(settings, table)
    src = f"read_parquet('{glob}')"
    total = _q(con, f"SELECT count(*) FROM {src}")[0][0]

    results = [CheckResult("non_empty", FAIL if total == 0 else PASS, total)]
    extras: dict = {}
    if total:
        for rule in RULES.get(table, []):
            share = _q(con, f"SELECT avg(CASE WHEN {rule.predicate} THEN 1 ELSE 0 END) "
                            f"FROM {src}")[0][0]
            status = FAIL if share > rule.fail else WARN if share > rule.warn else PASS
            results.append(CheckResult(rule.name, status, round(share, 6), rule.predicate))

        if keys := DUPLICATE_KEYS.get(table):
            cols = ", ".join(keys)
            dups = _q(con, f"SELECT count(*) - count(DISTINCT ({cols})) FROM {src}")[0][0]
            results.append(CheckResult(f"duplicate_{'_'.join(keys)}", WARN if dups else PASS,
                                       dups))

        if spec.partition_by:
            vol, extras = volume_checks(con, glob)
            results += vol
            if len(spec.files) > 1:
                # v2 refresh files overlap v1 in KKBox; silver must de-duplicate on this.
                overlap = _q(con, f"""
                    SELECT count(*) FROM (
                        SELECT msno, {spec.partition_by} FROM {src} GROUP BY ALL
                        HAVING count(DISTINCT source) > 1)
                """)[0][0]
                results.append(CheckResult("cross_source_overlap", WARN if overlap else PASS,
                                           overlap, f"(msno, {spec.partition_by}) in >1 source"))

    status = (FAIL if any(r.status == FAIL for r in results)
              else WARN if any(r.status == WARN for r in results) else PASS)
    return {
        "table": table,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": status,
        "rows": total,
        "checks": [asdict(r) for r in results],
        "columns": profile_columns(con, src) if total else {},
        **extras,
    }


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="Bronze data-quality report")
    parser.add_argument("--tables", nargs="+", choices=sorted(TABLES))
    args = parser.parse_args(argv)

    settings = get_settings()
    tables = args.tables or [t for t in TABLES if (settings.paths.bronze / t).exists()]
    out_dir = settings.paths.reports / "dq"
    out_dir.mkdir(parents=True, exist_ok=True)

    con = connect(settings)
    worst = PASS
    for table in tables:
        report = run_table(con, settings, table)
        (out_dir / f"{table}.json").write_text(json.dumps(report, indent=2, default=str))
        log.info("[%s] %s  rows=%s", table, report["status"], f"{report['rows']:,}")
        for c in report["checks"]:
            if c["status"] != PASS:
                log.warning("  %-28s %-4s %s %s", c["check"], c["status"], c["value"], c["detail"])
        if report["status"] == FAIL or (report["status"] == WARN and worst == PASS):
            worst = report["status"]
    log.info("overall: %s (reports in %s)", worst, out_dir)
    return 1 if worst == FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
