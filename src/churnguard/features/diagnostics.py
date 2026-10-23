"""Univariate feature report and leakage smell test.

For each numeric feature: null rate and single-feature AUC on decided training rows. A
single feature that separates churners almost perfectly is far more often a leak than a
discovery, so the report fails above LEAK_AUC.

Usage:
  python -m churnguard.features.diagnostics
"""

from __future__ import annotations

import argparse
import json
import sys

import duckdb

from churnguard.config import get_settings
from churnguard.db import connect
from churnguard.features.registry import registry
from churnguard.logging_utils import get_logger, setup_logging

log = get_logger(__name__)

LEAK_AUC = 0.95
NULL_SENTINEL = -1e18  # NULLs rank lowest


def feature_auc(con: duckdb.DuckDBPyConnection, rel: str, feature: str,
                label: str = "is_churn") -> float:
    """Exact ROC AUC of a single feature (ties count 1/2), computed in SQL by value groups."""
    return con.execute(f"""
        WITH g AS (
            SELECT coalesce({feature}::DOUBLE, {NULL_SENTINEL}) AS v,
                   count(*) FILTER (WHERE {label}) AS pos,
                   count(*) FILTER (WHERE NOT {label}) AS neg
            FROM {rel} GROUP BY 1
        ),
        c AS (
            SELECT pos, neg,
                   coalesce(sum(neg) OVER (ORDER BY v ROWS BETWEEN UNBOUNDED PRECEDING
                                           AND 1 PRECEDING), 0) AS neg_below
            FROM g
        )
        SELECT sum(pos * (neg_below + 0.5 * neg)) / (sum(pos) * sum(neg)) FROM c
    """).fetchone()[0]


def univariate_report(con: duckdb.DuckDBPyConnection, rel: str) -> list[dict]:
    rows = []
    for name, f in registry().items():
        if f.dtype == "str":
            continue
        auc = feature_auc(con, rel, name)
        null_rate = con.execute(f"SELECT avg(({name} IS NULL)::INT) FROM {rel}").fetchone()[0]
        rows.append({
            "feature": name,
            "family": f.family,
            "auc": round(max(auc, 1 - auc), 4),
            "direction": "+" if auc >= 0.5 else "-",
            "null_rate": round(null_rate, 4),
        })
    return sorted(rows, key=lambda r: -r["auc"])


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="Univariate feature report + leak check")
    parser.add_argument("--before", default="2017-03",
                        help="use decided label months strictly before this (training data)")
    args = parser.parse_args(argv)

    settings = get_settings()
    con = connect(settings)
    con.execute(f"""
        CREATE TEMP VIEW train AS
        SELECT * FROM read_parquet('{settings.paths.gold}/features/**/*.parquet',
                                   hive_partitioning = false)
        WHERE label_status = 'decided' AND label_month < '{args.before}'
    """)
    report = univariate_report(con, "train")
    out = settings.paths.reports / "features"
    out.mkdir(parents=True, exist_ok=True)
    (out / "univariate.json").write_text(json.dumps(report, indent=2))

    for r in report[:10]:
        log.info("%-26s AUC %.3f %s null %5.1f%%", r["feature"], r["auc"], r["direction"],
                 100 * r["null_rate"])
    leaks = [r["feature"] for r in report if r["auc"] >= LEAK_AUC]
    if leaks:
        log.error("possible leakage, single-feature AUC >= %.2f: %s", LEAK_AUC, leaks)
        return 1
    log.info("leak smell test passed: max single-feature AUC %.3f", report[0]["auc"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
