"""Build the offline feature store: gold/features/label_month=YYYY-MM/data.parquet.

One row per spine row: keys + label columns + every registry feature, cast to its
registry dtype so training and scoring always see the same schema.

Usage:
  python -m churnguard.features.build                       # all spine months
  python -m churnguard.features.build --months 2017-01:2017-02
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date

import duckdb

from churnguard.config import Settings, get_settings
from churnguard.data.fs import fresh_staging, publish
from churnguard.db import connect
from churnguard.features import families
from churnguard.features.registry import check_columns, registry
from churnguard.labels.spine import months_between
from churnguard.logging_utils import get_logger, setup_logging

log = get_logger(__name__)

KEY_COLS = ("member_id", "label_month", "as_of_date", "is_churn", "label_status")
SQL_TYPES = {"int": "INTEGER", "float": "DOUBLE", "str": "VARCHAR"}


def silver_sources(settings: Settings) -> dict[str, str]:
    s = settings.paths.silver
    return {
        "members": f"read_parquet('{s}/members/members.parquet')",
        "txn": f"read_parquet('{s}/transactions/transactions.parquet')",
        "logs": f"read_parquet('{s}/user_logs/**/*.parquet', hive_partitioning = false)",
    }


def feature_sql(pop: str, src: dict[str, str], as_of_min: date, as_of_max: date,
                passthrough: tuple[str, ...] = ()) -> str:
    """All registry features for population `pop` (member_id, as_of_date, last_expire)."""
    casts = ",\n       ".join(
        f"{f.name}::{SQL_TYPES[f.dtype]} AS {f.name}" for f in registry().values()
    )
    keep = "".join(f"p.{c}, " for c in passthrough)
    return f"""
    WITH m AS ({families.member_sql(pop, src['members'])}),
         b AS ({families.billing_sql(pop, src['txn'], as_of_max)}),
         e AS ({families.engagement_sql(pop, src['logs'], as_of_min, as_of_max)})
    SELECT {keep}p.member_id AS _member_id,
       {casts}
    FROM {pop} p
    JOIN m USING (member_id, as_of_date)
    JOIN b USING (member_id, as_of_date)
    JOIN e USING (member_id, as_of_date)
    """


def _feature_columns(con: duckdb.DuckDBPyConnection, path: str) -> list[str]:
    cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM '{path}'").fetchall()]
    return [c for c in cols if c not in KEY_COLS]


def build_month(con: duckdb.DuckDBPyConnection, settings: Settings, month: str,
                out_path) -> dict:
    spine = settings.paths.gold / "spine" / f"label_month={month}" / "data.parquet"
    con.execute(f"CREATE OR REPLACE TEMP TABLE pop AS SELECT * FROM '{spine}'")
    as_of = con.execute("SELECT min(as_of_date), max(as_of_date) FROM pop").fetchone()
    sql = feature_sql("pop", silver_sources(settings), as_of[0], as_of[1],
                      passthrough=("label_month", "as_of_date", "is_churn", "label_status"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"""
        COPY (SELECT _member_id AS member_id, * EXCLUDE (_member_id) FROM ({sql})
              ORDER BY member_id)
        TO '{out_path}' (FORMAT parquet, COMPRESSION zstd)
    """)
    check_columns(_feature_columns(con, out_path))

    n_pop = con.execute("SELECT count(*) FROM pop").fetchone()[0]
    n_out, n_dup = con.execute(
        f"SELECT count(*), count(*) - count(DISTINCT member_id) FROM '{out_path}'").fetchone()
    if n_out != n_pop or n_dup:
        raise RuntimeError(f"{month}: {n_out} feature rows for {n_pop} spine rows, {n_dup} dups")

    nulls = con.execute(f"""
        SELECT {", ".join(f"avg(({f} IS NULL)::INT) AS {f}" for f in registry())}
        FROM '{out_path}'""").fetchone()
    return {"rows": n_out,
            "null_rate": {f: round(v, 4) for f, v in zip(registry(), nulls, strict=True)}}


def build_features(settings: Settings, months: list[str]) -> dict:
    con = connect(settings)
    final = settings.paths.gold / "features"
    staging = fresh_staging(final)
    report: dict = {"n_features": len(registry()), "months": {}}
    for month in months:
        t0 = time.monotonic()
        out = staging / f"label_month={month}" / "data.parquet"
        report["months"][month] = build_month(con, settings, month, out)
        log.info("[features] %s: %s rows x %d features (%.1fs)", month,
                 f"{report['months'][month]['rows']:,}", len(registry()),
                 time.monotonic() - t0)
    (staging / "_manifest.json").write_text(json.dumps(report, indent=2))
    publish(staging, final)
    return report


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Build point-in-time features")
    parser.add_argument("--months", help="YYYY-MM:YYYY-MM (default: every spine month)")
    args = parser.parse_args(argv)

    settings = get_settings()
    months = (months_between(args.months) if args.months else sorted(
        p.name.removeprefix("label_month=")
        for p in (settings.paths.gold / "spine").glob("label_month=*")))
    build_features(settings, months)


if __name__ == "__main__":
    main()
