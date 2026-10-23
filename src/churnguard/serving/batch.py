"""Batch scorer: everyone due in a month, scored by the champion, written for the CRM.

Output gold/scores/score_month=YYYY-MM/data.parquet, one row per member due that month:
p_churn, value, contact decision, top-3 reasons, msno for the CRM, model name/version.

Usage:
  python -m churnguard.serving.batch                  # month after the last transaction
  python -m churnguard.serving.batch --month 2017-04
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

import duckdb
import pandas as pd

from churnguard import tracking
from churnguard.config import Settings, get_settings
from churnguard.data.fs import fresh_staging, publish
from churnguard.db import connect
from churnguard.features.build import silver_sources
from churnguard.labels.spine import due_population_sql
from churnguard.logging_utils import get_logger, setup_logging
from churnguard.serving.scoring import features_for, score_frame

log = get_logger(__name__)


def next_month(con: duckdb.DuckDBPyConnection, txn: str) -> str:
    last = con.execute(f"SELECT max(transaction_date) FROM {txn}").fetchone()[0]
    y, m = (last.year + 1, 1) if last.month == 12 else (last.year, last.month + 1)
    return f"{y:04d}-{m:02d}"


def score_month(settings: Settings, month: str, model, model_info: dict,
                con: duckdb.DuckDBPyConnection | None = None) -> dict:
    con = con or connect(settings)
    txn = silver_sources(settings)["txn"]
    con.execute(f"CREATE OR REPLACE TEMP TABLE batch_pop AS {due_population_sql(month, txn)}")
    raw = features_for(con, settings, "batch_pop")
    scored = score_frame(raw, model, settings.retention, settings.retention.budget)

    ids = con.execute(
        f"SELECT member_id, msno FROM '{settings.paths.silver}/member_ids/member_ids.parquet'"
    ).df()
    scored = scored.merge(ids, on="member_id", how="left")
    scored["score_month"] = month
    scored["model_name"], scored["model_version"] = model_info["name"], model_info["version"]
    scored["model_type"] = model_info["model_type"]
    scored["scored_at"] = datetime.now(UTC).isoformat()
    scored = scored.sort_values("ev", ascending=False, ignore_index=True)
    scored.insert(0, "ev_rank", range(1, len(scored) + 1))

    final = settings.paths.gold / "scores" / f"score_month={month}"
    staging = fresh_staging(final)
    scored.to_parquet(staging / "data.parquet", index=False)
    manifest = {
        "score_month": month, "rows": len(scored), "model": model_info,
        "contacted": int(scored.contact.sum()),
        "mean_p_churn": float(scored.p_churn.mean()),
        "expected_net_value_contacted": float(scored.loc[scored.contact, "ev"].sum()),
        "scored_at": scored.scored_at.iloc[0] if len(scored) else None,
    }
    (staging / "_manifest.json").write_text(json.dumps(manifest, indent=2))
    publish(staging, final)
    return {"manifest": manifest, "scores": scored}


def population_check(con: duckdb.DuckDBPyConnection, settings: Settings,
                     scores: pd.DataFrame) -> dict | None:
    """Overlap with Kaggle's April test population (sample_submission_v2), when present."""
    path = settings.paths.silver / "scoring_members" / "scoring_members.parquet"
    if not path.exists():
        return None
    con.register("scored_ids", scores[["member_id"]])
    n_kaggle, n_both = con.execute(f"""
        SELECT (SELECT count(*) FROM '{path}'),
               (SELECT count(*) FROM '{path}' k SEMI JOIN scored_ids s USING (member_id))
    """).fetchone()
    return {"kaggle_members": n_kaggle, "scored_members": len(scores), "overlap": n_both,
            "kaggle_recall": n_both / n_kaggle, "precision": n_both / len(scores)}


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--month", help="YYYY-MM to score (default: after last transaction)")
    args = parser.parse_args(argv)

    settings = get_settings()
    con = connect(settings)
    month = args.month or next_month(con, silver_sources(settings)["txn"])
    info = tracking.champion_info(settings)
    model = tracking.load_champion(settings)
    log.info("scoring %s with %s v%s (%s)", month, info["name"], info["version"],
             info["model_type"])
    out = score_month(settings, month, model, info, con)
    m = out["manifest"]
    log.info("%s: %s members, mean p %.4f, contact %s, expected net value %s NTD", month,
             f"{m['rows']:,}", m["mean_p_churn"], f"{m['contacted']:,}",
             f"{m['expected_net_value_contacted']:,.0f}")
    check = population_check(con, settings, out["scores"])
    if check and month == "2017-04":
        log.info("population vs Kaggle April test set: recall %.4f precision %.4f",
                 check["kaggle_recall"], check["precision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
