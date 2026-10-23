"""Scoring shared by the batch job and the API, so both produce identical rows.

population (member_id, as_of_date, last_expire) -> registry features (feature_sql) ->
prepare_features -> model probability -> expected value + contact decision -> reasons.
"""

from __future__ import annotations

from datetime import date

import duckdb
import numpy as np
import pandas as pd

from churnguard.config import RetentionConfig, Settings
from churnguard.decisioning.retention import expected_value, monthly_revenue, select
from churnguard.features.build import feature_sql, silver_sources
from churnguard.labels.spine import paid_through_sql
from churnguard.modeling.dataset import prepare_features

N_REASONS = 3
CHUNK_ROWS = 100_000


def features_for(con: duckdb.DuckDBPyConnection, settings: Settings, pop: str) -> pd.DataFrame:
    """Raw feature rows (with member_id, as_of_date) for population relation `pop`."""
    as_of = con.execute(f"SELECT min(as_of_date), max(as_of_date) FROM {pop}").fetchone()
    if as_of[0] is None:
        return pd.DataFrame()
    sql = feature_sql(pop, silver_sources(settings), as_of[0], as_of[1],
                      passthrough=("as_of_date", "last_expire"))
    return con.execute(f"SELECT _member_id AS member_id, * EXCLUDE (_member_id) "
                       f"FROM ({sql}) ORDER BY member_id, as_of_date").df()


def score_frame(raw: pd.DataFrame, model, policy: RetentionConfig,
                budget: float | None = None) -> pd.DataFrame:
    """Probability, value, decision and reason codes for raw feature rows."""
    X = prepare_features(raw)
    parts = []
    for start in range(0, len(X), CHUNK_ROWS):
        chunk = X.iloc[start:start + CHUNK_ROWS]
        p = model.predict_proba(chunk)
        reasons = (model.reason_codes(chunk, N_REASONS) if hasattr(model, "reason_codes")
                   else [[] for _ in range(len(chunk))])
        part = pd.DataFrame({"p_churn": p}, index=chunk.index)
        for i in range(N_REASONS):
            part[f"reason_{i + 1}"] = [r[i][0] if len(r) > i else None for r in reasons]
            part[f"reason_{i + 1}_logit"] = [r[i][1] if len(r) > i else np.nan
                                             for r in reasons]
        parts.append(part)
    scored = pd.concat(parts)
    revenue = monthly_revenue(X["last_amount_paid"], X["last_plan_days"])
    ev = expected_value(scored["p_churn"].to_numpy(), revenue, policy)
    ev.index = scored.index
    out = pd.concat([raw[["member_id", "as_of_date", "last_expire"]], scored,
                     ev.drop(columns="p_churn")], axis=1)
    out["contact"] = select(ev, budget).to_numpy()
    return out


def single_member_population(con: duckdb.DuckDBPyConnection, settings: Settings,
                             member_id: int, as_of: date) -> str:
    """Temp table with one row: the member's paid-through date as of `as_of`."""
    txn = silver_sources(settings)["txn"]
    con.execute("CREATE OR REPLACE TEMP TABLE _one AS SELECT ?::INTEGER AS member_id",
                [member_id])
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _pop AS
        SELECT member_id, DATE '{as_of}' AS as_of_date, last_expire
        FROM ({paid_through_sql(txn, as_of, members='_one')})
    """)
    return "_pop"
