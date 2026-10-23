"""Model performance once labels mature.

A month's outcome is known only ~30 days after each member's expiry, so performance is
always computed later, on the probabilities that were actually *served* (the batch scores
file) when they exist. Labels come from the same SQL as the training spine, run on whatever
transactions exist now, so they fill in as data arrives.

Status: pending (nothing decided), partial (< 95% decided), complete. Partial numbers are
biased: the first outcomes to resolve are renewals (non-churners), so they are reported but
never used to trigger anything.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from churnguard.config import Settings
from churnguard.features.build import silver_sources
from churnguard.labels.spine import label_month_sql
from churnguard.modeling.dataset import prepare_features
from churnguard.modeling.metrics import evaluate

COMPLETE_SHARE = 0.95


def scores_dir(settings: Settings, month: str):
    return settings.paths.gold / "scores" / f"score_month={month}"


def month_features(settings: Settings, month: str) -> pd.DataFrame | None:
    """Raw features for a month: the batch snapshot if it was scored, else the training
    feature store (same population definition and feature code)."""
    snap = scores_dir(settings, month) / "features.parquet"
    if snap.exists():
        return pd.read_parquet(snap)
    store = settings.paths.gold / "features" / f"label_month={month}" / "data.parquet"
    return pd.read_parquet(store) if store.exists() else None


def labels_now(con: duckdb.DuckDBPyConnection, settings: Settings, month: str) -> pd.DataFrame:
    txn = silver_sources(settings)["txn"]
    data_end = con.execute(f"SELECT max(transaction_date) FROM {txn}").fetchone()[0]
    return con.execute(f"SELECT member_id, is_churn, label_status "
                       f"FROM ({label_month_sql(month, txn, data_end)})").df()


def performance(settings: Settings, month: str, con: duckdb.DuckDBPyConnection,
                model=None) -> dict:
    served = scores_dir(settings, month) / "data.parquet"
    if served.exists():
        scores = pd.read_parquet(served, columns=["member_id", "p_churn"])
        source = "served"
    elif model is not None and (feats := month_features(settings, month)) is not None:
        scores = pd.DataFrame({"member_id": feats["member_id"],
                               "p_churn": model.predict_proba(prepare_features(feats))})
        source = "replayed"
    else:
        return {"month": month, "status": "no_scores"}

    df = scores.merge(labels_now(con, settings, month), on="member_id", how="left")
    decided = df[df.label_status == "decided"]
    share = len(decided) / len(df) if len(df) else 0.0
    status = ("pending" if decided.empty
              else "complete" if share >= COMPLETE_SHARE else "partial")
    out = {"month": month, "status": status, "scores": source, "members": len(df),
           "decided": len(decided), "decided_share": round(share, 4)}
    if not decided.empty and decided.is_churn.nunique() == 2:
        m = evaluate(decided.is_churn.astype(int).to_numpy(), decided.p_churn.to_numpy())
        out["metrics"] = {k: m[k] for k in ("roc_auc", "pr_auc", "log_loss", "ece",
                                            "base_rate", "mean_pred", "lift_top10")}
    return out
