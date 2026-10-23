"""Data access for the retention console. Pure functions over the gold/scores and
reports/monitoring outputs, so the UI stays thin and the logic is unit-tested."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from churnguard.config import Settings
from churnguard.decisioning.retention import select

SCORE_COLS = ["ev_rank", "member_id", "msno", "p_churn", "monthly_revenue", "clv", "cost", "ev",
              "contact", "reason_1", "reason_2", "reason_3", "model_name", "model_version",
              "model_type", "scored_at"]


def scored_months(settings: Settings) -> list[str]:
    return sorted((p.name.removeprefix("score_month=")
                   for p in (settings.paths.gold / "scores").glob("score_month=*")
                   if (p / "data.parquet").exists()), reverse=True)


def load_scores(settings: Settings, month: str) -> pd.DataFrame:
    path = settings.paths.gold / "scores" / f"score_month={month}" / "data.parquet"
    df = pd.read_parquet(path, columns=SCORE_COLS)
    return df.sort_values("ev_rank", ignore_index=True)


def campaign_summary(scores: pd.DataFrame) -> dict:
    c = scores[scores.contact]
    return {
        "members": len(scores),
        "contacted": len(c),
        "contact_share": len(c) / len(scores) if len(scores) else 0.0,
        "expected_net_value": float(c.ev.sum()),
        "campaign_cost": float(c.cost.sum()),
        "mean_p_churn": float(scores.p_churn.mean()) if len(scores) else 0.0,
        "model": f"{scores.model_name.iat[0]} v{scores.model_version.iat[0]} "
                 f"({scores.model_type.iat[0]})" if len(scores) else "-",
    }


def reason_mix(scores: pd.DataFrame) -> pd.DataFrame:
    """Share of contacted members by their top reason (explicit columns: charts given an
    x column avoid Streamlit's deprecated in-place index reset)."""
    c = scores[scores.contact]
    share = c.reason_1.fillna("(none)").value_counts(normalize=True)
    return pd.DataFrame({"reason": share.index, "share": share.to_numpy()})


def value_curve(scores: pd.DataFrame, points: int = 200) -> pd.DataFrame:
    """Cumulative expected net value when contacting members in EV order."""
    cum = scores.sort_values("ev", ascending=False).ev.cumsum().to_numpy()
    idx = np.unique(np.linspace(0, len(cum) - 1, min(points, len(cum))).astype(int))
    return pd.DataFrame({"contacted": idx + 1, "cumulative_expected_value": cum[idx]})


def budget_selection(scores: pd.DataFrame, budget: float | None) -> pd.DataFrame:
    """Re-run the EV/cost selection under a different budget."""
    chosen = select(scores[["ev", "cost"]], budget)
    return scores.assign(contact=chosen.to_numpy())


def crm_export(scores: pd.DataFrame) -> bytes:
    cols = ["ev_rank", "msno", "p_churn", "ev", "reason_1", "reason_2", "reason_3",
            "model_version"]
    return scores.loc[scores.contact, cols].to_csv(index=False).encode()


def find_member(scores: pd.DataFrame, query: str) -> pd.DataFrame:
    q = query.strip()
    if not q:
        return scores.iloc[0:0]
    hit = scores[scores.msno == q]
    if hit.empty and q.isdigit():
        hit = scores[scores.member_id == int(q)]
    return hit


def monitoring_report(settings: Settings, month: str) -> dict | None:
    path = settings.paths.reports / "monitoring" / f"month={month}.json"
    return json.loads(path.read_text()) if path.exists() else None
