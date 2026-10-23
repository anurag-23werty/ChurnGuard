"""Value-based retention targeting.

Contacting member i is worth

    EV_i = p_i * save_rate * CLV_i  -  cost_i

  p_i        calibrated churn probability (the model)
  save_rate  share of would-be churners an offer retains. KKBox has NO campaign data, so
             this is a scenario assumption, not an estimate; measure it with an A/B test
             (random holdout of targeted members) before trusting any money figure.
  CLV_i      value of keeping the member: monthly revenue x expected retained months over a
             finite horizon, discounted by the baseline monthly churn rate
  cost_i     contact cost + incentive (one free month here: proportional to revenue)

Targeting by EV under a budget is a knapsack; with costs this similar, greedy by EV/cost
is near-optimal. Members with EV <= 0 are never contacted, whatever the budget.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from churnguard.config import RetentionConfig as RetentionPolicy

DAYS_PER_MONTH = 30.0

__all__ = ["RetentionPolicy", "expected_value", "monthly_revenue", "realised_value",
           "retained_months", "select", "targeting_curve"]


def monthly_revenue(last_amount_paid, last_plan_days) -> np.ndarray:
    """Revenue per 30 days implied by the member's current plan (0 for free plans)."""
    paid = np.nan_to_num(np.asarray(last_amount_paid, dtype=float))
    days = np.asarray(last_plan_days, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rev = np.where(days > 0, paid / days * DAYS_PER_MONTH, 0.0)
    return np.clip(np.nan_to_num(rev), 0, None)


def retained_months(policy: RetentionPolicy) -> float:
    """Expected paying months over the horizon for a saved member: sum_t (1-q)^t."""
    q = policy.baseline_monthly_churn
    return float(sum((1 - q) ** t for t in range(policy.horizon_months)))


def expected_value(p, revenue, policy: RetentionPolicy) -> pd.DataFrame:
    p, revenue = np.asarray(p, dtype=float), np.asarray(revenue, dtype=float)
    clv = revenue * retained_months(policy)
    cost = policy.contact_cost + policy.incentive_months * revenue
    benefit = p * policy.save_rate * clv
    return pd.DataFrame({"p_churn": p, "monthly_revenue": revenue, "clv": clv,
                         "expected_benefit": benefit, "cost": cost, "ev": benefit - cost})


def select(ev: pd.DataFrame, budget: float | None = None) -> pd.Series:
    """Boolean mask of members to contact: positive EV, greedy by EV/cost within budget."""
    eligible = ev["ev"] > 0
    order = (ev["ev"] / ev["cost"]).where(eligible).sort_values(ascending=False).dropna()
    chosen = pd.Series(False, index=ev.index)
    if budget is None:
        chosen[order.index] = True
        return chosen
    spent = ev.loc[order.index, "cost"].cumsum()
    chosen[spent[spent <= budget].index] = True
    return chosen


def realised_value(ev: pd.DataFrame, chosen: pd.Series, y, policy: RetentionPolicy) -> float:
    """Value of a targeting decision against *actual* outcomes (backtest): each contacted
    member who really churned is saved with probability save_rate."""
    y = np.asarray(y, dtype=float)
    c = chosen.to_numpy()
    return float((y[c] * policy.save_rate * ev["clv"].to_numpy()[c]).sum()
                 - ev["cost"].to_numpy()[c].sum())


def targeting_curve(ev: pd.DataFrame, y, policy: RetentionPolicy,
                    fractions=(0.01, 0.02, 0.05, 0.10, 0.20, 0.30)) -> pd.DataFrame:
    """Realised net value when contacting the top x% under three rankings."""
    rng = np.random.default_rng(0)
    rankings = {
        "by_ev": ev["ev"].to_numpy(),
        "by_risk": ev["p_churn"].to_numpy(),
        "random": rng.random(len(ev)),
    }
    rows = []
    for frac in fractions:
        k = int(round(frac * len(ev)))
        row = {"contact_fraction": frac, "contacted": k}
        for name, score in rankings.items():
            top = np.zeros(len(ev), dtype=bool)
            top[np.argsort(-score, kind="stable")[:k]] = True
            row[name] = realised_value(ev, pd.Series(top, index=ev.index), y, policy)
        rows.append(row)
    return pd.DataFrame(rows)
