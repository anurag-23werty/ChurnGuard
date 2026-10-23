"""Out-of-time evaluation across every scored month, not just one.

For each scored month M, every candidate is trained on all earlier months only and scored
on M. One month can flatter a model (LightGBM won Feb 2017 by 0.022 AUC and lost Dec 2016
by 0.051), so selection uses the mean *and* the worst month.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from churnguard.logging_utils import get_logger
from churnguard.modeling.metrics import evaluate
from churnguard.modeling.tune import rolling_folds

log = get_logger(__name__)

SUMMARY_METRICS = ("roc_auc", "pr_auc", "log_loss", "ece", "lift_top10")


def out_of_time(candidates: dict[str, Callable[[], object]], X: pd.DataFrame, y: pd.Series,
                month: pd.Series, months: list[str]) -> tuple[pd.DataFrame, dict]:
    """Per (candidate, scored month) metrics, plus out-of-time predictions per candidate."""
    month = np.asarray(month)
    rows, oof = [], {name: [] for name in candidates}
    for fold in rolling_folds(months):
        tr, va = np.isin(month, fold.train_months), month == fold.valid_month
        for name, factory in candidates.items():
            p = factory().fit(X[tr], y[tr]).predict_proba(X[va])
            m = evaluate(y[va].to_numpy(), p)
            rows.append({"candidate": name, "month": fold.valid_month,
                         **{k: m[k] for k in SUMMARY_METRICS}})
            oof[name].append(pd.DataFrame({"month": fold.valid_month, "p": p,
                                           "y": y[va].to_numpy()}))
            log.info("[oot] %-20s %s AUC %.4f PR-AUC %.4f logloss %.4f", name,
                     fold.valid_month, m["roc_auc"], m["pr_auc"], m["log_loss"])
    return pd.DataFrame(rows), {k: pd.concat(v, ignore_index=True) for k, v in oof.items()}


def summarise(per_month: pd.DataFrame, baseline: str) -> pd.DataFrame:
    g = per_month.groupby("candidate")
    out = pd.DataFrame({
        "mean_roc_auc": g.roc_auc.mean(), "worst_roc_auc": g.roc_auc.min(),
        "mean_pr_auc": g.pr_auc.mean(), "worst_pr_auc": g.pr_auc.min(),
        "mean_log_loss": g.log_loss.mean(), "mean_ece": g.ece.mean(),
        "mean_lift_top10": g.lift_top10.mean(),
    })
    base = per_month[per_month.candidate == baseline].set_index("month")["pr_auc"]
    out["months_beating_baseline_pr_auc"] = (
        per_month.assign(b=per_month.month.map(base))
        .assign(win=lambda d: d.pr_auc > d.b).groupby("candidate").win.sum())
    return out.sort_values("mean_pr_auc", ascending=False)


def calibration_helps(oof: pd.DataFrame) -> tuple[bool, dict]:
    """Fit isotonic on all but the last scored month, test on the last. Calibrate only if
    it lowers both log loss and ECE out of time."""
    last = sorted(oof.month.unique())[-1]
    fit, hold = oof[oof.month != last], oof[oof.month == last]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(fit.p, fit.y)
    raw, cal = evaluate(hold.y.to_numpy(), hold.p.to_numpy()), evaluate(
        hold.y.to_numpy(), iso.predict(hold.p))
    detail = {"holdout_month": last, "raw_log_loss": raw["log_loss"], "raw_ece": raw["ece"],
              "cal_log_loss": cal["log_loss"], "cal_ece": cal["ece"]}
    return cal["log_loss"] < raw["log_loss"] and cal["ece"] < raw["ece"], detail
