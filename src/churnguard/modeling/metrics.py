"""Evaluation metrics shared by every model.

Ranking (ROC AUC, PR AUC), probability quality (log loss = competition metric, Brier, ECE)
and the business view: if retention can contact the top k% riskiest members, how many
churners does that reach (precision/recall/lift at k)?
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

EPS = 1e-15
TOP_FRACTIONS = (0.05, 0.10)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Equal-mass bins: weighted mean |observed rate - mean predicted| per bin."""
    order = np.argsort(p, kind="stable")
    bins = np.array_split(order, n_bins)
    return float(sum(len(b) * abs(y[b].mean() - p[b].mean()) for b in bins if len(b)) / len(y))


def top_k(y: np.ndarray, p: np.ndarray, frac: float) -> dict[str, float]:
    k = max(1, int(round(frac * len(y))))
    top = np.argsort(-p, kind="stable")[:k]
    hits = y[top].sum()
    base = y.mean()
    return {
        "precision": float(hits / k),
        "recall": float(hits / y.sum()) if y.sum() else float("nan"),
        "lift": float(hits / k / base) if base else float("nan"),
    }


def evaluate(y, p) -> dict:
    y = np.asarray(y, dtype=np.int8)
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    out = {
        "n": int(len(y)),
        "base_rate": float(y.mean()),
        "mean_pred": float(p.mean()),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
    }
    for frac in TOP_FRACTIONS:
        for k, v in top_k(y, p, frac).items():
            out[f"{k}_top{int(frac * 100)}"] = v
    return out
